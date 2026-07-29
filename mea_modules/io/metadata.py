"""Describe a Maxwell recording without reading its traces.

A Maxwell ``.raw.h5`` is almost entirely raw voltage — a 21-segment AxonTracking
scan runs to ~19 GB — but everything needed to say *what was recorded* sits in a
few kilobytes of settings alongside it. :func:`extract_metadata` reads only that
part for one segment: assay inputs, the data-store wall-clock window, the
recording's frame numbers, trigger settings, probe geometry, sampling rate,
gains, and the channel-to-electrode map.

Metadata lives in two places that do not always agree, so both are read and both
are reported rather than silently reconciled:

* the HDF5 file itself (``/assay``, ``/wellplate``, ``/data_store/dataNNNN``,
  ``/wells/<well>/<rec>``) — the instrument's own record, and
* the SpikeInterface extractor — what downstream analysis will actually see.

The one bulk dataset touched is ``frame_nos`` (one uint64 per sample, ~19 MB for
a two-minute segment). It is read because it is the only way to find dropped
samples: Maxwell records a gap in the frame counter rather than an explicit
marker, so a segment that looks contiguous by sample count can in fact be
several disjoint epochs. ``raw`` is never opened.

Everything returned is JSON-serializable, so the result can go straight to
:func:`save_metadata`.

Pure library: no argparse, no printing, no ``__main__``.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from .hdf5_plugin import set_plugin_path
from .load import describe_segment, list_maxwell_streams, load_maxwell

logger = logging.getLogger(__name__)

# Maxwell stores wall-clock timestamps as bare integers with no unit recorded,
# so magnitude is the only clue to the scale. Thresholds are ordered coarse-first.
_EPOCH_UNITS = ((10**18, 1e9, "ns"), (10**15, 1e6, "us"), (10**12, 1e3, "ms"))

# Two sampling rates count as the same rate when they agree this closely; the
# instrument reports 20000.0 in one place and a measured rate in another.
_SAMPLING_TOLERANCE_HZ = 0.5

# Triggered acquisition writes one contiguous epoch per trigger event — an
# AxonTracking segment routinely has thousands. Listing them all would make this
# description larger than everything else combined, so the list is capped and
# the distribution is summarized instead.
_MAX_EPOCHS_REPORTED = 64

# Directory levels MaxLab writes above the file: <dataset>/<date>/<plate>/<assay>/<run>/data.raw.h5
_PATH_CONTEXT_LEVELS = (("dataset", -6), ("date", -5), ("plate", -4), ("assay", -3), ("run", -2))

# Scalar settings under a recording's /settings group, mapped to output names.
_SETTINGS_FIELDS = (
    ("gain", "gain"),
    ("hpf", "hpf_hz"),
    ("lsb", "lsb_v"),
    ("sampling", "sampling_frequency_hz"),
    ("spike_threshold", "spike_threshold"),
)


# --------------------------------------------------------------------------- #
# small readers
# --------------------------------------------------------------------------- #


def _decode(value):
    """HDF5 fixed-width strings come back as bytes; everything else passes through."""
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    return value


def _jsonify(value):
    """Recursively turn numpy scalars/arrays and HDF5 bytes into JSON types."""
    import numpy as np

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (bytes, bytearray)):
        return _decode(value)
    if isinstance(value, np.ndarray):
        return _jsonify(value.tolist())
    if isinstance(value, np.generic):
        return _jsonify(value.item())
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonify(item) for item in value]
    return str(value)


def _read_scalar(group, path):
    """Value of a one-element HDF5 dataset, or None if absent or unreadable.

    Maxwell writes most settings as shape-(1,) datasets rather than attributes,
    and a missing key is normal across format versions — never an error here.
    """
    import numpy as np

    try:
        if path not in group:
            return None
        value = group[path][()]
    except (KeyError, OSError, TypeError, ValueError):
        return None

    if isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        value = value.ravel()[0]
    if isinstance(value, np.generic):
        value = value.item()
    return _decode(value)


def _read_scalars(group, names=None):
    """Every one-element dataset directly under `group`, as a plain dict."""
    if group is None:
        return {}
    try:
        keys = list(group.keys()) if names is None else list(names)
    except (AttributeError, OSError):
        return {}

    out = {}
    for key in keys:
        value = _read_scalar(group, key)
        if value is not None:
            out[str(key)] = value
    return out


def _as_int(value):
    """int(value) or None — used on fields that may legitimately be absent."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_positive_float(value):
    """float(value) when it is finite and > 0, else None.

    Rates and gains are only meaningful when positive; a zero or NaN from the
    file means "not recorded", not "zero".
    """
    import numpy as np

    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out) or out <= 0.0:
        return None
    return out


def _warn(warnings, message):
    """Record a non-fatal metadata gap in both the log and the payload."""
    logger.warning("%s", message)
    warnings.append(str(message))


# --------------------------------------------------------------------------- #
# identifiers and timestamps
# --------------------------------------------------------------------------- #


def _well_label(value):
    """Normalize a well identifier ("0", 0, "well0") to Maxwell's "wellNNN"."""
    text = str(value).strip()
    if text.lower().startswith("well") and text[4:].isdigit():
        return f"well{int(text[4:]):03d}"
    if text.isdigit():
        return f"well{int(text):03d}"
    return text


def _trailing_int(value):
    """Numeric suffix of an id like "well000" or "rec0007", or None."""
    digits = "".join(char for char in str(value or "") if char.isdigit())
    return int(digits) if digits else None


def _epoch_unit(values):
    """Infer the unit of Maxwell's epoch timestamps from their magnitude.

    The file records ``start_time``/``stop_time`` as bare integers. A 2026-era
    timestamp is ~1.8e9 in seconds and ~1.8e12 in milliseconds, so the number of
    digits identifies the scale unambiguously for any plausible recording date.
    """
    magnitudes = [abs(int(value)) for value in values if value is not None]
    biggest = max(magnitudes) if magnitudes else 0
    for threshold, divisor, unit in _EPOCH_UNITS:
        if biggest >= threshold:
            return divisor, unit
    return 1.0, "s"


def _utc_iso(value, divisor):
    """ISO-8601 UTC string for a raw epoch timestamp, or None if out of range."""
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(float(value) / float(divisor), tz=timezone.utc).isoformat()
    except (OSError, OverflowError, ValueError):
        return None


def _timestamp_block(start_raw, stop_raw):
    """Start/stop timestamps expanded into raw values, UTC strings and a duration."""
    divisor, unit = _epoch_unit([start_raw, stop_raw])
    block = {
        "timestamp_unit": f"{unit}_since_epoch",
        "start_time_raw": _as_int(start_raw),
        "stop_time_raw": _as_int(stop_raw),
        "start_time_utc": _utc_iso(start_raw, divisor),
        "stop_time_utc": _utc_iso(stop_raw, divisor),
    }
    if start_raw is not None and stop_raw is not None:
        block["duration_wall_clock_s"] = float((float(stop_raw) - float(start_raw)) / divisor)
    return block


def _path_context(h5_path):
    """Provenance MaxLab encodes in the directory tree but not in the file.

    Runs are written as ``<dataset>/<date>/<plate>/<assay>/<run>/data.raw.h5``,
    so the parent directories are the only record of which plate and assay a
    given ``data.raw.h5`` belongs to. Levels that do not exist are omitted.
    """
    resolved = Path(h5_path).expanduser().resolve()
    context = {"filename": resolved.name, "parent_dir": resolved.parent.name}
    parts = resolved.parts
    if len(parts) >= 6:
        for label, index in _PATH_CONTEXT_LEVELS:
            context[label] = parts[index]
    return context


# --------------------------------------------------------------------------- #
# per-block readers
# --------------------------------------------------------------------------- #


def _file_block(h5, h5_path):
    """File identity: path, size and the three version stamps at the root."""
    resolved = Path(h5_path).expanduser().resolve()
    block = {
        "path": str(resolved),
        "filename": resolved.name,
        "context": _path_context(resolved),
    }
    try:
        block["size_bytes"] = int(resolved.stat().st_size)
    except OSError:
        block["size_bytes"] = None
    # /version is a date-like integer written as a string; the other two are
    # dotted version strings and stay as text.
    version = _read_scalar(h5, "version")
    if version is not None:
        block["format_version"] = _as_int(version) or version
    for name in ("hdf_version", "mxw_version"):
        value = _read_scalar(h5, name)
        if value is not None:
            block[name] = value
    return block


def _assay_block(h5, stream_id, segment_index, warnings):
    """Assay identity plus the electrode configuration this segment recorded.

    ``/assay/inputs/electrodes`` is a JSON blob listing every configuration the
    whole assay will run — tens of thousands of electrode ids. It is summarized
    rather than copied wholesale: the assay's own properties, the "fixed"
    electrodes that stay routed for every configuration (an AxonTracking scan's
    seed sites), and the single scan configuration belonging to this segment.
    """
    if "assay" not in h5:
        return {}

    assay = h5["assay"]
    block = _read_scalars(assay, names=("run_id", "script_id"))

    record_time = _read_scalar(assay, "inputs/record_time")
    if record_time is not None:
        block["record_time_s"] = _as_int(record_time) or record_time

    raw_config = _read_scalar(assay, "inputs/electrodes")
    if raw_config is None:
        return block

    try:
        config = json.loads(raw_config)
    except (TypeError, ValueError) as exc:
        _warn(warnings, f"could not parse /assay/inputs/electrodes as JSON: {exc}")
        return block

    properties = config.get("properties")
    if isinstance(properties, dict):
        block["properties"] = properties

    wells = config.get("electrodes")
    if not isinstance(wells, dict):
        return block

    # The blob keys wells by bare id ("0"), not by the "well000" stream label.
    well_index = _trailing_int(stream_id)
    entry = wells.get(str(well_index)) if well_index is not None else None
    if entry is None and len(wells) == 1:
        entry = next(iter(wells.values()))
    if not isinstance(entry, dict):
        return block

    fixed = entry.get("fixed")
    if isinstance(fixed, list):
        block["fixed_electrode_groups"] = fixed
        block["fixed_electrode_count"] = int(sum(len(group) for group in fixed if isinstance(group, list)))

    scan = entry.get("scan")
    if isinstance(scan, list):
        block["scan_config_count"] = int(len(scan))
        if 0 <= segment_index < len(scan):
            block["scan_electrodes"] = scan[segment_index]
            block["scan_electrode_count"] = int(len(scan[segment_index]))
        else:
            _warn(
                warnings,
                f"segment index {segment_index} has no matching assay scan config "
                f"({len(scan)} configs declared)",
            )
    return block


def _wellplate_block(h5, stream_id):
    """Plate identity and the culture notes recorded against this well."""
    if "wellplate" not in h5:
        return {}
    plate = h5["wellplate"]
    block = _read_scalars(plate, names=("id", "variant", "version"))
    label = _well_label(stream_id)
    if label in plate:
        block["well"] = _read_scalars(plate[label])
    return block


def _data_store_block(h5, stream_id, rec_name, warnings):
    """Wall-clock window from ``/data_store``, for this segment and its stream.

    ``/data_store/dataNNNN`` is where MaxLab logs when acquisition actually ran.
    Entries are matched on ``well_id``/``recording_id`` rather than on position,
    because the ``dataNNNN`` ordering is not guaranteed to follow ``recNNNN``
    once a plate holds more than one well.
    """
    if "data_store" not in h5:
        return {}

    data_store = h5["data_store"]
    want_well = _trailing_int(stream_id)
    want_rec = _trailing_int(rec_name)

    matched_key = None
    matched_entry = None
    stream_starts = []
    stream_stops = []
    stream_durations = []

    for key in sorted(data_store.keys()):
        entry = data_store[str(key)]
        well_id = _as_int(_read_scalar(entry, "well_id"))
        if want_well is not None and well_id is not None and well_id != want_well:
            continue

        start_raw = _as_int(_read_scalar(entry, "start_time"))
        stop_raw = _as_int(_read_scalar(entry, "stop_time"))
        if start_raw is not None:
            stream_starts.append(start_raw)
        if stop_raw is not None:
            stream_stops.append(stop_raw)
        if start_raw is not None and stop_raw is not None:
            stream_durations.append(stop_raw - start_raw)

        rec_id = _as_int(_read_scalar(entry, "recording_id"))
        if matched_entry is None and (want_rec is None or rec_id is None or rec_id == want_rec):
            matched_key, matched_entry = str(key), entry

    if matched_entry is None:
        _warn(warnings, f"no /data_store entry for stream {stream_id!r} rec {rec_name!r}")
        return {}

    block = {
        "key": matched_key,
        "well_id": _as_int(_read_scalar(matched_entry, "well_id")),
        "recording_id": _as_int(_read_scalar(matched_entry, "recording_id")),
    }
    block.update(
        _timestamp_block(
            _read_scalar(matched_entry, "start_time"),
            _read_scalar(matched_entry, "stop_time"),
        )
    )

    if stream_starts and stream_stops:
        divisor, unit = _epoch_unit(stream_starts + stream_stops)
        block["stream"] = {
            "entry_count": int(len(stream_durations) or len(stream_starts)),
            "timestamp_unit": f"{unit}_since_epoch",
            "start_time_utc": _utc_iso(min(stream_starts), divisor),
            "stop_time_utc": _utc_iso(max(stream_stops), divisor),
            "span_s": float((max(stream_stops) - min(stream_starts)) / divisor),
            "recorded_duration_s": float(sum(stream_durations) / divisor),
        }
    return block


def _frames_block(routed, sampling_hz, warnings):
    """Frame numbers for the segment, split into contiguous epochs.

    ``frame_nos`` is the instrument's global sample counter. Where the counter
    jumps, the samples either side are not adjacent in time, so a run of
    consecutive frame numbers — not the sample count — is the real unit of
    continuous data. Under triggered acquisition every trigger event produces
    one such run, which is why a normal AxonTracking segment reports thousands
    of epochs and a large ``n_missing_frames``: that is the instrument skipping
    the untriggered stretches, not data loss.

    Each reported epoch carries its sample range and its offset in seconds from
    the start of the segment, which is what downstream splitting needs.
    """
    import numpy as np

    if routed is None or "frame_nos" not in routed:
        _warn(warnings, "segment has no routed/frame_nos dataset; frame timing unavailable")
        return {}

    frame_nos = np.asarray(routed["frame_nos"][()], dtype=np.int64)
    n_samples = int(frame_nos.size)
    block = {"n_samples": n_samples, "n_contiguous_epochs": 0, "epochs": []}
    if n_samples == 0:
        return block

    first = int(frame_nos[0])
    last = int(frame_nos[-1])
    frame_span = int(last - first + 1)
    block["frame_no_start"] = first
    block["frame_no_end"] = last
    block["frame_span"] = frame_span
    block["n_missing_frames"] = int(frame_span - n_samples)

    # A gap is any step in the counter other than +1; the split points bound the
    # contiguous runs between them.
    split_points = np.flatnonzero(np.diff(frame_nos) != 1) + 1
    run_starts = np.concatenate(([0], split_points)).astype(np.int64)
    run_ends = np.concatenate((split_points, [n_samples])).astype(np.int64)
    run_lengths = run_ends - run_starts
    keep = run_lengths > 0
    run_starts = run_starts[keep]
    run_ends = run_ends[keep]
    run_lengths = run_lengths[keep]

    block["n_contiguous_epochs"] = int(run_starts.size)
    if run_lengths.size:
        block["epoch_samples_min"] = int(run_lengths.min())
        block["epoch_samples_max"] = int(run_lengths.max())
        block["epoch_samples_median"] = float(np.median(run_lengths))

    epochs = []
    for epoch_index in range(min(int(run_starts.size), _MAX_EPOCHS_REPORTED)):
        run_start = int(run_starts[epoch_index])
        run_end = int(run_ends[epoch_index])
        epoch = {
            "epoch_index": int(epoch_index),
            "start_sample": run_start,
            "end_sample": run_end,
            "n_samples": int(run_end - run_start),
            "frame_no_start": int(frame_nos[run_start]),
            "frame_no_end": int(frame_nos[run_end - 1]),
        }
        if sampling_hz is not None:
            # Seconds relative to the segment's own first frame. The end bound is
            # inclusive of the last sample, hence the +1.
            start_s = float((int(frame_nos[run_start]) - first) / sampling_hz)
            end_s = float((int(frame_nos[run_end - 1]) - first + 1) / sampling_hz)
            epoch["relative_start_s"] = start_s
            epoch["relative_end_s"] = end_s
            epoch["duration_s"] = float(max(0.0, end_s - start_s))
        epochs.append(epoch)

    block["epochs"] = epochs
    block["epochs_reported"] = int(len(epochs))
    block["epochs_truncated"] = bool(len(epochs) < int(run_starts.size))
    if sampling_hz is not None:
        # Data actually captured, versus the wall-clock window it was drawn from.
        block["duration_samples_s"] = float(n_samples / sampling_hz)
        block["duration_span_s"] = float(frame_span / sampling_hz)
    return block


def _trigger_block(routed):
    """Spike-trigger settings recorded on the routed group."""
    if routed is None:
        return {}

    block = {}
    for name, cast in (
        ("triggered", _as_int),
        ("trigger_pre", _as_int),
        ("trigger_post", _as_int),
        ("trigger_minamp", float),
        ("trigger_maxamp", float),
    ):
        value = _read_scalar(routed, name)
        if value is None:
            continue
        try:
            block[name] = cast(value)
        except (TypeError, ValueError):
            continue

    if "triggered" in block:
        block["triggered"] = bool(block["triggered"])
    if "trigger_channels" in routed:
        try:
            channels = [int(value) for value in routed["trigger_channels"][()].ravel()]
        except (OSError, TypeError, ValueError):
            channels = []
        if channels:
            block["trigger_channels"] = channels
            block["n_trigger_channels"] = int(len(channels))
    return block


def _channel_map_from_h5(settings):
    """Channel/electrode/x/y mapping from the recording's own settings group.

    Preferred over SpikeInterface's ``contact_vector`` because it is present
    whether or not the extractor can be opened, and it is the mapping the
    instrument actually routed.
    """
    import numpy as np

    if settings is None or "mapping" not in settings:
        return {}
    try:
        mapping = np.asarray(settings["mapping"][()])
    except (OSError, TypeError, ValueError):
        return {}

    names = tuple(str(name) for name in (getattr(mapping.dtype, "names", None) or ()))
    if "channel" not in names:
        return {}

    block = {"source": "h5_settings_mapping", "channels": [int(v) for v in mapping["channel"]]}
    block["num_channels"] = int(len(block["channels"]))
    if "electrode" in names:
        block["electrodes"] = [int(v) for v in mapping["electrode"]]
    for axis in ("x", "y"):
        if axis in names:
            block[f"{axis}_um"] = [float(v) for v in mapping[axis]]
    return block


def _channel_map_from_recording(recording):
    """Fallback channel map built from SpikeInterface's contact_vector."""
    import numpy as np

    try:
        contacts = recording.get_property("contact_vector")
    except (AttributeError, KeyError, ValueError):
        return {}
    if contacts is None:
        return {}

    names = tuple(str(name) for name in (getattr(contacts.dtype, "names", None) or ()))
    block = {"source": "spikeinterface_contact_vector"}
    try:
        block["channels"] = [int(value) for value in recording.get_channel_ids()]
    except (AttributeError, TypeError, ValueError):
        block["channels"] = [str(value) for value in recording.get_channel_ids()]
    block["num_channels"] = int(len(block["channels"]))
    if "electrode" in names:
        block["electrodes"] = [int(value) for value in contacts["electrode"]]
    try:
        locations = np.asarray(recording.get_channel_locations(), dtype=float)
        block["x_um"] = [float(value) for value in locations[:, 0]]
        block["y_um"] = [float(value) for value in locations[:, 1]]
    except (AttributeError, IndexError, TypeError, ValueError):
        pass
    return block


def _probe_block(channel_map, recording):
    """Geometry of the routed electrodes: extent, pitch and contact shape.

    Extent and pitch come from the channel map so they are available without
    SpikeInterface; contact shape, shank count and manufacturer are extractor
    annotations and are added only when a recording was opened.
    """
    import numpy as np

    block = {"units": "um"}
    xs = channel_map.get("x_um")
    ys = channel_map.get("y_um")
    if xs and ys:
        block["num_contacts"] = int(len(xs))
        for axis, values in (("x", xs), ("y", ys)):
            array = np.asarray(values, dtype=float)
            block[f"{axis}_range_um"] = [float(array.min()), float(array.max())]
            block[f"{axis}_extent_um"] = float(array.max() - array.min())
            # Pitch is the smallest non-zero step between occupied coordinates —
            # the routed subset is sparse, so a mean spacing would be meaningless.
            steps = np.diff(np.unique(array))
            steps = steps[steps > 0]
            if steps.size:
                block[f"pitch_{axis}_um"] = float(steps.min())

    if recording is None:
        return block

    try:
        contacts = recording.get_property("contact_vector")
    except (AttributeError, KeyError, ValueError):
        contacts = None
    if contacts is not None:
        names = tuple(str(name) for name in (getattr(contacts.dtype, "names", None) or ()))
        if "contact_shapes" in names and len(contacts):
            block["contact_shape"] = str(_decode(contacts["contact_shapes"][0]))
        for name, key in (("width", "contact_width_um"), ("height", "contact_height_um")):
            if name in names and len(contacts):
                block[key] = float(contacts[name][0])
        if "shank_ids" in names and len(contacts):
            block["num_shanks"] = int(len({str(_decode(value)) for value in contacts["shank_ids"]}))

    probes_info = None
    get_annotation = getattr(recording, "get_annotation", None)
    if callable(get_annotation):
        try:
            probes_info = get_annotation("probes_info")
        except (KeyError, ValueError):
            probes_info = None
    if isinstance(probes_info, (list, tuple)) and probes_info and isinstance(probes_info[0], dict):
        manufacturer = probes_info[0].get("manufacturer")
        if manufacturer:
            block["manufacturer"] = str(manufacturer)
    return block


def _gain_block(settings_values, recording, warnings):
    """Everything needed to convert raw counts to microvolts.

    The file records ``lsb`` in volts; SpikeInterface exposes the same number as
    a per-channel ``gain_to_uV``. Both are reported so a mismatch is visible
    instead of being averaged away.
    """
    import numpy as np

    wanted = ("gain", "hpf_hz", "lsb_v", "spike_threshold")
    block = {key: settings_values[key] for key in wanted if key in settings_values}

    lsb_v = _as_positive_float(block.get("lsb_v"))
    if lsb_v is not None:
        block["gain_to_uV_from_lsb"] = float(lsb_v * 1e6)

    if recording is None:
        return block

    try:
        gains = np.asarray(recording.get_channel_gains(), dtype=float)
        offsets = np.asarray(recording.get_channel_offsets(), dtype=float)
    except (AttributeError, TypeError, ValueError) as exc:
        _warn(warnings, f"could not read channel gains from the extractor: {exc}")
        return block

    if gains.size:
        uniform = bool(np.allclose(gains, gains[0]))
        block["gain_uniform"] = uniform
        block["gain_to_uV"] = float(gains[0]) if uniform else [float(value) for value in gains]
    if offsets.size:
        uniform = bool(np.allclose(offsets, offsets[0]))
        block["offset_uniform"] = uniform
        block["offset_to_uV"] = float(offsets[0]) if uniform else [float(value) for value in offsets]

    lsb_gain = block.get("gain_to_uV_from_lsb")
    si_gain = block.get("gain_to_uV")
    if lsb_gain is not None and isinstance(si_gain, float):
        block["gain_matches_lsb"] = bool(abs(si_gain - lsb_gain) <= 1e-6 * max(1.0, abs(lsb_gain)))
    return block


def _stream_sampling_hz(h5, stream_id):
    """Sampling rate for the whole stream, read from ``/data_store``.

    Every ``dataNNNN`` entry for the well carries its own ``settings/sampling``;
    the median is used so a single corrupt entry cannot move the answer. Falls
    back to a sampling-frequency attribute at stream, wells or root scope for
    files that predate the settings group.
    """
    import numpy as np

    target = _well_label(stream_id)
    rates = []
    if "data_store" in h5:
        data_store = h5["data_store"]
        for key in sorted(str(name) for name in data_store.keys()):
            entry = data_store[key]
            well_id = _read_scalar(entry, "well_id")
            label = _well_label(well_id) if well_id is not None else None
            if label is not None and label != target:
                continue
            rate = _as_positive_float(_read_scalar(entry, "settings/sampling"))
            if rate is not None:
                rates.append(rate)
    if rates:
        return float(np.median(np.asarray(rates, dtype=float)))

    # Spellings seen across MaxLab versions; matched case-insensitively.
    attr_names = (
        "sampling_frequency",
        "sampling rate",
        "sampling_rate",
        "samplerate",
        "sample_rate",
        "sampling",
        "fs",
    )
    candidates = []
    if "wells" in h5:
        wells = h5["wells"]
        if target in wells:
            candidates.append(wells[target])
        candidates.append(wells)
    candidates.extend([h5.get("data_store"), h5])
    for obj in candidates:
        if obj is None:
            continue
        try:
            attrs = {str(key).lower(): value for key, value in obj.attrs.items()}
        except (AttributeError, OSError):
            continue
        for name in attr_names:
            rate = _as_positive_float(attrs.get(name))
            if rate is not None:
                return rate
    return None


def _sampling_block(rec_settings_hz, stream_hz, recording_hz):
    """Reconcile the sampling rates the file and the extractor each report."""
    block = {
        "h5_rec_settings_hz": rec_settings_hz,
        "h5_stream_median_hz": stream_hz,
        "spikeinterface_hz": recording_hz,
    }
    for source, value in (
        ("h5_rec_settings", rec_settings_hz),
        ("spikeinterface", recording_hz),
        ("h5_stream_median", stream_hz),
    ):
        if value is not None:
            block["sampling_frequency_hz"] = float(value)
            block["source"] = source
            break

    if rec_settings_hz is not None and recording_hz is not None:
        block["matches_spikeinterface"] = bool(abs(rec_settings_hz - recording_hz) <= _SAMPLING_TOLERANCE_HZ)
    return block


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #


def extract_metadata(h5_path, stream_id=None, rec_name=None, hdf5_plugin_path=None):
    """Describe one (well, recording) segment of a Maxwell file as a plain dict.

    `stream_id` is a well ("well000") and `rec_name` a recording within it
    ("rec0000"); either may be omitted, in which case the first available one is
    used — the same resolution :func:`~mea_modules.io.load.load_maxwell` applies.
    `hdf5_plugin_path` is the directory holding Maxwell's HDF5 compression
    plugin, needed only to open the SpikeInterface extractor.

    The result is JSON-serializable and grouped by topic: ``file``, ``assay``,
    ``wellplate``, ``data_store``, ``timing``, ``frames``, ``trigger``,
    ``sampling``, ``gains``, ``probe``, ``channel_map`` and ``recording``. Blocks
    that a given file does not carry are empty rather than missing, and anything
    that could not be read is described in ``warnings`` instead of raising — a
    partial description is more useful than none when triaging a suspect file.

    Traces are never read. The heaviest access is the segment's frame counter,
    so this stays fast on a multi-GB scan.
    """
    import h5py

    h5_path = Path(h5_path).expanduser()
    if not h5_path.exists():
        raise FileNotFoundError(f"no such Maxwell file: {h5_path}")

    # Resolve the plugin up front: h5py needs it for any compressed dataset, and
    # an explicitly wrong path should fail here rather than mid-read.
    set_plugin_path(hdf5_plugin_path)

    layout = list_maxwell_streams(h5_path)
    streams = layout["streams"]

    if stream_id is None:
        stream_id = next(iter(streams))
    elif stream_id not in streams:
        raise ValueError(f"stream {stream_id!r} not in file; available: {sorted(streams)}")

    rec_names = streams[stream_id]
    if rec_name is None:
        # The pre-2016 format carries no rec_name at all.
        rec_name = rec_names[0] if rec_names else None
    elif rec_names and rec_name not in rec_names:
        raise ValueError(f"rec_name {rec_name!r} not in stream {stream_id!r}; available: {rec_names}")
    segment_index = rec_names.index(rec_name) if rec_name in rec_names else 0

    logger.debug("extracting metadata for %s stream=%s rec=%s", h5_path, stream_id, rec_name)

    warnings = []
    metadata = {
        "stream_id": stream_id,
        "rec_name": rec_name,
        "segment_index": int(segment_index),
        "format_version": layout["version"],
        "segment_count": int(len(rec_names) or 1),
    }

    settings_values = {}
    channel_map = {}
    stream_hz = None

    with h5py.File(str(h5_path), mode="r") as h5:
        metadata["file"] = _file_block(h5, h5_path)
        metadata["assay"] = _assay_block(h5, stream_id, segment_index, warnings)
        metadata["wellplate"] = _wellplate_block(h5, stream_id)
        metadata["data_store"] = _data_store_block(h5, stream_id, rec_name, warnings)

        # /wells is the per-recording view; the old MaxOne format lacks it, so
        # fall back to the matching /data_store entry, which has the same layout.
        segment_group = None
        if "wells" in h5 and stream_id in h5["wells"]:
            well = h5["wells"][stream_id]
            if rec_name is not None and rec_name in well:
                segment_group = well[rec_name]
        if segment_group is None:
            key = metadata["data_store"].get("key")
            if key and "data_store" in h5 and key in h5["data_store"]:
                segment_group = h5["data_store"][key]
        if segment_group is None:
            _warn(warnings, f"no recording group found for stream {stream_id!r} rec {rec_name!r}")

        routed = None
        settings = None
        if segment_group is not None:
            routed = segment_group.get("groups/routed")
            settings = segment_group.get("settings")
            raw_settings = _read_scalars(settings, names=[name for name, _ in _SETTINGS_FIELDS])
            settings_values = {
                out_name: raw_settings[in_name]
                for in_name, out_name in _SETTINGS_FIELDS
                if in_name in raw_settings
            }

            metadata["timing"] = _timestamp_block(
                _read_scalar(segment_group, "start_time"),
                _read_scalar(segment_group, "stop_time"),
            )
            metadata["timing"]["well_id"] = _as_int(_read_scalar(segment_group, "well_id"))
            metadata["timing"]["recording_id"] = _as_int(_read_scalar(segment_group, "recording_id"))
        else:
            metadata["timing"] = {}

        rec_settings_hz = _as_positive_float(settings_values.get("sampling_frequency_hz"))
        stream_hz = _stream_sampling_hz(h5, stream_id)

        metadata["frames"] = _frames_block(routed, rec_settings_hz or stream_hz, warnings)
        metadata["trigger"] = _trigger_block(routed)
        channel_map = _channel_map_from_h5(settings)

        if routed is not None and "raw" in routed and not channel_map.get("num_channels"):
            try:
                channel_map["num_channels"] = int(routed["raw"].shape[0])
            except (OSError, TypeError, ValueError):
                pass

    recording = None
    try:
        recording = load_maxwell(
            h5_path,
            stream_id=stream_id,
            rec_name=rec_name,
            hdf5_plugin_path=hdf5_plugin_path,
        )
    except Exception as exc:
        # The h5-derived description stands on its own; losing the extractor
        # only costs the cross-check, so report it and carry on.
        _warn(warnings, f"could not open the SpikeInterface extractor: {exc}")

    recording_hz = None
    if recording is not None:
        metadata["recording"] = describe_segment(recording)
        try:
            metadata["recording"]["num_segments"] = int(recording.get_num_segments())
        except (AttributeError, TypeError, ValueError):
            pass
        recording_hz = _as_positive_float(metadata["recording"].get("fs_hz"))
        if not channel_map:
            channel_map = _channel_map_from_recording(recording)
    else:
        metadata["recording"] = {}

    metadata["sampling"] = _sampling_block(
        _as_positive_float(settings_values.get("sampling_frequency_hz")),
        stream_hz,
        recording_hz,
    )
    metadata["gains"] = _gain_block(settings_values, recording, warnings)
    metadata["channel_map"] = channel_map
    metadata["probe"] = _probe_block(channel_map, recording)
    metadata["warnings"] = warnings

    return _jsonify(metadata)


def save_metadata(metadata, path):
    """Write `metadata` to `path` as JSON and return the path written.

    Keys are sorted so two runs over the same recording produce byte-identical
    files; a diff then shows a real change rather than dict ordering. Parent
    directories are created.
    """
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonify(metadata), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.debug("wrote recording metadata to %s", path)
    return path


def find_common_electrodes(h5_path, stream_id, rec_names=None):
    """Electrodes routed in EVERY recording of one well.

    An AxonTracking scan routes a different electrode subset per recording, so
    the recordings do not share a channel set and cannot be concatenated as-is.
    SpikeInterface requires identical channel ids across concatenated segments,
    which makes this intersection the prerequisite for `concatenate_segments`.

    The intersection is taken WITHIN a single well: wells are physically
    separate arrays, so electrodes are only comparable inside one.

    Returns a JSON-serializable dict::

        {"well", "rec_names", "common_electrodes", "common_count",
         "union_count", "per_rec_counts", "retained_fraction"}

    `retained_fraction` is common/union — worth surfacing, because on a real
    scan it can be small (a few hundred of many thousand), and that is a
    scientific decision the caller should see rather than discover downstream.

    Read from the HDF5 channel mapping directly, so this stays fast and does not
    depend on the compression plugin being resolvable.
    """
    import h5py
    import numpy as np

    per_rec_counts = {}
    common = None
    union = set()

    with h5py.File(str(h5_path), mode="r") as h5:
        available = sorted(h5["wells"][stream_id].keys())
        names = list(rec_names) if rec_names is not None else available
        for rec in names:
            mapping = h5["wells"][stream_id][rec]["settings"]["mapping"]
            electrodes = np.asarray(mapping["electrode"])
            channels = np.asarray(mapping["channel"])
            # channel < 0 marks an electrode that is not actually routed out.
            routed = {int(value) for value in electrodes[channels >= 0]}
            per_rec_counts[str(rec)] = len(routed)
            union |= routed
            common = routed if common is None else (common & routed)

    common = sorted(common or set())
    return {
        "well": str(stream_id),
        "rec_names": [str(name) for name in names],
        "common_electrodes": common,
        "common_count": len(common),
        "union_count": len(union),
        "per_rec_counts": per_rec_counts,
        "retained_fraction": (len(common) / len(union)) if union else 0.0,
    }


__all__ = ["extract_metadata", "save_metadata", "find_common_electrodes"]
