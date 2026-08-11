"""Measure a Maxwell recording's geometry and gather its device/setup facts.

The pipeline's rule is *measure, don't assume* (Adam, 2026-08-11): a
diagnostics warning once assumed a fixed electrode-cluster size when the real
size is a GUI setting that changes per run, and the same class of bug hides in
every hardcoded pitch or well count. This module reads what the file actually
says — and measures what the file only implies — so downstream cluster/eps
logic can consume facts instead of folklore.

Two public entry points:

* :func:`measure_electrode_geometry` — pure geometry on one segment's routed
  electrode coordinates: bounding box, density, the nearest-neighbour distance
  distribution (its mode is the *effective* pitch of the routed subset — a
  config that routes every other electrode of a 17.5 µm array measures 35 µm,
  and that is the number downstream spatial logic needs), per-axis modal
  spacing, and — when electrode ids are given — the *physical* array grid
  derived from the id↔coordinate relation, which stays 17.5 µm even under
  sparse routing because electrode ids count the electrodes routing skipped.

* :func:`survey_well_device` — everything the h5 reliably says about the
  device and setup for one well: model/family (MaxOne vs MaxTwo), plate id,
  MaxWell software + HDF5 library versions, assay identity and GUI properties
  (the ``neighbors`` cluster size lives here), per-well plate annotations,
  environment (temperature/diagnosis) coverage, per-recording amplifier
  settings consensus, the derived ADC bit depth, and the per-segment geometry
  above. Every fact records which h5 path it came from (``provenance``);
  anything sought but not found is listed in ``absent`` — recorded as absent,
  never invented.

Traces are never read; the heaviest access is a ~1000-row channel mapping per
recording and the (small) environment logs. Pure library: no argparse, no
printing, no ``__main__``.
"""

import json
import logging

logger = logging.getLogger(__name__)

# Coordinates/pitches are reported to this many decimals (0.01 µm) — Maxwell
# writes exact multiples of the pitch, so this only strips float noise.
_DECIMALS = 2

# A nearest-neighbour mode counts as an integer multiple of the physical pitch
# when the ratio is within this of a whole number (5% — generous enough for
# float noise, tight enough that a diagonal neighbour at sqrt(2)x flags).
_MULTIPLE_TOL = 0.05

# Maxwell's amplifier full scale; used ONLY to derive the ADC bit depth from
# the recorded lsb and gain (bits = log2(full_scale / (lsb * gain))). The file
# records no bit depth anywhere, so the derivation — and this assumption — is
# stated in the output rather than silently applied.
_FULL_SCALE_V = 3.3

# How close the derived bit count must sit to an integer to be believed.
_BITS_TOL = 0.02

# Amplifier settings gathered per recording, mapped to output names.
_SETTINGS_FIELDS = (
    ("gain", "gain"),
    ("lsb", "lsb_v"),
    ("sampling", "sampling_hz"),
    ("hpf", "hpf_hz"),
    ("spike_threshold", "spike_threshold"),
)


# --------------------------------------------------------------------------- #
# small numeric helpers
# --------------------------------------------------------------------------- #


def _modal(values, decimals=_DECIMALS):
    """(modal value, fraction of values at the mode) — ties take the smallest."""
    import numpy as np

    arr = np.round(np.asarray(values, dtype=float), decimals)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None, 0.0
    uniq, counts = np.unique(arr, return_counts=True)
    best = int(np.argmax(counts))  # first == smallest among ties
    return float(uniq[best]), float(counts[best] / arr.size)


def _nn_distances(x, y, chunk=512):
    """Each electrode's distance to its nearest neighbour (brute force).

    Routed sets are <= ~1024 electrodes on Maxwell hardware, so a chunked
    O(n^2) pass is fast and keeps the working set a few MB — no KD-tree
    dependency for something this small.
    """
    import numpy as np

    pts = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    n = pts.shape[0]
    if n < 2:
        return np.empty(0)
    out = np.empty(n)
    for start in range(0, n, chunk):
        block = pts[start:start + chunk]
        d2 = ((block[:, None, :] - pts[None, :, :]) ** 2).sum(axis=-1)
        rows = np.arange(block.shape[0])
        d2[rows, start + rows] = np.inf  # mask self-distance
        out[start:start + block.shape[0]] = np.sqrt(d2.min(axis=1))
    return out


def _axis_pitch(primary, secondary, decimals=_DECIMALS):
    """Modal spacing along `primary` among electrodes sharing a `secondary` line.

    This is the EFFECTIVE routed spacing on that axis: electrodes are grouped
    into rows (columns) by their other coordinate and the consecutive gaps
    within each line are pooled. Sparse routing shows up honestly — an
    every-other-column config reports twice the physical pitch.
    """
    import numpy as np

    p = np.round(np.asarray(primary, float), decimals)
    s = np.round(np.asarray(secondary, float), decimals)
    order = np.lexsort((p, s))
    ps, ss = p[order], s[order]
    steps = []
    breaks = np.flatnonzero(np.diff(ss) != 0) + 1
    for line in np.split(ps, breaks):
        if line.size >= 2:
            gaps = np.diff(np.unique(line))
            if gaps.size:
                steps.append(gaps[gaps > 0])
    if not steps:
        return None, 0.0
    return _modal(np.concatenate(steps), decimals)


def _grid_from_ids(ids, x, y, decimals=_DECIMALS):
    """The physical array grid implied by the electrode-id numbering.

    Maxwell numbers electrodes row-major over the FULL array, so ids count the
    electrodes routing skipped: along a row, Δx/Δid is the physical x pitch no
    matter how sparse the routing. Column period (ids per row) comes from the
    gcd of id offsets after removing the column term, and the physical y pitch
    from Δy per derived row step.

    Returns {} when underivable (no co-row pairs, non-grid coordinates, …).
    Under pathological routing (e.g. only every other ROW routed) the y pitch
    degrades to the effective row spacing — the gcd cannot see rows nobody
    routed — which is exactly what "measured, not assumed" should report.
    """
    import numpy as np

    ids = np.asarray(ids)
    if ids.size < 2 or not np.issubdtype(ids.dtype, np.integer):
        return {}
    ids = ids.astype(np.int64)
    x = np.round(np.asarray(x, float), decimals)
    y = np.round(np.asarray(y, float), decimals)

    out = {}

    # Physical x pitch: consecutive same-row pairs, Δx over Δid.
    order = np.lexsort((ids, y))
    yi, idi, xi = y[order], ids[order], x[order]
    same_row = np.diff(yi) == 0
    did = np.diff(idi)[same_row]
    dx = np.diff(xi)[same_row]
    usable = (did > 0) & (dx > 0)
    if not usable.any():
        return {}
    pitch_x, frac_x = _modal(dx[usable] / did[usable], decimals)
    if not pitch_x or pitch_x <= 0:
        return {}
    out["pitch_x_um"] = pitch_x
    out["pitch_x_fraction"] = frac_x

    # Column index relative to the leftmost routed column; offsets cancel in
    # the differences below, so an unrouted column 0 does not matter.
    cols = (x - x.min()) / pitch_x
    if not np.allclose(cols, np.round(cols), atol=0.05):
        return out  # x not on the derived grid; stop at the x pitch
    v = ids - np.round(cols).astype(np.int64)  # = row * ncols + const

    uniq_v = np.unique(v)
    if uniq_v.size >= 2:
        stride = int(np.gcd.reduce(np.diff(uniq_v)))
        if stride > 0:
            out["row_stride_ids"] = stride
            rows = (v - v.min()) // stride
            # Physical y pitch: Δy per derived row step, over consecutive
            # distinct rows.
            order_r = np.argsort(rows, kind="stable")
            rr, yy = rows[order_r], y[order_r]
            keep = np.diff(rr) > 0
            if keep.any():
                pitch_y, frac_y = _modal(
                    np.diff(yy)[keep] / np.diff(rr)[keep], decimals
                )
                if pitch_y and pitch_y > 0:
                    out["pitch_y_um"] = pitch_y
                    out["pitch_y_fraction"] = frac_y
    return out


# --------------------------------------------------------------------------- #
# per-segment geometry
# --------------------------------------------------------------------------- #


def measure_electrode_geometry(x_um, y_um, electrode_ids=None):
    """Measured geometry of one segment's routed electrode set.

    `x_um`/`y_um` are the routed electrodes' coordinates; `electrode_ids`
    (optional) are their full-array electrode numbers, which unlock the
    physical-grid derivation described in :func:`_grid_from_ids`.

    Returns a JSON-serializable dict: electrode count, bounding box and
    extents, areal density, the nearest-neighbour distance distribution
    (min/median/max, modal value + the fraction of electrodes at the mode),
    the per-axis effective pitches, and — when derivable — the physical
    ``grid`` block. Keys that cannot be measured are absent, never invented.
    """
    import numpy as np

    x = np.asarray(x_um, dtype=float)
    y = np.asarray(y_um, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]

    block = {"n_electrodes": int(x.size), "units": "um"}
    if x.size == 0:
        return block

    block["x_range_um"] = [float(x.min()), float(x.max())]
    block["y_range_um"] = [float(y.min()), float(y.max())]
    extent_x = float(x.max() - x.min())
    extent_y = float(y.max() - y.min())
    block["x_extent_um"] = extent_x
    block["y_extent_um"] = extent_y
    area_mm2 = (extent_x / 1000.0) * (extent_y / 1000.0)
    if area_mm2 > 0:
        block["density_per_mm2"] = float(round(x.size / area_mm2, 2))

    if x.size < 2:
        return block

    nn = _nn_distances(x, y)
    modal, modal_fraction = _modal(nn)
    pitch = {
        "nn_min_um": float(round(float(nn.min()), _DECIMALS)),
        "nn_median_um": float(round(float(np.median(nn)), _DECIMALS)),
        "nn_max_um": float(round(float(nn.max()), _DECIMALS)),
        "nn_modal_um": modal,
        "nn_modal_fraction": float(round(modal_fraction, 4)),
    }
    x_pitch, x_fraction = _axis_pitch(x, y)
    if x_pitch is not None:
        pitch["x_pitch_um"] = x_pitch
        pitch["x_pitch_fraction"] = float(round(x_fraction, 4))
    y_pitch, y_fraction = _axis_pitch(y, x)
    if y_pitch is not None:
        pitch["y_pitch_um"] = y_pitch
        pitch["y_pitch_fraction"] = float(round(y_fraction, 4))
    block["pitch"] = pitch

    if electrode_ids is not None:
        ids = np.asarray(electrode_ids)[finite]
        try:
            grid = _grid_from_ids(ids, x, y)
        except (TypeError, ValueError, OverflowError):
            grid = {}
        if grid:
            block["grid"] = grid
    return block


# --------------------------------------------------------------------------- #
# h5 fact gathering
# --------------------------------------------------------------------------- #


def _scalar(h5, path):
    """Decoded value of a one-element dataset at `path`, or None."""
    import numpy as np

    try:
        if path not in h5:
            return None
        value = h5[path][()]
    except (KeyError, OSError, TypeError, ValueError):
        return None
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        value = value.ravel()[0]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    return value


def _take(facts, provenance, absent, h5, field, path, cast=None):
    """Record `field` from h5 `path` with provenance, or list it absent."""
    value = _scalar(h5, path)
    if value is None:
        absent.append(field)
        return None
    if cast is not None:
        try:
            value = cast(value)
        except (TypeError, ValueError):
            pass
    facts[field] = value
    provenance[field] = path
    return value


def _well_label(value):
    """Normalize a well identifier ("0", 0, "well0") to Maxwell's "wellNNN"."""
    text = str(value).strip()
    if text.lower().startswith("well") and text[4:].isdigit():
        return f"well{int(text[4:]):03d}"
    if text.isdigit():
        return f"well{int(text):03d}"
    return text


def _device_family(model):
    """"MaxOne"/"MaxTwo" parsed from a wellplate version string, else None."""
    text = str(model or "").lower().replace(" ", "").replace("-", "").replace("_", "")
    if "maxtwo" in text:
        return "MaxTwo"
    if "maxone" in text:
        return "MaxOne"
    return None


def _wells_in_model(model):
    """Well count a model string like "MaxTwo 6 multi-well MEA" declares."""
    family = _device_family(model)
    if family == "MaxOne":
        return 1
    for token in str(model or "").split():
        if token.isdigit():
            return int(token)
    return None


def _environment_block(h5, provenance):
    """Coverage of the acquisition-environment logs, plus temperature stats.

    MaxTwo rigs log chip temperatures and supply diagnostics for the whole
    session; MaxOne files carry the (empty) groups. Row counts distinguish
    "logged nothing" from "not present", and the wellplate/recording-unit
    temperatures get min/mean/max — the culture's thermal record.
    """
    import numpy as np

    block = {}
    for name in ("temperature", "diagnosis"):
        path = f"environment/{name}"
        if path not in h5:
            continue
        try:
            dataset = h5[path]
            fields = [str(f).strip() for f in (dataset.dtype.names or ())]
            entry = {"rows": int(dataset.shape[0]), "fields": fields}
            if name == "temperature" and dataset.shape[0]:
                table = dataset[()]
                for raw_name in dataset.dtype.names or ():
                    label = str(raw_name).strip()
                    if label.lower() in ("wellplate_temperature", "recording_unit"):
                        values = np.asarray(table[raw_name], dtype=float)
                        values = values[np.isfinite(values)]
                        if values.size:
                            entry[f"{label.lower()}_c"] = {
                                "min": float(round(float(values.min()), 2)),
                                "mean": float(round(float(values.mean()), 2)),
                                "max": float(round(float(values.max()), 2)),
                            }
            block[name] = entry
            provenance[f"environment.{name}"] = path
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("could not read %s: %s", path, exc)
    return block


def _assay_properties(h5, provenance, absent):
    """The assay's GUI configuration, from the /assay/inputs/electrodes blob.

    The blob's ``properties`` member is where run-shaping settings live —
    including ``neighbors``, the electrode-cluster size the GUI was configured
    with (9 by default, 12 on some runs): the fact whose assumption prompted
    this module.
    """
    raw = _scalar(h5, "assay/inputs/electrodes")
    if raw is None:
        absent.append("assay.properties")
        return None
    try:
        doc = json.loads(raw)
    except (TypeError, ValueError) as exc:
        logger.warning("could not parse /assay/inputs/electrodes as JSON: %s", exc)
        absent.append("assay.properties")
        return None
    properties = doc.get("properties")
    if not isinstance(properties, dict):
        absent.append("assay.properties")
        return None
    provenance["assay.properties"] = "assay/inputs/electrodes (JSON, key 'properties')"
    return properties


def _derive_adc_bits(settings_pairs):
    """ADC bit depth implied by each (lsb, gain) pair, if consistent.

    ``bits = log2(full_scale / (lsb * gain))`` with the amplifier's 3.3 V full
    scale. The file records lsb and gain but never a bit depth; when every
    recording's pair lands on the same integer, that integer is believable.
    """
    import numpy as np

    bits_seen = set()
    for lsb, gain in settings_pairs:
        try:
            lsb, gain = float(lsb), float(gain)
        except (TypeError, ValueError):
            continue
        if lsb <= 0 or gain <= 0:
            continue
        bits = float(np.log2(_FULL_SCALE_V / (lsb * gain)))
        if abs(bits - round(bits)) <= _BITS_TOL:
            bits_seen.add(int(round(bits)))
        else:
            return None  # a pair off-integer: don't guess
    if len(bits_seen) != 1:
        return None
    return bits_seen.pop()


def _segment_survey(rec_group, rec_name, warnings):
    """One recording's routing facts: mapping-derived geometry + settings."""
    import numpy as np

    entry = {}
    settings = rec_group.get("settings") if rec_group is not None else None
    mapping = None
    if settings is not None and "mapping" in settings:
        try:
            mapping = np.asarray(settings["mapping"][()])
        except (OSError, TypeError, ValueError) as exc:
            warnings.append(f"{rec_name}: unreadable settings/mapping: {exc}")

    if mapping is not None and mapping.dtype.names:
        names = set(mapping.dtype.names)
        entry["n_channels"] = int(mapping.shape[0])
        if {"channel", "x", "y"} <= names:
            routed = np.asarray(mapping["channel"]) >= 0
            entry["n_routed"] = int(routed.sum())
            ids = mapping["electrode"][routed] if "electrode" in names else None
            entry.update(
                measure_electrode_geometry(
                    mapping["x"][routed], mapping["y"][routed], electrode_ids=ids
                )
            )
    elif mapping is None:
        warnings.append(f"{rec_name}: no settings/mapping; geometry unmeasured")

    values = {}
    for h5_name, out_name in _SETTINGS_FIELDS:
        value = _scalar(settings, h5_name) if settings is not None else None
        if value is not None:
            values[out_name] = value
    if values:
        entry["settings"] = values

    raw = rec_group.get("groups/routed/raw") if rec_group is not None else None
    if raw is not None:
        entry["raw_dtype"] = str(raw.dtype)
    return entry


def _consensus(per_segment, key_path):
    """Sorted unique values of a (possibly nested) key across segments."""
    seen = set()
    for entry in per_segment.values():
        node = entry
        for key in key_path[:-1]:
            node = node.get(key) if isinstance(node, dict) else None
            if node is None:
                break
        if isinstance(node, dict) and key_path[-1] in node:
            value = node[key_path[-1]]
            if isinstance(value, (int, float, str)):
                seen.add(value)
    return sorted(seen)


def _pitch_sanity(per_segment, physical_pitch):
    """Check each segment's modal NN distance against the physical pitch.

    The modal nearest-neighbour distance of a grid-routed subset should be an
    integer multiple of the physical pitch (1x dense, 2x every-other, …).
    A non-multiple — a diagonal-neighbour config's sqrt(2)x, or a mapping that
    is not on the grid at all — is flagged as an oddity, never a failure.
    """
    sanity = {"physical_pitch_um": physical_pitch, "oddities": []}
    if not physical_pitch:
        sanity["checked"] = False
        return sanity
    sanity["checked"] = True
    for rec_name, entry in sorted(per_segment.items()):
        modal = (entry.get("pitch") or {}).get("nn_modal_um")
        if not modal:
            continue
        ratio = modal / physical_pitch
        if not (abs(ratio - round(ratio)) <= _MULTIPLE_TOL * max(1.0, round(ratio)) and round(ratio) >= 1):
            sanity["oddities"].append(
                {
                    "rec": rec_name,
                    "nn_modal_um": modal,
                    "ratio_to_physical": float(round(ratio, 3)),
                }
            )
    sanity["all_integer_multiples"] = not sanity["oddities"]
    return sanity


def survey_well_device(h5_path, stream_id, rec_names=None):
    """Device/setup facts for one well of a Maxwell file, measured not assumed.

    Returns ``{"device": {...}, "per_segment": {rec_name: {...}}}`` — the
    ``device`` block is the well-level summary the ingest manifest embeds, and
    ``per_segment`` holds each recording's routing facts (electrode count,
    bounding box, density, measured pitches, amplifier settings). Everything
    is JSON-serializable; every file-read fact carries its h5 path in
    ``device["provenance"]``; facts sought but not found are listed in
    ``device["absent"]``.

    Never raises on a missing group — a partial survey of a suspect file is
    more useful than none. Traces are not read.
    """
    import h5py

    facts = {}
    provenance = {}
    absent = []
    warnings = []

    with h5py.File(str(h5_path), mode="r") as h5:
        model = _take(facts, provenance, absent, h5, "model", "wellplate/version")
        _take(facts, provenance, absent, h5, "plate_id", "wellplate/id")
        _take(facts, provenance, absent, h5, "plate_variant", "wellplate/variant")
        family = _device_family(model)
        if family is not None:
            facts["family"] = family
            provenance["family"] = "derived from wellplate/version"
        else:
            absent.append("family")
        wells_in_model = _wells_in_model(model)
        if wells_in_model is not None:
            facts["wells_in_model"] = wells_in_model
            provenance["wells_in_model"] = "derived from wellplate/version"
        else:
            absent.append("wells_in_model")

        _take(facts, provenance, absent, h5, "mxw_version", "mxw_version")
        _take(facts, provenance, absent, h5, "hdf_version", "hdf_version")
        _take(facts, provenance, absent, h5, "format_version", "version", cast=int)

        assay = {}
        assay_prov = {}
        assay_absent = []
        _take(assay, assay_prov, assay_absent, h5, "run_id", "assay/run_id")
        _take(assay, assay_prov, assay_absent, h5, "script_id", "assay/script_id")
        _take(
            assay, assay_prov, assay_absent, h5,
            "record_time_s", "assay/inputs/record_time", cast=int,
        )
        properties = _assay_properties(h5, assay_prov, assay_absent)
        if properties is not None:
            assay["properties"] = properties
        if assay:
            facts["assay"] = assay
            for key, value in assay_prov.items():
                provenance[key if key.startswith("assay.") else f"assay.{key}"] = value
        absent.extend(
            name if name.startswith("assay.") else f"assay.{name}"
            for name in assay_absent
        )

        well_label = _well_label(stream_id)
        well_info_path = f"wellplate/{well_label}"
        if well_info_path in h5:
            info = {}
            group = h5[well_info_path]
            try:
                keys = sorted(group.keys())
            except (AttributeError, OSError):
                keys = []
            for key in keys:
                value = _scalar(group, key)
                if value is not None:
                    info[str(key)] = value
            if info:
                facts["well_info"] = info
                provenance["well_info"] = well_info_path
        else:
            absent.append("well_info")

        environment = _environment_block(h5, provenance)
        if environment:
            facts["environment"] = environment
        else:
            absent.append("environment")

        # Per-recording survey. `/wells/<well>/<rec>` is the modern layout;
        # the pre-2016 MaxOne format keeps a single implicit recording at the
        # file root, surveyed under its conventional empty rec name.
        per_segment = {}
        if "wells" in h5 and well_label in h5["wells"]:
            wells_group = h5["wells"][well_label]
            names = list(rec_names) if rec_names else sorted(wells_group.keys())
            for rec_name in names:
                if rec_name is None or rec_name not in wells_group:
                    if rec_name is not None:
                        warnings.append(f"recording {rec_name!r} not in /wells/{well_label}")
                    continue
                per_segment[str(rec_name)] = _segment_survey(
                    wells_group[rec_name], str(rec_name), warnings
                )
        elif "settings" in h5:  # old single-well format: root-level settings
            per_segment[""] = _segment_survey(h5, "(root)", warnings)
        else:
            warnings.append(f"no /wells/{well_label} group; per-recording survey unavailable")

    device = facts
    if per_segment:
        settings_values = {}
        for _, out_name in _SETTINGS_FIELDS:
            values = _consensus(per_segment, ("settings", out_name))
            if values:
                settings_values[out_name] = values
        if settings_values:
            device["settings"] = settings_values
            provenance["settings"] = "wells/<well>/<rec>/settings (unique values across recordings)"

        pairs = [
            (entry.get("settings", {}).get("lsb_v"), entry.get("settings", {}).get("gain"))
            for entry in per_segment.values()
            if isinstance(entry.get("settings"), dict)
        ]
        bits = _derive_adc_bits([p for p in pairs if None not in p])
        adc = {}
        raw_dtypes = _consensus(per_segment, ("raw_dtype",))
        if raw_dtypes:
            adc["raw_dtype"] = raw_dtypes if len(raw_dtypes) > 1 else raw_dtypes[0]
        if bits is not None:
            adc["bits_derived"] = bits
            adc["derivation"] = (
                f"log2({_FULL_SCALE_V} V / (lsb * gain)); the file records no bit depth"
            )
        else:
            absent.append("adc.bits")
        if adc:
            device["adc"] = adc

        geometry = {
            "nn_modal_um": _consensus(per_segment, ("pitch", "nn_modal_um")),
            "x_pitch_effective_um": _consensus(per_segment, ("pitch", "x_pitch_um")),
            "y_pitch_effective_um": _consensus(per_segment, ("pitch", "y_pitch_um")),
            "n_routed": _consensus(per_segment, ("n_routed",)),
        }
        grid_x = _consensus(per_segment, ("grid", "pitch_x_um"))
        grid_y = _consensus(per_segment, ("grid", "pitch_y_um"))
        stride = _consensus(per_segment, ("grid", "row_stride_ids"))
        grid = {}
        if len(grid_x) == 1:
            grid["pitch_x_um"] = grid_x[0]
        if len(grid_y) == 1:
            grid["pitch_y_um"] = grid_y[0]
        if len(stride) == 1:
            grid["row_stride_ids"] = stride[0]
        if grid:
            geometry["grid"] = grid
            provenance["geometry.grid"] = (
                "derived from settings/mapping electrode ids x coordinates"
            )
        candidates = [v for v in (grid.get("pitch_x_um"), grid.get("pitch_y_um")) if v]
        physical = min(candidates) if candidates else None
        geometry["pitch_sanity"] = _pitch_sanity(per_segment, physical)
        device["geometry"] = geometry
        if not geometry["pitch_sanity"]["all_integer_multiples"]:
            for oddity in geometry["pitch_sanity"]["oddities"]:
                warnings.append(
                    "pitch oddity: {rec} modal nearest-neighbour {nn} um is {ratio}x "
                    "the physical pitch (not an integer multiple)".format(
                        rec=oddity["rec"],
                        nn=oddity["nn_modal_um"],
                        ratio=oddity["ratio_to_physical"],
                    )
                )

    device["provenance"] = provenance
    device["absent"] = sorted(set(absent))
    if warnings:
        device["warnings"] = warnings
        for message in warnings:
            logger.warning("%s: %s", stream_id, message)

    return {"device": device, "per_segment": per_segment}


__all__ = ["measure_electrode_geometry", "survey_well_device"]
