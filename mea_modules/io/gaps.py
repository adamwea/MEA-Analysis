"""Where a Maxwell recording is missing time, and how much of it.

An AxonTracking scan is handed downstream as one contiguous block of samples,
but it was never acquired that way. Time goes missing in two places, and neither
leaves any marker in the traces:

* BETWEEN recordings in a well. The chip re-routes its electrodes between
  configurations, which costs wall clock that no sample covers — ~23 s per
  boundary on the P003454 reference scan, ~465 s across its 21 recordings.
* WITHIN a recording. ``groups/routed/frame_nos`` is the instrument's global
  sample counter, one entry per stored sample. Where it steps by more than 1 the
  acquisition stored nothing for that stretch, so the samples either side are
  adjacent in the array but not in time. rec0000 of the reference scan has 8548
  such breaks covering 3.56 M frames: 121 s of stored samples that actually span
  299 s of wall clock.

Both are read here, and only here, so the plotting side never has to open an
HDF5 file. Everything comes from the frame counter and the per-recording
start/stop timestamps — the counter is tens of MB per segment and ``raw`` is
never touched, so this stays fast on a multi-GB scan.

Results are JSON-serializable: :func:`well_gap_summary` is meant to be written
out beside a run as the record of how much time the analysis is missing.

Pure library: no argparse, no printing, no ``__main__``.
"""

import logging
from pathlib import Path

from .load import ROUTED_GROUP, list_maxwell_streams
from .metadata import _epoch_unit, _read_scalar, _stream_sampling_hz

logger = logging.getLogger(__name__)


def _resolve_rec_name(h5_path, stream_id, rec_name):
    """Validate a (well, recording) pair against the file, defaulting the rec.

    Same resolution rule as :func:`~mea_modules.io.load.load_maxwell`, so a name
    that opens there resolves here — this module is only ever used to describe a
    segment someone else is about to load.
    """
    streams = list_maxwell_streams(h5_path)["streams"]
    if stream_id not in streams:
        raise ValueError(f"stream {stream_id!r} not in file; available: {sorted(streams)}")

    rec_names = streams[stream_id]
    if not rec_names:
        raise ValueError(
            f"stream {stream_id!r} carries no recordings; the pre-2016 MaxOne format "
            "has no per-recording frame counter to read"
        )
    if rec_name is None:
        return rec_names[0]
    if rec_name not in rec_names:
        raise ValueError(f"rec_name {rec_name!r} not in stream {stream_id!r}; available: {rec_names}")
    return rec_name


def _routed_group(h5, stream_id, rec_name):
    """The ``groups/routed`` node of one segment, or None when it has none.

    A recording without a routed group is invisible to SpikeInterface too (see
    :func:`~mea_modules.io.load.segment_index`), so returning None here keeps the
    two views of the file agreeing rather than raising on a segment nothing else
    can read either.
    """
    try:
        return h5["wells"][stream_id][rec_name]["groups"][ROUTED_GROUP]
    except KeyError:
        return None


def _gaps_from_frame_nos(routed, with_breaks=True):
    """Break structure of one segment's frame counter, as a plain dict.

    Split out from :func:`frame_gaps` so a whole-well roll-up can reuse it
    against an already-open file instead of reopening once per segment. That
    roll-up wants the counts only, hence `with_breaks`: a well runs to ~205 k
    breaks, and turning them all into Python lists to throw away is wasted work.
    """
    import numpy as np

    empty = {
        "n_samples": 0,
        "n_breaks": 0,
        "missing_frames": 0,
        "break_sample_indices": [],
        "break_gap_frames": [],
    }
    if routed is None or "frame_nos" not in routed:
        return empty

    # Read the counter once, whole. It is one uint64 per sample — millions per
    # segment — so it is differenced in numpy; a Python loop over it would cost
    # seconds per segment where this costs tens of milliseconds.
    frame_nos = np.asarray(routed["frame_nos"][()], dtype=np.int64)
    n_samples = int(frame_nos.size)
    if n_samples < 2:
        return dict(empty, n_samples=n_samples)

    steps = np.diff(frame_nos)
    # +1 so an index names the first sample AFTER the break, i.e. the first
    # sample of the next contiguous run — the same split points
    # :func:`~mea_modules.io.metadata.extract_metadata` reports as epochs.
    break_at = np.flatnonzero(steps != 1) + 1
    break_steps = steps[break_at - 1]

    # A backwards step should not happen; if it ever does, counting it as
    # negative missing time would silently shorten the timeline instead of
    # showing that the file is odd.
    backwards = int(np.count_nonzero(break_steps < 1))
    if backwards:
        logger.warning("frame counter steps backwards at %d of %d breaks", backwards, int(break_at.size))

    missing = np.maximum(break_steps - 1, 0)
    return {
        "n_samples": n_samples,
        "n_breaks": int(break_at.size),
        "missing_frames": int(missing.sum()),
        "break_sample_indices": [int(value) for value in break_at] if with_breaks else [],
        "break_gap_frames": [int(value) for value in break_steps] if with_breaks else [],
    }


def frame_gaps(h5_path, stream_id, rec_name=None):
    """Locate every break in one segment's frame counter.

    ``/wells/<well>/<rec>/groups/routed/frame_nos`` holds the instrument's global
    sample counter, one entry per stored sample. Consecutive samples differ by 1;
    a larger step means the acquisition stored nothing across it, and that step
    is the only record of the missing time.

    `rec_name` defaults to the first recording in the well.

    Returns a JSON-serializable dict::

        {"n_samples", "n_breaks", "missing_frames",
         "break_sample_indices", "break_gap_frames"}

    ``break_sample_indices[k]`` is the index of the first sample AFTER break k,
    and ``break_gap_frames[k]`` is the counter step there — a step of ``d`` means
    ``d - 1`` frames are missing, so ``missing_frames`` is ``sum(d - 1)``. Both
    lists are in sample order and are exactly what
    :func:`mea_modules.diagnostics.timebase.real_time_axis` consumes.

    Only the counter is read; ``raw`` is never opened.
    """
    import h5py

    h5_path = Path(h5_path).expanduser()
    if not h5_path.exists():
        raise FileNotFoundError(f"no such Maxwell file: {h5_path}")

    rec_name = _resolve_rec_name(h5_path, stream_id, rec_name)
    with h5py.File(str(h5_path), mode="r") as h5:
        routed = _routed_group(h5, stream_id, rec_name)
        if routed is None:
            logger.warning("segment %s/%s has no %r group; no frame counter to read", stream_id, rec_name, ROUTED_GROUP)
        gaps = _gaps_from_frame_nos(routed)

    logger.debug(
        "%s/%s: %d breaks, %d missing frames over %d samples",
        stream_id,
        rec_name,
        gaps["n_breaks"],
        gaps["missing_frames"],
        gaps["n_samples"],
    )
    return gaps


def _timestamp_divisor(h5, stream_id):
    """Seconds-per-unit for this well's start/stop timestamps.

    Maxwell writes them as bare integers with no unit recorded, so the scale is
    inferred from their magnitude over the whole well at once — one divisor for
    every recording, otherwise two segments could end up on different scales and
    the gap between them would be nonsense. The reference scan stores
    milliseconds, but that is read off the file rather than assumed.
    """
    raw_values = []
    well = h5["wells"][stream_id]
    for rec in sorted(well.keys()):
        for name in ("start_time", "stop_time"):
            value = _read_scalar(well[rec], name)
            if value is not None:
                try:
                    raw_values.append(int(value))
                except (TypeError, ValueError):
                    continue
    divisor, unit = _epoch_unit(raw_values)
    logger.debug("stream %s timestamps read as %s since epoch", stream_id, unit)
    return float(divisor), str(unit)


def _segment_bounds(h5, stream_id):
    """Per-recording (rec, start_s, stop_s) in epoch seconds, in file order."""
    divisor, _unit = _timestamp_divisor(h5, stream_id)
    well = h5["wells"][stream_id]

    bounds = []
    for rec in sorted(well.keys()):
        start_raw = _read_scalar(well[rec], "start_time")
        stop_raw = _read_scalar(well[rec], "stop_time")
        start_s = float(start_raw) / divisor if start_raw is not None else None
        stop_s = float(stop_raw) / divisor if stop_raw is not None else None
        if start_s is None or stop_s is None:
            logger.warning("segment %s/%s has no start/stop timestamp; wall-clock gap unknown", stream_id, rec)
        bounds.append((str(rec), start_s, stop_s))
    return bounds


def segment_time_bounds(h5_path, stream_id):
    """Wall-clock window of every recording in one well, and the gap before it.

    Returns a list of dicts in file order::

        {"rec", "start_time", "stop_time", "gap_before_s"}

    ``start_time``/``stop_time`` are epoch SECONDS, converted from whatever unit
    the file uses (inferred from the timestamps' magnitude, not assumed — the
    P003454 reference scan stores milliseconds). Either is None when the file
    does not carry it.

    ``gap_before_s`` is the wall-clock time between the previous recording's stop
    and this one's start — the electrode re-routing that no sample covers — and
    is None for the first recording, which has nothing before it.

    No traces and no frame counters are read, so this is a metadata-speed call.
    """
    import h5py

    h5_path = Path(h5_path).expanduser()
    if not h5_path.exists():
        raise FileNotFoundError(f"no such Maxwell file: {h5_path}")

    _resolve_rec_name(h5_path, stream_id, None)
    with h5py.File(str(h5_path), mode="r") as h5:
        bounds = _segment_bounds(h5, stream_id)

    segments = []
    previous_stop = None
    for rec, start_s, stop_s in bounds:
        gap_before = None
        if previous_stop is not None and start_s is not None:
            gap_before = float(start_s - previous_stop)
        segments.append(
            {
                "rec": rec,
                "start_time": start_s,
                "stop_time": stop_s,
                "gap_before_s": gap_before,
            }
        )
        if stop_s is not None:
            previous_stop = stop_s
    return segments


def _seconds(frames, fs_hz):
    """frames / fs, or None when the sampling rate could not be established."""
    if fs_hz is None:
        return None
    return float(frames) / float(fs_hz)


def well_gap_summary(h5_path, stream_id, fs_hz=None):
    """Roll both gap kinds up into one JSON-serializable record for a well.

    This is the diagnostic artifact a capsule writes: it says how much of a
    concatenated timeline is real recorded time, how much time is missing inside
    the segments, how much is missing between them, and therefore how far the
    real elapsed-time axis runs past ``n_samples / fs``.

    `fs_hz` is read from the file's own ``/data_store`` settings when None. If it
    cannot be established, every seconds-valued field is None and the frame
    counts still stand.

    Returns::

        {"well", "n_segments", "fs_hz", "n_samples", "n_breaks",
         "missing_frames", "recorded_s", "missing_within_segments_s",
         "missing_between_segments_s", "missing_s", "real_span_s",
         "wall_clock_span_s", "segments": [...]}

    Each entry of ``segments`` carries ``rec``, ``segment_index``,
    ``start_sample`` (its offset on the concatenated timeline), ``n_samples``,
    ``recorded_s``, ``n_breaks``, ``missing_frames``, ``missing_within_s``,
    ``start_time``, ``stop_time`` and ``gap_before_s``. ``start_sample`` plus
    ``gap_before_s`` is exactly what
    :func:`mea_modules.diagnostics.timebase.real_time_axis` needs to place the
    between-segment gaps, so the summary can be handed straight to the plots.

    Per-break detail is deliberately not included — a well runs to ~180 k breaks,
    which does not belong in a summary. Call :func:`frame_gaps` per segment when
    you need it.
    """
    import h5py

    h5_path = Path(h5_path).expanduser()
    if not h5_path.exists():
        raise FileNotFoundError(f"no such Maxwell file: {h5_path}")

    _resolve_rec_name(h5_path, stream_id, None)

    with h5py.File(str(h5_path), mode="r") as h5:
        if fs_hz is None:
            fs_hz = _stream_sampling_hz(h5, stream_id)
            if fs_hz is None:
                logger.warning(
                    "no sampling rate found for stream %s; reporting frame counts only", stream_id
                )
        bounds = _segment_bounds(h5, stream_id)
        # One open, one pass: the counters are read per segment inside the same
        # file handle rather than reopening the file 21 times.
        per_rec_gaps = {
            rec: _gaps_from_frame_nos(_routed_group(h5, stream_id, rec), with_breaks=False)
            for rec, _start, _stop in bounds
        }

    return _summarize(stream_id, bounds, per_rec_gaps, fs_hz)


def _summarize(stream_id, bounds, per_rec_gaps, fs_hz):
    """Roll per-segment gap counts up into the summary record.

    Split out from :func:`well_gap_summary` so :func:`concatenated_gaps`, which
    has already read every counter, can produce the same summary without a
    second pass over the file — the counters are tens of MB per segment and
    reading them twice is the most expensive thing either function does.
    """
    segments = []
    start_sample = 0
    previous_stop = None
    total_missing_frames = 0
    total_breaks = 0
    missing_between_s = 0.0
    for segment_index, (rec, start_s, stop_s) in enumerate(bounds):
        gaps = per_rec_gaps[rec]
        gap_before = None
        if previous_stop is not None and start_s is not None:
            gap_before = float(start_s - previous_stop)
            missing_between_s += gap_before
        segments.append(
            {
                "rec": rec,
                "segment_index": int(segment_index),
                "start_sample": int(start_sample),
                "n_samples": int(gaps["n_samples"]),
                "recorded_s": _seconds(gaps["n_samples"], fs_hz),
                "n_breaks": int(gaps["n_breaks"]),
                "missing_frames": int(gaps["missing_frames"]),
                "missing_within_s": _seconds(gaps["missing_frames"], fs_hz),
                "start_time": start_s,
                "stop_time": stop_s,
                "gap_before_s": gap_before,
            }
        )
        start_sample += int(gaps["n_samples"])
        total_missing_frames += int(gaps["missing_frames"])
        total_breaks += int(gaps["n_breaks"])
        if stop_s is not None:
            previous_stop = stop_s

    starts = [start for _rec, start, _stop in bounds if start is not None]
    stops = [stop for _rec, _start, stop in bounds if stop is not None]
    wall_span = float(max(stops) - min(starts)) if starts and stops else None

    recorded_s = _seconds(start_sample, fs_hz)
    missing_within_s = _seconds(total_missing_frames, fs_hz)
    missing_s = None if missing_within_s is None else float(missing_within_s + missing_between_s)

    summary = {
        "well": str(stream_id),
        "n_segments": int(len(segments)),
        "fs_hz": float(fs_hz) if fs_hz is not None else None,
        "n_samples": int(start_sample),
        "n_breaks": int(total_breaks),
        "missing_frames": int(total_missing_frames),
        "recorded_s": recorded_s,
        "missing_within_segments_s": missing_within_s,
        "missing_between_segments_s": float(missing_between_s),
        "missing_s": missing_s,
        # What a real elapsed-time axis over the concatenated recording spans,
        # against which n_samples / fs is the understatement the plots correct.
        "real_span_s": None if (recorded_s is None or missing_s is None) else float(recorded_s + missing_s),
        "wall_clock_span_s": wall_span,
    }
    summary["segments"] = segments

    logger.info(
        "%s: %d segments, %d samples, %d breaks, %d missing frames",
        stream_id,
        summary["n_segments"],
        summary["n_samples"],
        summary["n_breaks"],
        summary["missing_frames"],
    )
    return summary


def concatenated_gaps(h5_path, stream_id, fs_hz=None):
    """Every gap in a well, indexed on the CONCATENATED sample timeline.

    :func:`well_gap_summary` reports how much time is missing but deliberately
    drops the per-break detail, so a plot handed the summary can only stretch
    its axis by the between-segment gaps. On the P003454 reference scan that is
    465 s of the ~3500 s actually missing — an axis still labelled "real elapsed
    time" while being wrong by most of what it exists to correct. This returns
    what such an axis actually needs: both gap kinds, with each segment's break
    indices offset by that segment's start on the concatenated timeline so they
    align with the sample indices a concatenated recording — and every spike
    train sorted from it — is addressed by.

    The result is the ``{"gaps": ..., "segment_gaps": ...}`` shape
    :func:`mea_modules.diagnostics.timebase.resolve_time_gaps` accepts, so it
    goes straight into any emitter's ``time_gaps``. The roll-up under
    ``summary`` is :func:`well_gap_summary`'s, for callers that want to record
    the totals beside the plot.

    Costs one pass over every segment's frame counter — the same read
    :func:`well_gap_summary` already does, keeping the breaks instead of
    discarding them, so a whole well is seconds. ``raw`` is never touched.
    """
    import h5py

    h5_path = Path(h5_path).expanduser()
    if not h5_path.exists():
        raise FileNotFoundError(f"no such Maxwell file: {h5_path}")

    _resolve_rec_name(h5_path, stream_id, None)

    with h5py.File(str(h5_path), mode="r") as h5:
        if fs_hz is None:
            fs_hz = _stream_sampling_hz(h5, stream_id)
        bounds = _segment_bounds(h5, stream_id)
        # with_breaks=True is the whole difference from well_gap_summary: same
        # single pass over the counters, keeping the per-break detail it drops.
        per_rec = {
            rec: _gaps_from_frame_nos(_routed_group(h5, stream_id, rec), with_breaks=True)
            for rec, _start, _stop in bounds
        }

    # The offset is the running sample count, which is exactly how concatenate
    # lays the segments end to end — so a break at segment-local sample k in the
    # third segment lands where the concatenated recording actually holds it.
    indices, steps = [], []
    segments = []
    start_sample = 0
    previous_stop = None
    for segment_index, (rec, start_s, stop_s) in enumerate(bounds):
        gaps = per_rec[rec]
        indices.extend(int(value) + start_sample for value in gaps["break_sample_indices"])
        steps.extend(int(value) for value in gaps["break_gap_frames"])

        gap_before = None
        if previous_stop is not None and start_s is not None:
            gap_before = float(start_s - previous_stop)
        segments.append({
            "rec": rec,
            "segment_index": int(segment_index),
            "start_sample": int(start_sample),
            "n_samples": int(gaps["n_samples"]),
            "gap_before_s": gap_before,
        })

        start_sample += int(gaps["n_samples"])
        if stop_s is not None:
            previous_stop = stop_s

    summary = _summarize(stream_id, bounds, per_rec, fs_hz)
    logger.info(
        "%s: %d break(s) within segments + %d gap(s) between them; %s s recorded "
        "spans %s s of real time",
        stream_id,
        len(indices),
        sum(1 for entry in segments if entry["gap_before_s"]),
        None if summary["recorded_s"] is None else round(summary["recorded_s"], 1),
        None if summary["real_span_s"] is None else round(summary["real_span_s"], 1),
    )

    return {
        "gaps": {"break_sample_indices": indices, "break_gap_frames": steps},
        "segment_gaps": segments,
        "summary": summary,
    }


__all__ = ["concatenated_gaps", "frame_gaps", "segment_time_bounds", "well_gap_summary"]
