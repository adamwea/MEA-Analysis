"""A recording's gap structure: normalising it, rescaling it, placing joins.

The cache stores gap tables and the concatenated well's collector rescales them;
the per-segment statistics place the segment joins. None of that draws, so it
lives here; :mod:`.timebase` (which shades the gaps on an axis) imports it back.
"""

import logging

logger = logging.getLogger(__name__)


def _empty_pair():
    """Two empty arrays with the dtypes the gap table uses."""
    import numpy as np

    return np.asarray([], dtype=np.int64), np.asarray([], dtype=float)


def _is_mapping(value):
    """True for dict-like inputs, without importing collections.abc everywhere."""
    return hasattr(value, "get") and hasattr(value, "keys")


def _normalize_gaps(gaps):
    """(break indices, missing frames) from any shape a caller naturally holds.

    Accepts the dict :func:`mea_modules.io.gaps.frame_gaps` returns, a sequence
    of ``(sample_index, gap_frames)`` pairs using that same convention (a counter
    step of ``d`` means ``d - 1`` missing frames), or a sequence of dicts with
    ``sample_index``/``gap_frames`` or an explicit ``missing_frames``.

    Being liberal here is deliberate: the gap structure crosses a JSON file on
    its way from the io layer to a plot, and round-tripping turns tuples into
    lists and dicts into dicts of whatever the writer felt like.
    """
    import numpy as np

    if gaps is None:
        return _empty_pair()

    if _is_mapping(gaps):
        indices = gaps.get("break_sample_indices")
        if indices is None:
            logger.warning("gap mapping has no 'break_sample_indices'; treating it as no gaps")
            return _empty_pair()
        steps = gaps.get("break_gap_frames")
        indices = np.asarray(list(indices), dtype=np.int64)
        if steps is None:
            logger.warning("gap mapping has no 'break_gap_frames'; treating every break as one missing frame")
            missing = np.ones(indices.size, dtype=float)
        else:
            missing = np.asarray(list(steps), dtype=float) - 1.0
        return indices, missing

    indices = []
    missing = []
    for item in gaps or ():
        if _is_mapping(item):
            index = item.get("sample_index", item.get("break_sample_index"))
            if "missing_frames" in item:
                frames = item.get("missing_frames")
            else:
                step = item.get("gap_frames", item.get("break_gap_frames"))
                frames = None if step is None else float(step) - 1.0
        else:
            index, step = tuple(item)[:2]
            frames = float(step) - 1.0
        if index is None or frames is None:
            continue
        indices.append(int(index))
        missing.append(float(frames))

    return np.asarray(indices, dtype=np.int64), np.asarray(missing, dtype=float)


def _normalize_segment_gaps(segment_gaps):
    """(boundary indices, missing seconds) for the gaps BETWEEN segments.

    Accepts the summary :func:`mea_modules.io.gaps.well_gap_summary` returns, its
    ``segments`` list, a sequence of ``(sample_index, gap_seconds)`` pairs, or
    dicts carrying a sample index and a gap in seconds.

    A boundary needs both halves to be placeable: ``gap_before_s`` says how long
    the gap is and ``start_sample`` says where on the concatenated timeline it
    falls. :func:`mea_modules.io.gaps.segment_time_bounds` gives only the first,
    which is why entries without an index are dropped with a warning rather than
    silently landing at sample 0.
    """
    import numpy as np

    if segment_gaps is None:
        return _empty_pair()

    if _is_mapping(segment_gaps):
        segment_gaps = segment_gaps.get("segments") or ()

    indices = []
    seconds = []
    unplaceable = 0
    for item in segment_gaps or ():
        if _is_mapping(item):
            index = item.get("start_sample", item.get("sample_index"))
            gap = item.get("gap_before_s", item.get("gap_s"))
        else:
            index, gap = tuple(item)[:2]
        if gap is None or float(gap) == 0.0:
            continue
        if index is None:
            unplaceable += 1
            continue
        indices.append(int(index))
        seconds.append(float(gap))

    if unplaceable:
        logger.warning(
            "%d segment gaps carry no sample index and were dropped; "
            "use well_gap_summary(), whose segments carry start_sample",
            unplaceable,
        )
    return np.asarray(indices, dtype=np.int64), np.asarray(seconds, dtype=float)


def _gap_table(fs_hz, gaps=(), segment_gaps=()):
    """Merge both gap kinds into one sorted (sample index, missing seconds) table.

    Within-segment breaks are counted in frames and between-segment gaps in
    seconds; converting the former with `fs_hz` is what lets a single table
    describe both. Entries at or before sample 0 are dropped — nothing can be
    missing before the first sample — as are negative gaps, which would run the
    axis backwards.
    """
    indices, seconds, _ = _gap_table_kinds(fs_hz, gaps=gaps, segment_gaps=segment_gaps)
    return indices, seconds


def _gap_table_kinds(fs_hz, gaps=(), segment_gaps=()):
    """:func:`_gap_table`, plus a mask saying which entries are between-segment.

    The two kinds have to be merged before the offsets can be computed — the
    real time of any break depends on every break before it, of either kind — so
    they cannot simply be tabulated separately. They do, however, mean completely
    different things to a reader: a within-segment entry is the frame counter
    skipping by microseconds, a between-segment entry is the instrument standing
    idle for half a minute while the chip re-routes. Carrying the origin through
    the merge is what lets a figure draw them differently without recomputing
    either.
    """
    import numpy as np

    frame_indices, missing_frames = _normalize_gaps(gaps)
    segment_indices, segment_seconds = _normalize_segment_gaps(segment_gaps)

    fs = float(fs_hz) if fs_hz else 0.0
    if frame_indices.size and fs <= 0.0:
        logger.warning("no usable sampling rate; frame gaps cannot be converted to seconds")
        frame_indices, missing_frames = _empty_pair()

    indices = np.concatenate((frame_indices, segment_indices)).astype(np.int64, copy=False)
    seconds = np.concatenate(
        (missing_frames / fs if fs > 0.0 else np.asarray([], dtype=float), segment_seconds)
    ).astype(float, copy=False)
    is_segment = np.concatenate(
        (np.zeros(frame_indices.size, dtype=bool), np.ones(segment_indices.size, dtype=bool))
    )

    keep = (indices > 0) & (seconds > 0.0)
    dropped = int(keep.size - np.count_nonzero(keep))
    if dropped:
        logger.debug("dropped %d gap entries at sample <= 0 or of non-positive length", dropped)
    indices = indices[keep]
    seconds = seconds[keep]
    is_segment = is_segment[keep]

    # Sorted so a cumulative sum over the table is the offset at each break, and
    # so searchsorted can answer "how much time is missing before sample s".
    order = np.argsort(indices, kind="stable")
    return indices[order], seconds[order], is_segment[order]


def sample_times(sample_indices, fs_hz, gaps=(), segment_gaps=()):
    """Real elapsed seconds for arbitrary sample indices on a gappy recording.

    Each index is mapped to ``index / fs`` plus every second of recorded-nothing
    time that falls before it, so the result is the sample's true position in
    elapsed time rather than its position in the array.

    `gaps` is the within-segment break structure from
    :func:`mea_modules.io.gaps.frame_gaps`; `segment_gaps` is the
    between-segment structure from :func:`mea_modules.io.gaps.well_gap_summary`.
    Either may be omitted, in which case only the other kind is applied and with
    neither this is just ``index / fs``.

    This is the entry point for plots, which draw a decimated subset: it costs
    one ``searchsorted`` over the gap table rather than materializing a time
    value for every sample in the recording.
    """
    import numpy as np

    frames = np.asarray(sample_indices, dtype=np.int64)
    fs = float(fs_hz) if fs_hz else 0.0
    times = frames.astype(float) / fs if fs > 0.0 else frames.astype(float)

    indices, seconds = _gap_table(fs_hz, gaps=gaps, segment_gaps=segment_gaps)
    if not indices.size or not frames.size:
        return times

    # searchsorted(..., "right") counts the breaks at or before each sample; the
    # leading zero makes "no breaks yet" a valid lookup instead of a special case.
    cumulative = np.concatenate(([0.0], np.cumsum(seconds)))
    return times + cumulative[np.searchsorted(indices, frames, side="right")]


def resolve_time_gaps(time_gaps):
    """Split a caller's ``time_gaps`` argument into ``(gaps, segment_gaps)``.

    The plot emitters take one ``time_gaps`` parameter rather than two, because
    what a caller has on hand varies: a single segment's
    :func:`mea_modules.io.gaps.frame_gaps`, a whole well's
    :func:`mea_modules.io.gaps.well_gap_summary`, or both together as
    ``{"gaps": ..., "segment_gaps": ...}``. All three are recognized by the keys
    they carry, so no caller has to remember which slot theirs belongs in.
    """
    if time_gaps is None:
        return (), ()

    if _is_mapping(time_gaps):
        if "gaps" in time_gaps or "segment_gaps" in time_gaps:
            return time_gaps.get("gaps") or (), time_gaps.get("segment_gaps") or ()
        if "break_sample_indices" in time_gaps:
            return time_gaps, ()
        if "segments" in time_gaps:
            return (), time_gaps["segments"]
        logger.warning("time_gaps mapping matched no known shape; plotting on the contiguous axis")
        return (), ()

    # A bare sequence: segment records are recognizable by their wall-clock gap
    # field, anything else is per-sample break structure.
    items = list(time_gaps)
    if items and _is_mapping(items[0]) and ("gap_before_s" in items[0] or "gap_s" in items[0]):
        return (), items
    return items, ()


def rescale_time_gaps(time_gaps, step):
    """The same gap structure, addressed on a timeline decimated by `step`.

    A cached series that keeps every `step`-th native sample is numbered in its
    OWN samples, so a gap table indexed by native frame would shade the wrong
    stretch of it. This re-addresses both gap kinds onto the decimated
    numbering without changing a single duration:

    * a within-segment break at native frame ``i`` applies to every kept sample
      at or after it, i.e. from decimated sample ``ceil(i / step)``, and its
      ``d - 1`` missing native frames become ``(d - 1) / step`` decimated ones
      -- the same seconds at the decimated rate;
    * a between-segment gap moves to ``ceil(start_sample / step)`` and keeps its
      ``gap_before_s``, which was always in seconds.

    Returns ``{"gaps": ..., "segment_gaps": ...}``, the shape every emitter's
    ``time_gaps`` takes. `step` 1 hands the same numbers back.
    """
    import numpy as np

    step = max(1, int(step))
    gaps, segment_gaps = resolve_time_gaps(time_gaps)
    indices, missing = _normalize_gaps(gaps)
    rescaled_gaps = {
        "break_sample_indices": [int(v) for v in -(-indices // step)],
        "break_gap_frames": [float(v) for v in (missing / step + 1.0)],
    }

    if _is_mapping(segment_gaps):
        segment_gaps = segment_gaps.get("segments") or ()
    rescaled_segments = []
    for item in segment_gaps or ():
        entry = dict(item) if _is_mapping(item) else {
            "start_sample": tuple(item)[0], "gap_before_s": tuple(item)[1],
        }
        index = entry.get("start_sample", entry.get("sample_index"))
        if index is not None:
            entry["start_sample"] = int(-(-int(index) // step))
            entry.pop("sample_index", None)
        rescaled_segments.append(entry)
    return {
        "gaps": rescaled_gaps,
        "segment_gaps": rescaled_segments,
        "frame_step": step,
    }


def join_marks(stitch_frames, fs_hz, gaps=(), segment_gaps=(), real_time=False):
    """Where each segment join lands on the axis a figure is drawing.

    Returns one ``(start_s, stop_s)`` pair per join, in the order given.

    On the FILE timeline a join is a single instant and ``start_s == stop_s``:
    the concatenation placed the later segment's first sample immediately after
    the earlier segment's last.

    On a REAL-ELAPSED timeline it is not an instant. The earlier segment stops,
    the instrument spends however long it takes to re-route its electrodes, and
    only then does the later segment start — so the pair brackets that gap:
    ``start_s`` is when recording stopped and ``stop_s`` when it resumed. One
    line there would pin the join to a single edge of the gap and leave a reader
    to assume the other edge is nothing (2026-09-19).

    Callers pass FRAMES on the concatenated timeline, always. That is the one
    convention: the manifest records frames, converting the handful of joins is
    cheaper than converting millions of event times, and it is the direction
    that cannot lose precision. A caller holding seconds has already lost the
    round trip.
    """
    fs_hz = float(fs_hz or 0.0)
    if fs_hz <= 0.0:
        raise ValueError(f"fs_hz must be positive to place joins in seconds; got {fs_hz}")

    frames = [int(frame) for frame in (stitch_frames or ())]
    if not frames:
        return []

    if not real_time:
        return [(frame / fs_hz, frame / fs_hz) for frame in frames]

    # A stitch frame is the FIRST frame of the later segment, so the earlier
    # segment's last frame is the one before it. Both go through the same gap
    # table as the rest of the axis, so the markers cannot drift from the data.
    sample_period = 1.0 / fs_hz
    ends = sample_times(
        [max(0, frame - 1) for frame in frames], fs_hz, gaps=gaps, segment_gaps=segment_gaps
    )
    starts = sample_times(frames, fs_hz, gaps=gaps, segment_gaps=segment_gaps)
    marks = []
    for end, start in zip(ends, starts):
        stop_s = float(start)
        # The earlier segment stops one sample period after its last sample
        # began. A join with no gap at it should give start == stop exactly, but
        # the two ends arrive from separate float accumulations, so they differ
        # by a few ULP; anything under half a sample period is that noise and is
        # snapped shut. Without the snap a gapless join reads as spanning, which
        # would put the wrong legend on the figure and draw a second rule on top
        # of the first.
        start_s = min(float(end) + sample_period, stop_s)
        if stop_s - start_s < 0.5 * sample_period:
            start_s = stop_s
        marks.append((start_s, stop_s))
    return marks
