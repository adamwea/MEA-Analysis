"""Turn a recording's gap structure into a time axis you can plot against.

A concatenated Maxwell scan hands the sorter — and every plot downstream — one
contiguous sample index, so ``sample / fs`` is the axis everything defaults to.
That axis is wrong: it silently deletes every stretch the instrument did not
record. On the P003454 reference scan a 3247 s contiguous timeline covers 6765 s
of wall clock, so a feature drawn 100 s in may in truth have happened minutes
later, and two events either side of a break look adjacent when they are not.

This module does the arithmetic that fixes it, and nothing else: it takes the
gap structure :mod:`mea_modules.io.gaps` reads off the file and produces

* :func:`sample_times` / :func:`real_time_axis` — real elapsed seconds per
  sample, i.e. the sample's own time plus all the time missing before it, and
* :func:`gap_spans` — the intervals of that axis where no data exists, which is
  what a plot shades or leaves blank.

Everything is vectorized. A gap table is a few hundred thousand entries on a
full well and a sample axis is tens of millions of points, so offsets are
accumulated with ``cumsum``/``searchsorted``, never a Python loop over samples.

No file access, no plotting, no state: the only inputs are numbers the caller
already holds. Pure library — no argparse, no printing, no ``__main__``.
"""

import logging

from . import figure_text

logger = logging.getLogger(__name__)

# All the bands go into one collection, so the ceiling is only a guard against
# pathological input; a whole 21-segment well is ~205 k spans and one segment is
# ~8.5 k, both of which draw in well under a second.
_MAX_SHADED_SPANS = 50_000

_GAP_SHADE_COLOR = "0.65"
_GAP_SHADE_ALPHA = 0.35

# Between-segment gaps get their own colour. Neutral grey for the thousands of
# microsecond frame-counter breaks, a light blue for the handful of acquisition
# gaps, chosen to stay legible next to the red join rules and to survive a
# greyscale print as a visibly different tone rather than a wider grey band.
_SEGMENT_GAP_SHADE_COLOR = "#a8c6e0"
_SEGMENT_GAP_SHADE_ALPHA = 0.55

# Re-exported from figure_text so the legend vocabulary has ONE home: this
# module owns where a band is drawn, figure_text owns what it is called.
GAP_LABEL_WITHIN = figure_text.GAP_WITHIN
GAP_LABEL_BETWEEN = figure_text.GAP_BETWEEN


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


def real_time_axis(n_samples, fs_hz, gaps=(), segment_gaps=()):
    """Real elapsed time, in seconds, of every sample from 0 to `n_samples`.

    Returns a float array of length `n_samples`, monotonically increasing, whose
    last value exceeds ``n_samples / fs`` by exactly the total missing time. This
    is the x axis a plot should use when it wants elapsed time rather than array
    position.

    `gaps` and `segment_gaps` are as in :func:`sample_times`.

    One value per sample means ~520 MB for a full 65 M-sample well, so prefer
    :func:`sample_times` on the decimated indices a plot actually draws; this
    exists for callers that genuinely need the dense axis.
    """
    import numpy as np

    n_samples = int(n_samples)
    if n_samples <= 0:
        return np.asarray([], dtype=float)

    fs = float(fs_hz) if fs_hz else 0.0
    times = np.arange(n_samples, dtype=float)
    if fs > 0.0:
        times /= fs

    indices, seconds = _gap_table(fs_hz, gaps=gaps, segment_gaps=segment_gaps)
    keep = indices < n_samples
    indices = indices[keep]
    seconds = seconds[keep]
    if not indices.size:
        return times

    # bincount + cumsum rather than searchsorted over an index array: it needs
    # one float array of n instead of an int64 one as well, which matters at
    # tens of millions of samples. bincount also sums duplicate indices, so two
    # gaps landing on the same sample both count.
    offsets = np.bincount(indices, weights=seconds, minlength=n_samples)[:n_samples]
    np.cumsum(offsets, out=offsets)
    times += offsets
    return times


def gap_spans(n_samples, fs_hz, gaps=(), segment_gaps=()):
    """Intervals of the real time axis, in seconds, that hold no data.

    Returns a list of ``(start_s, end_s)`` pairs in increasing order, positioned
    on the same axis :func:`real_time_axis` produces: a span starts one sample
    period after the last sample before the break and ends at the first sample
    after it. Spans beyond `n_samples` are dropped, so passing the end of the
    plotted window bounds the result to it.

    These are what a plot shades, or leaves as a genuine break in a line — they
    are the difference between "nothing was firing" and "nothing was recorded",
    which a reader cannot otherwise tell apart.

    A single segment has thousands of breaks, most of them microseconds wide, so
    a caller drawing them should filter or merge first (:func:`_shade_gap_spans`
    does).
    """
    import numpy as np

    indices, seconds = _gap_table(fs_hz, gaps=gaps, segment_gaps=segment_gaps)
    n_samples = int(n_samples)
    keep = indices < n_samples
    indices = indices[keep]
    seconds = seconds[keep]
    if not indices.size:
        return []

    fs = float(fs_hz) if fs_hz else 0.0
    if fs <= 0.0:
        logger.warning("no usable sampling rate; gap spans cannot be placed on a time axis")
        return []

    # Offset accumulated strictly BEFORE each break: the gap opens at the real
    # time the next sample would have had if nothing were missing.
    cumulative = np.cumsum(seconds)
    starts = indices.astype(float) / fs + (cumulative - seconds)
    ends = starts + seconds
    return [(float(start), float(end)) for start, end in zip(starts, ends)]


def gap_spans_by_kind(n_samples, fs_hz, gaps=(), segment_gaps=()):
    """:func:`gap_spans`, split into the two kinds rather than pooled.

    Returns ``{"within_segment": [...], "between_segment": [...]}``, each a list
    of ``(start_s, end_s)`` pairs on the same axis :func:`gap_spans` places them
    on — the offsets are computed over the merged table, exactly as before, and
    only the output is separated.

    The reason this exists: pooling makes a thirty-second acquisition gap
    indistinguishable from a microsecond frame-counter break, because the figure
    shades both in the same grey. On a scan with thousands of small breaks the
    one gap a reviewer actually needs to see disappears into the stipple. Drawing
    the between-segment kind in its own colour is what separates "the instrument
    paused to re-route" from "the frame counter skipped".
    """
    import numpy as np

    indices, seconds, is_segment = _gap_table_kinds(
        fs_hz, gaps=gaps, segment_gaps=segment_gaps
    )
    n_samples = int(n_samples)
    keep = indices < n_samples
    indices = indices[keep]
    seconds = seconds[keep]
    is_segment = is_segment[keep]
    empty = {"within_segment": [], "between_segment": []}
    if not indices.size:
        return empty

    fs = float(fs_hz) if fs_hz else 0.0
    if fs <= 0.0:
        logger.warning("no usable sampling rate; gap spans cannot be placed on a time axis")
        return empty

    cumulative = np.cumsum(seconds)
    starts = indices.astype(float) / fs + (cumulative - seconds)
    ends = starts + seconds

    def _pairs(mask):
        return [
            (float(start), float(end))
            for start, end in zip(starts[mask], ends[mask])
        ]

    return {
        "within_segment": _pairs(~is_segment),
        "between_segment": _pairs(is_segment),
    }


def shade_gap_kinds(
    axis,
    spans_by_kind,
    x_min=None,
    x_max=None,
    max_spans=_MAX_SHADED_SPANS,
):
    """Shade both gap kinds in their own colours; return a count for each.

    One call so that every emitter shades identically — the within-segment
    stipple underneath in neutral grey, the between-segment gaps on top in their
    own colour so they read as a different kind of thing rather than a wider
    example of the same thing.

    Returns ``{"within_segment": int, "between_segment": int}``: how many bands
    each kind actually drew, which is what a caller needs to decide whether the
    corresponding legend key belongs on the figure.
    """
    drawn = {}
    for kind, color, alpha in (
        ("within_segment", _GAP_SHADE_COLOR, _GAP_SHADE_ALPHA),
        ("between_segment", _SEGMENT_GAP_SHADE_COLOR, _SEGMENT_GAP_SHADE_ALPHA),
    ):
        drawn[kind] = _shade_gap_spans(
            axis,
            (spans_by_kind or {}).get(kind, ()),
            x_min=x_min,
            x_max=x_max,
            max_spans=max_spans,
            color=color,
            alpha=alpha,
        )
    return drawn


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


def _shade_gap_spans(
    axis,
    spans,
    x_min=None,
    x_max=None,
    max_spans=_MAX_SHADED_SPANS,
    color=_GAP_SHADE_COLOR,
    alpha=_GAP_SHADE_ALPHA,
):
    """Shade `spans` on a matplotlib axis as one full-height band collection.

    Shared by the trace and raster emitters. Takes the axis rather than importing
    matplotlib, so this module stays a pure-numeric one.

    Spans are clipped to `x_min`/`x_max` and overlapping ones merged, then drawn
    with ``broken_barh`` — a single collection rather than one patch per gap,
    which is what makes shading a segment's ~8.5 k breaks cost milliseconds. The
    bands are placed in axes coordinates vertically, so they span the full height
    whatever the y limits end up being and never drag the x limits around.

    Note that on a scan this gappy the honest render at full zoom IS a stipple:
    the gaps really do interleave with the data every few tens of milliseconds.
    The broken trace line is the companion signal — see :func:`plot_traces`.

    Returns the number of bands drawn.
    """
    import numpy as np

    bounds = np.asarray(list(spans), dtype=float).reshape(-1, 2)
    if not bounds.size:
        return 0

    lower = float(x_min) if x_min is not None else float(bounds[:, 0].min())
    upper = float(x_max) if x_max is not None else float(bounds[:, 1].max())
    if not upper > lower:
        return 0

    starts = np.clip(bounds[:, 0], lower, upper)
    ends = np.clip(bounds[:, 1], lower, upper)
    keep = ends > starts
    starts = starts[keep]
    ends = ends[keep]
    if not starts.size:
        return 0

    order = np.argsort(starts, kind="stable")
    starts = starts[order]
    ends = ends[order]

    # A new band begins where the previous one had already ended; the running
    # maximum handles spans nested inside earlier ones.
    running_end = np.maximum.accumulate(ends)
    new_band = np.concatenate(([True], starts[1:] > running_end[:-1]))
    band = np.cumsum(new_band) - 1
    merged_starts = starts[new_band]
    merged_ends = np.full(merged_starts.size, -np.inf, dtype=float)
    np.maximum.at(merged_ends, band, ends)

    if merged_starts.size > int(max_spans):
        # Widest first, so what survives is the missing time a reader can see.
        logger.warning(
            "shading the %d widest of %d gap spans", int(max_spans), int(merged_starts.size)
        )
        widest = np.argsort(merged_ends - merged_starts, kind="stable")[-int(max_spans) :]
        merged_starts = merged_starts[widest]
        merged_ends = merged_ends[widest]

    axis.broken_barh(
        list(zip(merged_starts.tolist(), (merged_ends - merged_starts).tolist())),
        (0.0, 1.0),
        transform=axis.get_xaxis_transform(),
        color=color,
        alpha=alpha,
        linewidth=0.0,
        zorder=0,
    )
    return int(merged_starts.size)


_JOIN_COLOR = "red"
_JOIN_LINEWIDTH = 0.8
_JOIN_LINESTYLE = ":"

JOIN_LABEL_INSTANT = figure_text.JOIN_INSTANT
JOIN_LABEL_SPANNING = figure_text.JOIN_SPANNING


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


def joins_span_time(marks):
    """Whether any join in `marks` has real elapsed time inside it.

    An exact comparison is enough because :func:`join_marks` has already snapped
    a gapless join shut; this stays strict so a genuine sub-sample gap is not
    silently rounded away twice.
    """
    return any(float(stop) > float(start) for start, stop in marks or ())


def draw_join_marks(
    axis,
    marks,
    color=_JOIN_COLOR,
    lw=_JOIN_LINEWIDTH,
    linestyle=_JOIN_LINESTYLE,
    alpha=0.85,
):
    """Draw every join from :func:`join_marks`; return how many lines were drawn.

    An instantaneous join is one line. A join with real time inside it is two —
    the earlier segment's end and the later one's start — so a reader can see
    that the space between them is elapsed time rather than a rendering gap.

    Takes the axis rather than importing matplotlib, keeping this module
    pure-numeric like the rest of it.
    """
    drawn = 0
    for start_s, stop_s in marks or ():
        axis.axvline(
            float(start_s), color=color, linewidth=lw, linestyle=linestyle, alpha=alpha
        )
        drawn += 1
        if float(stop_s) > float(start_s):
            axis.axvline(
                float(stop_s), color=color, linewidth=lw, linestyle=linestyle, alpha=alpha
            )
            drawn += 1
    return drawn


__all__ = [
    "real_time_axis",
    "gap_spans",
    "gap_spans_by_kind",
    "shade_gap_kinds",
    "GAP_LABEL_WITHIN",
    "GAP_LABEL_BETWEEN",
    "sample_times",
    "resolve_time_gaps",
    "join_marks",
    "joins_span_time",
    "draw_join_marks",
    "JOIN_LABEL_INSTANT",
    "JOIN_LABEL_SPANNING",
]
