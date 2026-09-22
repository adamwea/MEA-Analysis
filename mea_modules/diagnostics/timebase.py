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
# computed in .gap_table (the compute side imports no drawing code); re-exported here
from .gap_table import (  # noqa: F401
    _empty_pair,
    _is_mapping,
    _normalize_gaps,
    _normalize_segment_gaps,
    _gap_table,
    _gap_table_kinds,
    sample_times,
    resolve_time_gaps,
    rescale_time_gaps,
    join_marks,
)

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
