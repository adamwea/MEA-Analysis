"""Where each segment sits on the two timelines a concatenated well carries.

Concatenation lays separate acquisitions end to end, so the result has two
timelines that do not agree. The CONCATENATED one is what every sample index
downstream lives on. The WALL CLOCK is when the instrument was actually
recording. Between two segments the file advances by a single sample while real
time advances by however long the chip took to re-route its electrodes, which is
typically tens of seconds.

This module draws both, one bar per segment: the file on top, the clock below.
The two panels disagreeing is not a defect, it is the point of the figure — the
white space in the lower panel is time no sample covers. A segment that
contributed nothing is a zero-width bar in the upper panel, which is the other
read the figure exists for.

Both panels mark the segment joins, in the vocabulary
:mod:`mea_modules.diagnostics.timebase` sets for every emitter that marks them.
On the file timeline a join is one instant and one rule. On the clock it is not:
a PAIR of rules brackets the gap — the earlier segment's end and the later one's
start — so a reader can see that what lies between them is elapsed time rather
than a rendering gap.

The figure is a timeline, and it is drawn like one: thin bars, a fixed printed
height per segment row, and a canvas whose height follows the segment count
instead of a fixed sheet that is mostly empty at two segments and cramped at
twenty-one. `panels` selects which timeline is drawn, so the same code writes the
two-panel comparison and a dedicated single-panel figure per timeline; `annotate`
switches off the title and caption for a presentation render.

The wall-clock panel needs per-segment timestamps (``start_time`` / ``stop_time``,
the fields :mod:`mea_modules.io.gaps` reads off the source file). A source that
carries none still gets the concatenated panel, and the returned manifest says
which panels were drawn and why one is missing, rather than the figure quietly
arriving with half its content.

Figures are built straight from :class:`matplotlib.figure.Figure` on an Agg
canvas — no pyplot, so this is safe on a headless node and leaves no global
figure state behind. Pass `axes` to draw into panels the caller already owns,
which is how a composed sheet reuses this without its own figure.

Pure library: no argparse, no printing, no ``__main__``.
"""

import logging

from .channel_layout import (
    _add_caption,
    _fold_caption,
    _legend_line,
    _legend_patch,
    _new_figure,
    _save_and_release,
)
from .figure_style import LEGEND_FONTSIZE, LEGEND_FRAME_ALPHA, legend_corner
from .figure_text import CONTIGUOUS_AXIS, FILE_TIME_AXIS, REAL_TIME_AXIS, SEAM
from .timebase import (
    JOIN_LABEL_INSTANT,
    JOIN_LABEL_SPANNING,
    draw_join_marks,
    join_marks,
    joins_span_time,
)

logger = logging.getLogger(__name__)

_BOUNDARY_MAP_DPI = 180

# --- timeline geometry -----------------------------------------------------
#
# Every read this figure supports is horizontal — where a segment starts, how
# long it lasts, how far the next one is from it. A row therefore needs to be
# tall enough to read its label and no taller. The retired fixed 14 x 7.5 in
# sheet sized rows by dividing whatever was left over, which drew two segments
# as inch-thick slabs and twenty-one as a wall, with most of the canvas white
# either way (Adam, 2026-09-19).
#
# So the height is built up from the row count, and the per-row allowance is in
# INCHES rather than a share of the panel: that is what makes the drawn bar the
# same thickness at two segments as at twenty-one. 0.16 in is roughly twice the
# 6 pt row label, the tightest a labelled row stays readable at.
_FIGURE_WIDTH_IN = 12.0
_ROW_HEIGHT_IN = 0.16
# Floor on the row box, not on the figure: a two-segment map is drawn three rows
# tall and leaves the spare row empty, so its bars keep their printed thickness
# instead of inflating to fill the panel.
_MIN_ROWS = 3
# Per panel, for the x label and the tick labels. `annotate=False` — the
# presentation register every tool in this repo actually passes — never draws
# a title, so budgeting room for one unconditionally would leave that much of
# the canvas blank on every figure this repo renders; `_PANEL_CHROME_TITLED_IN`
# is what the two-panel comparison and the annotated single-panel figures use
# instead, sized for the title row `_finish_panel` adds under `annotate=True`.
_PANEL_CHROME_IN = 0.45
_PANEL_CHROME_TITLED_IN = 0.75
_LEGEND_BAND_IN = 0.35
_CAPTION_BAND_IN = 0.60
_TOP_MARGIN_IN = 0.15
# Nothing this repository produces carries more than 21 segments; the ceiling
# only stops a pathological manifest from asking for a canvas no viewer opens.
# `_add_caption` may still grow a figure past it to fit its own bottom matter,
# which is the right trade — annotation that covers the plot is worse.
_MAX_FIGURE_HEIGHT_IN = 16.0

# Fraction of a row the bar fills: thin enough to read as a timeline stroke,
# solid enough that a short segment is still a visible mark rather than a hair.
_BAR_HEIGHT = 0.45

# One colour per timeline, so the two panels cannot be confused at a glance.
_CONCATENATED_COLOR = "#4878cf"
_WALL_CLOCK_COLOR = "#6acc64"
# Matches every other emitter's join rule, drawn by `timebase.draw_join_marks`.
_JOIN_COLOR = "red"

_TITLE_FONTSIZE = 9
_AXIS_LABEL_FONTSIZE = 8
_TICK_FONTSIZE = 7
_SEGMENT_LABEL_FONTSIZE = 6
_GAP_LABEL_FONTSIZE = 5
_GAP_LABEL_COLOR = "#555555"

# `ax.margins(x=...)` in `_finish_panel`, named so the gap-label fit check
# below can reason about the SAME padding rather than a second copy of it.
_X_AXIS_MARGIN = 0.01

# Rough average glyph width for the gap-duration labels, as a fraction of their
# own font size — generous on purpose, so a label this estimates as "fits"
# really does rather than clipping the bar it was placed to clear. There is no
# renderer to measure against at the point this is used: the axis limits are
# not final until every bar in the panel is drawn, several bars from now.
_GAP_LABEL_CHAR_WIDTH_EM = 0.60
# Thin clearance either side of the label so it does not touch the bar it
# sits beside even when the estimate above is exactly right.
_GAP_LABEL_PAD_IN = 0.04
# The empty band between one row and the one above it: bars only fill
# `_BAR_HEIGHT` of their own row, so a label offset this far from its row
# clears both that row's bar and the row above it.
_GAP_LABEL_FALLBACK_OFFSET = 0.5

# Publication-shorthand legend keys for the two coloured bars. Distinct rather
# than a shared "segment" — the two-panel figure dedupes its keys by label, and
# collapsing these would drop one colour from the combined legend outright.
_CONCATENATED_LEGEND_KEY = "segment (file time)"
_WALL_CLOCK_LEGEND_KEY = "segment (elapsed time)"

_CONCATENATED_TITLE = "segment map — position in the recorded file"
# Two wordings because the paired one leans on the panel above it, and a
# standalone clock figure has nothing to be "the same segments" as.
_WALL_CLOCK_TITLE_PAIRED = (
    "the same segments in real time — white space is time nothing was recorded"
)
_WALL_CLOCK_TITLE = (
    "segment map — real elapsed time, white space is time nothing was recorded"
)

# Which panels each `panels` choice asks for, in drawing order.
_WANTED_PANELS = {
    "both": ("concatenated", "wall_clock"),
    "file": ("concatenated",),
    "clock": ("wall_clock",),
}

_NO_TIMESTAMPS_NOTE = (
    "wall-clock panel skipped: the source file carries no usable segment timestamps"
)
_ONE_AXES_NOTE = (
    "wall-clock panel skipped: one axes was supplied and the wall-clock panel needs "
    "a second"
)

# Plain wording for what the lower panel adds, in the register
# :mod:`.figure_text` sets: expand the idea, do not name our machinery.
_WALL_CLOCK_PANEL = (
    "The lower panel puts the same segments where they happened in real time, so a "
    "stretch of white between two bars is time the instrument was not recording at "
    "all — usually the chip re-routing its electrodes between one configuration and "
    "the next."
)
_WALL_CLOCK_ONLY = (
    "The segments are placed where they happened in real time, so a stretch of white "
    "between two bars is time the instrument was not recording at all — usually the "
    "chip re-routing its electrodes between one configuration and the next."
)
_NO_TIMESTAMPS_CAPTION = (
    "Only one timeline is drawn: this recording carries no usable per-segment "
    "timestamps, so when each segment actually ran is unknown."
)


def _as_axes_list(axes):
    """The caller's `axes` as a list, or None when none were supplied.

    Accepts a single axes or any sequence of them, which is what
    ``fig.subplots`` returns in its two shapes.
    """
    if axes is None:
        return None
    if hasattr(axes, "barh"):
        return [axes]
    return list(axes)


def _have_wall_clock(segments):
    """Whether every segment carries both timestamps the lower panel needs.

    All or nothing on purpose: a panel drawn from a partial set would place the
    segments it does have at offsets measured from a first segment whose own
    start is unknown, which is a wrong picture rather than an incomplete one.
    """
    return bool(segments) and all(
        entry.get("start_time") is not None and entry.get("stop_time") is not None
        for entry in segments
    )


def _figure_size(n_segments, n_panels, annotate):
    """Canvas for `n_segments` rows across `n_panels` panels, in inches.

    Height is composed rather than fixed: each panel gets its rows at
    :data:`_ROW_HEIGHT_IN` apiece plus one chrome allowance, and the figure gets
    a band for the legend, a band for the caption when there is one, and a top
    margin. Nothing here is a share of anything else, so the result holds its
    proportions from two segments to twenty-one.

    Dropping the caption band when `annotate` is false is the point of that
    switch as much as the missing text is: a presentation render should not
    carry an inch of margin reserved for a caption nobody asked for. The same
    reasoning picks the per-panel chrome allowance: `annotate=False` draws no
    title, so it gets the smaller of the two.
    """
    rows = max(int(n_segments), _MIN_ROWS)
    chrome_in = _PANEL_CHROME_TITLED_IN if annotate else _PANEL_CHROME_IN
    height_in = int(n_panels) * (rows * _ROW_HEIGHT_IN + chrome_in)
    height_in += _LEGEND_BAND_IN + _TOP_MARGIN_IN
    if annotate:
        height_in += _CAPTION_BAND_IN
    if height_in > _MAX_FIGURE_HEIGHT_IN:
        logger.warning(
            "%d segments over %d panel(s) want a %.1f in figure; capping at %.1f in, "
            "so the rows will be tighter than %.2f in each",
            int(n_segments),
            int(n_panels),
            height_in,
            _MAX_FIGURE_HEIGHT_IN,
            _ROW_HEIGHT_IN,
        )
        height_in = _MAX_FIGURE_HEIGHT_IN
    return (_FIGURE_WIDTH_IN, height_in)


def _concatenated_span_s(segments, fs_hz):
    """How far the last segment reaches on the file's own timeline, in seconds."""
    return max(
        (float(entry.get("start_frame", 0)) + float(entry.get("n_samples", 0))) / fs_hz
        for entry in segments
    )


def _wall_clock_span_s(segments):
    """How far the last segment reaches on the clock, from the first one's start.

    A segment whose stamps run backwards contributes its start and nothing more,
    matching the zero-width bar the panel draws for it.
    """
    origin = float(segments[0]["start_time"])
    return max(
        float(entry["start_time"])
        - origin
        + max(0.0, float(entry["stop_time"]) - float(entry["start_time"]))
        for entry in segments
    )


def _join_label(marks):
    """The legend wording for `marks`, from the shared vocabulary."""
    return JOIN_LABEL_SPANNING if joins_span_time(marks) else JOIN_LABEL_INSTANT


def _wall_clock_join_marks(segments, origin):
    """``(stop, start)`` of each consecutive pair, on the clock panel's own axis.

    Deliberately NOT :func:`mea_modules.diagnostics.timebase.join_marks`. That
    helper places a join on the real-elapsed axis built from the gap table, and
    this panel is not drawn on that axis: its bars come straight from the
    per-segment wall-clock stamps, which this emitter is handed and the gap table
    is not. Placing the marks from the same stamps as the bars is the only way
    the two cannot drift — a mark from the gap table would sit the accumulated
    within-segment breaks away from the bar edge it is supposed to touch.

    The pair shape, the drawing and the legend wording are still the shared ones,
    so both panels say the same thing about a join in the same way.
    """
    marks = []
    for earlier, later in zip(segments, segments[1:]):
        stop_s = float(later["start_time"]) - origin
        # Clamped as `join_marks` clamps: a stop stamp later than the next start
        # (clock skew, or an overlap the source never resolved) must not run the
        # pair backwards and draw the later rule first.
        start_s = min(float(earlier["stop_time"]) - origin, stop_s)
        marks.append((start_s, stop_s))
    return marks


def _finish_panel(ax, segments, xlabel, title, annotate):
    """Row labels, axis label, title, and the tight timeline geometry.

    Shared by both panels so they cannot drift apart in anything but colour and
    what they are a picture of.
    """
    ax.set_yticks(list(range(len(segments))))
    ax.set_yticklabels(
        # Rows are recording names, not indices: an index tells a reviewer
        # nothing they cannot already count, and the name is what the rest of
        # the run's artifacts are filed under.
        [str(entry.get("rec")) for entry in segments],
        fontsize=_SEGMENT_LABEL_FONTSIZE,
    )
    # A timeline should end where the data ends; matplotlib's default 5% padding
    # is dead space at both ends of every row.
    ax.margins(x=_X_AXIS_MARGIN)
    # Reading order: the segment recorded first belongs at the top. Expressed as
    # limits rather than `invert_yaxis` so the row PITCH is pinned as well — the
    # box is always at least `_MIN_ROWS` rows tall, which is what keeps two
    # segments from stretching into slabs.
    ax.set_ylim(max(len(segments), _MIN_ROWS) - 0.5, -0.5)
    # The row labels name the rows; tick marks next to them are noise.
    ax.tick_params(axis="y", length=0.0)
    ax.tick_params(axis="x", labelsize=_TICK_FONTSIZE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel(xlabel, fontsize=_AXIS_LABEL_FONTSIZE)
    if annotate and title:
        ax.set_title(title, fontsize=_TITLE_FONTSIZE)


def _gap_label_fits_inside(fig, text, gap_s, total_span_s):
    """Whether `text` is narrow enough to sit inside a `gap_s`-wide span.

    Both sides of the comparison are a FRACTION of the axes width, so neither
    depends on the view limits being final — at the point this runs (mid-draw,
    one bar at a time) `ax.get_xlim()` still reflects whatever autoscale saw
    last, not the panel's finished range. `total_span_s` (the whole panel's
    known reach, computed once up front from the segment stamps) plus the same
    `_X_AXIS_MARGIN` `_finish_panel` pads the axis with gives the gap's fraction
    without waiting for a layout pass; the label's fraction comes from its
    character count against the figure's own width, which is fixed the moment
    the figure is created.
    """
    if total_span_s <= 0.0:
        return False
    axes_width_in = float(fig.get_figwidth()) or 1.0
    label_width_in = (
        len(text) * _GAP_LABEL_CHAR_WIDTH_EM * _GAP_LABEL_FONTSIZE / 72.0
        + 2.0 * _GAP_LABEL_PAD_IN
    )
    required_fraction = label_width_in / axes_width_in
    gap_fraction = float(gap_s) / (total_span_s * (1.0 + 2.0 * _X_AXIS_MARGIN))
    return gap_fraction >= required_fraction


def _draw_concatenated(ax, segments, fs_hz, stitch_frames, title, annotate):
    """The file's own timeline: one bar per segment, joins as red rules.

    Returns ``(legend handles, joins drawn)`` so the caller can key the figure
    once instead of once per panel.
    """
    for position, entry in enumerate(segments):
        ax.barh(
            position,
            float(entry.get("n_samples", 0)) / fs_hz,
            left=float(entry.get("start_frame", 0)) / fs_hz,
            height=_BAR_HEIGHT,
            color=_CONCATENATED_COLOR,
            edgecolor="none",
        )

    # One rule per join: on this timeline the later segment's first sample sits
    # immediately after the earlier segment's last, so the join is an instant.
    marks = join_marks(stitch_frames, fs_hz, real_time=False)
    drawn_joins = draw_join_marks(ax, marks, color=_JOIN_COLOR)

    _finish_panel(ax, segments, FILE_TIME_AXIS, title, annotate)

    handles = [_legend_patch(_CONCATENATED_COLOR, _CONCATENATED_LEGEND_KEY)]
    if drawn_joins:
        handles.append(
            _legend_line(_JOIN_COLOR, _join_label(marks), lw=0.9, linestyle=":")
        )
    return handles, drawn_joins


def _draw_wall_clock(ax, segments, title, annotate):
    """The same segments at their real offsets, gaps annotated in place.

    Offsets are measured from the first segment's start rather than from the
    absolute timestamp: the absolute value is an acquisition clock nobody reads,
    and the quantity that matters is how far apart the segments are.

    Returns ``(legend handles, joins drawn)``.
    """
    origin = float(segments[0]["start_time"])
    # The panel's full reach, known from the stamps before a single bar is
    # drawn — used below to size the gap labels, since the axis's own view
    # limits are not final until every bar in the loop has been added.
    total_span_s = _wall_clock_span_s(segments)
    annotated = 0
    for position, entry in enumerate(segments):
        left = float(entry["start_time"]) - origin
        width = max(0.0, float(entry["stop_time"]) - float(entry["start_time"]))
        ax.barh(
            position,
            width,
            left=left,
            height=_BAR_HEIGHT,
            color=_WALL_CLOCK_COLOR,
            edgecolor="none",
        )

        # The first segment has nothing before it, so its gap is not a
        # measurement — only the joins carry one.
        gap = entry.get("gap_before_s")
        if gap is not None and position:
            gap = float(gap)
            label = f"+{gap:.1f}s gap"
            # Centred on the gap's own span (from the previous bar's end at
            # `left - gap` to this bar's start at `left`), not on the bar that
            # follows it — the earlier placement at `left` read as belonging
            # to the next segment instead of to the gap before it.
            center_x = left - gap / 2.0
            if _gap_label_fits_inside(ax.figure, label, gap, total_span_s):
                label_y = position
            else:
                # Too narrow to hold the label without touching a bar: drop it
                # into the empty band above this row instead of shrinking or
                # truncating it, so it stays legible and never sits on a bar.
                label_y = position - _GAP_LABEL_FALLBACK_OFFSET
            ax.text(
                center_x,
                label_y,
                label,
                va="center",
                ha="center",
                fontsize=_GAP_LABEL_FONTSIZE,
                color=_GAP_LABEL_COLOR,
            )
            annotated += 1

    # Two rules per join here, bracketing the gap. Every consecutive pair of
    # segments is a join and this panel holds the stamps to place both of its
    # edges, so the marks come from the segment list rather than from
    # `stitch_frames`, which is the FILE timeline's record of the same joins.
    marks = _wall_clock_join_marks(segments, origin)
    drawn_joins = draw_join_marks(ax, marks, color=_JOIN_COLOR)

    _finish_panel(ax, segments, REAL_TIME_AXIS, title, annotate)

    handles = [_legend_patch(_WALL_CLOCK_COLOR, _WALL_CLOCK_LEGEND_KEY)]
    if drawn_joins:
        handles.append(
            _legend_line(_JOIN_COLOR, _join_label(marks), lw=0.9, linestyle=":")
        )
    logger.debug(
        "wall-clock panel: annotated %d of %d joins, drew %d join rule(s)",
        annotated,
        len(segments) - 1,
        drawn_joins,
    )
    return handles, drawn_joins


def _dedupe_handles(handles):
    """Legend handles with one key per label, first occurrence winning.

    Both panels key their own join rule. When the two describe the same thing —
    a concatenation with no real time inside its joins — that is one encoding
    drawn one way, and it earns one key.
    """
    unique = {}
    for handle in handles:
        unique.setdefault(handle.get_label(), handle)
    return list(unique.values())


def _caption_parts(drawn, wall_clock_dropped, joins_drawn):
    """Caption fragments for the panels this figure actually ended up with."""
    parts = []
    if joins_drawn:
        parts.append(SEAM)
    if "concatenated" in drawn:
        parts.append(CONTIGUOUS_AXIS)
    if "wall_clock" in drawn:
        parts.append(_WALL_CLOCK_PANEL if len(drawn) > 1 else _WALL_CLOCK_ONLY)
    elif wall_clock_dropped:
        parts.append(_NO_TIMESTAMPS_CAPTION)
    return parts


def plot_segment_boundary_map(
    segments,
    fs_hz,
    out_path=None,
    *,
    stitch_frames=(),
    panels="both",
    title=None,
    wall_clock_title=None,
    annotate=True,
    axes=None,
    figsize=None,
    dpi=_BOUNDARY_MAP_DPI,
):
    """Draw one bar per segment on the requested timelines; return a manifest.

    Parameters
    ----------
    segments : sequence of dict
        One entry per segment, in concatenated order. The concatenated panel
        reads ``start_frame`` and ``n_samples`` (frames on the concatenated
        timeline) and ``rec`` (the row label); the wall-clock panel additionally
        reads ``start_time`` / ``stop_time`` and the optional ``gap_before_s``.
        This is the shape a concatenation manifest joined against
        :func:`mea_modules.io.gaps.concatenated_gaps` already has.
    fs_hz : float
        Sampling rate, used to turn the frame bookkeeping into seconds. Must be
        positive — an unknown rate makes the concatenated panel meaningless
        rather than merely imprecise.
    out_path : path-like or None
        Where to write the PNG. Required unless `axes` is given, in which case
        the caller owns the figure and nothing is written.
    stitch_frames : sequence of int
        Frame offsets of the segment joins on the concatenated timeline — the
        one convention across every emitter, see
        :func:`mea_modules.diagnostics.timebase.join_marks`. They place the
        joins on the concatenated panel. The wall-clock panel needs no such
        record: it derives its joins from the consecutive segments' own stamps,
        and marks each one with a PAIR of rules bracketing the gap.
    panels : {"both", "file", "clock"}
        Which timelines to draw. ``"both"`` is the two-panel comparison and the
        default; ``"file"`` and ``"clock"`` write one dedicated single-panel
        figure each, so a caller wanting all three drives this three times.
        ``"clock"`` needs per-segment timestamps and raises without them, since
        there is no honest half of that figure to fall back to.
    title, wall_clock_title : str or None
        Per-panel title overrides, one each so a single-panel call titles the
        panel it actually draws. Left None each panel uses its own default.
    annotate : bool
        True (the default) draws the titles and the caption block. False draws
        neither and drops the caption's reserved band from the canvas, leaving
        axes, units and legend — the presentation register, where the title's
        information lives in the filename instead.
    axes : matplotlib axes or sequence of them, or None
        Draw into the caller's panels instead of building a figure. The
        requested panels are filled in order, so one axes with ``panels="both"``
        gets the concatenated panel only. With `axes` given each panel keys
        itself, no caption is added and no file is written — the caller's figure
        owns its own margin.
    figsize : tuple or None
        Left None the canvas is computed from the segment count and the number
        of panels, which is what keeps a row the same printed height at two
        segments and at twenty-one. An explicit tuple overrides that. Ignored
        when `axes` is given.
    dpi : float
        Figure resolution. Ignored when `axes` is given.

    Returns
    -------
    dict
        JSON-serializable: the file written (``None`` when drawing into `axes`),
        which panels were drawn, the segment count, each timeline's span in
        seconds, and a ``note`` naming why the wall-clock panel is absent when
        it is. The spans describe the DATA, so a single-panel call still reports
        both and three figures of the same well stay comparable; ``panels`` is
        the record of what was drawn.

    Raises
    ------
    ValueError
        No segments, a non-positive `fs_hz`, an unknown `panels`, neither
        `out_path` nor `axes`, or ``panels="clock"`` on a source with no usable
        per-segment timestamps.
    """
    segments = list(segments or ())
    if not segments:
        raise ValueError("no segments to map")
    fs_hz = float(fs_hz or 0.0)
    if fs_hz <= 0.0:
        raise ValueError(f"fs_hz must be positive to place segments in seconds; got {fs_hz}")
    if panels not in _WANTED_PANELS:
        raise ValueError(
            f"panels must be one of {tuple(_WANTED_PANELS)}; got {panels!r}"
        )

    supplied = _as_axes_list(axes)
    if supplied is None and out_path is None:
        raise ValueError("pass out_path to write a figure, or axes to draw into one")
    if supplied is not None and not supplied:
        raise ValueError("axes was given but holds no axes to draw into")

    have_wall_clock = _have_wall_clock(segments)
    wanted = _WANTED_PANELS[panels]
    note = None

    if "wall_clock" in wanted and not have_wall_clock:
        if panels == "clock":
            raise ValueError(
                "panels='clock' draws the wall-clock timeline alone, and this source "
                "carries no usable per-segment timestamps to place it from"
            )
        wanted = tuple(name for name in wanted if name != "wall_clock")
        note = _NO_TIMESTAMPS_NOTE

    if supplied is not None and len(supplied) < len(wanted):
        wanted = wanted[: len(supplied)]
        note = _ONE_AXES_NOTE

    fig = None
    targets = supplied
    if supplied is None:
        size = _figure_size(len(segments), len(wanted), annotate) if figsize is None else figsize
        fig = _new_figure(size, dpi)
        # subplots returns a bare axes for a single panel and an array for two.
        created = fig.subplots(len(wanted), 1)
        targets = [created] if len(wanted) == 1 else list(created)

    clock_title = wall_clock_title or (
        _WALL_CLOCK_TITLE_PAIRED if len(wanted) > 1 else _WALL_CLOCK_TITLE
    )

    drawn = []
    panel_keys = []
    joins_drawn = 0
    for axis, name in zip(targets, wanted):
        if name == "concatenated":
            keys, drew = _draw_concatenated(
                axis, segments, fs_hz, stitch_frames, title or _CONCATENATED_TITLE, annotate
            )
        else:
            keys, drew = _draw_wall_clock(axis, segments, clock_title, annotate)
        drawn.append(name)
        panel_keys.append((axis, keys))
        joins_drawn += drew

    written = None
    if fig is None:
        # The caller owns the figure's margin, so each panel carries its key
        # inside its own axes; a figure-level legend would land in a band this
        # function has no right to reserve.
        for axis, keys in panel_keys:
            # Fixed rather than scored: `emptiest_corner` only sees Line2D and
            # collection artists (for the point-density scan it runs), and
            # this panel's data is `barh` patches, which it cannot see at all
            # — a dynamic score here would place the legend by the join rules
            # alone and ignore the bars. Geometrically the bars do not need
            # scoring anyway: segments are drawn in time order, one per row,
            # so row 0 (top) sits at the smallest x and the last row (bottom)
            # at the largest — a diagonal, not edge-to-edge fill — and that
            # makes the upper-right corner empty on every render.
            legend_corner(axis, handles=keys, loc="upper right", labelspacing=0.7)
    else:
        # One figure-level key instead of a box inside every panel: on a timeline
        # this tight a per-axes legend covers the bars it is explaining, and
        # `_add_caption` owns the bottom margin and reserves exactly the height
        # the caption and the legend measure.
        caption = ""
        if annotate:
            caption = _fold_caption(_caption_parts(drawn, note is not None, joins_drawn))
        _add_caption(
            fig,
            caption,
            legend_handles=_dedupe_handles(key for _, keys in panel_keys for key in keys),
            legend_fontsize=LEGEND_FONTSIZE,
            legend_framealpha=LEGEND_FRAME_ALPHA,
        )
        written = _save_and_release(fig, out_path)

    concatenated_span_s = _concatenated_span_s(segments, fs_hz)
    wall_clock_span_s = _wall_clock_span_s(segments) if have_wall_clock else None

    if note:
        logger.warning("segment boundary map: %s", note)
    logger.info(
        "segment boundary map: %d segment(s), %s panel(s), %d join rule(s), "
        "%.1f s in the file%s",
        len(segments),
        "+".join(drawn),
        joins_drawn,
        concatenated_span_s,
        "" if wall_clock_span_s is None else f", {wall_clock_span_s:.1f} s on the clock",
    )
    return {
        "files": {"png": None if written is None else str(written)},
        "panels": drawn,
        "n_segments": len(segments),
        "concatenated_span_s": float(concatenated_span_s),
        "wall_clock_span_s": (
            None if wall_clock_span_s is None else float(wall_clock_span_s)
        ),
        "note": note,
    }


__all__ = ["plot_segment_boundary_map"]
