"""Threshold-crossing raster: the fastest read on whether a chip is alive.

This deliberately does not sort. A spike sorter takes minutes to hours and can
fail for reasons that have nothing to do with the tissue, which is useless as a
QC gate. A per-electrode threshold takes seconds and answers the only question
being asked at review time: is anything firing, on how many electrodes, and does
it stop partway through the recording.

This module only DRAWS. The events come from the diagnostics' detector,
:func:`mea_modules.quality.detection.detect_events` (SpikeInterface's
``detect_peaks``, by channel, thresholds at k x our MAD-sigma), run by whoever
had the recording open -- a capsule, which caches them. Two home-grown
detectors used to live here and in ``quality``; both are gone, so a raster and
an activity rate can no longer disagree about what an event is.
"""

import logging

from ..quality.detection import (
    DEFAULT_DETECT_THRESHOLD,
    DEFAULT_EXCLUDE_SWEEP_MS,
    channel_labels,
)

from .channel_layout import (
    _add_caption,
    _fold_caption,
    _legend_dot,
    _legend_line,
    _legend_patch,
    _new_figure,
    _save_and_release,
)
from .figure_style import legend_corner
from .figure_text import (
    CONTIGUOUS_AXIS,
    NO_DATA_SHADING,
    REAL_ELAPSED_AXIS,
    SEAM,
    acronym_note,
)
from .timebase import (
    _GAP_SHADE_ALPHA,
    _GAP_SHADE_COLOR,
    JOIN_LABEL_INSTANT,
    JOIN_LABEL_SPANNING,
    GAP_LABEL_BETWEEN,
    GAP_LABEL_WITHIN,
    _SEGMENT_GAP_SHADE_ALPHA,
    _SEGMENT_GAP_SHADE_COLOR,
    draw_join_marks,
    gap_spans_by_kind,
    shade_gap_kinds,
    join_marks,
    joins_span_time,
    resolve_time_gaps,
    sample_times,
)
from .traces import (
    _CONTIGUOUS_XLABEL,
    _REAL_TIME_XLABEL,
    _frames_to_seconds,
    _has_time_vector,
    _resolve_frame_window,
    resolve_plot_quality,
)

logger = logging.getLogger(__name__)

_RASTER_FIGSIZE = (16.0, 8.0)
_RASTER_DPI = 180

# One Maxwell recording config routes at most ~1k electrodes; anything past that
# is a full-array view no one wants to raster in a single figure.
_DEFAULT_MAX_CHANNELS = 1024

# Default read window; a whole AxonTracking scan is far too much to threshold
# for a review plot.
_DEFAULT_DURATION_S = 60.0

# Past this many rows individual tick labels are unreadable.
_MAX_YTICKS = 64


def _select_channels(recording, channel_ids, max_channels):
    """Channel ids to threshold, thinned evenly across the array if needed.

    Thinning takes every k-th channel rather than the first N: channel order
    tracks position on the array, so a prefix would raster one corner of the
    chip and report the rest as silent.
    """
    import numpy as np

    if channel_ids is None:
        channel_ids = list(recording.get_channel_ids())
    channel_ids = list(channel_ids)
    if max_channels is None or max_channels <= 0 or len(channel_ids) <= int(max_channels):
        return channel_ids

    keep = np.linspace(0, len(channel_ids) - 1, num=int(max_channels)).astype(int)
    logger.warning(
        "rastering %d of %d channels (evenly spaced); raise max_channels to cover all",
        int(keep.size),
        len(channel_ids),
    )
    return [channel_ids[index] for index in np.unique(keep)]


def _channel_labels(channel_ids):
    """Numeric y-axis labels for the raster rows: the detector's own rule.

    Imported rather than restated, because the event labels a raster plots come
    out of :func:`mea_modules.quality.detection.channel_labels`, and a row axis
    built by a second rule would not line up with them.
    """
    return channel_labels(channel_ids)


def _seconds_to_frames(recording, times, fs, has_times):
    """Recover sample indices from event times so a gap axis can be applied.

    Cached events are seconds, but the gap tables are indexed by sample, so the
    mapping has to be undone before it can be redone against real elapsed time.
    Without a time vector the inverse is exact — the times came from
    ``frame / fs`` — and with one the recording is asked to invert its own
    mapping.
    """
    import numpy as np

    times = np.asarray(times, dtype=float)
    if has_times:
        try:
            return np.asarray(recording.time_to_sample_index(times), dtype=np.int64)
        except Exception:
            # Same tolerance as _frames_to_seconds: not every recording object
            # implements the inverse, and falling back only costs sub-sample
            # rounding on a recording whose time vector starts at zero.
            logger.debug("recording cannot invert its time vector; assuming frame / fs event times")
    return np.rint(times * float(fs)).astype(np.int64) if fs else times.astype(np.int64)


def plot_raster_threshold(
    recording,
    out_path,
    *,
    events,
    channel_ids=None,
    max_channels=_DEFAULT_MAX_CHANNELS,
    start_time_s=0.0,
    duration_s=_DEFAULT_DURATION_S,
    threshold_factor=DEFAULT_DETECT_THRESHOLD,
    exclude_sweep_ms=DEFAULT_EXCLUDE_SWEEP_MS,
    stitch_frames=(),
    title=None,
    figsize=_RASTER_FIGSIZE,
    dpi=_RASTER_DPI,
    time_gaps=None,
    quality=None,
    annotate=True,
):
    """Write a threshold-crossing raster to `out_path`; return the path.

    `events` is the ``(times_s, labels)`` pair a detection produced -- in
    seconds on the recording's own timeline, labelled with integer electrode
    ids -- and it is required: this emitter never detects. `recording` answers
    only for geometry, the sampling rate and the analysed span, so a
    :class:`mea_modules.diagnostics.cache.CachedProbe` is enough. Rows are the
    channels in `channel_ids` (default: every channel, thinned evenly to
    `max_channels`); events on other channels are simply not drawn.

    `threshold_factor` and `exclude_sweep_ms` are the detection's own settings,
    passed so the legend states what the dots are. They change nothing drawn.

    `stitch_frames` draws the segment joins as dotted red verticals — pass the
    concatenation's join offsets in FRAMES (the one convention across every
    emitter, see :func:`mea_modules.diagnostics.timebase.join_marks`) and the
    plot shows immediately whether activity dies in a particular segment. On a
    real-elapsed axis each join is drawn as TWO rules, the earlier segment's end
    and the later one's start, because minutes of wall clock sit between them.
    Rows are labelled individually only while there are few enough to read.

    `annotate` controls the explanatory chrome. True keeps the title and the
    caption block. False draws neither, leaving axes, units and the legend —
    the presentation-plot register the pipeline's own figures use, where the
    title's information lives in the filename instead.

    `time_gaps` chooses which timeline the x axis is. Left None it is the
    contiguous one — sample index over sampling rate, labelled ``time (s)`` —
    which is what every existing caller gets. Given the gap structure from
    :mod:`mea_modules.io.gaps` (a ``frame_gaps`` dict, a ``well_gap_summary``, or
    ``{"gaps": ..., "segment_gaps": ...}``) events are placed at their real
    elapsed time, the stretches holding no data are shaded, and the label says
    ``time (s, real elapsed)``. That distinction matters most here: read as a
    spike train, a raster on the contiguous axis quietly closes up every break,
    so inter-event intervals across one are wrong. `stitch_frames` are mapped
    onto whichever axis is in force, by the same gap table as the events.

    `quality` picks a render preset (:data:`mea_modules.diagnostics.traces.
    PLOT_QUALITY_PRESETS`): ``"draft"`` (the default) or ``"high"``, which
    raises the dots-per-inch so a dense raster survives zooming instead of
    collapsing into a block. An explicit `dpi` argument still wins.
    """
    import numpy as np

    # Only the dpi half of the preset applies here: a raster draws detected
    # events, not decimated samples, so there is no point budget to raise.
    if dpi == _RASTER_DPI:
        dpi = resolve_plot_quality(quality)["dpi"]

    channel_ids = _select_channels(recording, channel_ids, max_channels)
    if not channel_ids:
        raise ValueError("recording has no channels to raster")

    if events is None:
        raise ValueError("pass the detected events; this emitter does not detect")
    event_times, event_labels = (np.asarray(part) for part in events)
    labels = _channel_labels(channel_ids)
    # Only the rows this figure has: a thinned raster must not draw the events
    # of channels it left out on rows that belong to other electrodes.
    if event_labels.size:
        keep = np.isin(event_labels, labels)
        event_times, event_labels = event_times[keep], event_labels[keep]

    gaps, segment_gaps = resolve_time_gaps(time_gaps)
    real_time = time_gaps is not None
    window_start, window_end, fs = _resolve_frame_window(recording, start_time_s, duration_s)
    spans = {}
    shaded = {"within_segment": 0, "between_segment": 0}
    if real_time:
        # Events come back as seconds on whichever timeline the detector used;
        # re-index them to samples so the gap offsets can be applied.
        frames = _seconds_to_frames(recording, event_times, fs, _has_time_vector(recording))
        event_times = sample_times(frames, fs, gaps=gaps, segment_gaps=segment_gaps)
        # Split by kind rather than pooled: a scan carries thousands of
        # microsecond frame-counter breaks and a handful of half-minute
        # acquisition gaps, and one grey for both hides the second in the first.
        spans = gap_spans_by_kind(window_end, fs, gaps=gaps, segment_gaps=segment_gaps)

    fig = _new_figure(figsize, dpi)
    ax = fig.subplots()
    if event_times.size:
        # Rasterized: hundreds of thousands of markers as vectors make a PDF or
        # SVG unopenable, and the points carry no detail worth keeping vector.
        ax.scatter(
            event_times,
            event_labels,
            s=1.0,
            c="black",
            marker=".",
            linewidths=0,
            alpha=0.6,
            rasterized=True,
        )

    marks = join_marks(
        stitch_frames, fs, gaps=gaps, segment_gaps=segment_gaps, real_time=real_time
    )
    drawn_joins = draw_join_marks(ax, marks)

    # Span the window that was actually analysed, not the extent of the events
    # and NEVER the extent of the joins: a raster where firing stops halfway has
    # to show the silence. Clipping to the joins used to be the first branch
    # here, which on a two-segment concatenation is a single interior join and
    # therefore a degenerate `set_xlim(x, x)` that matplotlib silently widens by
    # ±5% — 95% of the data off-canvas with nothing on the figure saying so.
    edges_frames = [window_start, max(window_start, window_end - 1)]
    if real_time:
        edges = sample_times(edges_frames, fs, gaps=gaps, segment_gaps=segment_gaps)
    else:
        edges = _frames_to_seconds(recording, edges_frames, fs)
    if float(edges[1]) > float(edges[0]):
        ax.set_xlim(float(edges[0]), float(edges[1]))

    if real_time:
        # Shade after the limits are set, so the bands cover exactly what is on
        # screen and no more.
        x_lower, x_upper = ax.get_xlim()
        shaded = shade_gap_kinds(ax, spans, x_min=x_lower, x_max=x_upper)
        logger.info(
            "raster: shaded %d within-segment break(s) of %d and %d between-segment "
            "gap(s) of %d in window",
            shaded["within_segment"], len(spans.get("within_segment", ())),
            shaded["between_segment"], len(spans.get("between_segment", ())),
        )

    if labels.size:
        ax.set_ylim(float(labels.min()) - 1.0, float(labels.max()) + 1.0)
        if labels.size <= _MAX_YTICKS:
            ax.set_yticks(sorted(int(label) for label in labels))

    ax.set_xlabel(_REAL_TIME_XLABEL if real_time else _CONTIGUOUS_XLABEL)
    ax.set_ylabel("electrode id")
    if annotate and title:
        ax.set_title(title)
    ax.grid(False)

    # Legend every encoding (Adam, 2026-08-11). A raster is three different
    # marks — dots, dotted rules, grey bands — and none of them is self-evident:
    # a reader cannot otherwise tell a segment join from a dead stretch, or
    # "nothing recorded" from "nothing fired".
    handles = []
    # Conditional like every other key here: zero events means the scatter
    # above was never drawn, and a key for a mark that is not on the canvas
    # tells the reader something was plotted that was not.
    if event_times.size:
        handles.append(
            _legend_dot(
                "black",
                f"≥{float(threshold_factor):g}× MAD-σ peak "
                f"({float(exclude_sweep_ms):g} ms)",
                size=4.0,
            )
        )
    if drawn_joins:
        handles.append(
            _legend_line(
                "red",
                JOIN_LABEL_SPANNING if joins_span_time(marks) else JOIN_LABEL_INSTANT,
                lw=0.9,
                linestyle=":",
            )
        )
    if real_time and shaded["within_segment"]:
        handles.append(
            _legend_patch(_GAP_SHADE_COLOR, GAP_LABEL_WITHIN, alpha=_GAP_SHADE_ALPHA)
        )
    if real_time and shaded["between_segment"]:
        handles.append(
            _legend_patch(
                _SEGMENT_GAP_SHADE_COLOR, GAP_LABEL_BETWEEN, alpha=_SEGMENT_GAP_SHADE_ALPHA
            )
        )
    # A raster is a scatter of up to a few hundred thousand events, so
    # loc="best" reliably picked a central spot instead of a corner — this
    # scores the four corners against the drawn points and takes the emptiest
    # one, and must run after the scatter/join marks above are on the axis.
    # `handles` can now be empty (zero events, no join, no shading), which an
    # empty-list legend would still draw as an empty box.
    if handles:
        legend_corner(ax, handles=handles, labelspacing=0.7)

    caption = ""
    if annotate:
        caption_parts = [acronym_note("MAD")]
        if drawn_joins:
            caption_parts.append(SEAM)
        caption_parts.append(REAL_ELAPSED_AXIS if real_time else CONTIGUOUS_AXIS)
        if real_time and any(shaded.values()):
            caption_parts.append(NO_DATA_SHADING)
        caption_parts.append(
            "Detection is per-electrode peak detection, not spike sorting: one dot is one "
            "negative peak past threshold on one electrode, not one identified neuron."
        )
        caption = _fold_caption(caption_parts)
    _add_caption(fig, caption)

    out_path = _save_and_release(fig, out_path)
    logger.info(
        "wrote threshold raster: %s (%d channels, %d events)",
        out_path,
        len(channel_ids),
        int(event_times.size),
    )
    return out_path
