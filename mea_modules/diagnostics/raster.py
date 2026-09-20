"""Threshold-crossing raster: the fastest read on whether a chip is alive.

This deliberately does not sort. A spike sorter takes minutes to hours and can
fail for reasons that have nothing to do with the tissue, which is useless as a
QC gate. A per-channel MAD threshold with a refractory period takes seconds and
answers the only question being asked at review time: is anything firing, on how
many electrodes, and does it stop partway through the recording.

The detection rule ported from the working build is a negative-going local
minimum below `threshold_factor` * sigma, where sigma is estimated as
MAD / 0.6745 over a few evenly spaced windows. Events on the same channel closer
together than the refractory period are collapsed to the first.
"""

import logging

from ..quality.robust import mad_sigma

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
    _read_traces,
    _resolve_frame_window,
    resolve_plot_quality,
)

logger = logging.getLogger(__name__)

_RASTER_FIGSIZE = (16.0, 8.0)
_RASTER_DPI = 180

# Defaults ported from the working build.
_DEFAULT_THRESHOLD_FACTOR = 5.0
_DEFAULT_REFRACTORY_MS = 0.8
_DEFAULT_NOISE_WINDOW_FRAMES = 20_000
_DEFAULT_NOISE_WINDOWS = 4
_DEFAULT_DETECTION_CHUNK_FRAMES = 50_000

# MAD -> Gaussian sigma. Quartile-based so a few large spikes cannot inflate the
# noise estimate the way a plain std would.

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
    """Numeric y-axis labels for the raster rows.

    Maxwell channel ids are integer-like, and plotting the real id keeps the
    raster comparable with the layout plot. When ids are not numeric at all,
    fall back to row position for every channel so the axis stays consistent.
    """
    import numpy as np

    try:
        return np.asarray([int(channel_id) for channel_id in channel_ids], dtype=np.int64)
    except (TypeError, ValueError):
        return np.arange(len(channel_ids), dtype=np.int64)


def _seconds_to_frames(recording, times, fs, has_times):
    """Recover sample indices from event times so a gap axis can be applied.

    :func:`detect_threshold_crossings` reports seconds, but the gap tables are
    indexed by sample, so the mapping has to be undone before it can be redone
    against real elapsed time. Without a time vector the inverse is exact —
    the times came from ``frame / fs`` — and with one the recording is asked to
    invert its own mapping.
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


def estimate_channel_thresholds(
    recording,
    channel_ids=None,
    threshold_factor=_DEFAULT_THRESHOLD_FACTOR,
    window_frames=_DEFAULT_NOISE_WINDOW_FRAMES,
    n_windows=_DEFAULT_NOISE_WINDOWS,
    start_time_s=0.0,
    duration_s=None,
    return_in_uV=False,
):
    """Per-channel detection threshold, as `threshold_factor` * MAD sigma.

    Noise is estimated over `n_windows` evenly spaced windows of `window_frames`
    samples and combined with a median, so a burst landing inside one window
    cannot drag the threshold up. Returns a positive float array aligned with
    `channel_ids`; the caller applies it as a negative-going bound.
    """
    import numpy as np

    if channel_ids is None:
        channel_ids = list(recording.get_channel_ids())
    channel_ids = list(channel_ids)
    window_start, window_end, _fs = _resolve_frame_window(recording, start_time_s, duration_s)
    span = int(window_end - window_start)
    if span <= 0 or not channel_ids:
        return np.asarray([], dtype=float)

    window = max(256, min(int(window_frames), span))
    max_start = max(0, span - window)
    if max_start <= 0:
        starts = [window_start]
    else:
        starts = (
            window_start
            + np.unique(np.linspace(0, max_start, num=max(1, int(n_windows)), dtype=np.int64))
        ).tolist()

    estimates = []
    for start in starts:
        traces = _read_traces(recording, int(start), min(window_end, int(start) + window), channel_ids, return_in_uV)
        if traces.size == 0:
            continue
        estimates.append(mad_sigma(traces))

    if not estimates:
        return np.full(len(channel_ids), max(1e-6, float(threshold_factor)), dtype=float)
    thresholds = np.median(np.vstack(estimates), axis=0) * float(threshold_factor)
    # A perfectly flat (dead or saturated) channel gives sigma 0, which would
    # otherwise fire on every sample.
    return np.clip(np.asarray(thresholds, dtype=float), 1e-6, None)


def detect_threshold_crossings(
    recording,
    channel_ids=None,
    thresholds=None,
    threshold_factor=_DEFAULT_THRESHOLD_FACTOR,
    refractory_period_ms=_DEFAULT_REFRACTORY_MS,
    start_time_s=0.0,
    duration_s=_DEFAULT_DURATION_S,
    chunk_frames=_DEFAULT_DETECTION_CHUNK_FRAMES,
    return_in_uV=False,
):
    """Find negative threshold crossings; return (event_times_s, event_labels).

    An event is a sample that is below ``-threshold`` and is a local minimum
    (``x[i] <= x[i-1]`` and ``x[i] < x[i+1]``). Events within
    `refractory_period_ms` of the previous event *on the same channel* are
    dropped, which is what stops one spike from being counted five times.

    With `thresholds` None they are estimated over the same window via
    :func:`estimate_channel_thresholds`. Traces are read in `chunk_frames`
    chunks with a one-sample overlap so crossings on a chunk edge are not lost.

    Event times come from the recording's time vector when it has one, so a
    concatenated recording rasters on its real timeline.
    """
    import numpy as np

    if channel_ids is None:
        channel_ids = list(recording.get_channel_ids())
    channel_ids = list(channel_ids)
    window_start, window_end, fs = _resolve_frame_window(recording, start_time_s, duration_s)
    empty = (np.asarray([], dtype=float), np.asarray([], dtype=np.int64))
    if int(window_end - window_start) <= 0 or not channel_ids:
        return empty

    if thresholds is None:
        thresholds = estimate_channel_thresholds(
            recording,
            channel_ids=channel_ids,
            threshold_factor=threshold_factor,
            start_time_s=start_time_s,
            duration_s=duration_s,
            return_in_uV=return_in_uV,
        )
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.size != len(channel_ids):
        raise ValueError(f"thresholds has {thresholds.size} entries for {len(channel_ids)} channels")

    labels = _channel_labels(channel_ids)
    refractory_samples = max(1, int(round(fs * (float(refractory_period_ms) / 1000.0))))
    chunk = max(1024, int(chunk_frames))
    last_emitted = np.full(len(channel_ids), -refractory_samples - 1, dtype=np.int64)

    frame_parts = []
    label_parts = []
    for chunk_start in range(window_start, window_end, chunk):
        chunk_stop = min(window_end, chunk_start + chunk)
        # One sample of overlap on each side: a local minimum needs both
        # neighbours, and the ones on the chunk seam live in the next chunk.
        read_start = max(window_start, chunk_start - 1)
        read_stop = min(window_end, chunk_stop + 1)
        traces = _read_traces(recording, read_start, read_stop, channel_ids, return_in_uV).astype(float, copy=False)
        if traces.shape[0] < 3 or traces.shape[1] == 0:
            continue

        center = traces[1:-1, :]
        crossing = (center <= -thresholds[None, :]) & (center <= traces[:-2, :]) & (center < traces[2:, :])
        if not crossing.any():
            continue

        for channel_index in range(crossing.shape[1]):
            candidates = np.flatnonzero(crossing[:, channel_index]).astype(np.int64) + 1 + read_start
            # Drop candidates from the overlap so neighbouring chunks cannot
            # both claim the same sample.
            candidates = candidates[(candidates >= chunk_start) & (candidates < chunk_stop)]
            if candidates.size == 0:
                continue
            kept = []
            previous = int(last_emitted[channel_index])
            for frame in candidates.tolist():
                if frame - previous < refractory_samples:
                    continue
                kept.append(frame)
                previous = frame
            last_emitted[channel_index] = previous
            if kept:
                frame_parts.append(np.asarray(kept, dtype=np.int64))
                label_parts.append(np.full(len(kept), labels[channel_index], dtype=np.int64))

    if not frame_parts:
        return empty

    frames = np.concatenate(frame_parts)
    event_labels = np.concatenate(label_parts)
    order = np.argsort(frames, kind="stable")
    frames = frames[order]
    event_labels = event_labels[order]
    times = _frames_to_seconds(recording, frames, fs, has_times=_has_time_vector(recording))
    return times.astype(float, copy=False), event_labels


def plot_raster_threshold(
    recording,
    out_path,
    channel_ids=None,
    max_channels=_DEFAULT_MAX_CHANNELS,
    start_time_s=0.0,
    duration_s=_DEFAULT_DURATION_S,
    threshold_factor=_DEFAULT_THRESHOLD_FACTOR,
    refractory_period_ms=_DEFAULT_REFRACTORY_MS,
    chunk_frames=_DEFAULT_DETECTION_CHUNK_FRAMES,
    stitch_frames=(),
    title=None,
    return_in_uV=False,
    figsize=_RASTER_FIGSIZE,
    dpi=_RASTER_DPI,
    time_gaps=None,
    quality=None,
    annotate=True,
    events=None,
):
    """Write a threshold-crossing raster to `out_path`; return the path.

    Detects events with :func:`detect_threshold_crossings` over `duration_s`
    seconds from `start_time_s` (None for the whole recording) on at most
    `max_channels` channels, then scatters them as time vs electrode id.

    `events` supplies that detection's result -- the ``(times_s, labels)`` pair
    -- instead of running it. Detection is the expensive half of this figure,
    and how expensive depends entirely on the channel count: ~2 minutes over a
    segment's ~1000 routed channels, against ~3 seconds over a concatenated
    well's 266-channel shared electrode set. It is also run TWICE per segment
    today, because the real-elapsed twin re-detects the identical events. So a
    caller that has already detected passes the result here, and `recording`
    need only answer for geometry, sampling rate and the analysed span: a
    :class:`mea_modules.diagnostics.cache.CachedProbe` is enough. The drawing
    below is the same either way, which is the point -- one definition of this
    figure, two sources for its numbers.

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
    PLOT_QUALITY_PRESETS`): ``"draft"`` (the default, and what this development
    pass ships) or ``"high"``, which raises the dots-per-inch so a dense raster
    survives zooming instead of collapsing into a block. An explicit `dpi`
    argument still wins over the preset.
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
        event_times, event_labels = detect_threshold_crossings(
            recording,
            channel_ids=channel_ids,
            threshold_factor=threshold_factor,
            refractory_period_ms=refractory_period_ms,
            start_time_s=start_time_s,
            duration_s=duration_s,
            chunk_frames=chunk_frames,
            return_in_uV=return_in_uV,
        )
    else:
        event_times, event_labels = (np.asarray(part) for part in events)
    labels = _channel_labels(channel_ids)

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
                f"≥{float(threshold_factor):g}× MAD-σ crossing "
                f"({float(refractory_period_ms):g} ms)",
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
            "Detection is a threshold crossing count, not spike sorting: one dot is one "
            "downward crossing on one electrode, not one identified neuron."
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
