"""Trace review plots, and the representative-channel selection behind them.

Plotting every channel of a Maxwell recording is useless — a thousand traces
stacked in one figure is a grey block. What actually gets reviewed is a handful
of channels, and *which* handful is the whole question. The rule ported from the
working build is two-stage:

1. collapse the layout to one channel per electrode cluster, so no clump is
   over-represented just because it routed six electrodes (see
   :mod:`mea_modules.diagnostics.channel_layout`), then
2. rank those cluster representatives by activity RMS and keep the loudest.

Ranking by RMS is what makes the plot informative: a randomly chosen channel on
a healthy chip is usually silent, so an unranked sample looks identical to a
dead recording.

Reading traces is the point of this module, but every read is bounded — by a
time window, by a channel count, and by a downsample step — so a review plot can
never pull a multi-gigabyte recording into memory.
"""

import logging

from .channel_layout import (
    _channel_xy,
    _new_figure,
    _pick_cluster_representative,
    _save_and_release,
    detect_electrode_clusters,
)
from .timebase import _shade_gap_spans, gap_spans, resolve_time_gaps, sample_times

logger = logging.getLogger(__name__)

# Axis labels. The default one is deliberately unchanged: an x axis that says
# nothing about gaps is the contiguous sample timeline, which is what every
# existing plot already shows.
_CONTIGUOUS_XLABEL = "time (s)"
_REAL_TIME_XLABEL = "time (s, real elapsed)"

_TRACE_FIGSIZE = (13.33, 7.5)
_TRACE_DPI = 180

# Defaults ported from the working build.
_DEFAULT_MAX_POINTS = 150_000
_DEFAULT_BLOCK_FRAMES = 200_000
_DEFAULT_ACTIVITY_CHUNKS = 4
_DEFAULT_ACTIVITY_CHUNK_FRAMES = 4_000

# Above this the figure takes minutes to rasterize and reads as a solid smear.
_POINTS_WARN = 200_000

# A trace plot is only readable at single-digit channel counts.
_DEFAULT_MAX_CHANNELS = 8

# Default read window. None would mean "the whole recording", which on a
# concatenated AxonTracking scan is tens of gigabytes.
_DEFAULT_DURATION_S = 60.0


def _resolve_frame_window(recording, start_time_s=0.0, duration_s=None):
    """Clamp a (start, duration) request to a valid [start_frame, end_frame).

    Times are offsets from the first sample, not positions on the recording's
    own time vector, so this stays meaningful whether or not wall-clock times
    have been attached.
    """
    fs = float(recording.get_sampling_frequency())
    total = int(recording.get_num_samples())
    if total <= 0 or fs <= 0.0:
        return 0, 0, fs

    start_frame = max(0, int(round(float(start_time_s or 0.0) * fs)))
    start_frame = min(start_frame, max(0, total - 1))
    if duration_s is None:
        end_frame = total
    else:
        span = max(1, int(round(float(duration_s) * fs)))
        end_frame = min(total, start_frame + span)
    return start_frame, max(start_frame, end_frame), fs


def _has_time_vector(recording):
    """True when the recording carries explicit sample times."""
    try:
        return bool(recording.has_time_vector())
    except Exception:
        # Not every recording object implements it; absence just means "no".
        return False


def _frames_to_seconds(recording, frames, fs, has_times=None):
    """Convert sample indices to seconds, honouring an attached time vector."""
    import numpy as np

    frames = np.asarray(frames, dtype=np.int64)
    if has_times is None:
        has_times = _has_time_vector(recording)
    if has_times:
        try:
            return np.asarray(recording.sample_index_to_time(frames), dtype=float)
        except Exception:
            pass
    return frames.astype(float) / float(fs) if fs else frames.astype(float)


def _read_traces(recording, start_frame, end_frame, channel_ids, return_in_uV):
    """get_traces with a 2-D result guaranteed, even for a single channel."""
    import numpy as np

    traces = recording.get_traces(
        start_frame=int(start_frame),
        end_frame=int(end_frame),
        channel_ids=list(channel_ids),
        return_in_uV=bool(return_in_uV),
    )
    traces = np.asarray(traces)
    if traces.ndim == 1:
        traces = traces[:, None]
    return traces


def channel_activity_rms(
    recording,
    channel_ids=None,
    num_chunks=_DEFAULT_ACTIVITY_CHUNKS,
    chunk_frames=_DEFAULT_ACTIVITY_CHUNK_FRAMES,
    seed=0,
    start_time_s=0.0,
    duration_s=None,
    return_in_uV=False,
):
    """RMS amplitude per channel, sampled from a few random chunks.

    Returns a float array aligned with `channel_ids` (all channels by default).
    Chunk starts are drawn from a seeded generator and shared by every channel,
    so the scores are comparable to each other and stable across runs — that is
    what makes them usable as a ranking key.

    All channels are read together per chunk rather than one at a time: the
    samples, and therefore the scores, are identical, but a recording backed by
    a memmap is touched once per chunk instead of once per channel.
    """
    import numpy as np

    if channel_ids is None:
        channel_ids = list(recording.get_channel_ids())
    channel_ids = list(channel_ids)
    if not channel_ids:
        return np.asarray([], dtype=float)

    window_start, window_end, _fs = _resolve_frame_window(recording, start_time_s, duration_s)
    span = int(window_end - window_start)
    if span <= 0:
        return np.zeros(len(channel_ids), dtype=float)

    chunk_frames = max(100, int(chunk_frames))
    num_chunks = max(1, int(num_chunks))

    if span <= chunk_frames:
        starts = [window_start]
        chunk_frames = span
    else:
        rng = np.random.default_rng(int(seed))
        starts = (
            window_start + rng.integers(0, span - chunk_frames, size=num_chunks, endpoint=False)
        ).tolist()

    sum_squares = np.zeros(len(channel_ids), dtype=float)
    count = 0
    for start in starts:
        traces = _read_traces(recording, int(start), int(start) + chunk_frames, channel_ids, return_in_uV)
        if traces.size == 0:
            continue
        traces = traces.astype(float, copy=False)
        sum_squares += np.sum(traces * traces, axis=0)
        count += int(traces.shape[0])

    return np.sqrt(sum_squares / max(count, 1))


def select_representative_channels(
    recording,
    n_channels=_DEFAULT_MAX_CHANNELS,
    eps=None,
    max_cluster_size_warn=9,
    num_chunks=_DEFAULT_ACTIVITY_CHUNKS,
    chunk_frames=_DEFAULT_ACTIVITY_CHUNK_FRAMES,
    seed=0,
    start_time_s=0.0,
    duration_s=None,
    return_in_uV=False,
):
    """Pick the `n_channels` most active cluster representatives.

    One candidate is taken from each electrode cluster — the member nearest the
    cluster centroid — and the candidates are then ordered by activity RMS,
    loudest first. Pass `n_channels` <= 0 to get every candidate in that order.

    Falls back to the recording's own channel order when the probe carries no
    usable locations, so this never fails on a probe-less recording.

    Returns channel ids in the recording's own id type, ready to hand back to
    ``get_traces``.
    """
    channel_ids, xs, ys = _channel_xy(recording)
    if not channel_ids:
        return []

    if xs is None:
        logger.warning("no usable channel locations; falling back to recording channel order")
        return list(channel_ids) if n_channels <= 0 else list(channel_ids[: max(1, int(n_channels))])

    clusters = detect_electrode_clusters(xs, ys, eps=eps, max_cluster_size_warn=max_cluster_size_warn)
    candidates = [channel_ids[_pick_cluster_representative(xs, ys, cluster)] for cluster in clusters]
    if not candidates:
        candidates = list(channel_ids)

    scores = channel_activity_rms(
        recording,
        channel_ids=candidates,
        num_chunks=num_chunks,
        chunk_frames=chunk_frames,
        seed=seed,
        start_time_s=start_time_s,
        duration_s=duration_s,
        return_in_uV=return_in_uV,
    )
    ranked = [
        channel_id
        for channel_id, _score in sorted(zip(candidates, list(scores)), key=lambda item: item[1], reverse=True)
    ]

    logger.info(
        "selected %d representative channels from %d clusters over %d channels",
        len(ranked) if n_channels <= 0 else min(len(ranked), max(1, int(n_channels))),
        len(clusters),
        len(channel_ids),
    )
    if n_channels <= 0:
        return ranked
    return ranked[: max(1, int(n_channels))]


def _resolve_downsample_step(total_frames, fs, target_hz, max_points):
    """Frames to skip between plotted points, from both caps at once."""
    try:
        parsed_max_points = int(max_points)
    except (TypeError, ValueError):
        parsed_max_points = _DEFAULT_MAX_POINTS
    if parsed_max_points <= 0:
        step_by_points = 1
    else:
        # Below ~1k points the plot stops being a trace, so never cap harder.
        step_by_points = max(1, total_frames // max(1000, parsed_max_points))

    step_by_rate = 1
    if target_hz is not None:
        try:
            if float(target_hz) > 0.0 and fs > 0.0:
                step_by_rate = max(1, int(round(fs / float(target_hz))))
        except (TypeError, ValueError):
            step_by_rate = 1

    # Whichever cap is stricter wins.
    return max(step_by_points, step_by_rate)


def plot_traces(
    recording,
    out_path,
    channel_ids=None,
    max_channels=_DEFAULT_MAX_CHANNELS,
    start_time_s=0.0,
    duration_s=_DEFAULT_DURATION_S,
    target_hz=None,
    max_points=_DEFAULT_MAX_POINTS,
    stitch_frames=(),
    title=None,
    return_in_uV=False,
    block_frames=_DEFAULT_BLOCK_FRAMES,
    figsize=_TRACE_FIGSIZE,
    dpi=_TRACE_DPI,
    time_gaps=None,
):
    """Stack representative channel traces into one figure; return `out_path`.

    With `channel_ids` None the channels are chosen by
    :func:`select_representative_channels` — cluster representatives ranked by
    activity — which is almost always what you want for a review artifact.

    The read is bounded three ways: `duration_s` seconds from `start_time_s`
    (pass None for the whole recording, only sane on short segments),
    `max_channels` traces, and a decimation step derived from `target_hz` and
    `max_points`. Traces are pulled in `block_frames` blocks and decimated as
    they arrive, so peak memory is one block, not the window.

    `stitch_frames` draws a red vertical line per concatenation boundary. When
    the recording carries a time vector with gaps at those boundaries, the line
    is broken with NaN so the plot does not draw a fake ramp across the gap.

    `time_gaps` chooses which timeline the x axis is. Left None it is the
    contiguous one — sample index over sampling rate, labelled ``time (s)`` —
    which is what every existing caller gets and what SpikeInterface believes.
    Given the gap structure from :mod:`mea_modules.io.gaps` (a ``frame_gaps``
    dict, a ``well_gap_summary``, or ``{"gaps": ..., "segment_gaps": ...}``) the
    axis becomes real elapsed time instead: every sample is pushed right by the
    time missing before it, the stretches holding no data are shaded, and the
    label says ``time (s, real elapsed)`` so the two can never be confused. The
    supplied structure wins over any time vector on the recording, since the two
    would otherwise both correct for the same gaps.
    """
    import numpy as np

    window_start, window_end, fs = _resolve_frame_window(recording, start_time_s, duration_s)
    total = int(window_end - window_start)
    if total <= 0:
        raise ValueError("requested trace window contains no samples")

    if channel_ids is None:
        channel_ids = select_representative_channels(
            recording,
            n_channels=max_channels,
            return_in_uV=return_in_uV,
        )
    else:
        channel_ids = list(channel_ids)
        if max_channels is not None and max_channels > 0:
            channel_ids = channel_ids[: int(max_channels)]
    if not channel_ids:
        raise ValueError("no channels to plot")

    has_times = _has_time_vector(recording)
    step = _resolve_downsample_step(total, fs, target_hz, max_points)
    selected_frames = np.arange(window_start, window_end, step, dtype=np.int64)
    effective_hz = (fs / step) if step > 0 else fs
    logger.info(
        "plot traces: fs=%.2fHz step=%d effective=%.2fHz points/channel=%d channels=%d out=%s",
        fs,
        step,
        effective_hz,
        selected_frames.size,
        len(channel_ids),
        out_path,
    )
    if selected_frames.size > _POINTS_WARN:
        logger.warning(
            "plot traces: %d points/channel after decimation; lower target_hz or max_points",
            int(selected_frames.size),
        )

    gaps, segment_gaps = resolve_time_gaps(time_gaps)
    real_time = time_gaps is not None
    if real_time:
        time_vector = sample_times(selected_frames, fs, gaps=gaps, segment_gaps=segment_gaps)
        # Bounded to the window that is actually drawn, so a whole-well gap
        # structure costs only the spans inside this plot.
        spans = gap_spans(window_end, fs, gaps=gaps, segment_gaps=segment_gaps)
        logger.info(
            "plot traces: real elapsed axis spans %.3fs for %.3fs of samples",
            float(time_vector[-1] - time_vector[0]) if time_vector.size else 0.0,
            total / fs if fs else 0.0,
        )
    else:
        time_vector = _frames_to_seconds(recording, selected_frames, fs, has_times=has_times)
        spans = ()

    block = max(1, int(block_frames))
    total_blocks = max(1, (total + block - 1) // block)
    parts_per_channel = [[] for _ in channel_ids]
    for block_index, start in enumerate(range(window_start, window_end, block), start=1):
        end = min(window_end, start + block)
        traces = _read_traces(recording, start, end, channel_ids, return_in_uV)
        # Keep the global decimation phase across block boundaries, otherwise
        # the sample grid shifts every block and the x axis drifts.
        offset = (window_start - start) % step
        decimated = traces[offset::step, :]
        for channel_index in range(len(channel_ids)):
            parts_per_channel[channel_index].append(np.asarray(decimated[:, channel_index]))
        if block_index == 1 or block_index == total_blocks or block_index % max(1, total_blocks // 10) == 0:
            logger.debug("plot traces: read %d/%d blocks out=%s", block_index, total_blocks, out_path)

    fig = _new_figure(figsize, dpi)
    axes = fig.subplots(len(channel_ids), 1, sharex=True)
    if len(channel_ids) == 1:
        axes = [axes]

    # Boundary markers are positions on the same axis as the traces, so they go
    # through whichever mapping the traces did.
    if real_time:
        stitch_seconds = [
            float(sample_times([int(frame)], fs, gaps=gaps, segment_gaps=segment_gaps)[0])
            for frame in stitch_frames or ()
        ]
    else:
        stitch_seconds = [
            float(_frames_to_seconds(recording, [int(frame)], fs, has_times=has_times)[0])
            for frame in stitch_frames or ()
        ]

    shaded = 0
    for axis, channel_id, parts in zip(axes, channel_ids, parts_per_channel):
        y = np.concatenate(parts).astype(float, copy=False) if parts else np.asarray([], dtype=float)
        t = time_vector[: y.size]
        if (has_times or real_time) and y.size > 2:
            # A jump far larger than the sampling step is a gap between stitched
            # segments; NaN there so matplotlib lifts the pen.
            breaks = np.flatnonzero(np.diff(t) > 5.0 * max(step / fs if fs else 0.0, 1e-9))
            if breaks.size:
                y = y.copy()
                y[breaks + 1] = np.nan
        axis.plot(t, y, lw=0.2, color="black")
        if real_time and t.size:
            shaded = _shade_gap_spans(axis, spans, x_min=float(t[0]), x_max=float(t[-1]))
        for x_value in stitch_seconds:
            axis.axvline(x_value, color="red", lw=0.6, alpha=0.8)
        axis.set_ylabel(f"ch {channel_id}")
        axis.grid(False)

    if real_time:
        logger.info("plot traces: shaded %d gap spans of %d in window", shaded, len(spans))
    axes[-1].set_xlabel(_REAL_TIME_XLABEL if real_time else _CONTIGUOUS_XLABEL)
    if title:
        fig.suptitle(title)
    fig.tight_layout()

    out_path = _save_and_release(fig, out_path)
    logger.info("wrote traces: %s (%d channels)", out_path, len(channel_ids))
    return out_path
