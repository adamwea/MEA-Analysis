"""Raster of sorted units, ordered by firing rate.

The threshold raster upstream answers "where on the array is there signal?" —
its y axis is electrodes, and one spike lights up every electrode that sees it.
This one answers "what did the sorter actually find?": y is UNITS, so each row
is one putative neuron and a row is a spike train rather than a channel.

Rows are ordered by firing rate, which turns the plot into a summary of the
sort's yield: a healthy sort shows a smooth spread from a few fast units down to
a long tail of slow ones. A block of rows all firing at once is usually one
neuron split across several units, and a dense band at the very top is usually
noise that got clustered.

Reads only the Sorting — spike trains, not traces — so it costs nothing next to
the recording it came from.
"""

import logging

from ..diagnostics.channel_layout import _new_figure, _save_and_release
from ..diagnostics.timebase import _shade_gap_spans, gap_spans, resolve_time_gaps, sample_times

logger = logging.getLogger(__name__)

DEFAULT_FIGSIZE = (16.0, 9.0)
DEFAULT_DPI = 180


def unit_firing_rates(sorting, duration_s=None):
    """{unit_id: spikes per second}, using the sorting's own frame counts.

    `duration_s` defaults to the span of the last spike, which is the only
    duration a Sorting knows on its own; pass the recording's duration when the
    exact denominator matters.
    """
    fs = float(sorting.get_sampling_frequency())
    trains = {unit: sorting.get_unit_spike_train(unit) for unit in sorting.unit_ids}

    if duration_s is None:
        last = max((int(t[-1]) for t in trains.values() if len(t)), default=0)
        duration_s = (last / fs) if fs else 0.0
    duration_s = float(duration_s) or 1.0

    return {unit: len(train) / duration_s for unit, train in trains.items()}


def plot_unit_raster(
    sorting,
    out_path,
    duration_s=None,
    start_time_s=0.0,
    max_units=None,
    descending=True,
    segment_boundaries=(),
    time_gaps=None,
    title=None,
    marker_size=0.5,
    figsize=DEFAULT_FIGSIZE,
    dpi=DEFAULT_DPI,
):
    """Draw one row per unit, ordered by firing rate; return `out_path`.

    `descending` puts the fastest units at the top. `max_units` keeps only the
    fastest N — the tail of a large sort is mostly single-digit spike counts and
    crowds the figure without adding information.

    `segment_boundaries` (seconds on the recording's own timeline) draws the
    segment joins, and `time_gaps` redraws the x axis on real elapsed time with
    the missing stretches shaded — the same treatment the upstream rasters get,
    and it matters more here: an inter-spike interval that straddles a join is
    not a real interval.

    Note that `duration_s` is a window on the RECORDING timeline in both modes,
    never on the stretched one. On the reference scan a 3247 s recording covers
    6765 s of wall clock, so clipping real elapsed times against the recording's
    own duration would silently discard half the sort.
    """
    import numpy as np

    fs = float(sorting.get_sampling_frequency())
    rates = unit_firing_rates(sorting, duration_s=duration_s)
    order = sorted(rates, key=lambda unit: rates[unit], reverse=bool(descending))
    if max_units and max_units > 0:
        order = order[: int(max_units)]
    if not order:
        raise ValueError("sorting contains no units to raster")

    real_time = time_gaps is not None
    gaps, segment_gaps = resolve_time_gaps(time_gaps) if real_time else ((), ())

    # Windowing happens in FRAMES, before any gap mapping, so that both versions
    # of this plot cover exactly the same spikes and differ only in where those
    # spikes are drawn. Filtering after the mapping would make the window mean
    # something different in each mode — see the docstring.
    start_frame = int(round(float(start_time_s or 0.0) * fs))
    end_frame = start_frame + int(round(float(duration_s) * fs)) if duration_s else None

    x_parts, y_parts = [], []
    last_frame = start_frame
    for row, unit in enumerate(order):
        frames = np.asarray(sorting.get_unit_spike_train(unit), dtype=np.int64)
        if frames.size == 0:
            continue
        keep = frames >= start_frame
        if end_frame is not None:
            keep &= frames <= end_frame
        frames = frames[keep]
        if not frames.size:
            continue
        last_frame = max(last_frame, int(frames.max()))
        times = (
            sample_times(frames, fs, gaps=gaps, segment_gaps=segment_gaps)
            if real_time else frames / fs
        )
        x_parts.append(np.asarray(times, dtype=float))
        y_parts.append(np.full(frames.size, row, dtype=np.int32))

    # The frame the window actually ends at, which is what bounds the gap table:
    # gap_spans indexes the CONTIGUOUS timeline, so handing it a real-elapsed
    # value would shade stretches past the end of the recording.
    window_end_frame = end_frame if end_frame is not None else last_frame + 1

    fig = _new_figure(figsize, dpi)
    ax = fig.subplots(1, 1)
    if x_parts:
        # One scatter for everything: hundreds of per-unit calls is what makes a
        # raster of this size slow to draw and slow to open.
        ax.scatter(
            np.concatenate(x_parts),
            np.concatenate(y_parts),
            s=marker_size, c="black", marker="|", linewidths=0.3, rasterized=True,
        )
    n_events = int(sum(part.size for part in x_parts))

    # Boundaries arrive on the recording timeline, so in real-time mode they go
    # through the same mapping the spikes did — otherwise the joins would be
    # drawn at the one place the data is guaranteed not to be.
    boundary_times = [float(value) for value in (segment_boundaries or ())]
    if real_time and boundary_times:
        boundary_frames = [int(round(value * fs)) for value in boundary_times]
        boundary_times = [
            float(value)
            for value in sample_times(boundary_frames, fs, gaps=gaps, segment_gaps=segment_gaps)
        ]
    for boundary in boundary_times:
        ax.axvline(boundary, color="red", lw=0.5, ls=":", zorder=3)

    # Span the window that was analysed rather than the extent of the spikes: a
    # sort that goes quiet halfway has to show the silence.
    edge_frames = [start_frame, max(start_frame, window_end_frame - 1)]
    edges = (
        sample_times(edge_frames, fs, gaps=gaps, segment_gaps=segment_gaps)
        if real_time else np.asarray(edge_frames, dtype=float) / (fs or 1.0)
    )
    if float(edges[1]) > float(edges[0]):
        ax.set_xlim(float(edges[0]), float(edges[1]))

    if real_time:
        # Shaded after the limits are set, so the bands cover what is on screen
        # and no more. _shade_gap_spans merges and caps them — a full well is
        # ~205 k breaks, and one patch each would take minutes to draw.
        spans = gap_spans(window_end_frame, fs, gaps=gaps, segment_gaps=segment_gaps)
        x_lower, x_upper = ax.get_xlim()
        shaded = _shade_gap_spans(ax, spans, x_min=x_lower, x_max=x_upper)
        logger.info("unit raster: shaded %d gap span(s) of %d in window", shaded, len(spans))

    ax.set_xlabel("time (s, real elapsed)" if real_time else "time (s)")
    ax.set_ylabel(f"unit (ordered by firing rate, {'fastest' if descending else 'slowest'} first)")
    ax.set_ylim(-1, len(order))
    if title:
        ax.set_title(title)

    logger.info(
        "unit raster: %d unit(s), %d event(s), rates %.2f-%.2f Hz -> %s",
        len(order), n_events, min(rates[u] for u in order), max(rates[u] for u in order), out_path,
    )
    return _save_and_release(fig, out_path)


__all__ = ["plot_unit_raster", "unit_firing_rates"]
