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

    `segment_boundaries` (seconds) draws the segment joins, and `time_gaps`
    redraws the x axis on real elapsed time with the missing stretches shaded —
    the same treatment the upstream rasters get, and it matters more here: an
    inter-spike interval that straddles a join is not a real interval.
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
    if real_time:
        from ..diagnostics.timebase import gap_spans, resolve_time_gaps, sample_times

        gaps, segment_gaps = resolve_time_gaps(time_gaps)

    window_start = float(start_time_s or 0.0)
    window_end = None
    if duration_s:
        window_end = window_start + float(duration_s)

    x_parts, y_parts = [], []
    for row, unit in enumerate(order):
        frames = np.asarray(sorting.get_unit_spike_train(unit), dtype=np.int64)
        if frames.size == 0:
            continue
        times = sample_times(frames, fs, gaps=gaps, segment_gaps=segment_gaps) if real_time \
            else frames / fs
        times = np.asarray(times, dtype=float)
        keep = times >= window_start
        if window_end is not None:
            keep &= times <= window_end
        times = times[keep]
        if times.size:
            x_parts.append(times)
            y_parts.append(np.full(times.size, row, dtype=np.int32))

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

    for boundary in segment_boundaries or ():
        ax.axvline(float(boundary), color="red", lw=0.5, ls=":", zorder=3)

    if real_time:
        total = max((float(part.max()) for part in x_parts), default=0.0)
        spans = gap_spans(int(round(total * fs)), fs, gaps=gaps, segment_gaps=segment_gaps)
        for low, high in spans:
            ax.axvspan(low, high, color="0.85", lw=0, zorder=0)

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
