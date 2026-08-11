"""One unit's spikes shown IN the recording trace, not just averaged out of it.

The waveform plot answers "does the average look like a spike"; this one answers
the question that comes right before trusting that average: **are these events
actually visible in the raw trace, at the size the template claims?** A sorter
can emit a beautiful template assembled from sub-noise residue, and the only way
to see that is to look at the trace itself with the sorter's event times marked
on it.

So: a bounded window of the (preprocessed, concatenated) recording on the unit's
extremum channel, with every spike the sorter attributed to this unit marked at
its aligned sample. The window defaults to wherever the unit fires most — a
review artifact should show the unit at its best, and a unit that looks marginal
in its own densest second is a finding.

Descriptive only, like the rest of this package: nothing is scored and no spike
is filtered. Concatenation joins falling inside the window are drawn, because a
"spike train" straddling a join is two recordings glued together and any
inter-spike statement across it is fiction.

Figures are built on an Agg canvas without pyplot and released after writing;
this runs in a loop over hundreds of units.
"""

import logging

from ..diagnostics.channel_layout import (
    _add_caption,
    _fold_caption,
    _legend_line,
    _new_figure,
    _save_and_release,
    _wrap_label,
)
from ..diagnostics.figure_text import CONTIGUOUS_AXIS, SEAM, acronym_note
from ..diagnostics.traces import (
    _CONTIGUOUS_XLABEL,
    _COUNTS_LABEL,
    _UV_LABEL,
    _read_traces,
)

logger = logging.getLogger(__name__)

_TRACE_FIGSIZE = (12.0, 4.0)
_TRACE_DPI = 180

# Legend/caption defaults, matched to the diagnostics figures.
_LEGEND_FONTSIZE = 7
_LEGEND_FRAMEALPHA = 0.85


def _legend_ring(color, label, size=7.0, lw=0.9):
    """Legend key for a HOLLOW ring marker.

    The shared handle set in :mod:`mea_modules.diagnostics.channel_layout` has a
    filled dot and a line; the spike marks on this figure are open rings riding
    the trace, and a filled key would read as a different mark.
    """
    from matplotlib.lines import Line2D

    return Line2D(
        [],
        [],
        linestyle="none",
        marker="o",
        markersize=float(size),
        markerfacecolor="none",
        markeredgecolor=color,
        markeredgewidth=float(lw),
        label=_wrap_label(label),
    )

# Two seconds shows individual spikes at full sample resolution at either 10 or
# 20 kHz without decimation (40k points is nothing), and is long enough to read
# rhythm — burst vs tonic — off the marks.
_DEFAULT_WINDOW_S = 2.0

_TRACE_COLOR = "#4a4a4a"
_MARK_COLOR = "#c0392b"
_JOIN_COLOR = "red"


def densest_spike_window(frames, fs, window_s, n_samples):
    """``(start_frame, end_frame)`` of the fixed-width window holding the most spikes.

    Fixed bins on the recording's own timeline, densest bin wins, ties go to the
    earliest — deterministic, so re-running the capsule reviews the same second
    of data. Falls back to the start of the recording for an empty train, which
    keeps the caller's plot loop uniform (the plot then honestly shows a trace
    with zero marks).
    """
    import numpy as np

    n_samples = int(n_samples)
    span = max(1, int(round(float(window_s) * float(fs))))
    span = min(span, n_samples) if n_samples > 0 else span

    frames = np.asarray(frames, dtype=np.int64)
    if frames.size == 0 or n_samples <= 0:
        return 0, span

    bins = frames // span
    values, counts = np.unique(bins, return_counts=True)
    best = int(values[int(np.argmax(counts))])

    start = best * span
    # Clamp so the window never runs off the end of the recording — the last
    # bin is usually partial.
    start = max(0, min(start, n_samples - span))
    return start, start + span


def plot_unit_trace(
    recording,
    sorting,
    unit_id,
    channel_id,
    out_path,
    window_s=_DEFAULT_WINDOW_S,
    start_time_s=None,
    stitch_frames=(),
    title=None,
    return_in_uV=True,
    figsize=_TRACE_FIGSIZE,
    dpi=_TRACE_DPI,
    caption_extra=None,
):
    """Draw the trace on one channel with one unit's spike times marked; return `out_path`.

    `channel_id` is the caller's choice on purpose — the extremum channel from
    the analyzer is the one that makes sense, and the caller already has it.

    `start_time_s` None picks the unit's densest `window_s` stretch (see
    :func:`densest_spike_window`); pass a value to review a specific moment,
    e.g. the same window across several units.

    `stitch_frames` (concatenated-timeline frames, the concat manifest's own
    unit) draws each join that falls inside the window, exactly as the upstream
    trace plots do.

    The read is one channel over one bounded window, so this costs milliseconds
    next to anything that touches the whole binary. Units are microvolts when
    the recording can scale (`return_in_uV=True` falls back to the device's own
    counts with the y label and the caption saying so, rather than failing on an
    unscaleable recording).

    Every mark is legended (Adam, 2026-08-11) — a grey line, red rings and dotted
    red rules are three different statements — and the caption says which
    timeline the x axis is and expands anything it abbreviates.

    `caption_extra` appends one more caption sentence, which is where a caller
    names this unit's sibling figures by their real emitted filenames, e.g.::

        caption_extra=(
            "Same unit elsewhere: waveforms/unit_7.png (these spikes averaged), "
            "footprints/unit_7.png (where on the array)."
        )
    """
    import numpy as np

    fs = float(recording.get_sampling_frequency())
    n_samples = int(recording.get_num_samples())
    frames = np.asarray(sorting.get_unit_spike_train(unit_id), dtype=np.int64)

    if start_time_s is None:
        start_frame, end_frame = densest_spike_window(frames, fs, window_s, n_samples)
    else:
        start_frame = max(0, min(int(round(float(start_time_s) * fs)), max(0, n_samples - 1)))
        end_frame = min(n_samples, start_frame + max(1, int(round(float(window_s) * fs))))

    in_uv = bool(return_in_uV)
    try:
        traces = _read_traces(recording, start_frame, end_frame, [channel_id], in_uv)
    except Exception as exc:
        if not in_uv:
            raise
        # No gain/offset on the recording — show raw units rather than nothing.
        logger.info("cannot scale to uV (%s); plotting raw units", exc)
        in_uv = False
        traces = _read_traces(recording, start_frame, end_frame, [channel_id], False)
    trace = np.asarray(traces[:, 0], dtype=float)

    time_s = (start_frame + np.arange(trace.size, dtype=np.int64)) / fs

    inside = frames[(frames >= start_frame) & (frames < end_frame)]
    mark_times = inside / fs
    # The trace's own value at the aligned sample, so the marks ride the trace
    # instead of floating at an arbitrary height.
    mark_values = trace[inside - start_frame]

    fig = _new_figure(figsize, dpi)
    ax = fig.subplots()
    ax.plot(time_s, trace, color=_TRACE_COLOR, lw=0.5, zorder=2)
    if mark_times.size:
        ax.scatter(
            mark_times, mark_values,
            s=18, marker="o", facecolors="none", edgecolors=_MARK_COLOR,
            linewidths=0.9, zorder=3,
        )

    joins_drawn = 0
    for join in stitch_frames or ():
        join = int(join)
        if start_frame < join < end_frame:
            ax.axvline(join / fs, color=_JOIN_COLOR, lw=0.6, ls=":", zorder=1)
            joins_drawn += 1

    unit_label = _UV_LABEL if in_uv else _COUNTS_LABEL

    ax.set_xlim(float(time_s[0]), float(time_s[-1]))
    ax.set_xlabel(_CONTIGUOUS_XLABEL)
    ax.set_ylabel(f"amplitude ({unit_label})")
    ax.set_title(
        title
        or (
            f"unit {unit_id} - channel {channel_id} - {int(mark_times.size)} of "
            f"{int(frames.size)} spikes in the {window_s:g}s shown"
        )
    )

    # Every drawn encoding gets a legend key (Adam, 2026-08-11): without one the
    # rings are unexplained marks and the dotted rules could be anything.
    handles = [
        _legend_line(
            _TRACE_COLOR,
            f"recorded signal on electrode {channel_id}, amplitude in {unit_label}",
            lw=0.9,
        )
    ]
    if mark_times.size:
        handles.append(
            _legend_ring(
                _MARK_COLOR,
                f"a spike the sorter assigned to unit {unit_id}, drawn at the "
                f"trace's own value ({int(mark_times.size)} in this window)",
            )
        )
    if joins_drawn:
        handles.append(_legend_line(_JOIN_COLOR, "segment join", lw=0.9, linestyle=":"))
    ax.legend(
        handles=handles,
        loc="best",
        fontsize=_LEGEND_FONTSIZE,
        framealpha=_LEGEND_FRAMEALPHA,
        labelspacing=0.7,
    )

    caption_parts = [
        "The rings are the sorter's own event times, not a re-detection: nothing "
        "on this figure is thresholded, filtered or scored.",
    ]
    if joins_drawn:
        caption_parts.append(SEAM)
    caption_parts.append(CONTIGUOUS_AXIS)
    if not in_uv:
        caption_parts.append(
            "This recording carries no gain and offset, so amplitude is shown in "
            "the device's own counts rather than converted to microvolts."
        )
        caption_parts.append(acronym_note("ADC"))
    caption_parts.append(caption_extra)
    _add_caption(fig, _fold_caption(caption_parts))

    out_path = _save_and_release(fig, out_path)
    logger.info(
        "wrote unit trace: %s (unit=%s channel=%s window=[%.3f, %.3f]s marks=%d joins=%d uv=%s)",
        out_path, unit_id, channel_id,
        float(time_s[0]), float(time_s[-1]),
        int(mark_times.size), joins_drawn, in_uv,
    )
    return out_path


__all__ = ["plot_unit_trace", "densest_spike_window"]
