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

from ..diagnostics.channel_layout import _new_figure, _save_and_release
from ..diagnostics.traces import _read_traces

logger = logging.getLogger(__name__)

_TRACE_FIGSIZE = (12.0, 4.0)
_TRACE_DPI = 180

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
    the recording can scale (`return_in_uV=True` falls back to raw units with
    the y label saying so, rather than failing on an unscaleable recording).
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

    ax.set_xlim(float(time_s[0]), float(time_s[-1]))
    ax.set_xlabel("time (s, concatenated timeline)")
    ax.set_ylabel("amplitude (uV)" if in_uv else "amplitude (raw units)")
    ax.set_title(
        title
        or (
            f"unit {unit_id} - channel {channel_id} - {int(mark_times.size)} of "
            f"{int(frames.size)} spikes in the {window_s:g}s shown"
        )
    )
    fig.tight_layout()

    out_path = _save_and_release(fig, out_path)
    logger.info(
        "wrote unit trace: %s (unit=%s channel=%s window=[%.3f, %.3f]s marks=%d joins=%d uv=%s)",
        out_path, unit_id, channel_id,
        float(time_s[0]), float(time_s[-1]),
        int(mark_times.size), joins_drawn, in_uv,
    )
    return out_path


__all__ = ["plot_unit_trace", "densest_spike_window"]
