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

Amplitudes default to MICROVOLTS (Adam's unit ruling, 2026-08-10): raw device
counts around an arbitrary ADC mid-rail read as meaninglessly huge values in
review, and a figure that does not name its units invites exactly that
misreading. Callers wanting device units opt out with ``return_in_uV=False``;
either way the figure states the units actually drawn (y label + title block),
and a recording that cannot scale (`has_scaleable_traces()` False) downgrades
to device units with a warning rather than failing — a review plot in ADC
counts beats no plot.

Two emitters live here. :func:`plot_traces` is the atomic one — a stack of
channels and nothing else. :func:`plot_traces_with_layout` pairs that same stack
with the electrode layout of the very channels it draws, because a trace is hard
to place on the array from a second figure open in another window
(2026-09-19). Both are emitted; the composite does not replace the single panels.
It reuses them rather than redrawing either, by handing each one the axes it
should draw into.
"""

import logging

from .channel_layout import (
    _add_caption,
    _channel_xy,
    _fold_caption,
    _legend_line,
    _legend_patch,
    _new_figure,
    _pick_cluster_representative,
    _save_and_release,
    cluster_center_channels,
    detect_electrode_clusters,
    plot_channel_layout,
)
from .figure_text import (
    CONTIGUOUS_AXIS,
    FILE_TIME_AXIS,
    NO_DATA_SHADING,
    REAL_ELAPSED_AXIS,
    REAL_TIME_AXIS,
    REPRESENTATIVE_CHANNELS,
    SEAM,
    TRACED_CHANNELS,
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

logger = logging.getLogger(__name__)

# Axis labels. Aliased to :mod:`.figure_text`'s publication shorthand rather
# than redefined here, so every importer of these two names
# (`raster.py`, `unit_traces.py`, `unit_raster.py`) picks up the short label
# with no edit of its own. The default one is still the contiguous sample
# timeline, which is what every existing plot already shows.
_CONTIGUOUS_XLABEL = FILE_TIME_AXIS
_REAL_TIME_XLABEL = REAL_TIME_AXIS

_TRACE_FIGSIZE = (13.33, 7.5)
_TRACE_DPI = 180

# The composed sheet is the trace stack with the layout beside it, so it is
# wider than the stack alone and no taller. The layout axes spans every trace
# row (see `plot_traces_with_layout` below), so its HEIGHT already matches the
# whole plot area regardless of how many channels are drawn — only the WIDTH
# ratio decides its shape. A narrow ratio left it a tall sliver: the layout
# axes sets an equal aspect (electrode spacing is isotropic — see
# `channel_layout.plot_channel_layout`), and `adjustable="box"` then shrinks
# that box to its narrower dimension, so a tall cell drew the array cramped
# into a horizontal band with dead space above and below it (2026-09-19).
# The ratio below instead gives the layout column a width close to the plot
# area's height, so the equal-aspect box actually fills its cell; the figure
# width is raised to match so the trace column keeps roughly its standalone
# width (`_TRACE_FIGSIZE`) rather than shrinking to make room.
_TRACES_WITH_LAYOUT_FIGSIZE = (20.5, 7.5)
_TRACES_WITH_LAYOUT_WIDTH_RATIOS = (1.0, 1.9)

_LAYOUT_PANEL_TITLE = "where these channels sit on the array"
_TRACES_PANEL_TITLE = "what they recorded"

# Plain wording for what pairing the two panels adds, in the register
# :mod:`.figure_text` sets: expand the idea, do not name our machinery. Local
# rather than canonical because exactly one figure draws this pairing.
_PAIRED_PANELS = (
    "The red electrodes on the left are the same channels drawn on the right, so a "
    "trace can be placed on the array without opening a second figure."
)

# Named on the layout panel's red key. Publication shorthand collapses both
# cases to the same short key; the two names survive so a call
# site can still tell "we picked these" apart from "the caller picked these" if
# that distinction ever needs its own wording again.
_PAIRED_HIGHLIGHT_LABEL = TRACED_CHANNELS
_PAIRED_HIGHLIGHT_LABEL_GIVEN = TRACED_CHANNELS

# Defaults ported from the working build.
_DEFAULT_MAX_POINTS = 150_000
_DEFAULT_BLOCK_FRAMES = 200_000
_DEFAULT_ACTIVITY_CHUNKS = 4
_DEFAULT_ACTIVITY_CHUNK_FRAMES = 4_000

# Above this the figure takes minutes to rasterize and reads as a solid smear.
# Only reachable when a caller raises `max_points` itself: the decimation step
# rounds UP (see `_resolve_downsample_step`), so the default budget is a real
# ceiling and the default configuration cannot trip this (Adam, 2026-08-11 —
# the warning was firing on every segment, which made it noise).
_POINTS_WARN = 750_000

# A trace plot is only readable at single-digit channel counts. Lowered from 8
# to 6 (Adam, 2026-08-11): fewer stacked panels means each one is taller, and
# detail in an individual trace is what these figures are reviewed for. This is
# a knob everywhere it matters — nothing may hardcode the count, least of all
# legend text, since `channel_layout.png`'s highlighted set is exactly this many.
_DEFAULT_MAX_CHANNELS = 6

# --- Figure quality presets (Adam, 2026-08-11) ----------------------------
#
# "draft" is the default and stays the current low-resolution render: this
# development pass wants fast turnaround over pixel fidelity. "high" raises the
# dots-per-inch and the point budget together, so a reviewer can zoom into a
# trace or a raster and still see individual deflections rather than a smear.
# Raising dpi alone would not help — the decimation, not the raster size, is
# what destroys fine detail.
PLOT_QUALITY_PRESETS = {
    "draft": {"dpi": _TRACE_DPI, "max_points": _DEFAULT_MAX_POINTS},
    "high": {"dpi": 320, "max_points": 900_000},
}
DEFAULT_PLOT_QUALITY = "draft"


def resolve_plot_quality(quality=None):
    """Return the ``{"dpi", "max_points"}`` preset for `quality`.

    Unknown names fall back to the default with a warning rather than raising:
    a review figure rendered at draft quality beats a capsule that died on a
    typo in a command-line flag.
    """
    name = str(quality or DEFAULT_PLOT_QUALITY).strip().lower()
    if name not in PLOT_QUALITY_PRESETS:
        logger.warning(
            "unknown plot quality %r; falling back to %r (known: %s)",
            quality,
            DEFAULT_PLOT_QUALITY,
            ", ".join(sorted(PLOT_QUALITY_PRESETS)),
        )
        name = DEFAULT_PLOT_QUALITY
    return dict(PLOT_QUALITY_PRESETS[name])

# Amplitude unit labels. Every figure states which one it drew; the µV/counts
# decision is made once per figure by `_effective_uv` below.
_UV_LABEL = "µV"
_COUNTS_LABEL = "device counts (ADC)"

# Default read window. None would mean "the whole recording", which on a
# concatenated AxonTracking scan is tens of gigabytes.
_DEFAULT_DURATION_S = 60.0


def _effective_uv(recording, return_in_uV):
    """True when traces can actually be returned in microvolts.

    Honours the request but never fails on it: asking for µV from a recording
    that carries no gain/offset (``has_scaleable_traces()`` False) downgrades
    to device units with a warning, because a review plot in ADC counts beats
    no plot. Callers label axes from THIS value, so a figure always states the
    units it actually drew rather than the units that were requested.
    """
    if not return_in_uV:
        return False
    try:
        scaleable = bool(recording.has_scaleable_traces())
    except Exception:  # pragma: no cover - defensive; exotic recording objects
        scaleable = False
    if not scaleable:
        logger.warning(
            "return_in_uV requested but the recording carries no gain/offset; "
            "falling back to device units (ADC counts)"
        )
    return scaleable


def _shared_y_limits(parts_per_channel, margin=0.05):
    """Global (low, high) across every channel's samples, or None if empty.

    One y range for every panel. Per-panel autoscaling is actively misleading
    when traces are stacked: a quiet channel is stretched to fill its axes and
    reads like an active one, so channels cannot be compared by eye — which is
    the reason for stacking them in the first place.

    NaN-aware, because gap handling writes NaN at breaks. A small symmetric
    margin keeps peaks off the frame. Returns None when nothing finite is left
    to scale to, in which case the caller leaves matplotlib's autoscale alone.
    """
    import numpy as np

    low, high = np.inf, -np.inf
    for parts in parts_per_channel:
        for part in parts:
            values = np.asarray(part, dtype=float)
            if values.size == 0:
                continue
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                continue
            low = min(low, float(finite.min()))
            high = max(high, float(finite.max()))

    if not np.isfinite(low) or not np.isfinite(high):
        return None
    if high == low:  # a flat trace still needs a non-degenerate axis
        pad = abs(high) * margin or 1.0
        return low - pad, high + pad
    pad = (high - low) * margin
    return low - pad, high + pad


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
    return_in_uV=True,
):
    """RMS amplitude per channel, sampled from a few random chunks.

    Returns a float array aligned with `channel_ids` (all channels by default).
    Chunk starts are drawn from a seeded generator and shared by every channel,
    so the scores are comparable to each other and stable across runs — that is
    what makes them usable as a ranking key.

    Scores are in microvolts by default (device units on explicit opt-out, or
    automatically when the recording cannot scale). With a uniform gain the
    RANKING is identical either way; the µV default exists so the scores read
    in the same unit the trace figures draw.

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

    return_in_uV = _effective_uv(recording, return_in_uV)

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
    num_chunks=_DEFAULT_ACTIVITY_CHUNKS,
    chunk_frames=_DEFAULT_ACTIVITY_CHUNK_FRAMES,
    seed=0,
    start_time_s=0.0,
    duration_s=None,
    return_in_uV=True,
    restrict_to=None,
):
    """Pick the `n_channels` most active cluster representatives.

    Activity RMS is scored in microvolts by default (see
    :func:`channel_activity_rms`); with a uniform gain the selection is
    identical in either unit, so flipping `return_in_uV` never changes which
    channels a paired trace figure draws.

    One candidate is taken from each electrode cluster — the member nearest the
    cluster centroid, via
    :func:`mea_modules.diagnostics.channel_layout.cluster_center_channels` —
    and the candidates are then ordered by activity RMS, loudest first. Pass
    `n_channels` <= 0 to get every candidate in that order.

    `restrict_to` narrows the electrode pool BEFORE clustering. Pass the shared
    electrode set (:func:`.stitch_consistency.backbone_channel_ids`) to select over
    only the electrodes every segment kept, which is what a figure comparing
    segments must do.

    Falls back to the recording's own channel order when the probe carries no
    usable locations, so this never fails on a probe-less recording.

    Returns channel ids in the recording's own id type, ready to hand back to
    ``get_traces``.
    """
    channel_ids, xs, _ys = _channel_xy(recording)
    if not channel_ids:
        return []

    if xs is None:
        logger.warning("no usable channel locations; falling back to recording channel order")
        pool = list(channel_ids)
        if restrict_to is not None:
            wanted = {str(cid) for cid in restrict_to}
            pool = [cid for cid in pool if str(cid) in wanted] or pool
        return pool if n_channels <= 0 else pool[: max(1, int(n_channels))]

    candidates = cluster_center_channels(recording, channel_ids=restrict_to, eps=eps)
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
        len(candidates),
        len(channel_ids) if restrict_to is None else len(list(restrict_to)),
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
        budget = max(1000, parsed_max_points)
        # Round UP. Floor division made `max_points` a suggestion rather than a
        # cap: a window of 250k frames against a 150k budget gave step 1 and
        # drew all 250k points, which is what tripped the "too many points"
        # warning at default settings (Adam, 2026-08-11). Ceiling division makes
        # the budget true, so the warning below now only fires for a caller who
        # deliberately raised it.
        step_by_points = max(1, -(-int(total_frames) // budget))

    step_by_rate = 1
    if target_hz is not None:
        try:
            if float(target_hz) > 0.0 and fs > 0.0:
                step_by_rate = max(1, int(round(fs / float(target_hz))))
        except (TypeError, ValueError):
            step_by_rate = 1

    # Whichever cap is stricter wins.
    return max(step_by_points, step_by_rate)


def _resolve_channel_ids(recording, channel_ids, max_channels, in_uv):
    """(channels to draw, whether this function chose them).

    Both trace emitters need the same answer, and the composed sheet needs it
    BEFORE it can build its grid — one row per channel — and before it can tell
    the layout panel which electrodes to redden. Shared so the two emitters
    cannot disagree about which channels a figure is of.

    The second return is what the caption is allowed to claim: a caller that
    passed its own list never gets the representative-channel rule stated on its
    figure, because that figure did not apply it.
    """
    if channel_ids is None:
        return (
            select_representative_channels(
                recording,
                n_channels=max_channels,
                return_in_uV=in_uv,
            ),
            True,
        )

    channel_ids = list(channel_ids)
    if max_channels is not None and max_channels > 0:
        channel_ids = channel_ids[: int(max_channels)]
    return channel_ids, False


def _as_trace_axes(axes, n_channels):
    """The caller's panels as a list of exactly `n_channels` axes.

    A bare axes is accepted for a single-channel stack, which is the shape
    ``fig.subplots`` returns for one row. A count mismatch raises instead of
    drawing the first few channels and dropping the rest: the caller built its
    grid from a channel list, so a mismatch means the two have already drifted
    and the figure would be silently short a channel.
    """
    if hasattr(axes, "plot"):
        axes = [axes]
    axes = list(axes)
    if len(axes) != int(n_channels):
        raise ValueError(
            f"{len(axes)} axes cannot hold {int(n_channels)} channel(s); "
            "pass one axes per channel"
        )
    return axes


def plot_traces(
    recording,
    out_path=None,
    channel_ids=None,
    max_channels=_DEFAULT_MAX_CHANNELS,
    start_time_s=0.0,
    duration_s=_DEFAULT_DURATION_S,
    target_hz=None,
    max_points=_DEFAULT_MAX_POINTS,
    stitch_frames=(),
    title=None,
    return_in_uV=True,
    block_frames=_DEFAULT_BLOCK_FRAMES,
    figsize=_TRACE_FIGSIZE,
    dpi=_TRACE_DPI,
    time_gaps=None,
    caption_extra=None,
    quality=None,
    annotate=True,
    axes=None,
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

    `stitch_frames` marks each concatenation boundary — in FRAMES on the
    concatenated timeline, the one convention across every emitter (see
    :func:`mea_modules.diagnostics.timebase.join_marks`). On the contiguous axis
    a join is one instant and draws one rule. On the real-elapsed axis it is
    not: the earlier segment stops, the chip spends minutes re-routing, and only
    then does the later segment start, so the join draws TWO rules bracketing
    that gap — its end and the next one's start — rather than one rule pinned to
    an arbitrary edge of it (2026-09-19). The trace line is separately
    broken with NaN at the gaps so the plot does not draw a fake ramp across
    them; that is a different mechanic and both still apply.

    Amplitudes are MICROVOLTS by default; pass ``return_in_uV=False`` for
    device units, and a recording that cannot scale downgrades to device units
    with a warning instead of failing. The figure states the units it actually
    drew — a shared y label, and the title block (a ``[µV]`` / ``[device
    counts (ADC)]`` suffix, skipped when the given title already names the
    unit so callers can word it themselves). The y label is not annotation and
    is never dropped: an amplitude with no unit is unreadable, so it survives
    `annotate=False` and follows the stack into a caller's axes.

    `time_gaps` chooses which timeline the x axis is. Left None it is the
    contiguous one — sample index over sampling rate, labelled ``file time
    (s)`` — which is what every existing caller gets and what SpikeInterface
    believes. Given the gap structure from :mod:`mea_modules.io.gaps` (a
    ``frame_gaps`` dict, a ``well_gap_summary``, or ``{"gaps": ...,
    "segment_gaps": ...}``) the axis becomes real elapsed time instead: every
    sample is pushed right by the time missing before it, the stretches holding
    no data are shaded, and the label says ``elapsed time (s)`` so the two can
    never be confused. The supplied structure wins over any time vector on the
    recording, since the two would otherwise both correct for the same gaps.

    The figure explains itself (Adam, 2026-08-11): a legend keys every drawn
    encoding — the stacked traces, the red segment-join rules, the shaded
    stretches where nothing was recorded — and a caption states, in plain
    language, what a segment join is and which timeline the x axis is. A caller
    that picked the channels itself passes `caption_extra` to say how, ideally
    naming the sibling figure those channels are marked on, e.g.
    ``"These are the channels marked red in channel_layout.png."``

    `quality` picks a render preset (:data:`PLOT_QUALITY_PRESETS`): ``"draft"``
    (the default, and what this development pass ships) or ``"high"``, which
    raises the dots-per-inch AND the point budget together so a reviewer can
    zoom in on a deflection instead of a smear. An explicit `dpi` or
    `max_points` argument still wins over the preset.

    `annotate` controls the explanatory chrome. True keeps the title and the
    caption block. False draws neither, leaving the axes, the amplitude unit and
    the legend — the presentation-plot register, where the title's information
    lives in the filename instead.

    Pass `axes` — one per channel — to draw the stack into panels a caller
    already owns, which is how a composed sheet reuses it without its own
    figure. Nothing is written then, and `out_path`, `figsize`, `dpi` and
    `title`'s figure-level placement do not apply: the title goes on the top
    panel and the return is None. The bottom matter still comes from here,
    because the keys describe the stack and :func:`._add_caption` has to be the
    one owner of that margin — so a composing caller adds its own wording
    through `caption_extra` rather than calling ``_add_caption`` again, and
    draws anything else it needs BEFORE handing the panels over.
    :func:`plot_traces_with_layout` is the in-tree consumer.
    """
    import numpy as np

    preset = resolve_plot_quality(quality)
    # The preset fills only what the caller left at the module default, so an
    # explicit argument is never silently overridden by a quality flag.
    if dpi == _TRACE_DPI:
        dpi = preset["dpi"]
    if max_points == _DEFAULT_MAX_POINTS:
        max_points = preset["max_points"]

    if axes is None and out_path is None:
        raise ValueError("pass out_path to write a figure, or axes to draw into one")

    window_start, window_end, fs = _resolve_frame_window(recording, start_time_s, duration_s)
    total = int(window_end - window_start)
    if total <= 0:
        raise ValueError("requested trace window contains no samples")

    # Resolved once for the whole figure: selection, reads, and labels must
    # all describe the same unit.
    in_uv = _effective_uv(recording, return_in_uV)
    unit_label = _UV_LABEL if in_uv else _COUNTS_LABEL

    # Tracked so the caption only explains the representative-channel rule when
    # this figure actually applied it; a caller passing its own channel list
    # gets no claim it did not make.
    channel_ids, channel_ids_were_selected = _resolve_channel_ids(
        recording, channel_ids, max_channels, in_uv
    )
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
        # structure costs only the spans inside this plot. Split by kind, so the
        # handful of acquisition gaps do not disappear into the thousands of
        # microsecond frame-counter breaks drawn in the same grey.
        spans = gap_spans_by_kind(window_end, fs, gaps=gaps, segment_gaps=segment_gaps)
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
        traces = _read_traces(recording, start, end, channel_ids, in_uv)
        # Keep the global decimation phase across block boundaries, otherwise
        # the sample grid shifts every block and the x axis drifts.
        offset = (window_start - start) % step
        decimated = traces[offset::step, :]
        for channel_index in range(len(channel_ids)):
            parts_per_channel[channel_index].append(np.asarray(decimated[:, channel_index]))
        if block_index == 1 or block_index == total_blocks or block_index % max(1, total_blocks // 10) == 0:
            logger.debug("plot traces: read %d/%d blocks out=%s", block_index, total_blocks, out_path)

    supplied_axes = axes is not None
    if supplied_axes:
        panels = _as_trace_axes(axes, len(channel_ids))
        # The caller's figure, so that the bottom matter below lands under the
        # panels it describes rather than on a figure of our own.
        fig = panels[0].get_figure()
    else:
        fig = _new_figure(figsize, dpi)
        panels = fig.subplots(len(channel_ids), 1, sharex=True)
        if len(channel_ids) == 1:
            panels = [panels]

    # One y range for every panel, spanning the global min/max of everything
    # plotted. Per-panel autoscaling is actively misleading here: a quiet
    # channel gets stretched to fill its axes and reads like an active one, so
    # channels cannot be compared by eye — which is the whole point of stacking
    # them. Computed before drawing so every axis gets the same limits.
    y_limits = _shared_y_limits(parts_per_channel)

    # Boundary markers are positions on the same axis as the traces, so they go
    # through whichever mapping the traces did — one shared implementation, in
    # `timebase`, rather than a per-emitter loop over `frame / fs`.
    marks = join_marks(
        stitch_frames, fs, gaps=gaps, segment_gaps=segment_gaps, real_time=real_time
    )

    shaded = {"within_segment": 0, "between_segment": 0}
    drawn_joins = 0
    for axis, channel_id, parts in zip(panels, channel_ids, parts_per_channel):
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
        if y_limits is not None:
            axis.set_ylim(*y_limits)
        if real_time and t.size:
            shaded = shade_gap_kinds(axis, spans, x_min=float(t[0]), x_max=float(t[-1]))
        drawn_joins = draw_join_marks(axis, marks)
        axis.set_ylabel(f"ch {channel_id}")
        axis.grid(False)

    if real_time:
        logger.info(
            "plot traces: shaded %d within-segment break(s) of %d and %d "
            "between-segment gap(s) of %d in window",
            shaded["within_segment"], len(spans.get("within_segment", ())),
            shaded["between_segment"], len(spans.get("between_segment", ())),
        )
    panels[-1].set_xlabel(_REAL_TIME_XLABEL if real_time else _CONTIGUOUS_XLABEL)

    # The figure must state the units it drew (2026-08-10). That is an
    # AXIS label, not chrome, so `annotate=False` keeps it: a stack of
    # deflections with no amplitude unit cannot be read at all. With the caller
    # owning the figure there is no figure-level y label to hang it on, so it
    # rides on the bottom panel, which already carries the stack's shared x
    # label.
    if supplied_axes:
        panels[-1].set_ylabel(f"ch {channel_ids[-1]}\namplitude ({unit_label})")
    else:
        fig.supylabel(f"amplitude ({unit_label})", fontsize="small")

    if annotate:
        # A caller whose title already says the unit keeps its own wording.
        if title and not any(mark in title for mark in ("µV", "uV", "device counts")):
            title = f"{title} [{unit_label}]"
        if supplied_axes:
            if title:
                panels[0].set_title(title)
        else:
            fig.suptitle(title or f"amplitude in {unit_label}")

    # Every drawn encoding gets a legend key (Adam, 2026-08-11). Without one the
    # red rules and the grey bands are unexplained marks: a reader cannot tell a
    # segment join from an artifact, or "not recorded" from "silent".
    handles = [
        _legend_line(
            "black",
            f"trace / channel, shared y ({unit_label})",
            lw=0.9,
        )
    ]
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
    caption = ""
    if annotate:
        caption_parts = []
        if channel_ids_were_selected:
            caption_parts.append(REPRESENTATIVE_CHANNELS)
        if caption_extra:
            caption_parts.append(caption_extra)
        if drawn_joins:
            caption_parts.append(SEAM)
        caption_parts.append(REAL_ELAPSED_AXIS if real_time else CONTIGUOUS_AXIS)
        if real_time and any(shaded.values()):
            caption_parts.append(NO_DATA_SHADING)
        if not in_uv:
            caption_parts.append(acronym_note("ADC"))
        caption = _fold_caption(caption_parts)
    # Caption and legend go through ONE call: both live in the margin under the
    # panels, and `_add_caption` is what gives each its own band there. Pinning
    # the legend separately is what put it on top of the caption on capsule
    # 05's traces.png (Adam, 2026-08-11). The keys describe the whole stack, and
    # a legend inside any single panel would cover that channel's trace, so the
    # margin is where it belongs — just not on the caption's lines. An empty
    # caption still comes through here: `_add_caption` lays the legend out and
    # re-fits the axes, and only the caption band is skipped.
    _add_caption(fig, caption, legend_handles=handles)

    if supplied_axes:
        logger.debug(
            "drew trace stack into a caller's axes (%d channels, %s)",
            len(channel_ids),
            unit_label,
        )
        return None

    out_path = _save_and_release(fig, out_path)
    logger.info(
        "wrote traces: %s (%d channels, %s)", out_path, len(channel_ids), unit_label
    )
    return out_path


def plot_traces_with_layout(
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
    return_in_uV=True,
    block_frames=_DEFAULT_BLOCK_FRAMES,
    figsize=_TRACES_WITH_LAYOUT_FIGSIZE,
    dpi=_TRACE_DPI,
    time_gaps=None,
    caption_extra=None,
    quality=None,
    annotate=True,
):
    """Draw the layout and the traces of the SAME channels; return a manifest dict.

    The electrodes on the left, the traces on the right, one row per channel,
    and the reddened electrodes are exactly the rows beside them. A standalone
    layout panel cannot make that claim — highlighting channels on it only
    points at some other file — which is why the highlight belongs either here,
    where both halves are visible at once, or on a figure whose caller names the
    sibling artifact (2026-09-19).

    Neither half is redrawn here: the channels are resolved once, then
    :func:`mea_modules.diagnostics.channel_layout.plot_channel_layout` and
    :func:`plot_traces` are each handed the axes to draw into. This is the only
    composed figure in the pair — both single panels are still emitted on their
    own and are not replaced by it.

    Parameters
    ----------
    recording : RecordingExtractor
        Read for both halves: probe geometry for the layout panel, bounded
        blocks of samples for the traces.
    out_path : path-like
        Where to write the PNG. Required — a composed sheet owns its figure, so
        there is no axes to draw into instead.
    channel_ids : sequence or None
        The channels to draw and to redden. None picks them with
        :func:`select_representative_channels`, and only then does the caption
        state the representative-channel rule.
    max_channels : int
        Row cap, and therefore the height of the trace column.
    start_time_s, duration_s, target_hz, max_points, block_frames : float, int
        The read and decimation bounds, exactly as :func:`plot_traces` takes
        them.
    stitch_frames : sequence of int
        The concatenation joins, in FRAMES on the concatenated timeline, passed
        through to the trace panels.
    title : str or None
        Figure-level title over both panels. Each panel keeps its own short
        title naming what it shows.
    return_in_uV : bool
        Microvolts unless False or the recording cannot scale; resolved once
        here so the selection, the traces and the labels share one unit.
    figsize, dpi : tuple, float
        Figure geometry. `dpi` left at the module default follows `quality`.
    time_gaps : dict or sequence or None
        The gap structure that turns the trace x axis into real elapsed time,
        exactly as :func:`plot_traces` takes it.
    caption_extra : str or None
        Anything the caller wants added to the caption, after the note pairing
        the two panels.
    quality : str or None
        Render preset, as :func:`plot_traces` takes it.
    annotate : bool
        False drops the figure title, both panel titles and the caption block,
        keeping the axes, the units and both legends.

    Returns
    -------
    dict
        JSON-serializable: the file written, which panels it holds, the channel
        ids drawn in them, and whether this function chose those channels.

    Raises
    ------
    ValueError
        The recording has no channels to draw, or no usable locations to place
        them on.
    """
    preset = resolve_plot_quality(quality)
    if dpi == _TRACE_DPI:
        dpi = preset["dpi"]

    # Resolved here rather than inside each half: the layout must redden the
    # channels the traces actually drew, and the RMS ranking must score in the
    # unit the traces are read in.
    in_uv = _effective_uv(recording, return_in_uV)
    channel_ids, channel_ids_were_selected = _resolve_channel_ids(
        recording, channel_ids, max_channels, in_uv
    )
    if not channel_ids:
        raise ValueError("no channels to plot")

    fig = _new_figure(figsize, dpi)
    grid = fig.add_gridspec(
        len(channel_ids), 2, width_ratios=_TRACES_WITH_LAYOUT_WIDTH_RATIOS
    )
    # One tall panel for the array, one row per channel beside it.
    layout_ax = fig.add_subplot(grid[:, 0])
    panels = []
    for row in range(len(channel_ids)):
        panel = fig.add_subplot(grid[row, 1], sharex=panels[0] if panels else None)
        if row < len(channel_ids) - 1:
            # Only the bottom row carries the time axis, as in the standalone
            # stack; `sharex` alone does not hide the inner tick labels when the
            # axes are added one at a time.
            panel.tick_params(labelbottom=False)
        panels.append(panel)

    if annotate and title:
        fig.suptitle(title)

    plot_channel_layout(
        recording,
        ax=layout_ax,
        title=_LAYOUT_PANEL_TITLE,
        highlight_channel_ids=channel_ids,
        highlight_label=(
            _PAIRED_HIGHLIGHT_LABEL
            if channel_ids_were_selected
            else _PAIRED_HIGHLIGHT_LABEL_GIVEN
        ),
        annotate=annotate,
    )

    # The trace stack owns the bottom matter of this figure, so everything this
    # sheet has to say beyond what the stack already says travels as its caption
    # extra. The representative-channel rule is ours to state here: the stack
    # was handed an explicit channel list and so cannot claim it applied one.
    sheet_caption = [_PAIRED_PANELS]
    if channel_ids_were_selected:
        sheet_caption.insert(0, REPRESENTATIVE_CHANNELS)
    if caption_extra:
        sheet_caption.append(caption_extra)

    plot_traces(
        recording,
        axes=panels,
        channel_ids=channel_ids,
        max_channels=max_channels,
        start_time_s=start_time_s,
        duration_s=duration_s,
        target_hz=target_hz,
        max_points=max_points,
        stitch_frames=stitch_frames,
        title=_TRACES_PANEL_TITLE,
        return_in_uV=in_uv,
        block_frames=block_frames,
        time_gaps=time_gaps,
        caption_extra=_fold_caption(sheet_caption),
        quality=quality,
        annotate=annotate,
    )

    written = _save_and_release(fig, out_path)
    logger.info(
        "wrote traces with layout: %s (%d channels, %s)",
        written,
        len(channel_ids),
        _UV_LABEL if in_uv else _COUNTS_LABEL,
    )
    return {
        "files": {"png": str(written)},
        "panels": ["channel_layout", "traces"],
        "channel_ids": [str(channel_id) for channel_id in channel_ids],
        "n_channels": len(channel_ids),
        "representative_selection": bool(channel_ids_were_selected),
        "real_time": time_gaps is not None,
    }
