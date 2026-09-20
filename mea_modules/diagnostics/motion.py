"""SpikeInterface drift estimation over a concatenated recording.

Moved out of `capsules/concat_diagnostics/run_capsule.py` when that capsule was
retired in favour of the `mrp diag concat_diagnostics` tool (2026-09-19). The
tool had been carrying a stub saying this chain "has not been moved into
mea_modules yet"; this module is that move, and the numbers are ported
unchanged — same detector, same threshold, same rigid decentralized estimate —
so a figure drawn before and after the move is the same figure.

**Default OFF wherever it is wired.** It is expensive and it is unvalidated on
this data (the diagnostics catalogue defers it to a second pass), so nothing
enables it implicitly.

**The caveat that makes this readable.** Kilosort-style drift models assume a
continuous timeline. A concatenated well does not have one: between two
segments the chip spent minutes re-routing electrodes, so a displacement STEP
at a join is an electrode re-routing artifact until someone proves otherwise.
Read the trace within each segment band; treat a cross-join trend as suspect.
The joins are drawn on the figure for exactly that reason.

Two functions, the usual split: :func:`estimate_motion_over_recording` returns
the numbers and touches no canvas, :func:`plot_motion_estimate` draws only what
that returned. A number can therefore become a check without dragging a figure
along.
"""

import logging

from .channel_layout import _add_caption, _legend_line, _new_figure, _save_and_release
from .figure_style import legend_corner, tighten
from .figure_text import FILE_TIME_AXIS, JOIN_INSTANT
from .timebase import _JOIN_COLOR, draw_join_marks, join_marks

logger = logging.getLogger(__name__)

# Ported verbatim from the retired capsule. These ARE the scientific settings;
# changing one changes the result, so they are named here rather than inlined.
_DETECT_METHOD = "locally_exclusive"
_DETECT_THRESHOLD = 5.0
_LOCALIZE_METHOD = "center_of_mass"
_ESTIMATE_METHOD = "decentralized"
_RIGID = True

METHOD_CHAIN = (
    "detect_peaks(locally_exclusive) -> center_of_mass -> "
    "estimate_motion(decentralized, rigid)"
)

TIMELINE_CAVEAT = (
    "Displacement is on the concatenated file timeline, not real elapsed time. "
    "A step at a segment join is an electrode re-routing artifact until proven "
    "otherwise; read the trace within each segment band."
)

_MOTION_FIGSIZE = (12.0, 4.2)
_MOTION_DPI = 180
_DISPLACEMENT_YLABEL = "estimated displacement (µm)"


def estimate_motion_over_recording(recording, well=None):
    """Rigid drift estimate over `recording`. Returns a JSON-serializable dict.

    Keys: ``well``, ``method``, ``n_peaks``, ``max_abs_displacement_um``,
    ``temporal_bins_s``, ``displacement_um``, ``note``.

    Unit coherence, ported with the code because it is easy to get wrong: noise
    levels are taken in DEVICE COUNTS, because ``detect_peaks`` cuts the
    recording's UNSCALED traces against ``detect_threshold * noise``. Passing
    microvolt noise here would scale the threshold by the gain (~6.3x on this
    hardware) and miss real peaks. The figure's axes are µm against seconds and
    never see the unit either way.
    """
    import numpy as np
    from spikeinterface.core import get_noise_levels
    from spikeinterface.sortingcomponents.motion import estimate_motion
    from spikeinterface.sortingcomponents.peak_detection import detect_peaks
    from spikeinterface.sortingcomponents.peak_localization import localize_peaks

    noise = get_noise_levels(recording, return_in_uV=False)
    peaks = detect_peaks(
        recording,
        method=_DETECT_METHOD,
        detect_threshold=_DETECT_THRESHOLD,
        noise_levels=noise,
    )
    locations = localize_peaks(recording, peaks, method=_LOCALIZE_METHOD)
    motion = estimate_motion(
        recording, peaks, locations, method=_ESTIMATE_METHOD, rigid=_RIGID
    )

    displacement = np.atleast_1d(np.asarray(motion.displacement[0]).squeeze())
    bins_s = np.asarray(motion.temporal_bins_s[0])
    summary = {
        "well": well,
        "method": METHOD_CHAIN,
        "n_peaks": int(len(peaks)),
        "max_abs_displacement_um": (
            float(np.nanmax(np.abs(displacement))) if displacement.size else None
        ),
        "note": TIMELINE_CAVEAT,
        "temporal_bins_s": [float(b) for b in bins_s],
        "displacement_um": [float(d) for d in displacement],
    }
    logger.info(
        "motion estimate for %s: %d peaks, max |displacement| %s um",
        well, summary["n_peaks"], summary["max_abs_displacement_um"],
    )
    return summary


def plot_motion_estimate(
    summary,
    out_path,
    stitch_frames=(),
    fs_hz=None,
    title=None,
    figsize=_MOTION_FIGSIZE,
    dpi=_MOTION_DPI,
    annotate=True,
):
    """Draw the displacement trace from `summary`, joins marked. Returns the Path.

    Reads only what :func:`estimate_motion_over_recording` returned and
    recomputes nothing, so the figure and the JSON beside it cannot disagree.

    `stitch_frames` are FRAMES on the concatenated timeline, as everywhere else
    in this package; `fs_hz` converts them. Without a rate the joins are simply
    not drawn — a figure with joins at the wrong places is worse than one with
    none, and the caption says which happened.

    `annotate` False drops the title and the caption block, keeping the axes,
    the units and the legend.
    """
    import numpy as np

    bins_s = np.asarray(summary.get("temporal_bins_s") or [], dtype=float)
    displacement = np.asarray(summary.get("displacement_um") or [], dtype=float)

    fig = _new_figure(figsize, dpi)
    ax = fig.add_subplot(111)
    ax.plot(bins_s, displacement, linewidth=0.9, color="#2b6cb0")

    handles = []
    drawn_joins = 0
    if fs_hz:
        # The one join-geometry implementation, never a second copy of it. This
        # figure is on the FILE timeline, where a join is a single instant, so
        # no gaps are passed and `real_time` stays at its default. Routing
        # through timebase is what makes a later change to how a join is drawn
        # reach this figure too, instead of leaving it behind at the old style.
        drawn_joins = draw_join_marks(ax, join_marks(stitch_frames, fs_hz))
    if drawn_joins:
        handles.append(_legend_line(_JOIN_COLOR, JOIN_INSTANT, lw=0.9))

    ax.set_xlabel(FILE_TIME_AXIS)
    ax.set_ylabel(_DISPLACEMENT_YLABEL)
    if handles:
        legend_corner(ax, handles=handles)

    if annotate:
        if title:
            ax.set_title(title)
        # `drawn_joins` alone cannot tell the two zero cases apart: a
        # single-segment recording has no `stitch_frames` and is not missing
        # anything, while a concatenated one with `stitch_frames` but no
        # `fs_hz` genuinely has joins it could not place. Say the caveat only
        # when the second is true.
        missing_rate = bool(stitch_frames) and not fs_hz
        _add_caption(
            fig,
            TIMELINE_CAVEAT
            + ("  Segment joins are not drawn: no sampling rate." if missing_rate else ""),
        )
    tighten(fig)
    out_path = _save_and_release(fig, out_path)
    logger.info("wrote motion estimate: %s (%d joins drawn)", out_path, drawn_joins)
    return out_path
