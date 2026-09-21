"""How much of the array one well's recordings share, drawn two ways.

An AxonTracking scan routes a different electrode subset per recording, and only
the electrodes routed in EVERY recording survive concatenation. That cut can be
brutal -- a few percent of the routed union -- and it is a scientific fact about
the run, so it gets a picture rather than one number in a log:

* :func:`plot_electrode_coverage` -- every electrode any recording routed, at
  its position on the array, coloured by how many recordings routed it, with
  the survivors ringed. Whether the kept set is one patch of tissue or a
  scatter across the chip is visible at a glance.
* :func:`plot_segment_electrode_counts` -- one bar per recording, its own routed
  count, with the kept count drawn across all of them.

Both read the dict :func:`mea_modules.io.metadata.electrode_coverage` returns
and nothing else: the counting happened when the capsule read the file.
"""

import logging

import numpy as np

from .channel_layout import _legend_dot, _legend_line, _new_figure, _save_and_release, _wrap_label
from .figure_style import LEGEND_FONTSIZE, LEGEND_FRAME_ALPHA, legend_corner, tighten
from .figure_text import SEGMENT_AXIS
from .segment_event_rates import _bar_colors, _rates_figure_size

logger = logging.getLogger(__name__)

_COVERAGE_FIGSIZE = (7.5, 4.6)
_DPI = 180
_UM_LABEL = r"$\mu\mathrm{m}$"
_KEPT_EDGE = "black"
_KEPT_LINE = "tab:red"
_BAR_ALPHA = 0.55
_LABEL_FONTSIZE = 6


def plot_electrode_coverage(coverage, out_path, dpi=_DPI, ax=None):
    """Every routed electrode, coloured by how many recordings routed it.

    The colour scale is discrete, one step per possible count, because the
    count is an integer and a continuous map would imply values between them.
    Electrodes routed in every recording -- the ones concatenation keeps -- are
    ringed in black. Returns the written path, or None when drawing into `ax`.
    """
    import matplotlib as mpl

    ids = coverage["electrode_ids"]
    if not ids:
        raise ValueError("no routed electrodes to draw")
    x = np.asarray([np.nan if v is None else v for v in coverage["x_um"]], dtype=float)
    y = np.asarray([np.nan if v is None else v for v in coverage["y_um"]], dtype=float)
    counts = np.asarray(coverage["n_segments_routed"], dtype=int)
    kept = np.asarray(coverage["kept"], dtype=bool)
    n_segments = int(coverage["n_segments"])

    fig = None
    if ax is None:
        fig = _new_figure(_COVERAGE_FIGSIZE, dpi)
        ax = fig.subplots()
    bounds = np.arange(0.5, n_segments + 1.5, 1.0)
    cmap = mpl.colormaps["viridis"].resampled(max(1, n_segments))
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    scatter = ax.scatter(x, y, c=counts, s=5, cmap=cmap, norm=norm, linewidths=0)
    if kept.any():
        ax.scatter(
            x[kept], y[kept], s=12, facecolors="none", edgecolors=_KEPT_EDGE, linewidths=0.5,
        )
    ticks = np.unique(np.linspace(1, n_segments, num=min(n_segments, 8)).round().astype(int))
    bar = ax.figure.colorbar(scatter, ax=ax, ticks=ticks, fraction=0.04, pad=0.02)
    bar.set_label("recordings routing the electrode")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"x ({_UM_LABEL})")
    ax.set_ylabel(f"y ({_UM_LABEL})")
    handles = [
        _legend_dot("0.5", f"routed ({int(coverage['union_count'])})"),
    ]
    if kept.any():
        from matplotlib.lines import Line2D

        handles.append(Line2D(
            [], [], marker="o", linestyle="none", markerfacecolor="none",
            markeredgecolor=_KEPT_EDGE, markersize=5,
            label=_wrap_label(f"kept, in all {n_segments} ({int(coverage['common_count'])})"),
        ))
    legend_corner(ax, handles=handles, fontsize=LEGEND_FONTSIZE, framealpha=LEGEND_FRAME_ALPHA)
    if fig is None:
        return None
    tighten(fig)
    written = _save_and_release(fig, out_path)
    logger.info(
        "wrote electrode coverage: %s (%d routed, %d kept)",
        written, len(ids), int(kept.sum()),
    )
    return written


def plot_segment_electrode_counts(coverage, out_path, dpi=_DPI, ax=None):
    """One bar per recording, its own routed count, the kept count across them.

    Bars edge to edge in recording order, one colour each, the same house style
    as the per-segment activity bars. The dashed line is the shared set every
    recording contributes to the concatenated timeline: the distance from a
    bar's top down to it is what that recording loses. Returns the written
    path, or None when drawing into `ax`.
    """
    rows = coverage.get("segments") or []
    if not rows:
        raise ValueError("no recordings to draw")
    positions = np.arange(len(rows))
    heights = [int(row.get("n_electrodes") or 0) for row in rows]

    fig = None
    if ax is None:
        fig = _new_figure(_rates_figure_size(len(rows)), dpi)
        ax = fig.subplots()
    ax.bar(
        positions, heights, width=1.0, color=_bar_colors(len(rows)), alpha=_BAR_ALPHA,
        edgecolor="white", linewidth=0.4,
    )
    common = int(coverage.get("common_count") or 0)
    ax.axhline(common, color=_KEPT_LINE, lw=1.0, linestyle="--")
    ax.set_xticks(positions)
    ax.set_xticklabels([str(row.get("rec")) for row in rows], rotation=90, fontsize=_LABEL_FONTSIZE)
    ax.set_xlim(-0.5, len(rows) - 0.5)
    ax.set_ylim(0, max(heights + [common]) * 1.08 if heights else 1)
    ax.set_xlabel(SEGMENT_AXIS)
    ax.set_ylabel("routed electrodes")
    handles = [_legend_line(_KEPT_LINE, f"kept in all ({common})", lw=1.0, linestyle="--")]
    legend_corner(ax, handles=handles, fontsize=LEGEND_FONTSIZE, framealpha=LEGEND_FRAME_ALPHA)
    if fig is None:
        return None
    tighten(fig)
    written = _save_and_release(fig, out_path)
    logger.info("wrote per-recording electrode counts: %s (%d recordings)", written, len(rows))
    return written


__all__ = ["plot_electrode_coverage", "plot_segment_electrode_counts"]
