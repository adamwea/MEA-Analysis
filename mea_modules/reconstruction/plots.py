"""Render one unit's tracked axon as a PNG — the visualization step after
`track_unit_axon`/`save_reconstruction`.

Public API::

    from mea_modules.reconstruction import plot_unit_reconstruction

Takes the SAME `gtr` object `track_unit_axon` returns (or an unpickled
`gtr.pkl` `save_reconstruction` wrote) and a destination path; writes one PNG.
Nothing here reads a well, a run directory, or a manifest — `capsules/
plot_reconstructions/run_capsule.py` is the thin CLI adapter that discovers
`gtr.pkl` files on disk and calls this.

**What style this ports, and why.** Adam asked for this to match the
pre-rebuild build's LATEST reconstruction-plot style, not a new invention.
Investigated three candidates in
`~/dev/RBS-adamwea/pkgs/pre-rebuild-28Jul2026/`:

1. `axon_velocity/axon_velocity/plotting.py::plot_axon_summary` — the
   library's own free function (amplitude map + peak-latency map + one
   propagation panel per branch + a velocity panel, 2x2 grid). Never called
   anywhere in the old build (grep across
   `axon_reconstructor__loop-cleanup/` turns up zero references) — reasonable
   as a general-purpose utility, but not what this lab actually shipped.
2. `axon_reconstructor__loop-cleanup/.../reconstruct/core/plot_unit_summary.py`
   (`write_unit_summary_plot`, wired as `reconstruct.plot_unit_summary`) — a
   black-background 3-panel dashboard (circle-recon footprint + velocity +
   per-branch propagation row) built on an entire bespoke rendering subsystem
   (`CircleReconConfig`, `TemplateCirclesPlotConfig`, `FootprintMapConfig`,
   `branch_styles.py`, `templates/core/render.py`'s `render_template_circles_plot_v2`
   — thousands of lines across a dozen modules, none of it ported to this
   rebuild). Also opt-in and `enabled=False` by default in the old build's own
   config (`config.py`'s `_phase_enabled(plot_unit_summary_cfg, False)`).
   Faithfully porting this was judged infeasible for a same-hour, additive
   task and not clearly "the" default style since it shipped disabled.
3. `axon_reconstructor__loop-cleanup/.../reconstruct/core/diagnostic_plots.py`
   (`write_unit_axon_reconstruction_diagnostic_figure`, wired straight into
   `axon_velocity_gtrs.py` — the SAME phase that computes the `gtr`, not a
   separate opt-in dashboard phase) — a 2-row composite: top row is the
   tracker's own graph (nodes/edges colored by search heuristic, via `gtr`'s
   private `_plot_nodes`/`_plot_edges`), bottom row is `gtr.plot_raw_branches`
   (a PUBLIC, documented `GraphAxonTracking` method) beside a per-branch
   velocity fit built from `axon_velocity.plotting.plot_velocity` (a PUBLIC
   free function) plus `gtr._estimate_peaks_and_dists` /
   `gtr.robust_velocity_estimator`. This is squarely "compose the library's
   own plotting calls (public where available, the tracker's own private
   internals where the public surface doesn't cover it) with extra titles/
   colorbars/annotations" — exactly the pattern worth porting faithfully, and
   self-contained (one `gtr`, no separate config-object dependency tree).
   Covered by its own `tests/test_diagnostic_plots.py` in the old build.

Ported #3 near-verbatim: same two-row layout, same panel titles, same
`axon_velocity` calls in the same order. Differences, all deliberate:

- Old build wrote a PNG *and* SVG per figure, driven by a config object with
  independent `write_png`/`write_svg`/`dpi`/`invert_y_axis` fields; this
  module writes exactly one PNG at a caller-given path (this pipeline's
  `reconstruct_axons` persists `gtr.pkl` for anyone who wants a different
  rendering — see that module's docstring — so a second format here would be
  redundant, not a new capability).
- Old build's `_plot_channel_selection_panels` companion figure (channel
  amplitude/latency maps + per-threshold selection scatter, using `av.
  plot_amplitude_map`/`av.plot_peak_latency_map`) is NOT ported. It documents
  *channel selection*, a step upstream of tracking; `capsules/
  plot_reconstructions` is about the tracked *reconstruction* itself, which
  is exactly what the ported figure shows. Add it later as a second function
  in this module if a real consumer asks for it — the two are already
  factored as independent figures in the old build for exactly this reason.
- Old build's diagnostic figure carried no unit id or branch count in its
  static "Axonal Reconstruction Method" suptitle (the surrounding per-unit
  directory supplied that context). This module's caller writes one file per
  `<unit_id>/` directory the same way, but hundreds of these will be opened
  standalone tonight, so the title here is built from `unit_id` (when given)
  and `len(gtr.branches)` — a strict addition, not a style deviation.
"""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

RECONSTRUCTION_PLOT_FILENAME = "reconstruction.png"

# Matches the old build's `write_unit_axon_reconstruction_diagnostic_figure`
# defaults (figsize=(15.0, 12.0), dpi=300.0) except DPI, lowered to 150 per
# this task's guidance -- hundreds of units render tonight, and a diagnostic
# PNG (not a publication figure) does not need 300 DPI to be legible.
DEFAULT_FIGSIZE = (15.0, 12.0)
DEFAULT_DPI = 150.0


def _normalize_colorbar_limits(values: Any) -> tuple:
    """(vmin, vmax) for a colorbar, guarding the degenerate all-equal case.

    Ported verbatim from the old build's `diagnostic_plots._normalize_colorbar_limits`
    -- `matplotlib.colors.Normalize(vmin=vmax)` raises, so a flat heuristic
    array (a real possibility on a tiny/synthetic toy graph) gets a
    manufactured 1-wide range instead of crashing the whole figure.
    """
    import numpy as np

    array = np.asarray(values, dtype=float)
    if array.size <= 0:
        return 0.0, 1.0
    vmin = float(np.min(array))
    vmax = float(np.max(array))
    if vmax <= vmin:
        return vmin, vmin + 1.0
    return vmin, vmax


def _plot_graph_panels(*, fig, gtr, invert_y_axis: bool) -> None:
    """Top row: the tracker's search graph, nodes and edges by heuristic.

    Calls `gtr._plot_nodes` / `gtr._plot_edges` -- private on `GraphAxonTracking`
    (confirmed present on the installed `axon_velocity==0.1.2`, the same
    version this lab's old build ran against), because there is no public
    equivalent: `GraphAxonTracking.plot_graph` draws both onto one shared axis
    with one shared colorbar, whereas this wants two axes with two independent
    heuristic colorbars (init-heuristic for nodes, path-heuristic for edges) --
    the old build's own reason for reaching past the public method, not a
    corner cut here.
    """
    import matplotlib as mpl

    graph_spec = fig.add_gridspec(
        1, 4,
        left=0.05, right=0.95, bottom=0.56, top=0.92,
        width_ratios=(7.0, 0.6, 7.0, 0.6), wspace=0.18,
    )
    ax_nodes = fig.add_subplot(graph_spec[0, 0])
    ax_nodes_cb = fig.add_subplot(graph_spec[0, 1])
    ax_edges = fig.add_subplot(graph_spec[0, 2])
    ax_edges_cb = fig.add_subplot(graph_spec[0, 3])

    gtr._plot_nodes(cmap_nodes="viridis", node_searched_labels=False, ax=ax_nodes)
    node_vmin, node_vmax = _normalize_colorbar_limits(getattr(gtr, "_node_heuristic", (1.0,)))
    node_norm = mpl.colors.Normalize(vmin=node_vmin, vmax=node_vmax)
    mpl.colorbar.ColorbarBase(
        ax_nodes_cb, cmap=mpl.colormaps["viridis"], norm=node_norm, orientation="vertical",
    ).set_label("heuristic init (a.u.)")

    gtr._plot_edges(cmap_edges="YlGn", ax=ax_edges)
    edge_vmin, edge_vmax = _normalize_colorbar_limits(
        [data.get("heur", 0.0) for _, _, data in gtr.graph.edges.data()]
    )
    edge_norm = mpl.colors.Normalize(vmin=edge_vmin, vmax=edge_vmax)
    mpl.colorbar.ColorbarBase(
        ax_edges_cb, cmap=mpl.colormaps["YlGn"], norm=edge_norm, orientation="vertical",
    ).set_label("heuristic (a.u.)")

    ax_nodes.set_title("Nodes\n(by node heuristic)")
    ax_edges.set_title("Edges\n(by edge heuristic)")
    if invert_y_axis:
        ax_nodes.invert_yaxis()
        ax_edges.invert_yaxis()


def _plot_branch_velocity_panel(*, ax, gtr) -> None:
    """Bottom-right: one velocity fit per branch, `axon_velocity`'s own
    `plot_velocity` free function doing the actual drawing.

    Re-derives each branch's clean/inlier peak-time-vs-distance fit via
    `gtr._estimate_peaks_and_dists` + `gtr.robust_velocity_estimator` rather
    than trusting `branch['velocity']`/`branch['offset']` directly, matching
    the old build -- this also surfaces which points that branch's own
    outlier rejection dropped (plotted as diamonds), which the summary JSON's
    flat `branches` list does not carry.

    **One deliberate bug fix relative to the old build's `diagnostic_plots.py`
    (verified by rendering this exact function against a toy `gtr` before
    trusting it — the blank panel is what caught this).**
    `GraphAxonTracking.robust_velocity_estimator` returns `inlier_mask=None` in
    TWO different situations that the old build's skip condition
    (`if ... or inlier_mask is None or ...: continue`) could not tell apart:
    (a) outlier removal degenerated (<=2 points survived) -- genuinely nothing
    to plot, correctly skipped; and (b) outlier removal was never even
    ATTEMPTED because the branch's `r2` already cleared
    `r2_threshold_for_outliers` (production default 0.98) -- i.e. THE
    CLEANEST branches, whose `velocity`/`offset`/`dists`/`peaks` are all
    perfectly valid. The old build's condition drops (b) on the floor along
    with (a), silently leaving the velocity panel blank for exactly the
    best-tracked units -- confirmed empirically: the hand-built synthetic
    delay line this module's own toy test uses (a near-perfect linear
    propagation, r2 well above 0.98) hits case (b) on every branch. Fixed
    here by only testing the fields case (a) actually nulls out
    (`velocity`/`offset`/`dists_clean`/`peaks_clean`) and treating
    `inlier_mask is None` as its own case: "no outliers were removed" (draw
    the fit, skip the diamond overlay), not "nothing to draw".
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from axon_velocity.plotting import plot_velocity

    paths_raw = list(getattr(gtr, "_paths_raw", ()) or ())
    cm = plt.get_cmap("tab20")
    branch_colors = [cm(index / max(1, len(paths_raw))) for index, _ in enumerate(paths_raw)]

    for branch in getattr(gtr, "branches", ()) or ():
        raw_idx = int(branch.get("raw_path_idx", 0))
        if raw_idx >= len(paths_raw):
            continue
        color = branch_colors[raw_idx] if branch_colors else cm(0.0)
        path = np.asarray(paths_raw[raw_idx])[::-1][1:]
        peaks, dists = gtr._estimate_peaks_and_dists(path)
        _path_clean, velocity, offset, r2, _p_value, dists_clean, peaks_clean, inlier_mask = (
            gtr.robust_velocity_estimator(path, peaks, dists, True)
        )
        # See docstring: inlier_mask is None both when outlier removal wasn't
        # needed (fine — plot the fit as-is) and when it degenerated (not
        # fine — nothing valid to plot), so it is deliberately NOT part of
        # this skip test; only the fields robust_velocity_estimator nulls out
        # in the genuine-degenerate case are.
        if velocity is None or offset is None or dists_clean is None or peaks_clean is None:
            continue
        plot_velocity(
            peaks_clean, dists_clean, velocity, offset,
            color=color, r2=r2, ax=ax,
            markeredgecolor="k", alpha_markers=0.3, lw=2, markersize=12, fs=18,
            plot_markers=True,
        )
        if inlier_mask is not None:
            outlier_idxs = np.where(np.asarray(inlier_mask) == False)  # noqa: E712 (numpy bool mask, not a Python bool)
            if outlier_idxs[0].size > 0:
                ax.plot(
                    peaks[outlier_idxs], dists[outlier_idxs],
                    marker="d", ls="", color=color, markersize=18, markeredgecolor="k",
                    zorder=10, alpha=0.7,
                )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_unit_reconstruction(gtr, out_path, unit_id=None, **kwargs) -> Path:
    """Render one unit's tracked axon reconstruction to a PNG at `out_path`.

    `gtr` is a `GraphAxonTracking` instance -- either fresh from
    `track_unit_axon` or unpickled from a `gtr.pkl` `save_reconstruction`
    wrote (both are the identical live object; that is the whole point of
    pickling the full instance rather than a summary -- see
    `axon_velocity_track.save_reconstruction`'s docstring). `out_path` is
    created (parents included) if it does not exist; any existing file there
    is overwritten.

    `unit_id` is used only for the figure's suptitle -- purely cosmetic, not
    part of any output CONTRACT, since the caller already encodes the unit id
    in the directory `out_path` lives under.

    `kwargs`: `dpi` (default :data:`DEFAULT_DPI`), `figsize` (default
    :data:`DEFAULT_FIGSIZE`), `invert_y_axis` (default `True`, matching the
    old build's MEA convention of row 0 at the top). Unknown kwargs are
    ignored rather than raising -- unlike `track_unit_axon`'s strict
    kwarg-filtering contract, this is a plotting convenience, not a
    scientific-parameter surface where a silently-dropped override could hide
    a real mistake.

    Always closes the figure before returning (or raising) -- this runs over
    several hundred units in one capsule invocation tonight; leaking one
    matplotlib figure per unit would exhaust memory long before the well
    finishes.

    Returns `out_path` (as a `Path`), so a caller doesn't have to re-wrap it.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)  # headless: no display on the box this runs on
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    dpi = float(kwargs.get("dpi", DEFAULT_DPI))
    figsize = tuple(kwargs.get("figsize", DEFAULT_FIGSIZE))
    invert_y_axis = bool(kwargs.get("invert_y_axis", True))

    fig = plt.figure(figsize=figsize)
    try:
        _plot_graph_panels(fig=fig, gtr=gtr, invert_y_axis=invert_y_axis)

        lower_spec = fig.add_gridspec(1, 2, left=0.05, right=0.95, bottom=0.07, top=0.47, wspace=0.14)
        ax_raw = fig.add_subplot(lower_spec[0, 0])
        gtr.plot_raw_branches(
            cmap="tab20", plot_bp=True, plot_neighbors=True, plot_full_template=True, ax=ax_raw,
        )
        ax_raw.set_title("Raw branches", fontsize=16)
        if invert_y_axis:
            ax_raw.invert_yaxis()
        handles, labels = ax_raw.get_legend_handles_labels()
        if handles and labels:
            ax_raw.legend(fontsize=9, loc="best")

        ax_velocity = fig.add_subplot(lower_spec[0, 1])
        _plot_branch_velocity_panel(ax=ax_velocity, gtr=gtr)
        ax_velocity.set_title("Branch velocities", fontsize=16)

        n_branches = len(getattr(gtr, "branches", ()) or ())
        title = f"Axon reconstruction ({n_branches} branch(es))"
        if unit_id is not None:
            title = f"Unit {unit_id} — {title}"
        fig.suptitle(title, fontsize=18, y=0.97)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        logger.info(
            "wrote reconstruction plot%s: %d branch(es) -> %s",
            f" for unit {unit_id}" if unit_id is not None else "", n_branches, out_path,
        )
    finally:
        plt.close(fig)

    return out_path


__all__ = [
    "plot_unit_reconstruction",
    "RECONSTRUCTION_PLOT_FILENAME",
    "DEFAULT_FIGSIZE",
    "DEFAULT_DPI",
]
