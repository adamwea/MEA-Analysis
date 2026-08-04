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

**Correction, same session.** The above ported the WRONG style as the
*primary* output: `write_unit_axon_reconstruction_diagnostic_figure` is a
real, load-bearing style from the old build, but Adam's actual request was
the amplitude/latency circle-footprint style — his own words: "circles would
scale with amplitude size, colors would scale with timing of the signals,
and reconstruction would build on top of those." That style is
:func:`plot_unit_footprint_reconstruction` below, ported from the old
build's `unit_plots.py`/`templates/core/render.py`
(`write_unit_amplitude_map_png`, `render_template_circles_plot_v2`,
`_draw_branch_morphology_overlay`) — see that function's own docstring for
what was ported verbatim vs. simplified. `plot_unit_reconstruction` above is
NOT wrong to have and is NOT removed — the 4-panel search-graph diagnostic is
a real, useful secondary view (Adam: "kind of clunky" refers to the OLD
BUILD'S IMPLEMENTATION of the circles style, not to this 4-panel figure) —
it just moved to a different capsule (`capsules/recon_diagnostics`, not
`capsules/plot_reconstructions`) so `plot_reconstructions` renders only the
style Adam actually asked to see there.
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

FOOTPRINT_RECONSTRUCTION_PLOT_FILENAME = "footprint_reconstruction.png"

DEFAULT_FOOTPRINT_FIGSIZE = (7.5, 6.5)
DEFAULT_FOOTPRINT_DPI = 150.0

# Old build's `TemplateCirclesPlotConfig`/`TemplatePlotTemplatesV2PhaseConfig`
# defaults (`marker_min_size=8.0`, `marker_max_size=50.0` — circle DIAMETER in
# points, not area). Carried over near-verbatim: this is the one visually
# load-bearing knob here (too narrow a range makes every circle look the same
# regardless of amplitude), not something worth reinventing.
# The old build's own defaults (8-50pt) were tuned against a filtered,
# spatially-local channel subset. This module deliberately plots the FULL
# union array instead (every electrode any segment routed -- up to ~13k on
# this scan), so an 8pt FLOOR means even background/off-axon electrodes stay
# visibly large and, at Maxwell's 17.5um pitch, tile the whole canvas solid
# -- exactly the "clunky" failure mode Adam flagged. `linear` min-max
# normalization already sends genuinely low-amplitude channels toward the
# floor; the fix is a much smaller floor so THEY actually recede, letting
# real amplitude peaks (near the axon) stand out. `sqrt`/`log` scaling would
# make this WORSE, not better -- both are saturating curves that boost
# low-normalized values up, the opposite of what a sparse footprint needs.
DEFAULT_MARKER_MIN_DIAMETER_PT = 1.0
DEFAULT_MARKER_MAX_DIAMETER_PT = 45.0


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


def _channel_amplitude_and_latency(template, fs):
    """Per-channel amplitude and relative latency from a `(channels, samples)` template.

    Amplitude: `np.max(np.abs(template), axis=1)` — the exact computation the
    old build's `unit_plots.write_unit_amplitude_map_png` used
    (`amp_values = np.max(np.abs(template), axis=1)`), not the alternative
    `np.ptp` its OTHER circles renderer (`render_template_circles_plot_v2`)
    used for the same metric — the two disagree by a noise floor's worth on a
    real waveform, but this task named `np.max(np.abs(...))` explicitly as
    the computation to port, so that one wins over its sibling.

    Latency: the sample index of each channel's own peak (`argmax` of the
    same `abs(template)` amplitude is built from — keeping the "peak" in both
    metrics the same event, not two different ones), made RELATIVE to the
    largest-amplitude channel's own peak time. This matches
    `render_template_circles_plot_v2`'s own `reference_index = min_indices
    [peak_index]` convention, itself aligned with `axon_velocity`'s own
    `init_channel = argmax(amplitudes)` (confirmed in
    `axon_velocity.tracking_classes.AxonTracking.__init__`) — so "the
    channel the axon initiates from" reads as latency 0 here too, not an
    arbitrary reference. Converted to milliseconds via `fs` when `fs` is
    truthy; left in raw samples otherwise (a caller passing `fs=0`/`None`
    gets *a* number, not a crash or a silent unit lie).

    Returns `(amplitude, latency)`, both `(n_channels,)` float arrays.
    """
    import numpy as np

    template = np.asarray(template, dtype=float)
    abs_template = np.abs(template)
    amplitude = np.max(abs_template, axis=1) if abs_template.size else np.zeros(0)
    peak_sample = np.argmax(abs_template, axis=1) if abs_template.size else np.zeros(0, dtype=int)

    reference_sample = float(peak_sample[int(np.argmax(amplitude))]) if amplitude.size else 0.0
    latency_samples = peak_sample.astype(float) - reference_sample

    fs = float(fs) if fs else 0.0
    latency = (latency_samples / fs * 1000.0) if fs > 0 else latency_samples
    return amplitude, latency


def _marker_areas_pt2(values, *, min_diameter_pt, max_diameter_pt, scaling="linear"):
    """Diameters-in-points -> scatter `s` areas-in-points^2, normalized by `values`.

    Ported from the old build's `render.py::_template_plot_v2_marker_sizes_pt2`:
    min-max normalize the (non-negative, NaN-safe) values to `[0, 1]`, apply
    an optional saturating curve (`sqrt`/`log`, matching the old build's own
    formulas — a few huge-amplitude outlier channels should not flatten every
    other circle to the minimum size) and map onto a `[min_diameter_pt,
    max_diameter_pt]` diameter range, then convert diameter to area —
    `matplotlib.Axes.scatter`'s `s` parameter is AREA in points^2, not
    diameter, so skipping this conversion would make apparent circle size
    grow with the SQUARE of what was intended.
    """
    import numpy as np

    values = np.abs(np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0))
    min_d = float(max(0.0, min_diameter_pt))
    max_d = float(max(min_d, max_diameter_pt))

    finite = values[np.isfinite(values)]
    if finite.size == 0 or float(np.max(finite)) <= float(np.min(finite)):
        diameters = np.full(values.shape, max_d, dtype=float)
    else:
        vmin, vmax = float(np.min(finite)), float(np.max(finite))
        normalized = np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)
        token = str(scaling or "linear").strip().lower()
        if token == "sqrt":
            normalized = np.sqrt(normalized)
        elif token == "log":
            normalized = np.log1p(9.0 * normalized) / np.log(10.0)
        diameters = min_d + (max_d - min_d) * normalized

    return np.pi * np.square(np.maximum(diameters * 0.5, 0.0))


def _branch_channel_paths(gtr):
    """Each clean branch's electrode path, as a plain list of channel indices.

    Reads `gtr.branches` — the same public, documented attribute
    `save_reconstruction`'s JSON summary and this module's own
    `_plot_branch_velocity_panel` already trust — rather than the private
    `_paths_raw` the 4-panel diagnostic figure reaches for (see this module's
    top docstring for why that ONE panel had no public alternative). Each
    branch dict's `'channels'` key is a numpy array of channel indices in
    path order (confirmed against `axon_velocity.tracking_classes
    .GraphAxonTracking.split_paths`, the code that actually builds
    `gtr.branches`: `branch_dict['channels'] = path.astype(int)`) — exactly
    what the old build's own port target
    (`render_template_circles_plot_v2`/`_draw_branch_morphology_overlay`)
    draws its branch overlay from too (there, via `branch.get("channels")` —
    see `unit_plots._preferred_branch_ids`).

    A branch with fewer than 2 channels cannot draw an edge and is dropped —
    matches the old build's own `len(channels) < 2: continue` filters in both
    `_raw_branch_payload_from_gtr` and `_clean_branch_payload_from_gtr`.
    """
    paths = []
    for branch in getattr(gtr, "branches", None) or ():
        channels = branch.get("channels") if isinstance(branch, dict) else None
        if channels is None:
            continue
        channels = [int(c) for c in list(channels)]
        if len(channels) < 2:
            continue
        paths.append(channels)
    return paths


def plot_unit_footprint_reconstruction(
    gtr, out_path, unit_id=None, template=None, locations=None, **kwargs,
) -> Path:
    """Render one unit's amplitude/latency electrode footprint with its
    tracked axon reconstruction overlaid on top — the PRIMARY
    `plot_reconstructions` output. Adam's own words for the target visual:
    "circles would scale with amplitude size, colors would scale with timing
    of the signals, and reconstruction would build on top of those."

    **Where `template`/`locations` come from.** Confirmed directly against
    `axon_velocity.tracking_classes.AxonTracking.__init__` (the base class
    `GraphAxonTracking` inherits from): `self.template = template` and
    `self.locations = locations` are stored VERBATIM as instance attributes
    (only reassigned when `upsample > 1`, and this lab's production params —
    see `axon_velocity_track._PRODUCTION_DEFAULTS` — pin `upsample=1`),
    alongside `self.fs`. So the pickled `gtr` this module already takes as
    its whole input (matching `plot_unit_reconstruction`'s own contract, and
    what `capsules/plot_reconstructions` already unpickles) already carries
    everything this function needs — no separate load from the merge stage's
    output required. The `template=`/`locations=`/`fs=` kwargs below exist
    only for a caller with its own copy of those arrays (e.g. wanting to plot
    against a different channel subset than `gtr` was tracked on); they are
    NOT part of this function's normal call path.

    **What this ports from the old build's `unit_plots.py` /
    `templates/core/render.py` (`write_unit_amplitude_map_png`,
    `render_template_circles_plot_v2`, `_draw_branch_morphology_overlay`,
    `TemplateCirclesPlotConfig`/`FootprintMapConfig`), and what got
    simplified.**

    Ported: `np.max(np.abs(template), axis=1)` for amplitude (the exact
    computation this task named); latency as each channel's own peak sample
    index, relative-zeroed at the largest-amplitude channel (see
    :func:`_channel_amplitude_and_latency`); a REVERSED `viridis` colormap
    for latency (`config.color_by == "latency"` always reverses in the old
    renderer — kept here as the default rather than a rarely-touched config
    flag); a black background with white text/colorbar (every relevant
    old-build config class defaulted to `background="black"`); circle
    DIAMETER (not area) normalized between a min/max in points with an
    optional `linear`/`sqrt`/`log` saturating curve, matching the old
    build's own `_template_plot_v2_marker_sizes_pt2` — see
    :func:`_marker_areas_pt2` — at the SAME default 8-50pt diameter range;
    branches drawn ON TOP (high `zorder`, above the base scatter) with one
    distinct color per branch from a `tab20` palette, matching the old
    build's branch-color convention.

    Simplified away, deliberately, per this task's explicit "kind of clunky,
    feel free to simplify" license: the entire `TemplateCirclesPlotConfig`/
    `FootprintMapConfig`/`CircleReconConfig` dataclass tree (a dozen-plus
    fields this function hard-codes to one sensible default each instead of
    exposing as config — scale-circle annotations, propagation-order labels,
    colorbar zero-transition contrast bands, three interchangeable
    `base_mode`s for one plot call); the old build's pixel-space
    circle-edge-trimming for branch edges (`_draw_branch_morphology_overlay`
    transforms to display space so an edge line stops exactly at a circle's
    rim) — this function instead draws each branch as a plain `ax.plot`
    line+marker path directly between electrode CENTERS, one of the old
    renderer's own documented options ("line+dot markers for branch paths")
    and visually equivalent at the circle sizes used here; and channel-subset
    filtering (`channel_scope` in the old `CircleReconConfig` — restricting
    the footprint to only the tracker's selected/branch channels) — this
    function always plots EVERY electrode `template`/`locations` covers,
    matching Adam's own description ("each electrode is a circle") rather
    than a filtered subset.

    `kwargs`: `dpi` (default :data:`DEFAULT_FOOTPRINT_DPI`), `figsize`
    (default :data:`DEFAULT_FOOTPRINT_FIGSIZE`), `invert_y_axis` (default
    `True`, matching `plot_unit_reconstruction`'s MEA convention), `cmap`
    (default `"viridis_r"`), `marker_min_diameter_pt`/
    `marker_max_diameter_pt` (defaults :data:`DEFAULT_MARKER_MIN_DIAMETER_PT`/
    :data:`DEFAULT_MARKER_MAX_DIAMETER_PT`), `marker_size_scaling`
    (`"linear"`/`"sqrt"`/`"log"`, default `"linear"`), `fs` (override for
    `gtr.fs`), `background` (default `"black"`). Unknown kwargs are ignored,
    matching `plot_unit_reconstruction`'s own plotting-convenience contract.

    Always closes the figure before returning (or raising) — same
    several-hundred-units-per-well memory concern as `plot_unit_reconstruction`.
    Returns `out_path` as a `Path`.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)  # headless: no display on the box this runs on
    import matplotlib.pyplot as plt
    import numpy as np

    out_path = Path(out_path)

    template_source = template if template is not None else getattr(gtr, "template", None)
    if template_source is None:
        raise ValueError("no template= given and gtr has no .template attribute -- pass template= explicitly")
    locations_source = locations if locations is not None else getattr(gtr, "locations", None)
    if locations_source is None:
        raise ValueError("no locations= given and gtr has no .locations attribute -- pass locations= explicitly")
    template_arr = np.asarray(template_source, dtype=float)
    locations_arr = np.asarray(locations_source, dtype=float)
    if template_arr.shape[0] != locations_arr.shape[0]:
        raise ValueError(
            f"template has {template_arr.shape[0]} channel(s) but locations has "
            f"{locations_arr.shape[0]} -- must match 1:1 (the channel index space "
            "gtr.branches' channel indices assume)"
        )
    fs = kwargs.get("fs", getattr(gtr, "fs", None))

    dpi = float(kwargs.get("dpi", DEFAULT_FOOTPRINT_DPI))
    figsize = tuple(kwargs.get("figsize", DEFAULT_FOOTPRINT_FIGSIZE))
    invert_y_axis = bool(kwargs.get("invert_y_axis", True))
    cmap_name = str(kwargs.get("cmap", "viridis_r"))
    min_diameter_pt = float(kwargs.get("marker_min_diameter_pt", DEFAULT_MARKER_MIN_DIAMETER_PT))
    max_diameter_pt = float(kwargs.get("marker_max_diameter_pt", DEFAULT_MARKER_MAX_DIAMETER_PT))
    size_scaling = str(kwargs.get("marker_size_scaling", "linear"))
    background = str(kwargs.get("background", "black"))
    text_color = "white" if background.strip().lower() in {"black", "k", "#000", "#000000"} else "black"

    amplitude, latency = _channel_amplitude_and_latency(template_arr, fs)
    sizes_pt2 = _marker_areas_pt2(
        amplitude, min_diameter_pt=min_diameter_pt, max_diameter_pt=max_diameter_pt, scaling=size_scaling,
    )

    fig, ax = plt.subplots(figsize=figsize)
    try:
        fig.patch.set_facecolor(background)
        ax.set_facecolor(background)
        ax.set_aspect("equal", adjustable="box")

        vmin, vmax = _normalize_colorbar_limits(latency)
        scatter = ax.scatter(
            locations_arr[:, 0], locations_arr[:, 1],
            s=sizes_pt2, c=latency, cmap=cmap_name,
            vmin=vmin, vmax=vmax, alpha=0.6, linewidths=0.0, edgecolors="none",
        )
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.045, pad=0.03)
        cbar.set_label("Latency (ms)" if fs else "Latency (samples)", color=text_color)
        cbar.ax.tick_params(colors=text_color, labelsize=8)
        cbar.outline.set_edgecolor(text_color)

        branch_paths = _branch_channel_paths(gtr)
        n_branches = len(branch_paths)
        branch_cmap = plt.get_cmap("tab20")
        show_legend = 0 < n_branches <= 10
        for branch_idx, channels in enumerate(branch_paths):
            color = (
                branch_cmap(branch_idx / max(1, len(branch_paths) - 1))
                if len(branch_paths) > 1 else branch_cmap(0.0)
            )
            in_range = [ch for ch in channels if 0 <= ch < locations_arr.shape[0]]
            if len(in_range) < 2:
                continue
            xy = locations_arr[in_range, :]
            ax.plot(
                xy[:, 0], xy[:, 1],
                color=color, linewidth=2.2, marker="o", markersize=5.0,
                markerfacecolor=color, markeredgecolor="black", markeredgewidth=0.6,
                solid_capstyle="round", zorder=10, alpha=0.95,
                label=(f"branch {branch_idx}" if show_legend else None),
            )

        if invert_y_axis:
            ax.invert_yaxis()
        ax.set_xlabel("x (um)", color=text_color)
        ax.set_ylabel("y (um)", color=text_color)
        ax.tick_params(colors=text_color, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(text_color)

        if show_legend:
            legend = ax.legend(loc="best", fontsize=7, framealpha=0.85)
            if legend is not None:
                legend.get_frame().set_facecolor(background)
                for text in legend.get_texts():
                    text.set_color(text_color)

        n_channels = int(locations_arr.shape[0])
        amp_min = float(np.min(amplitude)) if amplitude.size else 0.0
        amp_max = float(np.max(amplitude)) if amplitude.size else 0.0
        title = (
            f"{n_branches} branch(es), {n_channels} electrode(s), "
            f"amplitude {amp_min:.1f}-{amp_max:.1f}"
        )
        if unit_id is not None:
            title = f"Unit {unit_id} — {title}"
        ax.set_title(title, color=text_color, fontsize=12)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        logger.info(
            "wrote footprint reconstruction plot%s: %d branch(es), %d electrode(s) -> %s",
            f" for unit {unit_id}" if unit_id is not None else "", n_branches, n_channels, out_path,
        )
    finally:
        plt.close(fig)

    return out_path


__all__ = [
    "plot_unit_reconstruction",
    "RECONSTRUCTION_PLOT_FILENAME",
    "DEFAULT_FIGSIZE",
    "DEFAULT_DPI",
    "plot_unit_footprint_reconstruction",
    "FOOTPRINT_RECONSTRUCTION_PLOT_FILENAME",
    "DEFAULT_FOOTPRINT_FIGSIZE",
    "DEFAULT_FOOTPRINT_DPI",
    "DEFAULT_MARKER_MIN_DIAMETER_PT",
    "DEFAULT_MARKER_MAX_DIAMETER_PT",
]
