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
# Bumped 150 -> 220 (third correction, 2026-08-04): at pitch-capped radii
# (max ~8.4um on this 17.5um-pitch array) and the OLD 150 DPI, even the
# max-amplitude circle rendered at only a few px across on a ~2900um-wide
# axes -- pixel resolution, not just the amplitude->radius mapping, was
# part of why low-amplitude circles read as "invisible." Data-space sizing
# (still pitch-capped, still non-overlapping) is unaffected by this -- DPI
# only changes how many pixels represent a given um span.
DEFAULT_FOOTPRINT_DPI = 220.0

# SUPERSEDED (2026-08-04, second correction): a points^2 `scatter(s=...)`
# size is fixed in the FIGURE's point space, not the axes' DATA space -- it
# has no relationship to how many um apart two electrodes actually are, which
# is exactly why an amplitude-scaled points-based circle could straddle past
# its own electrode's neighbor and visually overlap it. Points-sizing was
# already the wrong primitive even after the 8pt->1pt floor fix narrowed the
# problem down to "less densely solid" rather than "not overlapping" (Adam's
# real requirement: circles must NEVER overlap a neighbor, by construction,
# not just look sparser on average). Kept here, unused, only as the record of
# what was tried first -- see `_electrode_pitch_um`/`_marker_radii_um` for
# the data-space replacement, which computes the max radius from the ACTUAL
# electrode pitch (nearest-neighbor spacing) rather than a fixed points value.
DEFAULT_MARKER_MIN_DIAMETER_PT = 1.0
DEFAULT_MARKER_MAX_DIAMETER_PT = 45.0

# Fraction of the detected electrode pitch used as the maximum circle RADIUS
# (Adam: "cap the max circle diameter at the electrode pitch... so even the
# largest-amplitude circle just fits within its electrode spacing and never
# overlaps a neighbor"). 0.48 rather than the exact 0.5 boundary -- two
# neighboring electrodes each at the true max (radius == pitch/2) would only
# just TOUCH, not overlap, but float/render rounding at that exact seam can
# still paint a 1px visual overlap; 0.48 leaves a small, deliberate margin.
DEFAULT_MAX_RADIUS_PITCH_FRACTION = 0.48
# Minimum circle radius as a fraction of the max radius (not of pitch
# directly) -- "small floor" per Adam's "biggest and smallest amps set the
# scale" framing: the smallest-amplitude channel should read as a small but
# still-visible dot, not vanish, and this stays proportionate however large
# or small the detected pitch turns out to be.
# THIRD CORRECTION (2026-08-04): raised 0.06 -> 0.22 after Adam reviewed
# unit 154 and reported many circles reading as invisible. 0.06 of an
# already-small pitch-capped max radius (~8.4um at this array's 17.5um
# pitch) rounds to a fraction of a pixel at any reasonable DPI -- correct
# in data-space, but genuinely below the render's visible threshold. 0.22
# keeps a real visual gap from the max (still obviously smaller) while
# guaranteeing every electrode, even the very lowest amplitude, paints as
# an actually-visible dot.
DEFAULT_MIN_RADIUS_MAX_FRACTION = 0.22
# Curve applied to the normalized [0, 1] amplitude position before mapping
# onto [min_radius, max_radius] -- "linear" leaves it untouched, "sqrt"/"log"
# both PULL low-normalized values UP toward the max (compressing the huge
# dynamic range real amplitude data can have -- unit 154 alone spans
# 0.9-320.6, ~356x) so the many low-but-not-minimum electrodes in between
# read as visibly differentiated dots rather than clustering near the floor.
# Defaults to "log" per Adam's own suggestion ("perhaps do log scaling or
# set a floor... so everything remains basically visible") -- paired with
# the raised floor above so even the true minimum is never sub-visible.
DEFAULT_RADIUS_SCALING = "log"


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


def _electrode_pitch_um(locations):
    """Nearest-neighbor electrode spacing, in the same units as `locations`.

    Read from the geometry, never hard-coded — this module has no business
    assuming 17.5um (MaxOne's pitch) is the pitch of whatever array a future
    recording used. A KD-tree nearest-neighbor query (not a sorted-unique-
    coordinate diff) is deliberate: it holds for any electrode LAYOUT, not
    just an axis-aligned rectangular grid, and a per-electrode nearest
    neighbor is exactly "how close can a circle centered here get to a
    circle centered at its closest neighbor" -- the actual overlap
    constraint. Takes the MEDIAN nearest-neighbor distance across all
    electrodes (not the min) so one duplicated/degenerate coordinate pair
    (distance ~0) cannot collapse the whole array's pitch estimate to zero.

    Returns `float('inf')` for fewer than 2 locations (nothing to collide
    with) rather than raising -- a caller sizing circles against this should
    treat "no neighbor" as "no cap," not a crash.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    locations = np.asarray(locations, dtype=float)
    if locations.shape[0] < 2:
        return float("inf")

    tree = cKDTree(locations)
    # k=2: each point's own coordinate (distance 0) plus its nearest actual
    # neighbor.
    distances, _ = tree.query(locations, k=2)
    nearest = distances[:, 1]
    finite = nearest[np.isfinite(nearest) & (nearest > 0)]
    if finite.size == 0:
        return float("inf")
    return float(np.median(finite))


def _marker_radii_um(
    values, *, pitch_um,
    max_radius_pitch_fraction=DEFAULT_MAX_RADIUS_PITCH_FRACTION,
    min_radius_max_fraction=DEFAULT_MIN_RADIUS_MAX_FRACTION,
    scaling=DEFAULT_RADIUS_SCALING,
):
    """Per-channel circle RADIUS in DATA units (um), capped by electrode pitch.

    Replaces :func:`_marker_areas_pt2` (points^2, figure-space) after Adam's
    second correction: a points-sized `scatter` has no relationship to how
    far apart two electrodes actually are in data space, so an
    amplitude-scaled points circle could straddle past its own electrode's
    neighbor -- exactly the overlap Adam flagged. This function instead maps
    the amplitude data range DIRECTLY onto a radius range that is provably
    bounded by geometry: `max_radius = pitch_um * max_radius_pitch_fraction`
    (never exceeds roughly half the nearest-neighbor spacing, so even the
    single largest-amplitude circle cannot reach past its neighbor's own
    center) and `min_radius = max_radius * min_radius_max_fraction` (a small
    but visible floor, proportionate to whatever the max radius turns out to
    be).

    THIRD CORRECTION (2026-08-04): the amplitude->radius map is no longer
    strictly linear. Adam reviewed a real render (unit 154, amplitude range
    0.9-320.6 -- a ~356x spread) and reported many circles reading as
    invisible: a linear min-max map crams the huge majority of ordinary
    (non-peak) electrodes into the bottom sliver of that range, right next
    to the floor. `scaling="log"` (the new default, per Adam's own
    suggestion) applies `log1p` to the NORMALIZED [0, 1] position before
    the radius map, pulling low-but-not-minimum values up toward the max --
    the true minimum still lands exactly on the floor either way, which is
    why the floor itself was ALSO raised (0.06 -> 0.22, see
    `DEFAULT_MIN_RADIUS_MAX_FRACTION`) so nothing, including the true
    minimum, renders sub-visible. `"sqrt"` is available as a gentler
    compression; `"linear"` restores the original literal min-max map.

    Returns an `(n_channels,)` float array of radii, safe to pass straight
    into `matplotlib.collections.EllipseCollection(widths=2*radii,
    heights=2*radii, ..., units="xy")`.
    """
    import numpy as np

    values = np.abs(np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0))

    if not np.isfinite(pitch_um) or pitch_um <= 0:
        # No sensible neighbor distance (degenerate/single-electrode input) --
        # fall back to a small fixed radius rather than an unbounded one.
        max_radius = 1.0
    else:
        max_radius = float(pitch_um) * float(max_radius_pitch_fraction)
    min_radius = max_radius * float(min_radius_max_fraction)

    finite = values[np.isfinite(values)]
    if finite.size == 0 or float(np.max(finite)) <= float(np.min(finite)):
        return np.full(values.shape, max_radius, dtype=float)

    vmin, vmax = float(np.min(finite)), float(np.max(finite))
    normalized = np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)

    token = str(scaling or "linear").strip().lower()
    if token == "sqrt":
        normalized = np.sqrt(normalized)
    elif token == "log":
        normalized = np.log1p(9.0 * normalized) / np.log(10.0)

    return min_radius + (max_radius - min_radius) * normalized


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


def _nice_round_length(target):
    """Round `target` UP to a visually clean 1/2/5 * 10^k value.

    Ported concept from the old build's `_add_scale_bar` (there: a bare
    `100.0 if span_x >= 180.0 else 50.0` two-value special case); this
    generalizes it to any span via the standard "1-2-5 per decade" ladder
    engineering scale bars/axis ticks conventionally use, so a scale bar
    reads as a clean round number (e.g. "500 um") whatever the array's
    actual physical extent turns out to be, not just the two hard-coded
    values the old build's simpler version handled.
    """
    import math

    target = float(max(1e-9, target))
    exponent = math.floor(math.log10(target))
    for mantissa in (1.0, 2.0, 5.0, 10.0):
        candidate = mantissa * (10.0 ** exponent)
        if candidate >= target:
            return candidate
    return 10.0 * (10.0 ** exponent)


def _add_scale_bar_um(ax, *, color="white", fontsize=9):
    """A spatial scale bar (a round-um-length line + label), bottom-right.

    Simplified port of the old build's `_add_scale_bar` (`templates/core/
    render.py`) — that version's config surface (independently configurable
    alignment/offset/fontsize/considers-fontsize-padding/...) collapsed here
    to one fixed, sensible placement, per the same "kind of clunky, simplify"
    license this whole module already operates under. Bar length is picked
    via :func:`_nice_round_length` at ~18% of the x-axis span. Drawn with a
    blended transform (X in DATA units, so the bar's LENGTH is physically
    correct; Y in AXES-fraction, so its on-screen position is stable
    regardless of the data's y-range) — matching the old build's own
    transform choice, the one part of its implementation with no simpler
    honest alternative.
    """
    from matplotlib.transforms import blended_transform_factory

    x0, x1 = ax.get_xlim()
    span_x = abs(float(x1) - float(x0))
    if span_x <= 0:
        return
    x_dir = 1.0 if x1 >= x0 else -1.0

    bar_um = _nice_round_length(0.18 * span_x)
    margin_frac = 0.04
    x_right = float(x1) - x_dir * margin_frac * span_x
    x_left = x_right - x_dir * bar_um
    y_bar = margin_frac

    transform = blended_transform_factory(ax.transData, ax.transAxes)
    ax.plot([x_left, x_right], [y_bar, y_bar], color=color, lw=2.5, solid_capstyle="butt", transform=transform)
    ax.text(
        (x_left + x_right) / 2.0, y_bar + 0.015, f"{int(round(bar_um))} um",
        transform=transform, color=color, ha="center", va="bottom", fontsize=fontsize,
    )


def _add_scale_circle_um(ax, *, radius_um, reference_value, color="white", fontsize=9):
    """A reference circle (top-left) showing what the MAX amplitude circle
    looks like, labeled with the uV value it represents.

    Simplified port of the old build's `_add_scale_circle` (`templates/core/
    render.py`) — same config-surface collapse as :func:`_add_scale_bar_um`.
    That original converts a POINTS^2 scatter area into axes-fraction via
    `fig.dpi`/the axes' pixel bbox; this version's circles are already sized
    in DATA um (:func:`_marker_radii_um`), so the conversion is simpler and
    more direct: `radius_um / axes_data_span` in each of x and y separately
    (not assumed equal, even though `ax.set_aspect("equal")` should make
    them so — cheap to be exactly correct rather than assume it holds).
    References the plot's actual max-amplitude circle (not an arbitrary
    round value) — the same "biggest amp sets the scale" honesty already
    used for the size mapping itself.
    """
    from matplotlib.patches import Ellipse

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    span_x = abs(float(x1) - float(x0))
    span_y = abs(float(y1) - float(y0))
    if span_x <= 0 or span_y <= 0 or radius_um <= 0:
        return

    rx_axes = float(radius_um) / span_x
    ry_axes = float(radius_um) / span_y
    margin = 0.04
    cx = margin + rx_axes
    cy = 1.0 - margin - ry_axes

    patch = Ellipse(
        (cx, cy), width=2.0 * rx_axes, height=2.0 * ry_axes, transform=ax.transAxes,
        fill=False, edgecolor=color, linewidth=1.8,
    )
    ax.add_patch(patch)
    ax.text(
        cx, cy - ry_axes - 0.02, f"{reference_value:.1f}",
        transform=ax.transAxes, color=color, ha="center", va="top", fontsize=fontsize,
    )


def _render_footprint_core(
    template_arr, locations_arr, fs, *,
    dpi, figsize, invert_y_axis, cmap_name,
    max_radius_pitch_fraction, min_radius_max_fraction, radius_scaling, background,
):
    """The amplitude/latency circle footprint alone — no branches, no title,
    not yet saved. Shared by :func:`plot_unit_footprint_reconstruction`
    (adds the tracked-branch overlay on top) and :func:`plot_unit_footprint`
    (footprint only, no `gtr`/reconstruction needed at all — inserted as its
    own pipeline stage, `footprint_plots`, immediately after
    `merge_templates` so a unit's raw merged-template footprint is visible
    before `reconstruct_axons` ever runs on it).

    Extracted 2026-08-04 when `footprint_plots` needed the IDENTICAL circle
    rendering (size/color/scale-bar/scale-circle/axes chrome)
    `plot_unit_footprint_reconstruction` already had, minus only the branch
    overlay — duplicating ~100 lines of matplotlib setup across two
    functions was the wrong call once a second caller existed for exactly
    the same core.

    Returns `(fig, ax, amplitude, latency, pitch_um, radii_um, text_color)`
    — the caller finishes with its own branch overlay (or not), title, and
    `fig.savefig`/`plt.close`.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)  # headless: no display on the box this runs on
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.collections import EllipseCollection

    background = str(background)
    text_color = "white" if background.strip().lower() in {"black", "k", "#000", "#000000"} else "black"

    amplitude, latency = _channel_amplitude_and_latency(template_arr, fs)
    pitch_um = _electrode_pitch_um(locations_arr)
    radii_um = _marker_radii_um(
        amplitude, pitch_um=pitch_um,
        max_radius_pitch_fraction=max_radius_pitch_fraction,
        min_radius_max_fraction=min_radius_max_fraction,
        scaling=radius_scaling,
    )

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(background)
    ax.set_facecolor(background)
    ax.set_aspect("equal", adjustable="box")
    # Data limits must be set explicitly BEFORE adding the collection: an
    # EllipseCollection with units="xy" does not participate in
    # Axes.autoscale the way a scatter PathCollection does, so without this
    # the axes can be left at their default (0, 1) view.
    pad = float(np.max(radii_um)) if radii_um.size else 1.0
    ax.set_xlim(locations_arr[:, 0].min() - pad, locations_arr[:, 0].max() + pad)
    ax.set_ylim(locations_arr[:, 1].min() - pad, locations_arr[:, 1].max() + pad)

    vmin, vmax = _normalize_colorbar_limits(latency)
    diameters_um = 2.0 * radii_um
    footprint = EllipseCollection(
        diameters_um, diameters_um, np.zeros_like(diameters_um),
        units="xy", offsets=locations_arr, offset_transform=ax.transData,
        array=latency, cmap=cmap_name, clim=(vmin, vmax),
        alpha=0.75, linewidths=0.0, edgecolors="none",
    )
    ax.add_collection(footprint)
    cbar = fig.colorbar(footprint, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Latency (ms)" if fs else "Latency (samples)", color=text_color)
    cbar.ax.tick_params(colors=text_color, labelsize=8)
    cbar.outline.set_edgecolor(text_color)

    if invert_y_axis:
        ax.invert_yaxis()
    ax.set_xlabel("x (um)", color=text_color)
    ax.set_ylabel("y (um)", color=text_color)
    ax.tick_params(colors=text_color, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(text_color)

    # Scale circle (top left): the plot's own largest-amplitude circle,
    # labeled with the uV value it represents -- an honest reference since
    # it's the ACTUAL max radius drawn, not a separately recomputed one.
    # Scale bar (bottom right): a clean round-um spatial reference. Both
    # ASCII-simple ports of the old build's `_add_scale_circle`/
    # `_add_scale_bar` (see those functions' docstrings for what was
    # collapsed) -- placed in fixed opposite corners from wherever a caller
    # puts a branch legend (top right), so none of the three ever compete.
    amp_max = float(np.max(amplitude)) if amplitude.size else 0.0
    max_radius_drawn = float(np.max(radii_um)) if radii_um.size else 0.0
    _add_scale_circle_um(ax, radius_um=max_radius_drawn, reference_value=amp_max, color=text_color)
    _add_scale_bar_um(ax, color=text_color)

    return fig, ax, amplitude, latency, pitch_um, radii_um, text_color


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
    old-build config class defaulted to `background="black"`); branches
    drawn ON TOP (high `zorder`, above the base footprint) with one distinct
    color per branch from a `tab20` palette, matching the old build's
    branch-color convention.

    **Circle sizing (second correction, 2026-08-04)**: circle RADIUS is
    computed in DATA units (um) via :func:`_marker_radii_um`, capped by the
    array's own electrode pitch (:func:`_electrode_pitch_um`, read from
    `locations` — never hard-coded) so the largest-amplitude circle is
    geometrically guaranteed not to overlap its nearest neighbor, and drawn
    with `matplotlib.collections.EllipseCollection(..., units="xy")` rather
    than `Axes.scatter(s=...)` — a points^2/figure-space size has no
    relationship to um distance between electrodes, which is exactly why an
    earlier points-based version could overlap. See those two functions'
    docstrings for the full reasoning; this replaced the module's original
    `_marker_areas_pt2`-based approach, kept in the source only as a record
    of what was tried first.

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
    (default `"viridis_r"`), `max_radius_pitch_fraction`/
    `min_radius_max_fraction`/`radius_scaling` (defaults
    :data:`DEFAULT_MAX_RADIUS_PITCH_FRACTION`/
    :data:`DEFAULT_MIN_RADIUS_MAX_FRACTION`/:data:`DEFAULT_RADIUS_SCALING`
    — `radius_scaling` is `"linear"`/`"sqrt"`/`"log"`, see
    :func:`_marker_radii_um`), `fs` (override for `gtr.fs`), `background`
    (default `"black"`). Unknown kwargs are ignored, matching
    `plot_unit_reconstruction`'s own plotting-convenience contract.

    Always closes the figure before returning (or raising) — same
    several-hundred-units-per-well memory concern as `plot_unit_reconstruction`.
    Returns `out_path` as a `Path`.
    """
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
    background = str(kwargs.get("background", "black"))

    fig, ax, amplitude, latency, pitch_um, radii_um, text_color = _render_footprint_core(
        template_arr, locations_arr, fs,
        dpi=float(kwargs.get("dpi", DEFAULT_FOOTPRINT_DPI)),
        figsize=tuple(kwargs.get("figsize", DEFAULT_FOOTPRINT_FIGSIZE)),
        invert_y_axis=bool(kwargs.get("invert_y_axis", True)),
        cmap_name=str(kwargs.get("cmap", "viridis_r")),
        max_radius_pitch_fraction=float(
            kwargs.get("max_radius_pitch_fraction", DEFAULT_MAX_RADIUS_PITCH_FRACTION)
        ),
        min_radius_max_fraction=float(
            kwargs.get("min_radius_max_fraction", DEFAULT_MIN_RADIUS_MAX_FRACTION)
        ),
        radius_scaling=str(kwargs.get("radius_scaling", DEFAULT_RADIUS_SCALING)),
        background=background,
    )
    try:
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

        if show_legend:
            # Adam: branch legend must live in a fixed corner (top right) --
            # "best" auto-placement had it landing dead-center on unit 154,
            # on top of the very branches it was labeling. The scale circle
            # (top left) and scale bar (bottom right) already drawn by
            # `_render_footprint_core` occupy the other two corners.
            legend = ax.legend(loc="upper right", fontsize=7, framealpha=0.85)
            if legend is not None:
                legend.get_frame().set_facecolor(background)
                for text in legend.get_texts():
                    text.set_color(text_color)

        n_channels = int(locations_arr.shape[0])
        amp_min = float(np.min(amplitude)) if amplitude.size else 0.0
        amp_max = float(np.max(amplitude)) if amplitude.size else 0.0
        pitch_label = f"{pitch_um:.1f}um" if np.isfinite(pitch_um) else "n/a"
        title = (
            f"{n_branches} branch(es), {n_channels} electrode(s), "
            f"amplitude {amp_min:.1f}-{amp_max:.1f}, pitch {pitch_label}"
        )
        if unit_id is not None:
            title = f"Unit {unit_id} — {title}"
        ax.set_title(title, color=text_color, fontsize=12)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=float(kwargs.get("dpi", DEFAULT_FOOTPRINT_DPI)), bbox_inches="tight", facecolor=fig.get_facecolor())
        logger.info(
            "wrote footprint reconstruction plot%s: %d branch(es), %d electrode(s) -> %s",
            f" for unit {unit_id}" if unit_id is not None else "", n_branches, n_channels, out_path,
        )
    finally:
        plt.close(fig)

    return out_path


FOOTPRINT_PLOT_FILENAME = "footprint.png"


def plot_unit_footprint(template, locations, out_path, unit_id=None, fs=None, **kwargs) -> Path:
    """Render one unit's amplitude/latency electrode footprint alone — the
    `footprint_plots` output, inserted as its own pipeline stage right
    after `merge_templates`. "Basically the same as the recon plot ...
    but without the branches" (Adam) — this is exactly
    :func:`plot_unit_footprint_reconstruction` minus the `gtr`/tracked-
    branch overlay, sharing the identical circle/color/scale-bar/
    scale-circle rendering via :func:`_render_footprint_core`. Deliberately
    does NOT need `reconstruct_axons` to have run at all: `template`/
    `locations` come straight from `merge_templates`' own output
    (`merged_templates.npy`/`channel_locations_xy.npy`), so a unit's raw
    merged-template footprint is visible immediately, before axon_velocity
    tracking is ever attempted on it.

    `template` is `(n_channels, n_samples)`, `locations` is
    `(n_channels, 2)` — the same contract `merge_templates`' own per-unit
    slice already is. `fs` is optional (as in
    `plot_unit_footprint_reconstruction`, a falsy `fs` reports latency in
    raw samples rather than ms).

    `kwargs`: identical surface to `plot_unit_footprint_reconstruction`'s
    own kwargs (`dpi`/`figsize`/`invert_y_axis`/`cmap`/
    `max_radius_pitch_fraction`/`min_radius_max_fraction`/`radius_scaling`/
    `background`) — see that function's docstring.

    No legend (nothing to label without branches) — the scale circle
    (top left) and scale bar (bottom right) from `_render_footprint_core`
    are the only chrome. Always closes the figure before returning (or
    raising). Returns `out_path` as a `Path`.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    out_path = Path(out_path)
    template_arr = np.asarray(template, dtype=float)
    locations_arr = np.asarray(locations, dtype=float)
    if template_arr.shape[0] != locations_arr.shape[0]:
        raise ValueError(
            f"template has {template_arr.shape[0]} channel(s) but locations has "
            f"{locations_arr.shape[0]} -- must match 1:1"
        )

    dpi = float(kwargs.get("dpi", DEFAULT_FOOTPRINT_DPI))
    fig, ax, amplitude, latency, pitch_um, radii_um, text_color = _render_footprint_core(
        template_arr, locations_arr, fs,
        dpi=dpi,
        figsize=tuple(kwargs.get("figsize", DEFAULT_FOOTPRINT_FIGSIZE)),
        invert_y_axis=bool(kwargs.get("invert_y_axis", True)),
        cmap_name=str(kwargs.get("cmap", "viridis_r")),
        max_radius_pitch_fraction=float(
            kwargs.get("max_radius_pitch_fraction", DEFAULT_MAX_RADIUS_PITCH_FRACTION)
        ),
        min_radius_max_fraction=float(
            kwargs.get("min_radius_max_fraction", DEFAULT_MIN_RADIUS_MAX_FRACTION)
        ),
        radius_scaling=str(kwargs.get("radius_scaling", DEFAULT_RADIUS_SCALING)),
        background=str(kwargs.get("background", "black")),
    )
    try:
        n_channels = int(locations_arr.shape[0])
        amp_min = float(np.min(amplitude)) if amplitude.size else 0.0
        amp_max = float(np.max(amplitude)) if amplitude.size else 0.0
        pitch_label = f"{pitch_um:.1f}um" if np.isfinite(pitch_um) else "n/a"
        title = f"{n_channels} electrode(s), amplitude {amp_min:.1f}-{amp_max:.1f}, pitch {pitch_label}"
        if unit_id is not None:
            title = f"Unit {unit_id} — {title}"
        ax.set_title(title, color=text_color, fontsize=12)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        logger.info(
            "wrote footprint plot%s: %d electrode(s) -> %s",
            f" for unit {unit_id}" if unit_id is not None else "", n_channels, out_path,
        )
    finally:
        plt.close(fig)

    return out_path


FOOTPRINT_DIAGNOSTIC_PLOT_FILENAME = "footprint_diagnostic.png"

DEFAULT_FOOTPRINT_DIAGNOSTIC_FIGSIZE = (10.0, 4.5)
DEFAULT_FOOTPRINT_DIAGNOSTIC_DPI = 120.0


def plot_unit_footprint_diagnostic(template, locations, out_path, unit_id=None, fs=None, **kwargs) -> Path:
    """Quick, cheap 2-panel look at one unit's raw merged template — the
    `footprint_diagnostics` output, inserted right after `merge_templates`
    (same slot as `footprint_plots`, an independent reader of the same
    input). Mirrors `recon_diagnostics`' role relative to
    `plot_reconstructions` one stage later: "light, judge quality per unit,
    NOT presentation-grade" — so this is deliberately NOT
    `_render_footprint_core`'s pitch-capped, non-overlapping, chrome-laden
    rendering (that machinery exists specifically because `footprint_plots`
    needs to look good; a diagnostic only needs to look INFORMATIVE, fast,
    over potentially hundreds of units).

    Two plain `Axes.scatter` panels side by side, fixed small marker size
    (no per-channel size encoding, no pitch/overlap math): LEFT = amplitude
    (`viridis`), RIGHT = latency (`viridis_r`) — the same two underlying
    per-channel metrics `_channel_amplitude_and_latency` already computes
    for the presentation plot, just rendered the cheap way. No scale bar,
    no scale circle, no legend — this is a sanity-check view, not a figure
    meant to stand alone.

    **Amplitude color scale (Adam, 2026-08-04)**: amplitude is heavy-tailed
    in real data (unit 154's own real values span 0.9-320.6, a ~356x
    range) — under a LINEAR color scale, a handful of peak channels near
    the axon soak up the whole colorbar and every other channel reads as
    one flat, indistinguishable color (exactly what the first real render
    of this panel showed). `amplitude_scaling="log"` (the default, per
    Adam's own request; "linear" restores the old behavior, kept
    available/optional for the future rather than hard-coded either way)
    switches ONLY the amplitude panel to `matplotlib.colors.LogNorm` — the
    latency panel stays linear regardless, since latency is signed/
    zero-centered and a log color scale has no sensible meaning there.
    Non-positive amplitude values (a channel with a literal zero
    max-abs-sample -- degenerate but not impossible) are clipped to a small
    positive floor before the log norm is built, since `LogNorm` cannot
    represent zero or negative values.

    `kwargs`: `dpi` (default :data:`DEFAULT_FOOTPRINT_DIAGNOSTIC_DPI`),
    `figsize` (default :data:`DEFAULT_FOOTPRINT_DIAGNOSTIC_FIGSIZE`),
    `invert_y_axis` (default `True`), `background` (default `"black"`),
    `amplitude_scaling` (`"log"`/`"linear"`, default `"log"`).

    Always closes the figure before returning (or raising). Returns
    `out_path` as a `Path`.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import numpy as np

    out_path = Path(out_path)
    template_arr = np.asarray(template, dtype=float)
    locations_arr = np.asarray(locations, dtype=float)
    if template_arr.shape[0] != locations_arr.shape[0]:
        raise ValueError(
            f"template has {template_arr.shape[0]} channel(s) but locations has "
            f"{locations_arr.shape[0]} -- must match 1:1"
        )

    dpi = float(kwargs.get("dpi", DEFAULT_FOOTPRINT_DIAGNOSTIC_DPI))
    figsize = tuple(kwargs.get("figsize", DEFAULT_FOOTPRINT_DIAGNOSTIC_FIGSIZE))
    invert_y_axis = bool(kwargs.get("invert_y_axis", True))
    background = str(kwargs.get("background", "black"))
    text_color = "white" if background.strip().lower() in {"black", "k", "#000", "#000000"} else "black"
    amplitude_scaling = str(kwargs.get("amplitude_scaling", "log")).strip().lower()

    amplitude, latency = _channel_amplitude_and_latency(template_arr, fs)

    fig, (ax_amp, ax_lat) = plt.subplots(1, 2, figsize=figsize)
    try:
        fig.patch.set_facecolor(background)
        for ax, values, cmap, panel_label, use_log in (
            (ax_amp, amplitude, "viridis", "Amplitude", amplitude_scaling == "log"),
            (ax_lat, latency, "viridis_r", "Latency", False),
        ):
            ax.set_facecolor(background)
            ax.set_aspect("equal", adjustable="box")
            if use_log:
                positive = values[np.isfinite(values) & (values > 0)]
                floor = float(np.min(positive)) if positive.size else 1e-3
                plotted_values = np.clip(values, floor, None)
                vmax = float(np.max(plotted_values)) if plotted_values.size else 1.0
                norm = mcolors.LogNorm(vmin=max(floor, 1e-6), vmax=max(vmax, floor * 1.001))
                scatter = ax.scatter(
                    locations_arr[:, 0], locations_arr[:, 1],
                    s=4.0, c=plotted_values, cmap=cmap, norm=norm, linewidths=0.0,
                )
            else:
                scatter = ax.scatter(
                    locations_arr[:, 0], locations_arr[:, 1],
                    s=4.0, c=values, cmap=cmap, linewidths=0.0,
                )
            cbar = fig.colorbar(scatter, ax=ax, fraction=0.045, pad=0.03)
            cbar.ax.tick_params(colors=text_color, labelsize=7)
            if invert_y_axis:
                ax.invert_yaxis()
            ax.set_xlabel("x (um)", color=text_color, fontsize=8)
            ax.set_ylabel("y (um)", color=text_color, fontsize=8)
            ax.tick_params(colors=text_color, labelsize=7)
            for spine in ax.spines.values():
                spine.set_color(text_color)
            ax.set_title(panel_label, color=text_color, fontsize=10)

        n_channels = int(locations_arr.shape[0])
        title = f"{n_channels} electrode(s)"
        if unit_id is not None:
            title = f"Unit {unit_id} — {title}"
        fig.suptitle(title, color=text_color, fontsize=12)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        logger.info(
            "wrote footprint diagnostic%s: %d electrode(s) -> %s",
            f" for unit {unit_id}" if unit_id is not None else "", n_channels, out_path,
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
    "DEFAULT_MAX_RADIUS_PITCH_FRACTION",
    "DEFAULT_MIN_RADIUS_MAX_FRACTION",
    "DEFAULT_RADIUS_SCALING",
    "plot_unit_footprint",
    "FOOTPRINT_PLOT_FILENAME",
    "plot_unit_footprint_diagnostic",
    "FOOTPRINT_DIAGNOSTIC_PLOT_FILENAME",
    "DEFAULT_FOOTPRINT_DIAGNOSTIC_FIGSIZE",
    "DEFAULT_FOOTPRINT_DIAGNOSTIC_DPI",
]
