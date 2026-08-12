"""All reconstructed arbors of one well on ONE canvas — the well-level
companion to `plots.plot_unit_footprint_reconstruction`'s per-unit figure.

Public API::

    from mea_modules.reconstruction import (
        unit_arbor_record,
        plot_all_reconstructions,
    )

`capsules/all_recon_overlay/run_capsule.py` (pipeline repo, capsule 26) is the
thin CLI adapter: it walks `reconstruct_axons`' per-unit `gtr.pkl` files,
squeezes each one into a lightweight :func:`unit_arbor_record` (so 40+ dense
full-array templates are never held in memory together — one pickled `gtr`
carries the whole `(n_channels, n_samples)` dense template, ~10+ MB each on a
26 400-electrode MaxOne array), and hands the records to
:func:`plot_all_reconstructions`.

**What this replaces (Adam: "there's some existing code for this in previous
build but its primitive").** The old build's
`axon_reconstructor/src/axon_recon/pipeline/stages/reconstruct/core/
report_full_chip_layout.py` (`write_full_chip_layout_plot`) drew every unit's
branches as bare `ax.plot` lines on a white canvas: no electrode/array context
(only a `Rectangle` around the data extents), no initiation-site marker, no
scale bar, a legend that was off by default and unlabeled encodings
throughout. Two things it DID get right are ported deliberately:

* **The one-color-per-neuron palette** — its `distinct_hsv` strategy
  (golden-ratio hue stepping, saturation/value tiers every 24 units) is
  ported near-verbatim as :func:`distinct_unit_colors`. Golden-angle hue
  spacing keeps consecutive colors maximally separated for ANY unit count
  (43 on the current validation well; a colormap sampled at N points would
  make neighbors nearly identical past ~20), stays deterministic with no
  RNG, and never repeats exactly (S/V tiers vary past 24). Colors are
  assigned in unit-id-sorted order (numeric-aware, ported `_unit_sort_key`),
  so the same well always renders the same colors.
* **The units-sorted, manifest-everything discipline** — the returned
  manifest records every unit's color and branch count plus every exclusion,
  matching that phase's own `manifest_units` bookkeeping.

Everything else follows the CURRENT per-unit style
(`plots.plot_unit_footprint_reconstruction` / `plots._render_footprint_core`),
per Adam's ask ("make it more like the individual reconstructions we have —
one color per neuron is the main difference"): black background with white
chrome, µm axes with equal aspect and inverted y (MEA row 0 at top), the
array's own electrodes drawn as faint dots for spatial context (the overlay's
analogue of the footprint circles — amplitude/latency belong to the per-unit
figures, identity belongs here), branches as line+round-cap paths, the
`_add_scale_bar_um` scale bar, and a plain-language caption that explains
every encoding on the figure itself (Adam's standing legend rule,
2026-08-11 — no insider jargon, acronyms defined, n shown vs n excluded and
why).

**Poster note (Adam via Rowan, 2026-08-12).** This figure goes on a printed
poster, so `dpi`, `figsize`, `linewidth`, `alpha` and marker sizes are all
first-class knobs, and the caller may request a vector SVG alongside the PNG
(`svg_path=`) — every artist here is a true vector artist (line paths, one
scatter collection, text), so the SVG scales losslessly. Defaults stay modest
for routine pipeline runs; a poster render passes `dpi=300+`.
"""

import colorsys
import logging
import textwrap
from pathlib import Path

logger = logging.getLogger(__name__)

ALL_RECONSTRUCTIONS_PLOT_FILENAME = "all_reconstructions.png"
ALL_RECONSTRUCTIONS_SVG_FILENAME = "all_reconstructions.svg"

# MaxOne's active area is wide (~3850 x 2100 um), so the default canvas is
# landscape; bbox_inches="tight" trims whatever the actual array shape leaves
# unused. 16 in x 200 dpi = 3200 px on the long edge for a routine run; a
# poster render passes dpi=300+ (16 in x 350 dpi = 5600 px).
DEFAULT_OVERLAY_FIGSIZE = (16.0, 10.0)
DEFAULT_OVERLAY_DPI = 200.0
# Thicker than a hairline on purpose: at poster blow-up (~A3 from a 16 in
# canvas) a 0.5 pt line all but vanishes; 1.8 pt survives print. Knob-exposed.
DEFAULT_OVERLAY_LINEWIDTH = 1.8
DEFAULT_OVERLAY_ALPHA = 0.9
# Initiation-site marker: a filled diamond ("D"), per Adam's review of the
# first render ("just dont use the stars as the soma-points lol. Use
# diamonds."). The WIDE variant ("D", not the thin "d") — at poster scale the
# thin diamond collapses toward a tick mark. 9 pt: a diamond carries more ink
# than the 12 pt star it replaced, so the size drops to keep the same visual
# weight over the branch lines.
DEFAULT_OVERLAY_SOMA_MARKERSIZE = 9.0
# Electrode-context dots: small enough that at MaxOne's 17.5 um pitch the
# backdrop reads as a faint grid, not a filled plane (dot diameter ~1.6 pt vs
# ~4.6 pt of pitch on the default 16 in canvas).
DEFAULT_ELECTRODE_DOT_SIZE = 2.0
# Above this many drawn units the per-unit id legend is dropped in "auto"
# mode (the caption still explains the color encoding; identity would need
# interactivity at that density).
DEFAULT_MAX_LEGEND_UNITS = 60

_GOLDEN_RATIO = 0.618033988749895


def _unit_sort_key(unit_id):
    """Numeric-aware sort key: unit '10' sorts after unit '5', and any
    non-numeric id sorts after every numeric one, lexicographically. Ported
    verbatim from the old build's `report_full_chip_layout._unit_sort_key` —
    this is the ONE ordering both color assignment and the legend use, so a
    unit's color is a pure function of the well's unit-id set.
    """
    try:
        return (0, f"{int(str(unit_id)):08d}")
    except (TypeError, ValueError):
        return (1, str(unit_id))


def distinct_unit_colors(n):
    """`n` visually distinct hex colors, deterministic, one per neuron.

    Ported from the old build's `report_full_chip_layout._resolve_unit_palette`
    (`distinct_hsv` strategy, its default): hue advances by the golden ratio
    (so consecutive indices land maximally far apart on the color wheel, for
    any `n`), while saturation and value step through tiers every 24 colors so
    hues that eventually revisit a neighborhood still differ in weight. All
    values keep floors (S >= 0.55, V >= 0.62) that stay clearly visible on
    this module's black default background.

    Chosen over a sampled colormap deliberately: `tab20` hard-repeats past 20,
    and any continuous map sampled at 40+ points makes adjacent samples nearly
    indistinguishable — the validation well alone has 43 reconstructed units.
    """
    colors = []
    for idx in range(max(0, int(n))):
        tier = idx // 24
        hue = (0.17 + idx * _GOLDEN_RATIO) % 1.0
        saturation = max(0.55, 0.86 - 0.10 * (tier % 3))
        value = max(0.62, 0.95 - 0.12 * ((tier // 3) % 2))
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append("#{0:02x}{1:02x}{2:02x}".format(
            *(int(max(0, min(255, round(channel * 255.0)))) for channel in rgb)
        ))
    return colors


def unit_arbor_record(gtr, unit_id=None):
    """Squeeze one tracked unit into the small dict the overlay draws from.

    Reads only public, documented `GraphAxonTracking` surface — the same
    attributes `plots._branch_channel_paths` and `save_reconstruction`'s JSON
    summary already trust: `gtr.branches` (each branch dict's `'channels'` is
    the electrode path in order), `gtr.locations` (µm xy per channel, stored
    verbatim by `AxonTracking.__init__`), and `gtr.init_channel` (the
    largest-amplitude electrode — `np.argmax(self.amplitudes)` in
    `axon_velocity.tracking_classes`, the tracker's own initiation site and
    the closest thing to a soma position this data carries).

    Returns a dict::

        unit_id             as given (cosmetic + manifest key)
        branch_paths        list of (N, 2) float arrays, µm, one per DRAWABLE
                            branch (>= 2 in-range channels — the same filter
                            the per-unit figure and the old build both apply)
        init_xy             (x, µm y) of the initiation site, or None
        n_branches_total    branches the tracker reported, drawable or not
        locations           the unit's full (n_channels, 2) µm array — kept so
                            a caller can pick the densest electrode backdrop
                            without holding any gtr

    Branch coordinates are resolved against THIS unit's own `locations`, so
    units tracked on different channel subsets still land in the same physical
    µm frame — nothing in the overlay assumes a shared channel index space.
    """
    import numpy as np

    locations = np.asarray(getattr(gtr, "locations"), dtype=float)
    if locations.ndim != 2 or locations.shape[1] < 2:
        raise ValueError(
            f"gtr.locations has shape {locations.shape}; expected (n_channels, 2+)"
        )

    branch_paths = []
    n_branches_total = 0
    for branch in getattr(gtr, "branches", None) or ():
        channels = branch.get("channels") if isinstance(branch, dict) else None
        if channels is None:
            continue
        n_branches_total += 1
        in_range = [int(ch) for ch in list(channels) if 0 <= int(ch) < locations.shape[0]]
        if len(in_range) < 2:
            continue
        branch_paths.append(np.asarray(locations[in_range, :2], dtype=float))

    init_xy = None
    init_channel = getattr(gtr, "init_channel", None)
    if init_channel is not None:
        try:
            init_index = int(init_channel)
        except (TypeError, ValueError):
            init_index = -1
        if 0 <= init_index < locations.shape[0]:
            init_xy = (float(locations[init_index, 0]), float(locations[init_index, 1]))

    return {
        "unit_id": unit_id,
        "branch_paths": branch_paths,
        "init_xy": init_xy,
        "n_branches_total": n_branches_total,
        "locations": locations[:, :2],
    }


def _compose_caption(*, well_label, n_drawn, n_no_branch, n_upstream,
                     upstream_reason, n_electrodes, width=118):
    """The plain-language explainer the figure carries at its bottom edge.

    One sentence per encoding, no insider jargon, counts stated with their
    reasons — Adam's standing legend rule (2026-08-11), the same contract the
    diagnostics figures satisfy via `mea_modules.diagnostics.figure_text`.
    Returned (and stored in the manifest) so tests and readers see the exact
    text the figure carries.
    """
    where = f" in {well_label}" if well_label else ""
    n_total = int(n_drawn) + int(n_no_branch) + int(n_upstream)
    parts = [
        f"Every reconstructed neuron{where}, drawn together over the recording array.",
        "Each coloured line traces one axonal branch — the electrode-to-electrode path a "
        "neuron's signal followed. Colour identifies the neuron: all lines of one colour "
        "belong to one neuron (colours are assigned in unit-id order and carry no other "
        "meaning).",
        "A diamond marks each neuron's initiation site — its largest-signal electrode, "
        "the closest available stand-in for the soma (cell body).",
        f"Faint grey dots are the array's {int(n_electrodes)} recording electrodes.",
        f"{int(n_drawn)} of {n_total} reconstructed neuron(s) are drawn.",
    ]
    if n_no_branch:
        parts.append(
            f"{int(n_no_branch)} excluded: no tracked branch spanned 2 or more electrodes, "
            "so there is no path to draw."
        )
    if n_upstream:
        reason = upstream_reason or "not loadable"
        parts.append(f"{int(n_upstream)} excluded upstream: {reason}.")
    parts.append("Axes are micrometres (µm); the white bar gives the physical scale.")
    return textwrap.fill(" ".join(parts), width=width)


def plot_all_reconstructions(records, out_path, *, locations=None,
                             well_label=None, svg_path=None,
                             n_units_excluded_upstream=0,
                             excluded_upstream_reason=None, **kwargs):
    """Render every unit's arbor on one canvas; return the manifest dict.

    `records` is a sequence of :func:`unit_arbor_record` dicts (order does not
    matter — they are re-sorted by unit id so color assignment is stable).
    `out_path` gets the PNG; `svg_path`, when given, additionally gets a
    losslessly scalable SVG of the same figure (poster/vector use — every
    artist drawn here is vector). `locations` is the electrode backdrop; when
    omitted, the densest `locations` among the records is used.

    `n_units_excluded_upstream` / `excluded_upstream_reason` let the calling
    capsule fold ITS exclusions (a unit dir with no `gtr.pkl`, an unpicklable
    file) into the figure's own n-shown-vs-n-excluded accounting — the figure
    states every exclusion and why, per Adam's legend rule. Units whose record
    holds zero drawable branches are excluded (and counted) here.

    `kwargs` (unknown keys ignored — same plotting-convenience contract as the
    rest of this package):

    `dpi` (:data:`DEFAULT_OVERLAY_DPI`; pass 300+ for print),
    `figsize` (:data:`DEFAULT_OVERLAY_FIGSIZE`),
    `linewidth` (:data:`DEFAULT_OVERLAY_LINEWIDTH`),
    `alpha` (:data:`DEFAULT_OVERLAY_ALPHA`),
    `soma_markersize` (:data:`DEFAULT_OVERLAY_SOMA_MARKERSIZE`),
    `electrode_dot_size` (:data:`DEFAULT_ELECTRODE_DOT_SIZE`),
    `unit_legend` (`"auto"` default — per-unit id legend shown while
    `n_drawn <=` `max_legend_units`; `True`/`False` force it),
    `max_legend_units` (:data:`DEFAULT_MAX_LEGEND_UNITS`),
    `invert_y_axis` (default `True`, the package's MEA convention),
    `background` (default `"black"`, matching the per-unit figures).

    Deterministic: Agg backend, unit-id-sorted color assignment, no RNG.
    Always closes the figure before returning (or raising).

    The returned manifest carries, per unit: id, hex color, branches drawn vs
    reported; plus the exclusion counts, the exact caption text, and the
    render parameters — everything the capsule needs to write a JSON manifest
    without re-deriving any of it.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)  # headless: no display on the box this runs on
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    from .plots import _add_scale_bar_um

    out_path = Path(out_path)
    dpi = float(kwargs.get("dpi", DEFAULT_OVERLAY_DPI))
    figsize = tuple(kwargs.get("figsize", DEFAULT_OVERLAY_FIGSIZE))
    linewidth = float(kwargs.get("linewidth", DEFAULT_OVERLAY_LINEWIDTH))
    alpha = float(min(1.0, max(0.0, float(kwargs.get("alpha", DEFAULT_OVERLAY_ALPHA)))))
    soma_markersize = float(kwargs.get("soma_markersize", DEFAULT_OVERLAY_SOMA_MARKERSIZE))
    electrode_dot_size = float(kwargs.get("electrode_dot_size", DEFAULT_ELECTRODE_DOT_SIZE))
    unit_legend = kwargs.get("unit_legend", "auto")
    max_legend_units = int(kwargs.get("max_legend_units", DEFAULT_MAX_LEGEND_UNITS))
    invert_y_axis = bool(kwargs.get("invert_y_axis", True))
    background = str(kwargs.get("background", "black"))
    text_color = "white" if background.strip().lower() in {"black", "k", "#000", "#000000"} else "black"
    electrode_color = "#3c3c3c" if text_color == "white" else "#c8c8c8"

    ordered = sorted(records, key=lambda record: _unit_sort_key(record.get("unit_id")))
    drawn = [record for record in ordered if record.get("branch_paths")]
    no_branch = [record for record in ordered if not record.get("branch_paths")]

    if locations is None:
        candidates = [record.get("locations") for record in ordered
                      if record.get("locations") is not None]
        if not candidates:
            raise ValueError("no locations= given and no record carries one")
        locations = max(candidates, key=lambda arr: np.asarray(arr).shape[0])
    locations_arr = np.asarray(locations, dtype=float)[:, :2]

    colors = distinct_unit_colors(len(drawn))
    caption = _compose_caption(
        well_label=well_label,
        n_drawn=len(drawn),
        n_no_branch=len(no_branch),
        n_upstream=int(n_units_excluded_upstream),
        upstream_reason=excluded_upstream_reason,
        n_electrodes=locations_arr.shape[0],
    )

    fig, ax = plt.subplots(figsize=figsize)
    try:
        fig.patch.set_facecolor(background)
        ax.set_facecolor(background)
        ax.set_aspect("equal", adjustable="box")

        pad_x = 0.02 * max(1.0, float(np.ptp(locations_arr[:, 0])))
        pad_y = 0.02 * max(1.0, float(np.ptp(locations_arr[:, 1])))
        ax.set_xlim(locations_arr[:, 0].min() - pad_x, locations_arr[:, 0].max() + pad_x)
        ax.set_ylim(locations_arr[:, 1].min() - pad_y, locations_arr[:, 1].max() + pad_y)

        # The array itself, underneath everything: one faint dot per electrode
        # — the overlay's spatial context, standing in for the per-unit
        # figures' amplitude/latency circles (those encode signal; here the
        # electrodes only say where the array is).
        ax.scatter(
            locations_arr[:, 0], locations_arr[:, 1],
            s=electrode_dot_size, c=electrode_color, linewidths=0.0, zorder=1,
        )

        n_branches_drawn = 0
        manifest_units = []
        for record, color in zip(drawn, colors):
            for points in record["branch_paths"]:
                ax.plot(
                    points[:, 0], points[:, 1],
                    color=color, linewidth=linewidth, alpha=alpha,
                    solid_capstyle="round", solid_joinstyle="round", zorder=10,
                )
            n_branches_drawn += len(record["branch_paths"])
            if record.get("init_xy") is not None:
                # "D" (wide diamond), Adam's marker of choice — see
                # DEFAULT_OVERLAY_SOMA_MARKERSIZE for the star -> diamond story.
                ax.plot(
                    [record["init_xy"][0]], [record["init_xy"][1]],
                    marker="D", markersize=soma_markersize, ls="",
                    markerfacecolor=color, markeredgecolor=text_color,
                    markeredgewidth=0.7, zorder=12,
                )
            manifest_units.append({
                "unit_id": record.get("unit_id"),
                "color": color,
                "n_branches_drawn": len(record["branch_paths"]),
                "n_branches_total": int(record.get("n_branches_total", 0)),
                "has_init_marker": record.get("init_xy") is not None,
            })

        if invert_y_axis:
            ax.invert_yaxis()
        ax.set_xlabel("x (µm)", color=text_color)
        ax.set_ylabel("y (µm)", color=text_color)
        ax.tick_params(colors=text_color, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(text_color)

        _add_scale_bar_um(ax, color=text_color, fontsize=10)

        show_legend = (unit_legend is True) or (
            str(unit_legend).strip().lower() == "auto"
            and 0 < len(drawn) <= max_legend_units
        )
        if show_legend and drawn:
            handles = [
                Line2D([0], [0], color=entry["color"], linewidth=max(2.0, linewidth),
                       label=f"unit {entry['unit_id']}")
                for entry in manifest_units
            ]
            # AXES legend anchored OUTSIDE the axes to the right — never over
            # the data (the same fixed-placement policy the per-unit figure
            # adopted after "best" auto-placement landed a legend on top of
            # its own branches), and deliberately NOT a figure-level legend:
            # `_add_caption` (diagnostics/channel_layout) is the sole owner of
            # figure-level bottom matter repo-wide (guarded by
            # test_no_emitter_pins_its_own_figure_legend), and its fixed
            # white-background styling does not fit this black-canvas figure.
            # An axes legend through `ax.legend` is that contract's own
            # sanctioned path; bbox_inches="tight" grows the canvas around it.
            legend = ax.legend(
                handles=handles, loc="center left",
                bbox_to_anchor=(1.01, 0.5), ncol=max(1, (len(handles) + 29) // 30),
                fontsize=7, framealpha=0.85, title="neuron identity",
                title_fontsize=8,
            )
            legend.get_frame().set_facecolor(background)
            legend.get_title().set_color(text_color)
            for text in legend.get_texts():
                text.set_color(text_color)

        title = f"All reconstructed axon arbors — {len(drawn)} neuron(s), {n_branches_drawn} branch(es)"
        if well_label:
            title = f"{title} — {well_label}"
        ax.set_title(title, color=text_color, fontsize=14, pad=12)

        # Caption at the bottom edge — the figure explains itself (Adam's
        # legend rule). Anchored just BELOW the canvas (negative y, va="top"):
        # bbox_inches="tight" expands the saved frame to include it, and
        # nothing else lives down there — a caption placed INSIDE the canvas
        # at y~0 landed on the x-axis label on the first real render.
        fig.text(0.5, -0.012, caption, ha="center", va="top",
                 color=text_color, fontsize=8, linespacing=1.35)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        written = {"png": str(out_path)}
        if svg_path is not None:
            svg_path = Path(svg_path)
            svg_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(svg_path, bbox_inches="tight", facecolor=fig.get_facecolor())
            written["svg"] = str(svg_path)

        logger.info(
            "wrote all-reconstructions overlay%s: %d neuron(s) drawn, %d branch(es), "
            "%d excluded (no drawable branch), %d excluded upstream -> %s",
            f" for {well_label}" if well_label else "", len(drawn), n_branches_drawn,
            len(no_branch), int(n_units_excluded_upstream), out_path,
        )
    finally:
        plt.close(fig)

    return {
        "n_units_drawn": len(drawn),
        "n_units_excluded_no_drawable_branch": len(no_branch),
        "excluded_no_drawable_branch": [record.get("unit_id") for record in no_branch],
        "n_units_excluded_upstream": int(n_units_excluded_upstream),
        "excluded_upstream_reason": excluded_upstream_reason,
        "n_branches_drawn": int(n_branches_drawn),
        "n_electrodes": int(locations_arr.shape[0]),
        "color_strategy": (
            "golden-ratio HSV (deterministic, one hue step of 0.618 per unit, "
            "S/V tiers past 24 units; ported from the legacy "
            "report_full_chip_layout distinct_hsv palette), assigned in "
            "unit-id order"
        ),
        "units": manifest_units,
        "caption": caption,
        "params": {
            "dpi": dpi, "figsize": list(figsize), "linewidth": linewidth,
            "alpha": alpha, "soma_markersize": soma_markersize,
            "electrode_dot_size": electrode_dot_size,
            "unit_legend": str(unit_legend), "background": background,
            "invert_y_axis": invert_y_axis,
        },
        "files": written,
    }


__all__ = [
    "ALL_RECONSTRUCTIONS_PLOT_FILENAME",
    "ALL_RECONSTRUCTIONS_SVG_FILENAME",
    "DEFAULT_OVERLAY_FIGSIZE",
    "DEFAULT_OVERLAY_DPI",
    "DEFAULT_OVERLAY_LINEWIDTH",
    "DEFAULT_OVERLAY_ALPHA",
    "DEFAULT_OVERLAY_SOMA_MARKERSIZE",
    "DEFAULT_ELECTRODE_DOT_SIZE",
    "DEFAULT_MAX_LEGEND_UNITS",
    "distinct_unit_colors",
    "unit_arbor_record",
    "plot_all_reconstructions",
]
