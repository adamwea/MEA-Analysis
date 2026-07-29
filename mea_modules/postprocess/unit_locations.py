"""Where the sorter thinks each unit sits, drawn on the array it was seen with.

SpikeInterface's ``unit_locations`` extension estimates a position per unit from
its template. Plotted against the electrodes those templates came from, it
answers two things at a glance: whether units land inside the array at all, and
whether they cluster where electrodes are dense or spread out over the whole
plate.

The electrode layer matters more here than it looks. On an AxonTracking scan the
electrodes fall into two very different populations — a fixed anchor set routed
in every segment, and a much larger tiled set routed once each — and a unit
located over one is supported by ~21x more data than a unit located over the
other. Both are drawn, faintly, and distinguished from each other, so the
reader can see which kind of ground a location is standing on without the plot
asserting anything about it.

**Descriptive only.** Nothing is thresholded, filtered, or ranked; this runs
before any post-sort QC. A location estimated from three spikes is drawn exactly
like one estimated from three thousand — the coverage diagnostics are where that
distinction lives.
"""

import logging

from ..diagnostics.channel_layout import _new_figure, _save_and_release

logger = logging.getLogger(__name__)

DEFAULT_FIGSIZE = (11.0, 9.0)
DEFAULT_DPI = 180


def plot_unit_locations(electrode_positions, unit_locations, out_path,
                        fixed_mask=None, values=None, value_label=None,
                        unit_ids=None, title=None,
                        figsize=DEFAULT_FIGSIZE, dpi=DEFAULT_DPI):
    """Scatter unit positions over the electrode layout; return `out_path`.

    `electrode_positions` is ``(n_electrodes, 2)``; `unit_locations` is
    ``(n_units, 2)`` or ``(n_units, 3)`` — SpikeInterface's monopolar
    triangulation returns x, y, z and only the first two are plotted.

    `fixed_mask` marks the electrodes routed in every segment (the sorting
    backbone). They are drawn slightly darker than the rest, which is enough to
    read the two populations apart without the background competing with the
    units.

    `values` optionally colours the units by a per-unit quantity the caller
    already holds (spike count, peak amplitude, estimated depth). Left None the
    units are one colour, which is the honest default: any colouring is a
    choice about what to draw attention to.
    """
    import numpy as np

    electrodes = np.asarray(electrode_positions, dtype=float)[:, :2]
    locations = np.asarray(unit_locations, dtype=float)
    if locations.ndim != 2 or locations.shape[1] < 2:
        raise ValueError(f"unit_locations must be (n_units, >=2); got {locations.shape}")
    xy = locations[:, :2]

    fig = _new_figure(figsize, dpi)
    ax = fig.subplots(1, 1)

    # --- the array, faint, two populations distinguished -------------------
    if fixed_mask is None:
        ax.scatter(electrodes[:, 0], electrodes[:, 1], s=3, c="0.86",
                   marker="s", linewidths=0, rasterized=True,
                   label=f"electrodes ({electrodes.shape[0]})")
    else:
        fixed_mask = np.asarray(fixed_mask, dtype=bool)
        ax.scatter(electrodes[~fixed_mask, 0], electrodes[~fixed_mask, 1],
                   s=3, c="0.88", marker="s", linewidths=0, rasterized=True,
                   label=f"routed in some segments ({int((~fixed_mask).sum())})")
        ax.scatter(electrodes[fixed_mask, 0], electrodes[fixed_mask, 1],
                   s=9, c="0.62", marker="s", linewidths=0, rasterized=True,
                   label=f"routed in every segment ({int(fixed_mask.sum())})")

    # --- the units --------------------------------------------------------
    if values is None:
        ax.scatter(xy[:, 0], xy[:, 1], s=22, c="crimson", marker="o",
                   edgecolors="black", linewidths=0.3, alpha=0.75,
                   label=f"units ({xy.shape[0]})", zorder=3)
    else:
        values = np.asarray(values, dtype=float)
        scatter = ax.scatter(xy[:, 0], xy[:, 1], s=22, c=values, cmap="viridis",
                             marker="o", edgecolors="black", linewidths=0.3,
                             zorder=3, label=f"units ({xy.shape[0]})")
        fig.colorbar(scatter, ax=ax, label=value_label or "value", shrink=0.75)

    # Units placed outside the electrode extent are worth counting: a location
    # estimate is an extrapolation once it leaves the array, so this is a fact
    # about the estimate rather than a verdict on the unit.
    lo, hi = electrodes.min(axis=0), electrodes.max(axis=0)
    outside = int(((xy < lo) | (xy > hi)).any(axis=1).sum())

    ax.set_aspect("equal")
    ax.set_xlabel(
        f"x (um)\n{xy.shape[0]} units   |   {outside} located outside the electrode extent"
    )
    ax.set_ylabel("y (um)")
    ax.set_title(title or "unit locations on the array")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()

    logger.info(
        "unit locations: %d unit(s) over %d electrode(s); %d outside the array extent",
        xy.shape[0], electrodes.shape[0], outside,
    )
    return _save_and_release(fig, out_path)


def unit_location_array(analyzer, method="monopolar_triangulation"):
    """Compute (or reuse) the ``unit_locations`` extension; return the array.

    Kept beside the plot so both capsules get the same estimator rather than
    each picking one. The extension is reused when already present, since it is
    derived from templates and costs a pass over them.
    """
    if not analyzer.has_extension("unit_locations"):
        analyzer.compute("unit_locations", method=method)
    return analyzer.get_extension("unit_locations").get_data()


__all__ = ["plot_unit_locations", "unit_location_array"]
