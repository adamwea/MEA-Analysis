"""Spatial footprints: the unit's template drawn on the electrode layout.

A waveform on the extremum channel tells you a unit is real. It tells you
nothing about what an AxonTracking scan was run for, which is *where the signal
goes*: the soma deflection on one electrode clump, then progressively smaller and
later deflections on clumps a few hundred micrometres away as the axon
propagates. That is only visible if the templates are drawn at their true
micrometre positions instead of stacked in channel order, which is what this
module does — one miniature trace per channel, placed at
``get_channel_locations()``.

Two things make it readable rather than a pile of spaghetti:

* **the layout's own scale.** Trace width and height are multiples of the median
  electrode pitch (see
  :func:`mea_modules.diagnostics.estimate_electrode_pitch`), so the same defaults
  work on a different chip generation without anyone editing a constant. A
  Maxwell config is clumps of tightly packed electrodes separated by empty space,
  so traces necessarily overlap inside a clump and never between clumps — the
  overlap is what shows the clump, and the separation is what shows the spread.
* **amplitude as colour.** Every trace is coloured by its own peak-to-peak
  amplitude. Without it the distant, low-amplitude axonal traces are visually
  indistinguishable from a flat channel that just happens to be in the mask.

The full electrode layout is drawn underneath in grey so a footprint is always
read in context: a unit covering two clumps in the corner of the array looks
very different from one covering two clumps in the middle.
"""

import logging

from ..diagnostics.channel_layout import (
    _new_figure,
    _save_and_release,
    estimate_electrode_pitch,
)

logger = logging.getLogger(__name__)

_FOOTPRINT_FIGSIZE = (7.5, 7.0)
_FOOTPRINT_DPI = 180
_GRID_PANEL_SIZE = (3.4, 3.2)
_GRID_DPI = 150

# Trace geometry as multiples of the electrode pitch. Width just over one pitch
# means neighbouring traces in a clump touch but stay individually traceable;
# height of two pitches gives the soma channel room to be obviously the soma.
_WIDTH_IN_PITCHES = 1.6
_HEIGHT_IN_PITCHES = 2.0

_CONTEXT_COLOR = "#cccccc"
_EXTREMUM_COLOR = "#c0392b"
_AMPLITUDE_CMAP = "viridis"

# Room around the drawn channels when zooming to the unit, again in pitches.
_ZOOM_MARGIN_PITCHES = 4.0


def _layout(analyzer):
    """``(channel_ids, locations)`` for the whole analyzer, as arrays.

    Read from the analyzer rather than its recording: the analyzer keeps the
    probe, so footprints still render for an analyzer whose source binary is on
    another filesystem.
    """
    import numpy as np

    channel_ids = list(analyzer.channel_ids)
    locations = np.asarray(analyzer.get_channel_locations(), dtype=float)
    if locations.ndim != 2 or locations.shape[0] != len(channel_ids) or locations.shape[1] < 2:
        raise ValueError(
            "analyzer has no usable 2-D channel locations; a footprint is meaningless without them"
        )
    return channel_ids, locations[:, :2]


def _draw_footprint(
    ax,
    analyzer,
    unit_id,
    locations,
    pitch,
    width_um,
    height_um,
    amplitude_scale=None,
    show_all_electrodes=True,
    mark_extremum=True,
    linewidth=0.7,
):
    """Draw one unit's footprint into `ax`; return a stats dict.

    Shared by the single-unit figure and the grid so the two can never drift
    apart. `amplitude_scale` fixes the micrometres-per-microvolt conversion —
    pass the same value across panels to make them comparable, leave it None to
    let each unit fill its own trace height.
    """
    import numpy as np
    from matplotlib import colormaps
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    from .analyzer import unit_channel_ids, unit_template
    from .waveforms import _unit_extremum_channel

    template = unit_template(analyzer, unit_id)  # (n_samples, n_unit_channels), uV
    channel_ids = unit_channel_ids(analyzer, unit_id)
    channel_indices = np.asarray(analyzer.channel_ids_to_indices(channel_ids), dtype=int)
    xy = locations[channel_indices]

    n_samples = int(template.shape[0])
    peak_to_peak = np.ptp(template, axis=0) if n_samples else np.zeros(len(channel_ids))
    largest = float(np.nanmax(np.abs(template))) if template.size else 0.0

    if amplitude_scale is None:
        # A unit with one spike can have an all-zero template; scaling by zero
        # would put NaNs into the vertex array and blank the whole figure.
        amplitude_scale = (height_um / 2.0) / largest if largest > 0 else 0.0

    if show_all_electrodes:
        ax.scatter(
            locations[:, 0], locations[:, 1], s=3, c=_CONTEXT_COLOR, linewidths=0, zorder=0
        )

    offsets = np.linspace(-width_um / 2.0, width_um / 2.0, n_samples) if n_samples else np.zeros(0)
    segments = [
        np.column_stack((xy[i, 0] + offsets, xy[i, 1] + template[:, i] * amplitude_scale))
        for i in range(len(channel_ids))
    ]

    norm = Normalize(vmin=0.0, vmax=float(np.nanmax(peak_to_peak)) if peak_to_peak.size else 1.0)
    collection = LineCollection(
        segments,
        array=np.asarray(peak_to_peak, dtype=float),
        cmap=colormaps[_AMPLITUDE_CMAP],
        norm=norm,
        linewidths=linewidth,
        zorder=2,
    )
    ax.add_collection(collection)

    extremum_channel = _unit_extremum_channel(analyzer, unit_id)
    if mark_extremum and extremum_channel in channel_ids:
        position = xy[channel_ids.index(extremum_channel)]
        ax.scatter(
            [position[0]],
            [position[1]],
            s=90,
            facecolors="none",
            edgecolors=_EXTREMUM_COLOR,
            linewidths=1.2,
            zorder=3,
        )

    margin = _ZOOM_MARGIN_PITCHES * pitch + max(width_um, height_um)
    ax.set_xlim(float(xy[:, 0].min()) - margin, float(xy[:, 0].max()) + margin)
    ax.set_ylim(float(xy[:, 1].min()) - margin, float(xy[:, 1].max()) + margin)
    # Electrode spacing is isotropic; a stretched aspect turns a round footprint
    # into a smear and makes propagation distances unreadable.
    ax.set_aspect("equal", adjustable="box")

    return {
        "unit_id": unit_id,
        "n_channels": len(channel_ids),
        "extremum_channel": extremum_channel,
        "peak_amplitude_uV": largest,
        "extent_x_um": float(xy[:, 0].max() - xy[:, 0].min()),
        "extent_y_um": float(xy[:, 1].max() - xy[:, 1].min()),
        "amplitude_scale_um_per_uV": float(amplitude_scale),
        "collection": collection,
        "duration_ms": (n_samples / float(analyzer.sampling_frequency)) * 1000.0,
    }


def _add_scale_bar(ax, width_um, height_um, duration_ms, amplitude_scale):
    """Corner marker saying what one trace's width and height mean.

    Without it the figure has micrometre axes and millivolt-shaped squiggles and
    no way to tell how big either actually is.
    """
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x0 = x_min + 0.04 * (x_max - x_min)
    y0 = y_min + 0.06 * (y_max - y_min)

    ax.plot([x0, x0 + width_um], [y0, y0], color="black", lw=1.2, zorder=4)
    ax.plot([x0, x0], [y0, y0 + height_um], color="black", lw=1.2, zorder=4)
    ax.text(
        x0 + width_um / 2.0, y0 - 0.015 * (y_max - y_min), f"{duration_ms:.1f} ms",
        ha="center", va="top", fontsize=7, zorder=4,
    )
    uv = (height_um / amplitude_scale) if amplitude_scale > 0 else float("nan")
    ax.text(
        # Offset in trace widths, not axis fractions: the vertical bar is drawn
        # in data units, so an axis-fraction pad collides with it whenever the
        # aspect-equal box is tall and narrow.
        x0 - 0.6 * width_um, y0 + height_um / 2.0, f"{uv:.0f} uV",
        ha="center", va="center", fontsize=7, rotation=90, zorder=4,
    )


def plot_unit_footprint(
    analyzer,
    unit_id,
    out_path,
    title=None,
    width_um=None,
    height_um=None,
    amplitude_scale=None,
    show_all_electrodes=True,
    mark_extremum=True,
    show_colorbar=True,
    zoom=True,
    figsize=_FOOTPRINT_FIGSIZE,
    dpi=_FOOTPRINT_DPI,
):
    """Draw one unit's template across its electrodes at true positions; return `out_path`.

    Every channel in the unit's sparsity mask gets a miniature trace centred on
    its electrode, coloured by peak-to-peak amplitude, with the extremum channel
    ringed in red and the rest of the array in grey underneath. The spatial
    extent of the drawn traces *is* the footprint — how far the unit's signal
    reaches across the chip.

    `width_um` and `height_um` set the box one trace occupies; both default to
    multiples of the layout's own median electrode pitch, so they adapt to the
    probe instead of assuming Maxwell's 17.5 um. `amplitude_scale` (micrometres
    per microvolt) is derived per unit by default so each footprint fills its
    trace height; fix it to compare units directly.

    `zoom` True frames the unit's own channels; False frames the whole array,
    which is the right choice when several PNGs will be compared side by side.

    Note that the mask, not the plot, bounds what can be seen: a footprint can
    only be as wide as the sparsity radius the analyzer was built with. If axons
    look truncated at a suspiciously round distance, raise
    ``sparsity radius`` in :func:`mea_modules.postprocess.analyzer.build_analyzer`.
    """
    channel_ids, locations = _layout(analyzer)
    pitch = estimate_electrode_pitch(locations[:, 0], locations[:, 1]) or 1.0
    width_um = float(width_um) if width_um else _WIDTH_IN_PITCHES * pitch
    height_um = float(height_um) if height_um else _HEIGHT_IN_PITCHES * pitch

    fig = _new_figure(figsize, dpi)
    ax = fig.subplots()

    stats = _draw_footprint(
        ax,
        analyzer,
        unit_id,
        locations,
        pitch,
        width_um,
        height_um,
        amplitude_scale=amplitude_scale,
        show_all_electrodes=show_all_electrodes,
        mark_extremum=mark_extremum,
    )

    if not zoom:
        margin = _ZOOM_MARGIN_PITCHES * pitch
        ax.set_xlim(float(locations[:, 0].min()) - margin, float(locations[:, 0].max()) + margin)
        ax.set_ylim(float(locations[:, 1].min()) - margin, float(locations[:, 1].max()) + margin)

    _add_scale_bar(
        ax, width_um, height_um, stats["duration_ms"], stats["amplitude_scale_um_per_uV"]
    )

    if show_colorbar:
        bar = fig.colorbar(stats["collection"], ax=ax, fraction=0.04, pad=0.02)
        bar.set_label("peak-to-peak amplitude (uV)")

    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title(
        title
        or (
            f"unit {unit_id} footprint - {stats['n_channels']} channels - "
            f"{stats['extent_x_um']:.0f} x {stats['extent_y_um']:.0f} um - "
            f"peak {stats['peak_amplitude_uV']:.0f} uV"
        )
    )
    fig.tight_layout()

    out_path = _save_and_release(fig, out_path)
    logger.info(
        "wrote unit footprint: %s (unit=%s channels=%d extent=%.0fx%.0f um peak=%.1f uV)",
        out_path,
        unit_id,
        stats["n_channels"],
        stats["extent_x_um"],
        stats["extent_y_um"],
        stats["peak_amplitude_uV"],
    )
    return out_path


def plot_footprint_grid(
    analyzer,
    unit_ids,
    out_path,
    n_cols=4,
    title=None,
    width_um=None,
    height_um=None,
    amplitude_scale=None,
    show_all_electrodes=True,
    zoom=False,
    panel_size=_GRID_PANEL_SIZE,
    dpi=_GRID_DPI,
):
    """Small multiples of several footprints in one figure; return `out_path`.

    The summary view: whether a batch of units sit on top of each other or tile
    the array, at a glance, without opening 800 PNGs. `zoom` defaults to False
    here — with every panel framed on the whole array the panels are directly
    comparable, which is the only reason to put them in a grid.

    Keep the unit count small (a couple of dozen); past that the panels are too
    small to read and the per-unit PNGs are the better artifact.
    """
    import numpy as np

    unit_ids = list(unit_ids)
    if not unit_ids:
        raise ValueError("no unit ids to plot")

    channel_ids, locations = _layout(analyzer)
    pitch = estimate_electrode_pitch(locations[:, 0], locations[:, 1]) or 1.0
    width_um = float(width_um) if width_um else _WIDTH_IN_PITCHES * pitch
    height_um = float(height_um) if height_um else _HEIGHT_IN_PITCHES * pitch

    n_cols = max(1, min(int(n_cols), len(unit_ids)))
    n_rows = int(np.ceil(len(unit_ids) / n_cols))
    fig = _new_figure((panel_size[0] * n_cols, panel_size[1] * n_rows), dpi)
    axes = np.atleast_1d(fig.subplots(n_rows, n_cols, squeeze=False)).ravel()

    margin = _ZOOM_MARGIN_PITCHES * pitch
    for ax, unit_id in zip(axes, unit_ids):
        stats = _draw_footprint(
            ax,
            analyzer,
            unit_id,
            locations,
            pitch,
            width_um,
            height_um,
            amplitude_scale=amplitude_scale,
            show_all_electrodes=show_all_electrodes,
            mark_extremum=True,
            linewidth=0.5,
        )
        if not zoom:
            ax.set_xlim(float(locations[:, 0].min()) - margin, float(locations[:, 0].max()) + margin)
            ax.set_ylim(float(locations[:, 1].min()) - margin, float(locations[:, 1].max()) + margin)
        ax.set_title(
            f"unit {unit_id} - {stats['n_channels']} ch - {stats['peak_amplitude_uV']:.0f} uV",
            fontsize=8,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    # Panels past the unit count would otherwise render as empty framed boxes.
    for ax in axes[len(unit_ids):]:
        ax.set_visible(False)

    if title:
        fig.suptitle(title)
    fig.tight_layout()

    out_path = _save_and_release(fig, out_path)
    logger.info("wrote footprint grid: %s (%d units)", out_path, len(unit_ids))
    return out_path
