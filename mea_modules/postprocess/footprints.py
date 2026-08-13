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
    _add_caption,
    _fold_caption,
    _legend_dot,
    _legend_line,
    _new_figure,
    _save_and_release,
    _wrap_label,
    estimate_electrode_pitch,
)
from ..diagnostics.figure_text import PROXY_NOT_MODEL, acronym_note

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

# Legend/caption defaults, matched to the diagnostics figures.
_LEGEND_FONTSIZE = 7
_LEGEND_FRAMEALPHA = 0.85

# Mid-viridis: the legend key for the coloured traces has to be a colour that
# actually occurs in the colormap, or the key points at nothing on the figure.
_CMAP_MID_COLOR = "#21918c"


def _legend_ring(color, label, size=8.0, lw=1.2):
    """Legend key for a HOLLOW ring marker.

    The shared handle set in :mod:`mea_modules.diagnostics.channel_layout` has a
    filled dot and a line; the extremum marker on this figure is an open ring
    drawn over the electrode, and a filled key beside the filled grey electrode
    dots would read as a third population rather than as the same ring.
    """
    from matplotlib.lines import Line2D

    return Line2D(
        [],
        [],
        linestyle="none",
        marker="o",
        markersize=float(size),
        markerfacecolor="none",
        markeredgecolor=color,
        markeredgewidth=float(lw),
        label=_wrap_label(label),
    )


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
    normalize="shared",
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

    from .analyzer import _extremum_index, unit_channel_ids, unit_template

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
    if normalize == "per_channel":
        # The soma is routinely 50x the axonal deflection, so on one shared
        # scale every channel but the soma is a flat line and the footprint
        # reads as "no spread at all". Scaling each channel to its own peak
        # trades amplitude for shape and is the only way to see propagation;
        # the colour still carries the true amplitude, so nothing is lost.
        per_channel = np.where(peak_to_peak > 0, peak_to_peak, 1.0)
        drawn = template / per_channel[None, :] * height_um
    else:
        drawn = template * amplitude_scale
    segments = [
        np.column_stack((xy[i, 0] + offsets, xy[i, 1] + drawn[:, i]))
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

    # From the template already in hand rather than re-slicing the dense
    # templates array; identical to extremum_channels() by construction.
    extremum_index = _extremum_index(template)
    extremum_channel = channel_ids[extremum_index]
    if mark_extremum:
        position = xy[extremum_index]
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
        "normalize": normalize,
        "collection": collection,
        "duration_ms": (n_samples / float(analyzer.sampling_frequency)) * 1000.0,
    }


def _add_scale_bar(ax, width_um, height_um, duration_ms, amplitude_scale, normalize="shared",
                   time_label=None, color="black"):
    """Corner marker saying what one trace's width and height mean.

    Without it the figure has micrometre axes and millivolt-shaped squiggles and
    no way to tell how big either actually is. Under per-channel normalisation
    the height has no single microvolt value, and saying so is the point — a
    number there would be a lie.

    `time_label` overrides the horizontal-bar text for the caller that has no
    sampling rate and honestly reports samples rather than milliseconds
    (:func:`plot_unit_waveform_footprint`); left None, the bar keeps its
    historical ``"<duration> ms"`` wording.

    Returns the height label it drew, so the legend key can quote the same words
    the bar itself carries instead of recomputing them.
    """
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x0 = x_min + 0.04 * (x_max - x_min)
    y0 = y_min + 0.06 * (y_max - y_min)

    ax.plot([x0, x0 + width_um], [y0, y0], color=color, lw=1.2, zorder=4)
    ax.plot([x0, x0], [y0, y0 + height_um], color=color, lw=1.2, zorder=4)
    ax.text(
        x0 + width_um / 2.0, y0 - 0.015 * (y_max - y_min),
        time_label if time_label is not None else f"{duration_ms:.1f} ms",
        ha="center", va="top", fontsize=7, zorder=4, color=color,
    )
    if normalize == "per_channel":
        uv_label = "each trace's own peak"
    else:
        uv = (height_um / amplitude_scale) if amplitude_scale > 0 else float("nan")
        uv_label = f"{uv:.0f} µV"
    ax.text(
        # Offset in trace widths, not axis fractions: the vertical bar is drawn
        # in data units, so an axis-fraction pad collides with it whenever the
        # aspect-equal box is tall and narrow.
        x0 - 0.6 * width_um, y0 + height_um / 2.0, uv_label,
        ha="center", va="center", fontsize=7, rotation=90, zorder=4, color=color,
    )
    return uv_label


def plot_unit_footprint(
    analyzer,
    unit_id,
    out_path,
    title=None,
    width_um=None,
    height_um=None,
    amplitude_scale=None,
    normalize="shared",
    show_all_electrodes=True,
    mark_extremum=True,
    show_colorbar=True,
    zoom=True,
    figsize=_FOOTPRINT_FIGSIZE,
    dpi=_FOOTPRINT_DPI,
    extremum_label=None,
    caption=None,
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

    `normalize` decides what a trace's height means. ``"shared"`` (the default)
    puts every channel on one microvolt scale, which is physically honest and
    shows at a glance how much of the unit is soma. ``"per_channel"`` scales each
    trace to its own peak, which is what you want when actually tracking an axon:
    a soma 50x larger than its axonal deflections flattens every distant channel
    to a line under shared scaling, and the propagating waveform only becomes
    visible once each channel is given its own height. Colour stays true
    amplitude either way, so the two together still tell you the real numbers.

    `zoom` True frames the unit's own channels; False frames the whole array,
    which is the right choice when several PNGs will be compared side by side.

    Note that the mask, not the plot, bounds what can be seen: a footprint can
    only be as wide as the sparsity radius the analyzer was built with. If axons
    look truncated at a suspiciously round distance, raise
    ``sparsity radius`` in :func:`mea_modules.postprocess.analyzer.build_analyzer`.

    Every encoding is legended and the colour bar names its quantity and unit
    (Adam, 2026-08-11). The red ring in particular is only fully meaningful once
    you know *which other figure* that electrode was drawn into, so
    `extremum_label` should name the sibling artifact by its real emitted
    filename, e.g.::

        extremum_label="loudest electrode — its waveform is waveforms/unit_7.png"

    This module cannot know that name; only the capsule that writes both files
    can, so it passes it in rather than this module guessing. `caption` appends
    one more caption sentence for anything a legend key is too short to hold.
    """
    _channel_ids, locations = _layout(analyzer)
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
        normalize=normalize,
        show_all_electrodes=show_all_electrodes,
        mark_extremum=mark_extremum,
    )

    if not zoom:
        margin = _ZOOM_MARGIN_PITCHES * pitch
        ax.set_xlim(float(locations[:, 0].min()) - margin, float(locations[:, 0].max()) + margin)
        ax.set_ylim(float(locations[:, 1].min()) - margin, float(locations[:, 1].max()) + margin)

    uv_label = _add_scale_bar(
        ax,
        width_um,
        height_um,
        stats["duration_ms"],
        stats["amplitude_scale_um_per_uV"],
        normalize=normalize,
    )

    if show_colorbar:
        bar = fig.colorbar(stats["collection"], ax=ax, fraction=0.04, pad=0.02)
        bar.set_label("peak-to-peak (PTP) amplitude of that electrode's trace (µV)")

    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title(
        title
        or (
            f"unit {unit_id} footprint - {stats['n_channels']} channels - "
            f"{stats['extent_x_um']:.0f} x {stats['extent_y_um']:.0f} µm - "
            f"peak {stats['peak_amplitude_uV']:.0f} µV"
        )
    )

    # Every encoding gets a legend key (Adam, 2026-08-11): grey dots, coloured
    # squiggles, a red ring and a black corner bracket are four different
    # statements and only the colour has a bar to explain it.
    colour_key = "one miniature copy of this unit's average waveform per electrode"
    colour_key += (
        ", coloured by that trace's peak-to-peak (PTP) amplitude — see the colour bar"
        if show_colorbar
        else ", coloured by that trace's peak-to-peak (PTP) amplitude in microvolts (µV)"
    )
    handles = [_legend_line(_CMAP_MID_COLOR, colour_key, lw=1.4)]
    if show_all_electrodes:
        handles.append(
            _legend_dot(
                _CONTEXT_COLOR,
                "every other electrode on the array, drawn for position only",
            )
        )
    if mark_extremum:
        handles.append(
            _legend_ring(
                _EXTREMUM_COLOR,
                extremum_label
                or (
                    f"electrode where this unit's signal is largest "
                    f"(channel {stats['extremum_channel']})"
                ),
            )
        )
    if normalize == "per_channel":
        scale_key = (
            f"scale bar: one trace box spans {stats['duration_ms']:.1f} ms across; "
            "its height is each trace's own peak, so heights are not comparable "
            "between electrodes"
        )
    else:
        scale_key = (
            f"scale bar: one trace box spans {stats['duration_ms']:.1f} ms across "
            f"and {uv_label} top to bottom"
        )
    handles.append(_legend_line("black", scale_key, lw=1.2))
    ax.legend(
        handles=handles,
        loc="best",
        fontsize=_LEGEND_FONTSIZE,
        framealpha=_LEGEND_FRAMEALPHA,
        labelspacing=0.7,
    )

    caption_parts = [
        "Each electrode that this unit reaches carries a miniature copy of the "
        "unit's average waveform, drawn at that electrode's real position on the "
        "array (axes are micrometres, µm); how far those traces spread IS the "
        "footprint.",
        acronym_note("PTP"),
    ]
    if normalize == "per_channel":
        caption_parts.append(
            "Trace HEIGHT is scaled electrode by electrode so small, distant "
            "deflections stay visible; colour still carries the true amplitude in "
            "microvolts (µV), so the two together give the real numbers."
        )
    else:
        caption_parts.append(
            "Every trace is drawn on one shared microvolt (µV) scale, so trace "
            "heights are directly comparable between electrodes."
        )
    caption_parts.append(
        "Only the electrodes stored for this unit are drawn, so the footprint "
        "cannot extend past that stored neighbourhood."
    )
    caption_parts.append(PROXY_NOT_MODEL)
    caption_parts.append(caption)
    _add_caption(fig, _fold_caption(caption_parts))

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
    normalize="shared",
    show_all_electrodes=True,
    zoom=False,
    panel_size=_GRID_PANEL_SIZE,
    dpi=_GRID_DPI,
    caption=None,
):
    """Small multiples of several footprints in one figure; return `out_path`.

    The summary view: whether a batch of units sit on top of each other or tile
    the array, at a glance, without opening 800 PNGs. `zoom` defaults to False
    here — with every panel framed on the whole array the panels are directly
    comparable, which is the only reason to put them in a grid.

    Keep the unit count small (a couple of dozen); past that the panels are too
    small to read and the per-unit PNGs are the better artifact.

    Panel ticks are omitted, so the legend (below the panels, where it cannot
    cover a footprint), the colour bar and the caption are the only place the
    encodings are named (Adam, 2026-08-11). The colour bar is deliberately
    RELATIVE: :func:`_draw_footprint` normalises colour to each unit's own
    largest trace, so one shared microvolt scale across panels would be a lie —
    each panel's real peak is printed in its own title instead.

    `caption` appends one more caption sentence, which is where a caller names
    the per-unit figures this sheet summarises by their real emitted filenames,
    e.g.::

        caption="Each panel has its own full-size figure at footprints/unit_<id>.png."
    """
    import numpy as np
    from matplotlib import colormaps
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    unit_ids = list(unit_ids)
    if not unit_ids:
        raise ValueError("no unit ids to plot")

    _channel_ids, locations = _layout(analyzer)
    pitch = estimate_electrode_pitch(locations[:, 0], locations[:, 1]) or 1.0
    width_um = float(width_um) if width_um else _WIDTH_IN_PITCHES * pitch
    height_um = float(height_um) if height_um else _HEIGHT_IN_PITCHES * pitch

    n_cols = max(1, min(int(n_cols), len(unit_ids)))
    n_rows = int(np.ceil(len(unit_ids) / n_cols))
    # One extra, narrow gridspec column carries the colour bar. Building it as a
    # subplot of the SAME gridspec as the panels — rather than letting
    # fig.colorbar(ax=...) steal space from them — is what keeps the figure
    # compatible with the tight_layout `_add_caption` re-runs; the stolen-space
    # form leaves the bar's label running off the edge of the page.
    _CBAR_WIDTH_RATIO = 0.05
    fig = _new_figure(
        (panel_size[0] * (n_cols + _CBAR_WIDTH_RATIO * 4.0), panel_size[1] * n_rows), dpi
    )
    grid = fig.add_gridspec(
        n_rows, n_cols + 1, width_ratios=[1.0] * n_cols + [_CBAR_WIDTH_RATIO]
    )
    axes = np.asarray(
        [
            fig.add_subplot(grid[row, column])
            for row in range(n_rows)
            for column in range(n_cols)
        ]
    )
    cax = fig.add_subplot(grid[:, n_cols])

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
            normalize=normalize,
            show_all_electrodes=show_all_electrodes,
            mark_extremum=True,
            linewidth=0.5,
        )
        if not zoom:
            ax.set_xlim(float(locations[:, 0].min()) - margin, float(locations[:, 0].max()) + margin)
            ax.set_ylim(float(locations[:, 1].min()) - margin, float(locations[:, 1].max()) + margin)
        ax.set_title(
            f"unit {unit_id} - {stats['n_channels']} electrodes - "
            f"peak {stats['peak_amplitude_uV']:.0f} µV",
            fontsize=8,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    # Panels past the unit count would otherwise render as empty framed boxes.
    for ax in axes[len(unit_ids):]:
        ax.set_visible(False)

    if title:
        fig.suptitle(title)

    # One colour bar for the sheet, on the RELATIVE scale the panels actually
    # use: each panel is normalised to its own largest trace, so a shared
    # microvolt axis here would claim a comparability the drawing does not have.
    mappable = ScalarMappable(
        norm=Normalize(vmin=0.0, vmax=1.0), cmap=colormaps[_AMPLITUDE_CMAP]
    )
    bar = fig.colorbar(mappable, cax=cax)
    # Short enough to fit the bar's height: the caption carries the sentence.
    bar.set_label("peak-to-peak (PTP) amplitude ÷ that panel's largest", fontsize=8)

    frame_text = (
        "Each panel frames its own unit, so panels are NOT positionally comparable."
        if zoom
        else "Every panel frames the whole array, so a unit's position in one panel "
        "is the same position in every other."
    )
    caption_parts = [
        "Every electrode a unit reaches carries a miniature copy of that unit's "
        "average waveform, drawn at the electrode's real position; panel axes are "
        "the array's own geometry in micrometres (µm), with ticks omitted because "
        "they are unreadable at this size. " + frame_text,
        acronym_note("PTP"),
        "Colour is relative WITHIN a panel — the colour bar reads as a fraction of "
        "the largest trace in the SAME panel, so it is comparable inside a panel "
        "and not between panels; each panel's own peak amplitude in microvolts "
        "(µV) is printed in its title.",
    ]
    if normalize == "per_channel":
        caption_parts.append(
            "Trace height is scaled electrode by electrode so small, distant "
            "deflections stay visible; colour still carries the true amplitude."
        )
    caption_parts.append(PROXY_NOT_MODEL)
    caption_parts.append(caption)

    # Terse keys — the caption carries the sentences. Both go in the bottom
    # margin, which `_add_caption` divides between them, so the legend can
    # cover neither a footprint nor a caption line.
    handles = [
        _legend_line(_CMAP_MID_COLOR, "one trace per electrode reached", lw=1.4)
    ]
    if show_all_electrodes:
        handles.append(_legend_dot(_CONTEXT_COLOR, "every other electrode"))
    handles.append(
        _legend_ring(_EXTREMUM_COLOR, "largest-signal electrode", size=7.0)
    )
    _add_caption(fig, _fold_caption(caption_parts), legend_handles=handles)

    out_path = _save_and_release(fig, out_path)
    logger.info("wrote footprint grid: %s (%d units)", out_path, len(unit_ids))
    return out_path


# --------------------------------------------------------------------------- #
# waveform-footprint on DENSE template arrays (no analyzer required)
# --------------------------------------------------------------------------- #

WAVEFORM_FOOTPRINT_PLOT_FILENAME = "waveform_footprint.png"

DEFAULT_WAVEFORM_TRACE_THRESHOLD = 0.05
# Presentation default is deliberately LOWER than the diagnostic 5 %: at 5 %
# only a handful of channels clear the bar on a full-chip dense template, so
# the drawn traces read as a few hairlines lost on the grey backdrop — exactly
# the "much more sparse than the circles plot" Adam flagged (2026-08-12). 2 %
# pulls the mid-amplitude ring of channels in so the frame reads dense and
# legible like the circle reconstruction plot; tuned on unit 18 of the P005843
# review run (5 %: 91 traces; 2 %: ~300, filling the zoomed frame without
# turning to mush). Still an exposed knob — override per unit if wanted.
DEFAULT_WAVEFORM_TRACE_THRESHOLD_PRESENTATION = 0.02
DEFAULT_WAVEFORM_LINEWIDTH = 0.7
# Bolder traces so they stay visible at poster scale (the diagnostic 0.7 pt
# hairline all but vanishes once the figure is printed a foot wide).
DEFAULT_WAVEFORM_LINEWIDTH_PRESENTATION = 1.3
DEFAULT_WAVEFORM_MAX_TRACES = 500
DEFAULT_WAVEFORM_FOOTPRINT_DPI = 180
_WAVEFORM_FOOTPRINT_FIGSIZE = (7.5, 7.0)


def _dense_pitch_um(locations):
    """Median nearest-neighbour distance, safe on a 13k-channel union layout.

    :func:`estimate_electrode_pitch` builds the full O(n²) distance matrix —
    ~1.4 GB at Maxwell's 13 377-electrode union — so the dense path goes
    through a KD-tree (O(n log n), exact same statistic). Falls back to the
    shared estimator when SciPy is unavailable, and to 0.0 below 2 points,
    matching the shared estimator's own degenerate-layout contract.
    """
    import numpy as np

    points = np.asarray(locations, dtype=float)
    if points.shape[0] < 2:
        return 0.0
    try:
        from scipy.spatial import cKDTree
    except ImportError:  # pragma: no cover - scipy ships with spikeinterface
        return estimate_electrode_pitch(points[:, 0], points[:, 1])
    distances, _ = cKDTree(points).query(points, k=2)
    return float(np.median(distances[:, 1]))


def plot_unit_waveform_footprint(
    template,
    locations,
    out_path,
    unit_id=None,
    fs=None,
    all_locations=None,
    trace_threshold=None,
    trace_radius_um=None,
    max_traces=DEFAULT_WAVEFORM_MAX_TRACES,
    width_pitches=_WIDTH_IN_PITCHES,
    height_pitches=_HEIGHT_IN_PITCHES,
    amplitude_scale=None,
    normalize="shared",
    linewidth=None,
    dpi=DEFAULT_WAVEFORM_FOOTPRINT_DPI,
    figsize=_WAVEFORM_FOOTPRINT_FIGSIZE,
    zoom=True,
    mark_extremum=True,
    show_colorbar=True,
    title=None,
    caption=None,
    style="diagnostic",
    show_backdrop=None,
    backdrop_alpha=None,
    zoom_pad_pitches=None,
    zoom_bbox=None,
):
    """The waveform-footprint style — miniature waveform traces at their true
    electrode positions — rendered from DENSE template arrays instead of a
    SortingAnalyzer; returns `out_path`.

    This is the same visual language as :func:`plot_unit_footprint` (capsule
    07's per-unit footprint, the style Adam named **waveform-footprint**):
    one small copy of the unit's average waveform per electrode, placed at
    that electrode's micrometre position, coloured by its own peak-to-peak
    amplitude, extremum ringed in red, the rest of the array in grey
    underneath, with the corner scale bar saying what one trace's width and
    height mean. What changes is the input contract: `template` is a plain
    ``(n_channels, n_samples)`` array and `locations` ``(n_channels, 2)`` —
    the stitch/merge bundle's own per-unit slice (`mea_modules.templates.
    load_unit_inputs`) — so the FULL-chip stitched templates can be drawn,
    not only the sorter's sparse backbone subset an analyzer carries.

    **Why a channel subset exists at all.** A dense bundle covers the whole
    union layout (13 377 electrodes on a Maxwell chip); a trace on every one
    of them is spaghetti and minutes of render time. Traces are drawn only
    where the unit actually reaches: channels whose peak-to-peak (PTP)
    amplitude is at least ``trace_threshold`` of the unit's own loudest
    channel (default 5 %), optionally also within ``trace_radius_um`` of the
    extremum, hard-capped at the ``max_traces`` loudest (default 500). Every
    other electrode is drawn as a grey position-only dot, and the figure
    says how many were omitted and why — the omission rule is part of the
    figure, not a silent choice.

    `all_locations` supplies the full-array context layer when `locations`
    itself is already a covered-channel subset; left None, `locations` is
    the context. `width_pitches`/`height_pitches` size one trace's box in
    multiples of the layout's own median electrode pitch (KD-tree estimate,
    dense-safe); `amplitude_scale` (µm per µV) is derived so the loudest
    DRAWN trace fills its box unless fixed by the caller; `normalize`
    behaves exactly as in :func:`plot_unit_footprint` (``"shared"`` is
    physically honest, ``"per_channel"`` trades amplitude for shape).
    Rendering is deterministic: Agg canvas, no randomness, channels drawn in
    stable ascending-amplitude order so the loudest trace always lands on
    top. A falsy `fs` reports the trace window in samples rather than
    milliseconds, honestly.

    **`style` — diagnostic (default) vs presentation (Adam, 2026-08-12).**
    ``"diagnostic"`` is the verbose, self-documenting figure above, unchanged:
    the full prose caption, the figure-wide legend keying every encoding, the
    stats-carrying title, the whole-array grey backdrop. ``"presentation"`` is
    the deck-ready cut of the SAME plot — no prose caption block, no framed
    legend (only a compact one-entry inline key for the extremum ring), the
    title is the unit id alone with no per-unit stats baked in, a compact
    ``"PTP (µV)"`` colour-bar label, the grey position-only backdrop heavily
    lightened so the traces dominate it, and — to fix the sparsity Adam flagged
    — a lower trace threshold (more channels carry a trace) and bolder traces,
    framed to the active bounding box. The commentary that used to live on the
    figure (n-of-N electrode counts, peak amplitude, spread) moves to the slide
    text.

    Every difference presentation mode makes is also an independent exposed
    knob, so a caller can dial any of them in either style: ``trace_threshold``
    (``None`` → :data:`DEFAULT_WAVEFORM_TRACE_THRESHOLD` diagnostic /
    :data:`DEFAULT_WAVEFORM_TRACE_THRESHOLD_PRESENTATION` presentation),
    ``linewidth`` (``None`` → :data:`DEFAULT_WAVEFORM_LINEWIDTH` /
    :data:`DEFAULT_WAVEFORM_LINEWIDTH_PRESENTATION`), ``show_backdrop``
    (``None`` → on in both styles; pass ``False`` to drop the grey grid
    entirely), ``backdrop_alpha`` (``None`` → 1.0 diagnostic / 0.30
    presentation), and ``zoom_pad_pitches`` (``None`` →
    :data:`_ZOOM_MARGIN_PITCHES`, the padding in electrode pitches around the
    drawn bounding box).

    **`zoom_bbox` — crop to a supplied box, e.g. the arbor (Adam, 2026-08-12).**
    By default the zoomed frame is the bounding box of the DRAWN traces, which
    for an axonal unit at the low presentation threshold spans nearly the whole
    chip (the template crosses threshold on scattered far-field electrodes), so
    the frame reads noisy. Pass ``zoom_bbox=(xmin, xmax, ymin, ymax)`` in
    micrometres to crop the axes to that box plus the ``zoom_pad_pitches``
    margin instead; it WINS over ``zoom`` (and over the drawn-trace bbox). Its
    intended source is the unit's own ARBOR extent — the bounding box of the
    reconstruction's tracked branch-node electrode positions,
    :func:`mea_modules.reconstruction.arbor_bbox_from_reconstruction` — so the
    footprint frames the same dense arbor region the reconstruction plot frames
    and the scattered far-field crossings drop out of view. Traces outside the
    box are still computed and drawn into the collection; matplotlib simply
    clips them at the axes edge, so nothing about the selection or colour scale
    changes — only the view. Left ``None``, the historical drawn-trace-bbox
    framing is unchanged.
    """
    from pathlib import Path

    import numpy as np
    from matplotlib import colormaps
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    out_path = Path(out_path)
    presentation = str(style).strip().lower() == "presentation"
    # Resolve the style-varying knobs. Each stays independently overridable: a
    # non-None argument wins over the style default, so presentation is just a
    # bundle of defaults, never a lock.
    if trace_threshold is None:
        trace_threshold = (
            DEFAULT_WAVEFORM_TRACE_THRESHOLD_PRESENTATION
            if presentation
            else DEFAULT_WAVEFORM_TRACE_THRESHOLD
        )
    if linewidth is None:
        linewidth = (
            DEFAULT_WAVEFORM_LINEWIDTH_PRESENTATION
            if presentation
            else DEFAULT_WAVEFORM_LINEWIDTH
        )
    if show_backdrop is None:
        # Kept in both styles, but presentation heavily LIGHTENS it (alpha
        # below): a faint full-chip electrode grid fills the frame so it does
        # not read as mostly-empty white — the sparsity Adam flagged — while
        # the bold coloured traces still dominate it, exactly the structure
        # that makes the circle reconstruction plot read dense. Pass
        # show_backdrop=False to drop it entirely.
        show_backdrop = True
    if backdrop_alpha is None:
        backdrop_alpha = 0.30 if presentation else 1.0
    if zoom_pad_pitches is None:
        zoom_pad_pitches = _ZOOM_MARGIN_PITCHES
    template_arr = np.nan_to_num(np.asarray(template, dtype=float), nan=0.0)
    locations_arr = np.asarray(locations, dtype=float)[:, :2]
    if template_arr.ndim != 2 or template_arr.shape[0] != locations_arr.shape[0]:
        raise ValueError(
            f"template has {template_arr.shape[0]} channel row(s) but locations has "
            f"{locations_arr.shape[0]} -- must match 1:1"
        )
    context = (
        np.asarray(all_locations, dtype=float)[:, :2]
        if all_locations is not None
        else locations_arr
    )

    n_channels, n_samples = template_arr.shape
    peak_to_peak = np.ptp(template_arr, axis=1)
    peak = float(peak_to_peak.max()) if peak_to_peak.size else 0.0
    if peak <= 0:
        raise ValueError(
            f"unit {unit_id!r}: template is flat (peak-to-peak 0 on every "
            "channel) -- nothing to draw"
        )
    extremum_index = int(np.argmax(peak_to_peak))
    extremum_xy = locations_arr[extremum_index]

    # --- the drawing subset: reach threshold, optional radius, loudest cap ---
    keep = peak_to_peak >= float(trace_threshold) * peak
    if trace_radius_um:
        keep &= (
            np.linalg.norm(locations_arr - extremum_xy[None, :], axis=1)
            <= float(trace_radius_um)
        )
    selected = np.flatnonzero(keep)
    capped = False
    if max_traces and selected.size > int(max_traces):
        loudest = np.argsort(-peak_to_peak[selected], kind="stable")[: int(max_traces)]
        selected = selected[np.sort(loudest)]
        capped = True
    # Stable ascending-amplitude draw order: the loudest trace lands on top.
    selected = selected[np.argsort(peak_to_peak[selected], kind="stable")]
    n_drawn = int(selected.size)
    if n_drawn == 0:
        raise ValueError(
            f"unit {unit_id!r}: no channel clears trace_threshold="
            f"{trace_threshold!r} (peak {peak:.1f} µV) -- nothing to draw"
        )

    pitch = _dense_pitch_um(context) or 1.0
    width_um = float(width_pitches) * pitch
    height_um = float(height_pitches) * pitch

    drawn_max = float(np.max(np.abs(template_arr[selected]))) if n_drawn else 0.0
    if amplitude_scale is None:
        amplitude_scale = (height_um / 2.0) / drawn_max if drawn_max > 0 else 0.0

    fig = _new_figure(figsize, dpi)
    ax = fig.subplots()

    # Presentation renders on a BLACK canvas to match the circle reconstruction
    # plot (Adam, 2026-08-12: "same or similar colour schemes as circles plot,
    # dark backgrounds"). Diagnostic is unchanged — white, part of the pipeline's
    # review-figure family that shares this module's channel_layout chrome.
    dark = presentation
    bg = "black" if dark else "white"
    text_color = "white" if dark else "black"
    if dark:
        fig.set_facecolor(bg)
        ax.set_facecolor(bg)

    if show_backdrop:
        ax.scatter(
            context[:, 0], context[:, 1], s=2, c=_CONTEXT_COLOR, linewidths=0,
            alpha=float(backdrop_alpha), rasterized=True, zorder=0,
        )

    offsets = np.linspace(-width_um / 2.0, width_um / 2.0, n_samples)
    sub = template_arr[selected]  # (n_drawn, n_samples)
    if normalize == "per_channel":
        per_channel = peak_to_peak[selected]
        per_channel = np.where(per_channel > 0, per_channel, 1.0)
        drawn_traces = sub / per_channel[:, None] * height_um
    else:
        drawn_traces = sub * amplitude_scale
    xy = locations_arr[selected]
    segments = [
        np.column_stack((xy[i, 0] + offsets, xy[i, 1] + drawn_traces[i]))
        for i in range(n_drawn)
    ]
    norm = Normalize(vmin=0.0, vmax=float(peak_to_peak[selected].max()))
    collection = LineCollection(
        segments,
        array=np.asarray(peak_to_peak[selected], dtype=float),
        cmap=colormaps[_AMPLITUDE_CMAP],
        norm=norm,
        linewidths=float(linewidth),
        zorder=2,
    )
    ax.add_collection(collection)

    if mark_extremum:
        ax.scatter(
            [extremum_xy[0]], [extremum_xy[1]], s=90, facecolors="none",
            edgecolors=_EXTREMUM_COLOR, linewidths=1.2, zorder=3,
        )

    margin = float(zoom_pad_pitches) * pitch + max(width_um, height_um)
    if zoom_bbox is not None:
        # Explicit crop box (e.g. the unit's ARBOR extent from its
        # reconstruction, arbor_bbox_from_reconstruction) wins over both zoom
        # modes: frame to the supplied (xmin, xmax, ymin, ymax) + the same
        # pitch-scaled margin, so the dense arbor cluster fills the frame like
        # the reconstruction plot and the scattered far-field threshold-
        # crossings drop out of view. Traces outside the box are still in the
        # collection — matplotlib clips them at the axes edge — so nothing is
        # recomputed, only reframed. min/max guard a box given in either order.
        bx0, bx1, by0, by1 = (float(v) for v in zoom_bbox)
        ax.set_xlim(min(bx0, bx1) - margin, max(bx0, bx1) + margin)
        ax.set_ylim(min(by0, by1) - margin, max(by0, by1) + margin)
    else:
        frame = xy if zoom else context
        ax.set_xlim(float(frame[:, 0].min()) - margin, float(frame[:, 0].max()) + margin)
        ax.set_ylim(float(frame[:, 1].min()) - margin, float(frame[:, 1].max()) + margin)
    ax.set_aspect("equal", adjustable="box")

    if fs:
        duration_ms = (n_samples / float(fs)) * 1000.0
        time_words = f"{duration_ms:.1f} ms ({n_samples} samples)"
        time_label = None  # the bar's default "<duration> ms" is correct
    else:
        duration_ms = float(n_samples)
        time_words = f"{n_samples} samples (no sampling rate supplied)"
        time_label = f"{n_samples} samples"
    uv_label = _add_scale_bar(
        ax, width_um, height_um, duration_ms, amplitude_scale,
        normalize=normalize, time_label=time_label, color=text_color,
    )

    if show_colorbar:
        bar = fig.colorbar(collection, ax=ax, fraction=0.04, pad=0.02)
        bar.set_label(
            "PTP (µV)"
            if presentation
            else "peak-to-peak (PTP) amplitude of that electrode's trace (µV)",
            color=text_color,
        )
        if dark:
            bar.ax.tick_params(colors=text_color)
            bar.outline.set_edgecolor(text_color)

    ax.set_xlabel("x (µm)", color=text_color)
    ax.set_ylabel("y (µm)", color=text_color)
    if dark:
        ax.tick_params(colors=text_color)
        for spine in ax.spines.values():
            spine.set_color(text_color)
    extent_x = float(xy[:, 0].max() - xy[:, 0].min())
    extent_y = float(xy[:, 1].max() - xy[:, 1].min())
    # Standard title only. Presentation mode = the unit id alone (Adam: "keep
    # the plots standard ... put that [per-unit commentary] on the slide
    # text"), so no n-of-N electrode count and no peak µV baked into the plot.
    # Diagnostic keeps the two SHORT lines it had — the title is centred on
    # the AXES, and an aspect-equal frame routinely sits off-centre in the
    # figure, so a long title runs past the figure edge and matplotlib clips
    # rather than wraps it; the spread stats live in the log line either way.
    if title is not None:
        title_text = title
    elif presentation:
        title_text = f"Unit {unit_id}" if unit_id is not None else ""
    else:
        title_text = (
            f"unit {unit_id} waveform footprint\n{n_drawn} of {n_channels} "
            f"electrodes drawn - peak {peak:.0f} µV"
        )
    ax.set_title(title_text, fontsize=13 if presentation else 11, color=text_color)

    if presentation:
        # Presentation (Adam, 2026-08-12): no prose caption, no framed legend.
        # A single compact inline key for the extremum ring — the one genuinely
        # non-obvious glyph, exactly the "diamond = soma" case Adam allowed a
        # small key for — and nothing else. The omission rule, the encoding
        # sentences and the per-unit numbers all move to the slide text.
        if mark_extremum:
            leg = ax.legend(
                handles=[_legend_ring(_EXTREMUM_COLOR, "loudest electrode", size=8.0)],
                loc="upper right",
                fontsize=8,
                framealpha=0.6,
                handletextpad=0.4,
                borderpad=0.4,
            )
            if dark and leg is not None:
                leg.get_frame().set_facecolor(bg)
                leg.get_frame().set_edgecolor(text_color)
                for t in leg.get_texts():
                    t.set_color(text_color)
        # No `_add_caption` to run `tight_layout` for us in this branch.
        fig.tight_layout()
    else:
        # Every encoding gets a legend key (Adam, 2026-08-11), and the omission
        # rule is an encoding: a reader must know why most electrodes carry no
        # trace before trusting the spread they see.
        colour_key = (
            "one miniature copy of this unit's average waveform per drawn "
            "electrode, coloured by that trace's peak-to-peak (PTP) amplitude"
        )
        colour_key += " — see the colour bar" if show_colorbar else " in microvolts (µV)"
        subset_words = (
            f"electrodes below {trace_threshold:.0%} of the unit's loudest "
            "peak-to-peak (PTP) amplitude"
        )
        if trace_radius_um:
            subset_words += f", or farther than {float(trace_radius_um):.0f} µm from the ringed electrode"
        handles = [
            _legend_line(_CMAP_MID_COLOR, colour_key, lw=1.4),
            _legend_dot(
                _CONTEXT_COLOR,
                f"every other electrode ({int(context.shape[0]) - n_drawn}), drawn for "
                f"position only — {subset_words} carry no trace",
            ),
        ]
        if mark_extremum:
            handles.append(
                _legend_ring(
                    _EXTREMUM_COLOR,
                    f"electrode where this unit's signal is largest (peak-to-peak {peak:.0f} µV)",
                )
            )
        if normalize == "per_channel":
            scale_key = (
                f"scale bar: one trace box spans {time_words} across; its height is "
                "each trace's own peak, so heights are not comparable between electrodes"
            )
        else:
            scale_key = (
                f"scale bar: one trace box spans {time_words} across and "
                f"{uv_label} top to bottom"
            )
        handles.append(_legend_line("black", scale_key, lw=1.2))

        cap_sentence = (
            f"Traces are drawn only where this unit's peak-to-peak (PTP) amplitude "
            f"reaches at least {trace_threshold:.0%} of its loudest electrode"
        )
        if trace_radius_um:
            cap_sentence += f" and within {float(trace_radius_um):.0f} µm of that electrode"
        if capped:
            cap_sentence += f", capped at the {int(max_traces)} loudest"
        cap_sentence += (
            f"; the other {int(context.shape[0]) - n_drawn} electrodes are grey "
            "position-only dots, omitted so a full-chip template stays readable."
        )
        caption_parts = [
            "Each drawn electrode carries a miniature copy of the unit's average "
            "waveform at that electrode's real position on the array (axes are "
            "micrometres, µm); how far those traces spread IS the footprint.",
            cap_sentence,
            acronym_note("PTP"),
        ]
        if normalize == "per_channel":
            caption_parts.append(
                "Trace HEIGHT is scaled electrode by electrode so small, distant "
                "deflections stay visible; colour still carries the true amplitude "
                "in microvolts (µV), so the two together give the real numbers."
            )
        else:
            caption_parts.append(
                "Every trace is drawn on one shared microvolt (µV) scale, so trace "
                "heights are directly comparable between electrodes."
            )
        caption_parts.append(PROXY_NOT_MODEL)
        caption_parts.append(caption)
        # Legend goes through `_add_caption`'s bottom margin, not `ax.legend`: on a
        # full-chip frame there is no in-axes corner these four wrapped keys fit in
        # without covering array context or the corner scale bar (seen on the first
        # real render), and the margin owner guarantees data stays uncovered.
        _add_caption(fig, _fold_caption(caption_parts), legend_handles=handles)

    out_path = _save_and_release(
        fig, out_path, facecolor=(fig.get_facecolor() if dark else None)
    )
    logger.info(
        "wrote waveform footprint: %s (unit=%s drawn=%d/%d extent=%.0fx%.0f um peak=%.1f uV)",
        out_path, unit_id, n_drawn, n_channels, extent_x, extent_y, peak,
    )
    return out_path
