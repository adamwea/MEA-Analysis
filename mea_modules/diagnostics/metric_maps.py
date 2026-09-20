"""Per-channel metrics painted on the probe geometry.

A bar chart carries the same numbers a metric map does, but only the spatial
version answers the question a reviewer actually has: are the loud channels one
corner of the array — a seating or grounding problem — or scattered speckle,
which is ordinary noise; and is the activity one clump of live tissue or
nothing at all. Position is the whole point of these figures.

Two pieces, split so the clip is testable without rendering anything:

* :func:`robust_color_limits` — the percentile clip. One shorted electrode
  reading a hundred times the array median would otherwise push every other
  channel onto a single colour and flatten the map.
* :func:`plot_metric_maps` — the shared scatter engine, one panel per metric on
  shared axes, plus the named figures built on it: the two-panel composite
  (:func:`plot_noise_activity_map`) and its single-panel twins
  (:func:`plot_noise_map`, :func:`plot_firing_rate_map`) — one drawing routine,
  :func:`_draw_metric_panel`, behind all three.

The values painted here come from :mod:`mea_modules.quality` — ``mad_noise``
and ``activity_rate`` — and are only comparable across recordings when both
were measured the same way; see :mod:`.channel_flags` for the dtype that
decides whether the noise numbers mean anything at all.

This is not :func:`.activity_map.plot_whole_chip_activity`, which draws one
presentation figure of a derived field over the dense union. These are review
panels over the channels a single recording actually routed: white background,
percentile-clipped linear colour, no grid inference.

Figures are built straight from :class:`matplotlib.figure.Figure` on an Agg
canvas — no pyplot — so they are safe on a headless node and leave no global
figure state behind when called in a loop over segments.
"""

import logging
from pathlib import Path

import numpy as np

from .channel_layout import _legend_dot, _new_figure, _wrap_label
from .figure_style import LEGEND_FONTSIZE, LEGEND_FRAME_ALPHA, legend_corner, tighten
from .figure_text import acronym_note

logger = logging.getLogger(__name__)

DEFAULT_FIGSIZE = (11.0, 4.8)
DEFAULT_DPI = 180

# Colour limits are clipped to this percentile range: one shorted electrode
# reading 100x the array median would otherwise flatten the whole map.
DEFAULT_PERCENTILES = (2.0, 98.0)

# Micrometres as real mathtext, the way spectra.py renders its PSD units,
# rather than the ASCII "um" this module printed before (2026-09-19). A
# bare "u" reads as the letter u, not the Greek micro sign, on every backend
# that does not happen to substitute it — mathtext is what makes the glyph
# render rather than merely hoping the font covers it.
_UM_LABEL = r"$\mu\mathrm{m}$"

# The legend that names each panel's quantity sits inside the axes (like the
# PSD panels' key), so its title has to fold narrow: a title wider than the
# swatch label beneath it widens the whole box out over the data.
_LEGEND_TITLE_WIDTH = 38
_LEGEND_TITLE_FONTSIZE = 6

# The colour bar gets its OWN axes, inset beside the panel, and is never made
# with `colorbar(ax=...)`. That call takes its space out of the parent axes,
# and it takes a different amount on each side -- measured 25% of the width at
# `location="left"` against 20% at `"right"` -- so the rule that puts a
# left-hand panel's bar on the left and a right-hand panel's on the right was
# leaving the two panels 6.25% different in size, which is precisely what a
# side-by-side comparison must not do. An inset never resizes its parent, so
# every panel keeps exactly the box the subplot grid gave it, and the
# `bbox_inches="tight"` save keeps the bars in frame.
_COLORBAR_WIDTH = 0.035
# Clear of the y tick labels, which `sharey` puts on the leftmost panel only.
_COLORBAR_PAD_LEFT = 0.17
_COLORBAR_PAD_RIGHT = 0.045
# Matches the shrink the old `colorbar(..., shrink=0.85)` applied.
_COLORBAR_Y0 = 0.075
_COLORBAR_HEIGHT = 0.85


def _save_tight(fig, out_path):
    """Write `fig` with a tight bounding box, then drop its artists.

    Not :func:`.channel_layout._save_and_release`: every figure in this module
    carries colorbars under a suptitle, which push the axis labels past the
    default bounding box, so the tight box is load-bearing rather than
    cosmetic.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(out_path, bbox_inches="tight")
    finally:
        # Called once per recording across many tasks; releasing artists keeps
        # peak RSS flat rather than growing with the figure count.
        fig.clear()
    return out_path


def robust_color_limits(values, percentiles=DEFAULT_PERCENTILES):
    """Colour limits clipped to a percentile range, or ``(None, None)``.

    ``(None, None)`` means "let matplotlib autoscale": it is returned both when
    nothing finite is left to scale to and when the clipped range is degenerate
    (every channel identical, or all zeros), because forcing ``vmin == vmax``
    renders one flat colour over the whole array.
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None, None
    low, high = (float(v) for v in np.percentile(finite, percentiles))
    return (low, high) if high > low else (None, None)


def _metric_swatch_color(cmap_name):
    """A representative colour from `cmap_name`, for a legend swatch.

    Each panel paints a continuous colormap, which has no single colour to put
    in a legend key. This picks one (60% up the scale, where most cmaps are
    saturated and legible against white) so the key reads as "this is the kind
    of thing that map paints", not as a claimed value.
    """
    import matplotlib as mpl

    return mpl.colormaps[cmap_name](0.65)


def _draw_metric_panel(
    ax, locations, values, panel_title, bar_label, cmap, percentiles,
    colorbar_side, legend_label, acronyms, annotate,
):
    """Draw one metric onto `ax`: the scatter, its colour bar, and its key.

    The one drawing routine every panel in this module goes through, whether it
    ends up alone in a single-panel figure (:func:`plot_noise_map`,
    :func:`plot_firing_rate_map`) or beside another panel in the composite
    (:func:`plot_noise_activity_map`) — :func:`plot_metric_maps` calls this once
    per panel either way, so there is only one place that draws a metric map.

    Parameters
    ----------
    colorbar_side : {"left", "right"}
        RULE (review ruling, 2026-09-19): the colour bar sits on the SAME
        side of the figure as its panel — a panel in the left half gets its
        bar on the left, one in the right half keeps it on the right — so a
        reader's eye never has to cross the panel to find the scale that
        explains it. The caller derives this from the panel's own column
        index rather than a hardcoded side, so it stays correct at one
        panel, two, or more.
    legend_label, acronyms : str or None, sequence of str
        A short legend key naming the quantity painted (a review ruling:
        "needing legends describing metric used"), with any acronym it uses
        expanded once in the key's title — the way
        :func:`.spectra.plot_spectra_panels` expands PSD.
        None/empty skips the key, for a caller that has nothing to add beyond
        the colour-bar label.
    """
    vmin, vmax = robust_color_limits(values, percentiles)
    scatter = ax.scatter(
        locations[:, 0], locations[:, 1],
        c=values, s=6, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0,
    )
    # An inset, not `colorbar(ax=ax)` -- see _COLORBAR_WIDTH for why the
    # latter made the two panels different sizes.
    on_left = colorbar_side == "left"
    cax = ax.inset_axes(
        (
            -_COLORBAR_PAD_LEFT - _COLORBAR_WIDTH if on_left else 1.0 + _COLORBAR_PAD_RIGHT,
            _COLORBAR_Y0,
            _COLORBAR_WIDTH,
            _COLORBAR_HEIGHT,
        )
    )
    bar = ax.figure.colorbar(scatter, cax=cax)
    if on_left:
        # Ticks and label outboard of the bar, so they read from the figure
        # edge inwards and never collide with the panel they belong to.
        cax.yaxis.set_ticks_position("left")
        cax.yaxis.set_label_position("left")
    bar.set_label(bar_label)
    if annotate:
        ax.set_title(panel_title)
    ax.set_aspect("equal", adjustable="box")

    # RULE (review ruling, 2026-09-19): panels laid out left/right genuinely
    # SHARE the y axis (same electrode y-positions), so plot_metric_maps
    # gives that ONE shared label. The x axis is NOT shared the same way
    # here — a single fig.supxlabel over two side-by-side panels (round 2)
    # read oddly and cost more white space than it saved — so every panel
    # gets its OWN x label.
    # Do not reintroduce a shared supxlabel; see plot_metric_maps.
    ax.set_xlabel(f"x ({_UM_LABEL})")

    if legend_label:
        handle = _legend_dot(_metric_swatch_color(cmap), legend_label)
        legend_kwargs = {}
        if acronyms:
            legend_kwargs["title"] = _wrap_label(
                acronym_note(*acronyms, short=True), width=_LEGEND_TITLE_WIDTH
            )
            legend_kwargs["title_fontsize"] = _LEGEND_TITLE_FONTSIZE
        legend_corner(
            ax, handles=[handle], fontsize=LEGEND_FONTSIZE, framealpha=LEGEND_FRAME_ALPHA,
            **legend_kwargs,
        )


def plot_metric_maps(
    recording,
    channel_ids,
    panels,
    out_path=None,
    title=None,
    percentiles=DEFAULT_PERCENTILES,
    figsize=DEFAULT_FIGSIZE,
    dpi=DEFAULT_DPI,
    annotate=True,
    axes=None,
):
    """Draw one geometry-scatter panel per metric to `out_path`; return the Path.

    Every panel scatters the same electrode positions and differs only in the
    values colouring them, on shared x/y axes with an equal aspect — electrode
    spacing is isotropic, and a stretched aspect makes clumps unreadable.

    Parameters
    ----------
    recording
        The recording whose ``get_channel_locations()`` supplies the geometry.
    channel_ids : sequence
        The ids the metric arrays are aligned to, in recording order. Their
        count must match the location count, which is the one thing that can
        silently paint a metric onto the wrong electrodes.
    panels : sequence of tuple
        One ``(values, panel_title, colorbar_label, cmap, legend_label,
        acronyms)`` per panel, where ``values`` is a per-channel array aligned
        to `channel_ids`. ``legend_label`` (str or None) and ``acronyms``
        (sequence of str) drive the per-panel key — see
        :func:`_draw_metric_panel`.
    out_path : path-like or None
        Where to write the PNG. Required unless `axes` is given, in which case
        the caller owns the figure and nothing is written.
    title : str or None
        Figure suptitle.
    percentiles : tuple
        Passed to :func:`robust_color_limits`, per panel.
    figsize, dpi : tuple, float
        A single panel gets a little over half the width — the same height, so
        a one-panel and a two-panel map read at the same scale. Ignored when
        `axes` is given.
    annotate : bool
        The explanatory chrome, as on every other emitter. True draws the
        suptitle and each panel's title. False draws neither, leaving the axes,
        the geometry, the colour bars and each panel's own legend key — the
        presentation register, where the title's information lives in the file
        name instead.
    axes : matplotlib axes or sequence of them, or None
        Draw into the caller's panels instead of building a figure — the same
        house pattern as :func:`.channel_layout.plot_channel_layout`'s `ax` and
        :func:`.segment_boundary_map.plot_segment_boundary_map`'s `axes`, so a
        future composed sheet can reuse this without a second copy of the
        drawing code. One axes is normalised to a one-item list. With `axes`
        given no figure-level label is added (the caller's figure owns that
        margin), no file is written, and the return is None.

    Raises
    ------
    ValueError
        When the channel locations do not line up with the metric arrays, or
        neither `out_path` nor `axes` was given.
    """
    locations = np.asarray(recording.get_channel_locations(), dtype=float)
    if locations.ndim != 2 or locations.shape[0] != len(channel_ids) or locations.shape[1] < 2:
        raise ValueError("channel locations do not line up with the metric arrays")

    supplied = None
    if axes is not None:
        supplied = [axes] if hasattr(axes, "scatter") else list(axes)
    if supplied is None and out_path is None:
        raise ValueError("pass out_path to write a figure, or axes to draw into one")
    if supplied is not None and len(supplied) < len(panels):
        raise ValueError("axes was given but holds fewer axes than panels")

    fig = None
    if supplied is None:
        width = figsize[0] if len(panels) > 1 else figsize[0] / 2 + 0.6
        fig = _new_figure((width, figsize[1]), dpi)
        target_axes = np.atleast_1d(fig.subplots(1, len(panels), sharex=True, sharey=True))
    else:
        target_axes = supplied[: len(panels)]

    n_panels = len(panels)
    for index, (ax, panel) in enumerate(zip(target_axes, panels)):
        values, panel_title, bar_label, cmap, legend_label, acronyms = panel
        # Column index decides the side, never a hardcoded per-call list — see
        # the rule spelled out on `_draw_metric_panel`.
        side = "left" if index < n_panels // 2 else "right"
        _draw_metric_panel(
            ax, locations, values, panel_title, bar_label, cmap, percentiles,
            side, legend_label, acronyms, annotate,
        )

    if fig is None:
        logger.debug("drew %d metric panel(s) into caller-owned axes", n_panels)
        return None

    # One label per SHARED axis. Every panel scatters the same electrode
    # positions on shared y, so a per-panel copy is the same word repeated
    # across the figure with nothing to distinguish them. x is NOT shared this
    # way — each panel already carries its own x label (see
    # `_draw_metric_panel`) — so only y gets the figure-level label.
    fig.supylabel(f"y ({_UM_LABEL})")

    if annotate and title:
        fig.suptitle(title)
    tighten(fig)

    out_path = _save_tight(fig, out_path)
    logger.info("wrote geometry map: %s", out_path)
    return out_path


def _noise_panel(noise):
    """The noise panel spec, shared by the standalone and composite figures."""
    return (
        np.asarray(noise["noise"], dtype=float),
        "MAD noise",
        # The estimator's name rides in the colour-bar label itself (a review
        # ruling: "needing legends describing metric used for noise") — under
        # annotate=False the panel title above is gone, so this is the only
        # place left that says the number is a MAD, not just "noise".
        f"MAD noise ({noise['unit']})",
        "viridis",
        "MAD noise",
        ("MAD",),
    )


def _activity_panel(activity):
    """The activity panel spec, shared by the standalone and composite figures."""
    threshold = activity["threshold_sd"]
    return (
        np.asarray(activity["rate_hz"], dtype=float),
        f"Activity rate ({threshold:g} sd crossings)",
        # Same reasoning as the noise panel: the threshold that defines a
        # "crossing" has to survive annotate=False, so it is in the bar label,
        # not only in the title.
        f"≥{threshold:g} SD crossing rate (events / s)",
        "magma",
        f"≥{threshold:g} SD crossings",
        ("SD",),
    )


def plot_noise_activity_map(recording, noise, activity, out_path, title=None, **kwargs):
    """Paint noise and activity onto the geometry, one panel each.

    `noise` and `activity` are :func:`mea_modules.quality.mad_noise` and
    :func:`mea_modules.quality.activity_rate` results; the channel ids come
    from `noise`, so both must have been measured on the same recording.
    Remaining keyword arguments go to :func:`plot_metric_maps`.

    Each metric is ALSO available on its own — :func:`plot_noise_map` and
    :func:`plot_firing_rate_map` — because a reviewer scanning many segments
    often wants one metric across several recordings side by side, which this
    combined figure cannot give them.
    """
    panels = [_noise_panel(noise), _activity_panel(activity)]
    return plot_metric_maps(
        recording, list(noise["channel_ids"]), panels, out_path, title, **kwargs
    )


def plot_noise_map(recording, noise, out_path, title=None, **kwargs):
    """MAD noise on the geometry as its own single-panel figure.

    The same panel :func:`plot_noise_activity_map` draws on the left, standing
    alone for a review deck — the individual-panel twin of
    :func:`plot_firing_rate_map`, which already does this for activity
    (review ruling, 2026-09-19: "make sure we have individual panel plots
    for each of these in addition to the multipanel").
    """
    panels = [_noise_panel(noise)]
    return plot_metric_maps(
        recording, list(noise["channel_ids"]), panels, out_path, title, **kwargs
    )


def plot_firing_rate_map(recording, activity, out_path, title=None, **kwargs):
    """Activity rate on the geometry as its own single-panel figure.

    The same panel :func:`plot_noise_activity_map` draws on the right, standing
    alone for a review deck. Crossings are counted at whatever threshold
    `activity` was measured at, so this map and the QC report that shares that
    dict always describe the same events.
    """
    panels = [_activity_panel(activity)]
    return plot_metric_maps(
        recording, list(activity["channel_ids"]), panels, out_path, title, **kwargs
    )
