"""Welch power spectra over the shared electrode set, and the raw-vs-filtered panel.

The figure this module draws is a pair of panels over the SAME channels on the
SAME axes, one before preprocessing and one after, so the difference the eye
reads is the filter's transfer function: the high-pass shoulder, drift power
gone below it, and any 50/60 Hz line that survived. Identical panels mean the
filter did nothing; a line spike in the filtered panel means interference the
chain does not remove.

Split in two, because the spectra are numbers before they are a picture:

* :func:`welch_spectra` — one recording and a channel list in, a per-channel
  periodogram out. Bounded by a time window, and testable against a synthetic
  tone without rendering anything.
* :func:`plot_spectra_panels` — one panel per named source, sharing axes.

The default Welch segment length resolves to roughly 2.4 Hz at 10 kHz and
4.9 Hz at 20 kHz — enough to separate line noise from the spike band and to see
a high-pass corner, without paying for a long read.

Figures are built straight from :class:`matplotlib.figure.Figure` on an Agg
canvas — no pyplot — so they are safe on a headless node and leave no global
figure state behind.
"""

import logging

import numpy as np

from .channel_layout import _legend_line, _new_figure, _renderer, _wrap_label
from .figure_style import legend_corner, tighten
from .figure_text import BACKBONE_KEY, REPRESENTATIVE_KEY, acronym_note
from .metric_maps import _save_tight
# computed in .welch (the compute side imports no drawing code); re-exported here
from .welch import (  # noqa: F401
    DEFAULT_NPERSEG,
    DEFAULT_POWER_FLOOR,
    _DEFAULT_DURATION_S,
    _UV_PSD_UNIT,
    _COUNTS_PSD_UNIT,
    welch_spectra,
)

logger = logging.getLogger(__name__)


_PSD_PANEL_WIDTH = 6.0
_PSD_FIGURE_PAD = 1.0
_PSD_FIGURE_HEIGHT = 4.8
_PSD_DPI = 180

# Clearance kept between the shared y label and the tick labels beside it —
# see `_clear_supylabel_of_ticks`. Matches the order of magnitude of
# `.channel_layout`'s own bottom-margin pads.
_PSD_SUPYLABEL_PAD_IN = 0.08
# A reservation this large means the width measurement is not to be trusted
# (or the label text is absurdly long); cap it rather than let one figure's
# axes collapse to a sliver.
_PSD_MAX_SUPYLABEL_RESERVE_FRAC = 0.4

# Panel identity is drawn inside the axes rather than in a title band above it;
# the reasoning is with the call that draws it, in `plot_spectra_panels`.
_PANEL_LABEL_FONTSIZE = 7.5

# Thin enough that ~30 overlapping curves (one per electrode cluster in the
# shared set — see `plot_spectra_panels`) still read as separate lines rather
# than a filled band. A handful of curves reads fine at this width too, so
# there is no count-dependent branch.
_PSD_LINE_WIDTH = 0.35

# The legend now carries a single key naming the whole channel family, not one
# key per channel (see `plot_spectra_panels`), so the box is only as wide as
# that one short line. The title has to fold to about the same width or it
# widens the whole box out over the data — the same failure a wider body used
# to guard against, just at a narrower body now.
_LEGEND_TITLE_WIDTH = 26
_LEGEND_TITLE_FONTSIZE = 6


# The same two units as they are DRAWN. The note that stood here held that ASCII
# was deliberate on the axis as well, on the grounds that a caret exponent reads
# fine beside the rest of a label. On the rendered figure it does not: psd.png
# printed a literal "uV^2/Hz", caret and all, where a reader expects a
# superscript (2026-09-19). The axis therefore carries mathtext and the
# data field above keeps the ASCII, which is the split the old note was missing.
_PSD_AXIS_UNITS = {
    _UV_PSD_UNIT: r"$\mu\mathrm{V}^{2}/\mathrm{Hz}$",
    _COUNTS_PSD_UNIT: r"$\mathrm{ADC\ counts}^{2}/\mathrm{Hz}$",
}


def _psd_y_label(sources):
    """``(y label, acronyms to expand)`` for the axis the panels share.

    The panels are drawn with ``sharey``, so there is one y axis between them
    and it can carry exactly one label. The unit is read off the results rather
    than taken from the caller, so the label cannot claim microvolts for a
    recording that degraded to device counts.

    Parameters
    ----------
    sources : sequence of tuple
        ``(source name, welch_spectra result)`` in panel order.
    """
    units = list(dict.fromkeys(str(result.get("unit") or "") for _, result in sources))
    if len(units) > 1:
        # `sharey` has already put these on one axis, so the mismatch is on the
        # figure whether or not the label admits it. Naming both is the honest
        # render; naming the first would label device counts as microvolts.
        logger.warning("spectra panels report different units (%s)", ", ".join(units))

    rendered = [_PSD_AXIS_UNITS.get(unit, unit) for unit in units if unit]
    # Only the acronyms this figure actually prints, per the rule in
    # :mod:`.figure_text`: ADC appears only when a panel degraded to counts.
    acronyms = ["PSD"] + (["ADC"] if _COUNTS_PSD_UNIT in units else [])
    if not rendered:
        return "PSD", acronyms
    return f"PSD ({' / '.join(rendered)})", acronyms


def _clear_supylabel_of_ticks(fig, axis, label, pad_in=_PSD_SUPYLABEL_PAD_IN):
    """Tighten `fig`, keeping `label` (a `fig.supylabel`) clear of `axis`'s ticks.

    `fig.supylabel` fixes its own x at a hardcoded fraction of the figure, and
    `tight_layout` never moves it — tight_layout only measures each Axes' own
    tightbbox, so a figure-level label is invisible to it. Left unreserved,
    the axes tighten right up to the figure edge and a wide log-scale tick
    (e.g. ``3 x 10^-3``) prints straight through the label's own text
    (2026-09-19).

    Reserves the label's actual rendered width as a left margin before
    tightening, then slides the label by whatever gap is still missing once
    the axes have settled into that margin — a pure translation of the
    anchor, so it holds regardless of the label's own alignment or rotation.
    Falls back to a plain, unreserved `tighten(fig)` when nothing can be
    measured (no renderer), which is the behaviour this replaces.
    """
    fig_width_in = float(fig.get_figwidth()) or 1.0
    renderer = _renderer(fig)
    reserved_frac = 0.0
    if renderer is not None:
        try:
            label_width_in = label.get_window_extent(renderer).width / fig.dpi
        except Exception:  # noqa: BLE001 - label not yet drawable; no reservation
            label_width_in = 0.0
        if label_width_in > 0.0:
            reserved_frac = min(
                _PSD_MAX_SUPYLABEL_RESERVE_FRAC, (label_width_in + pad_in) / fig_width_in
            )

    tighten(fig, rect=(reserved_frac, 0.0, 1.0, 1.0) if reserved_frac else None)

    # Re-measure: `tighten` just moved the axes, so the old extents are stale.
    renderer = _renderer(fig)
    if renderer is None:
        return
    try:
        gap_in = (
            axis.get_tightbbox(renderer).x0 - label.get_window_extent(renderer).x1
        ) / fig.dpi
    except Exception:  # noqa: BLE001 - leaves the label wherever tighten put it
        return
    if gap_in < pad_in:
        label.set_x(label.get_position()[0] + (pad_in - gap_in) / fig_width_in)


def plot_spectra_panels(
    spectra, out_path, title=None, figsize=None, dpi=_PSD_DPI, annotate=True
):
    """Draw one PSD panel per named source to `out_path`; return the Path.

    Parameters
    ----------
    spectra : mapping
        Source name -> :func:`welch_spectra` result, in panel order (a plain
        dict preserves it). Typically ``{"raw": ..., "preprocessed": ...}``;
        one entry is a valid figure, which is what a recording with no
        preprocessing chain to compare against gets.
    out_path : path-like
        Where to write the PNG.
    title : str or None
        Figure suptitle.
    figsize : tuple or None
        Defaults to one panel's width per source plus a margin, so a two-panel
        figure is twice as wide rather than half as legible.
    dpi : float
        Figure DPI.
    annotate : bool
        The explanatory chrome. True draws the suptitle. False draws neither it
        nor any caption, leaving the axes, the units, the legend — including the
        PSD expansion, which lives in the legend title precisely so it survives
        here — and each panel's own identity label.

    Raises
    ------
    ValueError
        When `spectra` is empty — there is no such thing as a zero-panel figure.
    """
    sources = list(spectra.items())
    if not sources:
        raise ValueError("no spectra to plot")

    if figsize is None:
        figsize = (_PSD_PANEL_WIDTH * len(sources) + _PSD_FIGURE_PAD, _PSD_FIGURE_HEIGHT)
    fig = _new_figure(figsize, dpi)
    # Shared axes are the point: the panels are only comparable if the eye can
    # read the difference between them as the filter rather than as a rescale.
    axes = np.atleast_1d(fig.subplots(1, len(sources), sharex=True, sharey=True))

    # Colour carries per-channel IDENTITY, not a measured value, and it is
    # built once from the first source rather than per panel: every panel in
    # this figure draws the SAME channel_ids in the SAME order (see the module
    # docstring), so index i gets the same colour in both, and a reader can
    # follow one electrode's curve from the raw panel to the filtered one.
    # `resampled` gives exactly one colour per channel — the same pattern
    # `spike_sensitivity` uses for its own per-series family — and a
    # perceptually even map keeps ~30 overlapping curves separable where a
    # handful of arbitrary hues would start repeating or clashing.
    from matplotlib import colormaps

    n_channels = len(sources[0][1]["channel_ids"])
    palette = colormaps["viridis"].resampled(max(n_channels, 2))

    for ax, (source_name, result) in zip(axes, sources):
        channel_ids = list(result["channel_ids"])
        power = np.asarray(result["power"])
        for index, cid in enumerate(channel_ids):
            ax.semilogy(
                result["freqs"], power[:, index], lw=_PSD_LINE_WIDTH, color=palette(index)
            )
        # Panel identity inside the axes rather than in a title band over them.
        # The panels already share both axes, so a strip of chrome above each
        # one buys a reader nothing the same four words in the corner do, and it
        # costs a row of figure height per panel. Boxed and on top, because a
        # raw spectrum's low-frequency shoulder runs underneath it.
        ax.text(
            0.02,
            0.98,
            f"{source_name} ({len(channel_ids)} {REPRESENTATIVE_KEY})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=_PANEL_LABEL_FONTSIZE,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.5},
            zorder=5,
        )
        ax.grid(True, which="both", alpha=0.25, lw=0.4)
        # Its own x label per panel rather than a figure-wide one: these panels
        # sit side by side, sharing an x RANGE (via `sharex`), and two
        # "frequency (Hz)"s under them read fine — a single `fig.supxlabel`
        # bought nothing here but an extra reserved band of white space
        # (2026-09-19). The y axis below stays shared: that one genuinely saves
        # a repeated label, because the panels are stacked on the SAME y range.
        ax.set_xlabel("frequency (Hz)")

    # One label for the SHARED y axis. Both panels are drawn on the same y, so
    # a per-panel label would be the same word printed twice — which is what
    # psd.png did with "PSD (uV^2/Hz)" (2026-09-19).
    y_label, acronyms = _psd_y_label(sources)
    y_label_artist = fig.supylabel(y_label)

    # PSD is expanded in the legend, not in a caption band: that is where the
    # review asked for it, and it means the definition survives
    # `annotate=False`, which a caption would not. The wording comes from
    # :mod:`.figure_text` so the figure, its README and the report generator
    # cannot drift apart.
    #
    # The legend carries exactly ONE key for the whole coloured family rather
    # than one per channel: at ~30 channels a "ch <id>" entry per curve would
    # be an unreadable column of text, and the count is what a reader actually
    # needs — which channel is which is carried by colour instead (the palette
    # above), not by a legend key. No colour bar goes with it: the index a
    # channel gets is its position in the shared electrode set, not a
    # physical quantity, so a bar next to it would claim a scale that is not
    # there.
    family_handle = _legend_line(
        palette(n_channels // 2), f"{BACKBONE_KEY} (n={n_channels})", lw=1.2
    )
    legend_corner(
        axes[0],
        handles=[family_handle],
        title=_wrap_label(acronym_note(*acronyms, short=True), width=_LEGEND_TITLE_WIDTH),
        title_fontsize=_LEGEND_TITLE_FONTSIZE,
    )

    if annotate and title:
        fig.suptitle(title)
    _clear_supylabel_of_ticks(fig, axes[0], y_label_artist)

    out_path = _save_tight(fig, out_path)
    logger.info("wrote PSD: %s", out_path)
    return out_path
