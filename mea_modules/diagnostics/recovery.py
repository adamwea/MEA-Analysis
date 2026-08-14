"""Look at what the honestly-stitched template actually measured.

Two questions a reviewer has about a stitched sorting, and one plot each:

* **Where is the array sampled well?** :func:`plot_coverage_map` colours every
  electrode by how many segments routed it. On an AxonTracking scan this is
  bimodal by design — a small anchor set routed in every configuration, and the
  rest tiled one configuration each — and that shape is worth seeing, because it
  is also the map of how reliable each electrode's template is.

* **How much data backs this unit's footprint, electrode by electrode?**
  :func:`plot_footprint_gain` draws one unit's coverage-weighted footprint —
  the measured amplitude at every electrode, alongside how many spikes or
  segments were averaged into each value — so a reader can tell a
  well-supported measurement from one resting on a single spike.

Pure plotting: every function takes arrays the caller already holds, writes a
PNG, and returns its path. No file reading, no argparse.
"""

import logging

from .channel_layout import (
    _add_caption,
    _fold_caption,
    _legend_line,
    _legend_patch,
    _new_figure,
    _save_and_release,
    _wrap_label,
)
from .figure_text import (
    BACKBONE_CHANNELS,
    DENSE_STITCH,
    PROXY_NOT_MODEL,
)

logger = logging.getLogger(__name__)

DEFAULT_FIGSIZE = (14.0, 9.0)
DEFAULT_DPI = 170

# Matched to channel_layout's, so every figure a capsule emits reads as one
# family rather than five different house styles.
_LEGEND_FONTSIZE = 7
_LEGEND_FRAMEALPHA = 0.85
_AXIS_NOTE_FONTSIZE = 8
_COLORBAR_LABEL_FONTSIZE = 8

# --------------------------------------------------------------------------- #
# Reader-facing wording
# --------------------------------------------------------------------------- #
#
# Named constants rather than inline literals, for the reason `figure_text`
# exists at all (Adam, 2026-08-11): the same quantity is drawn on this
# module's footprint figure and again in `stitch_wiring`, and a reader who
# meets it under two different names has to work out for themselves whether
# it is the same number. Every one of these carries a unit or says explicitly
# that it has none.

SEGMENTS_ROUTED_COLORBAR_LABEL = (
    "segments that routed this electrode\n(count of segments, not a segment number)"
)

# Colour-bar labels are drawn rotated, so their LENGTH is vertical and a long
# one runs past the end of its bar into the next row of panels. Kept short
# enough to fit the shortest bar any of these figures draws.
AMPLITUDE_COLORBAR_LABEL = "peak absolute amplitude (µV, logarithmic scale)"

COVERAGE_COLORBAR_LABEL = (
    "data behind this electrode's average\n"
    "(count of spikes, or of segments under uniform weighting)"
)

#: Plain-English name for the always-routed electrode set. The word "backbone"
#: is ours, not the field's, so the legend says what the set IS and the caption
#: carries `figure_text.BACKBONE_CHANNELS` for the full gloss.
BACKBONE_LEGEND_LABEL = "electrodes routed in every segment"

#: LogNorm masks anything <= 0, so a zero-amplitude electrode lands on the
#: colormap's "bad" colour exactly like a grid position with no electrode. Both
#: read as white, and a reader cannot tell them apart — so the key says both.
NO_ELECTRODE_LEGEND_LABEL = (
    "white: no electrode at this grid position, or an electrode with no measured "
    "signal (zero is off a logarithmic scale)"
)

#: The amplitude colour scales start above the true minimum. Saying so is not
#: optional: a colour bar whose low end is clipped without a word about it tells
#: the reader the faint electrodes were measured at the floor value.
AMPLITUDE_FLOOR_NOTE = (
    "Colour floor: the amplitude colour scale starts at the 1st percentile of the "
    "non-zero amplitudes (or one ten-thousandth of the peak, whichever is larger), "
    "so the faintest electrodes are drawn AT the floor colour rather than dropped; "
    "the top of the scale is the true maximum, unclipped."
)

MARKER_IS_ELECTRODE_NOTE = "One square marker is one electrode on the array."


def _caption_width(figsize):
    """Characters per caption line for a figure this wide.

    ``_fold_caption``'s default is tuned for the narrow layout plot. These
    figures are 14-16 inches across, and folding at that default both wastes
    half the width and doubles the caption's line COUNT — which is what
    :func:`_add_caption` reserves vertical space by, so a needlessly tall
    caption eats the panels.
    """
    return max(60, int(float(figsize[0]) * 13))


def plot_coverage_map(positions, n_segments_routed, out_path, title=None,
                      routing=None, segment_labels=None,
                      figsize=DEFAULT_FIGSIZE, dpi=DEFAULT_DPI,
                      highlight_label=None, caption=None):
    """Electrodes coloured by how many segments routed them; returns `out_path`.

    Three panels, because two different "coverage" questions get confused
    otherwise:

    * where on the array each electrode sits, coloured by how many segments
      reached it;
    * how many ELECTRODES fall into each coverage count — this axis is a count
      of segments, NOT a segment index, and its empty bins mean "no electrode is
      routed by exactly this many segments", not "this segment routed nothing";
    * how many electrodes EACH SEGMENT routes, which is the per-segment view and
      the one that would actually reveal a segment contributing nothing.

    `routing` is the ``(n_segments, n_electrodes)`` boolean table; without it the
    third panel is omitted.

    The red rule on the third panel marks the electrode set that every segment
    routed — the set capsule 12 compares across segments in a sibling figure —
    so `highlight_label` names that sibling by its real emitted filename, the
    way :func:`~mea_modules.diagnostics.channel_layout.plot_channel_layout`
    does::

        highlight_label="electrodes routed in every segment — compared across "
                        "segments in backbone_agreement.png"

    `caption` appends a line under the axes (e.g. where the routing table came
    from, when it is a surrogate rather than a measurement).
    """
    import numpy as np

    positions = np.asarray(positions, dtype=float)[:, :2]
    counts = np.asarray(n_segments_routed, dtype=int)

    fig = _new_figure(figsize, dpi)
    if routing is None:
        left, right = fig.subplots(1, 2, width_ratios=[2.0, 1.0])
        bottom = None
    else:
        left, right, bottom = fig.subplots(1, 3, width_ratios=[2.0, 1.0, 1.2])

    scatter = left.scatter(
        positions[:, 0], positions[:, 1], c=counts, s=4, marker="s",
        cmap="viridis", linewidths=0, rasterized=True,
    )
    left.set_aspect("equal")
    left.set_xlabel("x (µm)")
    left.set_ylabel("y (µm)")
    left.set_title("electrodes, coloured by segments routed")
    cbar = fig.colorbar(scatter, ax=left, shrink=0.8)
    cbar.set_label(SEGMENTS_ROUTED_COLORBAR_LABEL, fontsize=_COLORBAR_LABEL_FONTSIZE)

    # Log y: the always-routed set is a few hundred electrodes against ~13k
    # routed once, and on a linear axis its bar simply disappears.
    bins = np.arange(0.5, counts.max() + 1.5)
    right.hist(counts, bins=bins, color="0.3")
    right.set_yscale("log")
    # Spelled out because the obvious short label ("segments routing an
    # electrode") reads as a segment index, and then the empty bins look like
    # segments that contributed nothing.
    right.set_xlabel("how many segments route an electrode\n(count of segments, NOT a segment number)")
    right.set_ylabel("electrodes (count, logarithmic axis)")
    right.set_title("electrodes per coverage count")
    right.legend(
        handles=[_legend_patch("0.3", "electrodes with this coverage count")],
        loc="best", fontsize=_LEGEND_FONTSIZE, framealpha=_LEGEND_FRAMEALPHA,
    )

    once = int((counts == 1).sum())
    every = int((counts == counts.max()).sum())
    right.text(
        0.97, 0.72,
        f"routed once: {once}\nrouted by all {counts.max()}: {every}\n"
        f"empty bins = no electrode\nhas that coverage",
        transform=right.transAxes, ha="right", va="top", fontsize=8,
    )

    if bottom is not None:
        routing = np.asarray(routing, dtype=bool)
        per_segment = routing.sum(axis=1)
        labels = list(segment_labels) if segment_labels is not None \
            else [str(i) for i in range(per_segment.size)]
        y = np.arange(per_segment.size)
        bottom.barh(y, per_segment, color="0.35")
        bottom.set_yticks(y)
        bottom.set_yticklabels(labels, fontsize=6)
        bottom.invert_yaxis()
        bottom.set_xlabel("electrodes this segment routed (count)")
        bottom.set_ylabel("segment")
        bottom.set_title("electrodes routed, per segment")
        # The floor every segment shares: the set that makes concatenation and
        # sorting possible in the first place.
        anchor = int((counts == counts.max()).sum())
        bottom.axvline(anchor, color="red", ls="--", lw=1.0)
        bottom.text(anchor, per_segment.size * 0.5, f" every segment: {anchor}",
                    color="red", fontsize=8, rotation=90, va="center")
        bottom.text(0.97, 0.02,
                    f"min {per_segment.min()}  max {per_segment.max()}",
                    transform=bottom.transAxes, ha="right", va="bottom", fontsize=8)
        bottom.legend(
            handles=[
                _legend_patch("0.35", "electrodes this segment routed (count)"),
                _legend_line(
                    "red",
                    f"{highlight_label or BACKBONE_LEGEND_LABEL} (n={anchor})",
                    lw=1.0, linestyle="--",
                ),
            ],
            loc="best", fontsize=_LEGEND_FONTSIZE, framealpha=_LEGEND_FRAMEALPHA,
        )

    if title:
        fig.suptitle(title)

    caption_parts = [
        MARKER_IS_ELECTRODE_NOTE,
        "A segment is one recording configuration; each one routes its own subset "
        "of the electrodes, and an electrode's coverage is simply how many of them "
        "reached it (a count, no unit).",
        "The middle panel's x axis is a COUNT of segments, so an empty bin means no "
        "electrode has that coverage — it never means a segment recorded nothing.",
    ]
    if bottom is not None:
        caption_parts.append(BACKBONE_CHANNELS)
    if caption:
        caption_parts.append(caption)
    _add_caption(fig, _fold_caption(caption_parts, width=_caption_width(figsize)))

    logger.info(
        "coverage map: %d electrodes, %d routed once, %d routed by all %d segments",
        counts.size, once, every, int(counts.max()),
    )
    return _save_and_release(fig, out_path)


def plot_footprint_gain(positions, template, out_path, backbone_mask=None,
                        unit_id=None, coverage=None, title=None,
                        figsize=(15.0, 7.0), dpi=DEFAULT_DPI,
                        highlight_label=None, weight_label=None, caption=None):
    """One unit's coverage-weighted footprint: what was measured, and how much data backs it.

    **Descriptive only — nothing is thresholded, masked, or hidden.** This runs
    before any post-sort QC, so every electrode is drawn at whatever amplitude
    it carries and the reader decides what to believe. There is deliberately no
    "electrodes above X% of peak" count: choosing that percentage would be a
    claim about where the footprint ends, which is exactly the judgement this
    capsule refuses to make.

    Two maps of the same array, which is what makes the plot readable without a
    threshold:

    * **amplitude** — peak |value| per electrode, all of them, from the honest
      per-channel stitch: each electrode's value is a weighted average over
      only the segments that actually measured it, nothing extrapolated onto
      an electrode the unit was never recorded on;
    * **coverage** — how many spikes (or segments, under uniform weighting)
      stand behind each electrode's average.

    Read together they answer the only question that matters here: a distant
    electrode showing signal is interesting if its coverage is high and is
    probably one noisy snippet if its coverage is 1. Neither map decides that
    for the reader; putting them side by side lets the reader decide.

    `backbone_mask`, when supplied, outlines the electrodes routed in every
    segment, so a caller can compare the footprint against that anchor set.
    That outlined set is the one a sibling figure compares across segments, so
    `highlight_label` names the sibling by its real emitted filename::

        highlight_label="electrodes routed in every segment — compared across "
                        "segments in backbone_agreement.png"

    `weight_label` overrides the coverage colour bar's default wording
    (:data:`COVERAGE_COLORBAR_LABEL`); `caption` appends a line under the axes.
    """
    import numpy as np

    positions = np.asarray(positions, dtype=float)[:, :2]
    peak = np.abs(np.asarray(template, dtype=float)).max(axis=0)

    fig = _new_figure(figsize, dpi)
    axes = fig.subplots(1, 2 if coverage is not None else 1)
    axes = np.atleast_1d(axes)

    # --- amplitude, every electrode, no threshold -------------------------
    # Log colour, because the dynamic range is the whole problem: a soma at
    # ~700 uV beside axonal signal at ~5-50 uV puts the entire arbor in the
    # bottom 7% of a linear scale, i.e. black. This shows MORE than a linear
    # scale, it does not hide anything — every electrode is still drawn.
    from matplotlib.colors import LogNorm

    ax = axes[0]
    positive = peak[peak > 0]
    amp_norm = None
    if positive.size:
        floor = max(float(np.percentile(positive, 1)), float(peak.max()) * 1e-4)
        amp_norm = LogNorm(vmin=floor, vmax=max(float(peak.max()), floor * 10))
    scatter = ax.scatter(
        positions[:, 0], positions[:, 1], c=np.maximum(peak, 1e-12),
        s=6, cmap="magma", marker="s", linewidths=0, norm=amp_norm, rasterized=True,
    )
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label(AMPLITUDE_COLORBAR_LABEL, fontsize=_COLORBAR_LABEL_FONTSIZE)
    ax.set_title("measured amplitude (every electrode drawn)")
    _finish_axis(ax, backbone_mask, positions, backbone_label=highlight_label)
    # Below the axes, not inside them: the legend sits top-right and the array
    # fills the frame, so an in-axes box lands on top of one or the other.
    ax.set_xlabel(
        f"x (µm)\npeak {peak.max():.1f} µV   |   non-zero on "
        f"{int((peak > 0).sum())} of {peak.size} electrodes",
        fontsize=_AXIS_NOTE_FONTSIZE,
    )

    # --- coverage, every electrode ----------------------------------------
    if coverage is not None:
        coverage = np.asarray(coverage)
        ax = axes[1]
        # Log colour: coverage spans 1 to thousands, and on a linear scale the
        # single-spike electrodes are indistinguishable from the well-sampled.
        shown = np.where(coverage > 0, coverage, np.nan).astype(float)
        from matplotlib.colors import LogNorm
        finite = shown[np.isfinite(shown)]
        norm = LogNorm(vmin=max(finite.min(), 1), vmax=max(finite.max(), 2)) \
            if finite.size else None
        sc2 = ax.scatter(positions[:, 0], positions[:, 1], c=shown, s=6,
                         cmap="viridis", marker="s", linewidths=0,
                         norm=norm, rasterized=True)
        cbar2 = fig.colorbar(sc2, ax=ax, shrink=0.8)
        cbar2.set_label(weight_label or COVERAGE_COLORBAR_LABEL,
                        fontsize=_COLORBAR_LABEL_FONTSIZE)
        ax.set_title("how much data backs each electrode's average")
        _finish_axis(ax, backbone_mask, positions, backbone_label=highlight_label)
        live = coverage[coverage > 0]
        ax.set_xlabel(
            f"x (µm)\nmedian {int(np.median(live)) if live.size else 0}   |   "
            f"min {int(live.min()) if live.size else 0}   |   "
            f"n=1 on {int((coverage == 1).sum())} electrodes",
            fontsize=_AXIS_NOTE_FONTSIZE,
        )

    if title:
        fig.suptitle(title)

    caption_parts = [
        DENSE_STITCH,
        MARKER_IS_ELECTRODE_NOTE
        + " Every electrode is drawn at whatever amplitude it carries: nothing is "
          "thresholded, masked or hidden, and no line is drawn round 'the footprint'.",
        AMPLITUDE_FLOOR_NOTE,
    ]
    if coverage is not None:
        caption_parts.append(
            "Read the two maps together: a distant electrode showing signal is "
            "interesting when a lot of data backs it and is probably one noisy "
            "snippet when only one spike does."
        )
    if backbone_mask is not None:
        caption_parts.append(BACKBONE_CHANNELS)
    caption_parts.append(PROXY_NOT_MODEL)
    if caption:
        caption_parts.append(caption)
    _add_caption(fig, _fold_caption(caption_parts, width=_caption_width(figsize)))

    logger.info(
        "footprint unit %s: peak %.1f uV, non-zero on %d of %d electrodes%s",
        unit_id, float(peak.max()), int((peak > 0).sum()), peak.size,
        "" if coverage is None else
        f", coverage median {int(np.median(coverage[coverage > 0])) if (coverage > 0).any() else 0}",
    )
    return _save_and_release(fig, out_path)


def _finish_axis(ax, backbone_mask, positions, backbone_label=None):
    """Shared axis dressing for the footprint panels.

    The cyan outline is its own encoding, so it gets its own legend key. The
    scatter artist itself is the handle — an open cyan square is what the reader
    sees on the array, and a filled dot from :func:`_legend_dot` would be a
    different mark. The label is wrapped, never truncated, so a caller naming a
    sibling artifact keeps its filename intact.
    """
    import numpy as np

    if backbone_mask is not None:
        backbone_mask = np.asarray(backbone_mask, dtype=bool)
        label = f"{backbone_label or BACKBONE_LEGEND_LABEL} (n={int(backbone_mask.sum())})"
        artist = ax.scatter(
            positions[backbone_mask, 0], positions[backbone_mask, 1],
            s=18, facecolors="none", edgecolors="cyan", linewidths=0.35,
            label=_wrap_label(label), rasterized=True,
        )
        ax.legend(handles=[artist], loc="upper right", fontsize=_LEGEND_FONTSIZE,
                  framealpha=_LEGEND_FRAMEALPHA, labelspacing=0.7)
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")


__all__ = [
    "plot_coverage_map",
    "plot_footprint_gain",
    # Reader-facing wording, exported so a caller (or a test) can assert on the
    # exact string a figure prints instead of re-typing it.
    "AMPLITUDE_COLORBAR_LABEL",
    "AMPLITUDE_FLOOR_NOTE",
    "BACKBONE_LEGEND_LABEL",
    "COVERAGE_COLORBAR_LABEL",
    "MARKER_IS_ELECTRODE_NOTE",
    "NO_ELECTRODE_LEGEND_LABEL",
    "SEGMENTS_ROUTED_COLORBAR_LABEL",
]
