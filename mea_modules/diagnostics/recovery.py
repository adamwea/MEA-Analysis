"""Look at what the recovery actually recovered.

Three questions a reviewer has about a synthesized sorting, and one plot each:

* **Where is the array sampled well?** :func:`plot_coverage_map` colours every
  electrode by how many segments routed it. On an AxonTracking scan this is
  bimodal by design — a small anchor set routed in every configuration, and the
  rest tiled one configuration each — and that shape is worth seeing, because it
  is also the map of how reliable each electrode's template is.

* **Did the correction do anything?** :func:`plot_rescale_effect` shows the
  per-electrode factor between the raw union-recording average and the
  corrected one. Anything far from 1.0 is an electrode that only some segments
  saw, and the histogram is the honest summary of how wrong the uncorrected
  template would have been.

* **Does a unit's footprint reach past the backbone?** :func:`plot_footprint_gain`
  draws one unit twice — on the electrodes the sort could see, and on the union —
  which is the whole point of the exercise made visible.

Pure plotting: every function takes arrays the caller already holds, writes a
PNG, and returns its path. No file reading, no argparse.
"""

import logging

from .channel_layout import (
    _add_caption,
    _fold_caption,
    _legend_dot,
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
    acronym_note,
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
# exists at all (Adam, 2026-08-11): the same quantity is drawn on three figures
# here and two more in `stitch_wiring`, and a reader who meets it under three
# different names has to work out for themselves whether it is the same number.
# Every one of these carries a unit or says explicitly that it has none.

#: What `coverage` / `contributing_weight` actually counts — and it is NOT
#: always spikes. Capsule 10's ``averaging_method="uniform"`` stores the number
#: of contributing SEGMENTS in the very same array, so a figure that says
#: "spikes" unconditionally is wrong on every uniform stitch. A caller that
#: knows its weighting passes a narrower `weight_label`.
COVERAGE_WEIGHT_LABEL = (
    "data behind one unit-electrode average\n"
    "(count: spikes averaged in, or contributing segments under uniform weighting)"
)

#: Same quantity, one line, for panels whose x axis is already narrow.
COVERAGE_WEIGHT_SHORT_LABEL = (
    "data behind the average (count of spikes,\nor of segments under uniform weighting)"
)

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
    routed — the set capsule 11 compares across segments in a sibling figure —
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


def plot_rescale_effect(coverage, totals, out_path, templates=None, nbefore=None,
                        title=None, figsize=(16.0, 5.4), dpi=DEFAULT_DPI,
                        weight_label=None, caption=None):
    """How much data stands behind each averaged unit-electrode value.

    The rescale factor on its own answers nothing: a factor of 21 backed by 100
    spikes is a good estimate, and a factor of 2000 backed by 1 spike is noise
    at full amplitude. Both are "large corrections". What decides trust is how
    many spikes stand behind each unit-electrode average, so that is what these
    panels are keyed on.

    1. **How much data backs each estimate.** The coverage distribution, with
       the counts a reader would threshold on.
    2. **What that buys, measured.** The pre-spike baseline of a template is
       signal-free, so its RMS IS the residual noise left after averaging. Plotted
       against coverage it shows the precision actually achieved — no assumption
       about the noise level, and it should fall as 1/sqrt(n).
    3. **What a threshold costs.** How much of the array survives at each
       minimum-coverage cut, which is the decision being made.

    `templates` is ``(n_units, n_samples, n_channels)``; without it panel 2 is
    skipped. `nbefore` is how many leading samples precede the spike (default:
    a third of the window).

    `weight_label` overrides the x-axis wording for the coverage quantity. The
    default (:data:`COVERAGE_WEIGHT_LABEL`) is deliberately hedged, because the
    same array holds a spike count under ``spike_count`` weighting and a segment
    count under ``uniform``; a caller that knows which one it has should say so
    outright. `caption` appends a line under the axes.
    """
    import numpy as np

    coverage = np.asarray(coverage, dtype=float)
    totals = np.asarray(totals, dtype=float)
    live = coverage > 0
    counts = coverage[live]

    weight_axis = weight_label or COVERAGE_WEIGHT_LABEL
    weight_axis_short = weight_label or COVERAGE_WEIGHT_SHORT_LABEL

    fig = _new_figure(figsize, dpi)
    axes = fig.subplots(1, 3)

    # --- 1. how many spikes back each estimate ---------------------------
    ax = axes[0]
    bins = np.logspace(0, np.log10(max(counts.max(), 10.0)), 40)
    ax.hist(counts, bins=bins, color="0.35")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(weight_axis, fontsize=_AXIS_NOTE_FONTSIZE)
    ax.set_ylabel("unit-electrode pairs (count, logarithmic axis)")
    ax.set_title("how much data backs each averaged value")
    # The summary numbers ride IN the legend label rather than in a floating text
    # box, the way plot_channel_layout carries its counts: a separate box has to
    # be placed somewhere, and wherever that is, the legend wants it too.
    handles = [_legend_patch(
        "0.35",
        f"unit-electrode pairs with this much data behind them "
        f"(median {int(np.median(counts))}, minimum {int(counts.min())}; "
        f"{int((counts == 1).sum()):,} rest on a single one)",
    )]
    for n, c in ((1, "red"), (10, "darkorange"), (100, "green")):
        if counts.max() >= n:
            ax.axvline(n, color=c, ls="--", lw=1.0)
            ax.text(n, ax.get_ylim()[1] * 0.5, f" n={n}", color=c, fontsize=8, rotation=90)
            handles.append(_legend_line(
                c, f"reference mark at n = {n}", lw=1.0, linestyle="--",
            ))
    ax.legend(handles=handles, loc="best", fontsize=_LEGEND_FONTSIZE,
              framealpha=_LEGEND_FRAMEALPHA)

    # --- 2. measured precision vs coverage -------------------------------
    ax = axes[1]
    if templates is not None:
        templates = np.asarray(templates)
        pre = int(nbefore if nbefore is not None else max(2, templates.shape[1] // 3))
        # The window before the spike carries no signal, so whatever is left in
        # it is the noise that survived averaging.
        baseline = np.sqrt((templates[:, :pre, :] ** 2).mean(axis=1))   # (units, chans)
        noise = baseline[live]
        edges = np.unique(np.round(np.logspace(0, np.log10(max(counts.max(), 10.0)), 18)))
        mids, meds = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (counts >= lo) & (counts < hi)
            if m.sum() > 20:
                mids.append(np.sqrt(lo * hi)); meds.append(float(np.median(noise[m])))
        if mids:
            mids, meds = np.array(mids), np.array(meds)
            ax.loglog(mids, meds, "o-", color="0.2")
            # 1/sqrt(n) anchored on the first bin — the shape averaging should give.
            ax.loglog(mids, meds[0] * np.sqrt(mids[0] / mids), "--", color="red")
            ax.legend(
                handles=[
                    _legend_line(
                        "0.2",
                        "noise left in the pre-spike baseline, median of the pairs "
                        "in each bin (µV RMS)",
                        lw=1.4,
                    ),
                    _legend_line(
                        "red",
                        r"1/$\sqrt{n}$ reference curve, anchored on the first bin "
                        "— drawn, not fitted",
                        lw=1.4, linestyle="--",
                    ),
                ],
                loc="best", fontsize=_LEGEND_FONTSIZE, framealpha=_LEGEND_FRAMEALPHA,
            )
        ax.set_xlabel(weight_axis_short, fontsize=_AXIS_NOTE_FONTSIZE)
        ax.set_ylabel("noise left after averaging (µV RMS)")
        ax.set_title("precision actually achieved")
    else:
        ax.set_axis_off()
        ax.text(0.5, 0.5,
                "The measured-precision panel needs the\naveraged waveforms; "
                "they were not supplied\nfor this run.",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)

    # --- 3. what a coverage threshold costs -------------------------------
    ax = axes[2]
    thresholds = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500])
    thresholds = thresholds[thresholds <= counts.max()]
    kept = [100.0 * (counts >= t).sum() / coverage.size for t in thresholds]
    ax.plot(thresholds, kept, "o-", color="0.2")
    ax.set_xscale("log")
    ax.set_xlabel("minimum amount of data required to keep a value\n"
                  "(count, the same quantity as the left panel's x axis)",
                  fontsize=_AXIS_NOTE_FONTSIZE)
    ax.set_ylabel("unit-electrode pairs kept (% of all pairs)")
    ax.set_title("what a minimum-data cut would cost")
    ax.grid(alpha=0.3)
    ax.legend(
        handles=[_legend_line(
            "0.2", "share of unit-electrode pairs surviving this minimum (%)", lw=1.4,
        )],
        loc="best", fontsize=_LEGEND_FONTSIZE, framealpha=_LEGEND_FRAMEALPHA,
    )
    for t, k in zip(thresholds, kept):
        if t in (1, 10, 100):
            ax.annotate(f"{k:.0f}%", (t, k), textcoords="offset points",
                        xytext=(4, 6), fontsize=8)

    # The correction's size belongs here as context, not as its own panel: it
    # is what the rescale bought, but it is not what decides trust.
    with np.errstate(divide="ignore", invalid="ignore"):
        retained = np.where(live, coverage / totals[:, None], np.nan)
    med = float(np.nanmedian(retained))
    if title:
        fig.suptitle(f"{title}\nwithout the coverage correction each value would have "
                     f"kept a median {med:.4f} of its corrected amplitude "
                     f"(a fraction between 0 and 1, no unit)", fontsize=10)

    caption_parts = [
        "Every panel is keyed on the same quantity: how much data was averaged "
        "into one (unit, electrode) value. Nothing here is filtered, thresholded "
        "or hidden — the dashed lines are reference marks, and the right-hand "
        "panel only asks what a cut WOULD cost.",
        DENSE_STITCH,
    ]
    if templates is not None:
        caption_parts.append(acronym_note("RMS"))
    caption_parts.append(PROXY_NOT_MODEL)
    if caption:
        caption_parts.append(caption)
    _add_caption(fig, _fold_caption(caption_parts, width=_caption_width(figsize)))

    logger.info(
        "estimate reliability: coverage median %d, min %d, %d pair(s) rest on a "
        "single spike; uncorrected would have kept a median %.4f of amplitude",
        int(np.median(counts)), int(counts.min()), int((counts == 1).sum()), med,
    )
    return _save_and_release(fig, out_path)


def plot_footprint_gain(positions, template, out_path, backbone_mask=None,
                        unit_id=None, coverage=None, title=None,
                        figsize=(15.0, 7.0), dpi=DEFAULT_DPI,
                        highlight_label=None, weight_label=None, caption=None):
    """One unit on the union array: what was measured, and how much backs it.

    **Descriptive only — nothing is thresholded, masked, or hidden.** This runs
    before any post-sort QC, so every electrode is drawn at whatever amplitude
    it carries and the reader decides what to believe. There is deliberately no
    "electrodes above X% of peak" count: choosing that percentage would be a
    claim about where the footprint ends, which is exactly the judgement this
    capsule refuses to make.

    Two maps of the same array, which is what makes the plot readable without a
    threshold:

    * **amplitude** — peak |value| per electrode, all of them;
    * **coverage** — how many spikes stand behind each electrode's average.

    Read together they answer the only question that matters here: a distant
    electrode showing signal is interesting if its coverage is high and is
    probably one noisy snippet if its coverage is 1. Neither map decides that
    for the reader; putting them side by side lets the reader decide.

    `backbone_mask` outlines the electrodes the sorter could actually see, so
    the recovered extent is visible as the difference. That outlined set is the
    one a sibling figure compares across segments, so `highlight_label` names
    the sibling by its real emitted filename::

        highlight_label="electrodes routed in every segment — compared across "
                        "segments in backbone_agreement.png"

    `weight_label` overrides the coverage colour bar's wording (see
    :func:`plot_rescale_effect`); `caption` appends a line under the axes.
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


def plot_rescale_before_after(positions, templates, coverage, totals, out_path,
                              unit_ids=None, title=None, dpi=DEFAULT_DPI,
                              caption=None):
    """The same units with and without the coverage correction, side by side.

    The uncorrected template is recovered exactly, without recomputing anything:
    the correction is a per-(unit, electrode) multiply, so dividing it back out
    reproduces what SpikeInterface originally averaged.

    This is the plot that shows what the correction is FOR. An uncorrected
    template keeps only the fraction of its amplitude contributed by segments
    that routed each electrode — about 1/21 on the tiled electrodes of an
    AxonTracking scan — so the arbor sits at the noise floor and the footprint
    collapses to whatever happens to sit on the always-on electrodes. The
    always-on electrodes themselves are untouched by the correction (their
    coverage IS the total, so the factor is exactly 1), which is why the two
    panels agree there and diverge everywhere else.

    Each unit gets ONE colour scale shared by its before and after panel, so the
    difference is the data and not the normalisation — which is why every panel
    carries its own colour bar rather than one shared bar for the whole figure.

    `caption` appends a line under the axes.
    """
    import numpy as np

    positions = np.asarray(positions, dtype=float)[:, :2]
    templates = np.asarray(templates)
    coverage = np.asarray(coverage, dtype=float)
    totals = np.asarray(totals, dtype=float)
    n_units = templates.shape[0]
    labels = list(unit_ids) if unit_ids is not None else list(range(n_units))

    xs, ys = np.unique(positions[:, 0]), np.unique(positions[:, 1])
    col = np.searchsorted(xs, positions[:, 0])
    row = np.searchsorted(ys, positions[:, 1])
    extent = [xs[0] - 8.75, xs[-1] + 8.75, ys[0] - 8.75, ys[-1] + 8.75]

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#ffffff")

    fig = _new_figure((4.1 * n_units, 8.6), dpi)
    axes = np.atleast_2d(fig.subplots(2, n_units))
    if axes.shape[0] == 1:
        axes = axes.T if n_units == 1 else axes

    for j in range(n_units):
        corrected = templates[j]
        with np.errstate(divide="ignore", invalid="ignore"):
            undo = np.where(coverage[j] > 0, coverage[j] / max(totals[j], 1.0), 0.0)
        uncorrected = corrected * undo[None, :]

        pc = np.abs(corrected).max(axis=0)
        pu = np.abs(uncorrected).max(axis=0)
        vmax = max(pc.max(), pu.max())
        live = pc[pc > 0]
        vmin = max(np.percentile(live, 1) if live.size else vmax * 1e-4, vmax * 1e-4)
        norm = LogNorm(vmin=vmin, vmax=vmax)

        for k, (peak, label) in enumerate(
            ((pu, "before the coverage correction"), (pc, "after the coverage correction"))
        ):
            ax = axes[k, j]
            img = np.full((ys.size, xs.size), np.nan)
            img[row, col] = peak
            image = ax.imshow(img, origin="lower", extent=extent, cmap=cmap, norm=norm,
                              interpolation="nearest", aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
            if k == 0:
                ax.set_title(f"unit {labels[j]}", fontsize=12, pad=5)
            ax.set_xlabel(f"{label}\npeak {peak.max():.0f} µV", fontsize=8.5)
            # One colour bar per panel: the two panels of a column share a norm,
            # but no two COLUMNS do, so a single figure-wide bar would be a lie.
            bar = fig.colorbar(image, ax=ax, shrink=0.82, pad=0.02)
            bar.set_label(AMPLITUDE_COLORBAR_LABEL, fontsize=6.5)
            bar.ax.tick_params(labelsize=6)

    axes[0, 0].set_ylabel(
        "BEFORE\n(plain average over the segments\nthat measured each electrode)",
        fontsize=9.5,
    )
    axes[1, 0].set_ylabel(
        "AFTER\n(corrected for how many segments\ncould see each electrode)",
        fontsize=9.5,
    )
    # The one categorical colour on the figure. It is white on a white page, so
    # the key spells out that white is the mark rather than relying on the swatch.
    axes[0, -1].legend(
        handles=[_legend_patch("#ffffff", NO_ELECTRODE_LEGEND_LABEL)],
        loc="upper right", fontsize=6, framealpha=_LEGEND_FRAMEALPHA,
    )
    if title:
        fig.suptitle(title, fontsize=13)

    caption_parts = [
        "Both rows are the same unit on the same array; only the correction "
        "differs. Each column's two panels share one colour scale, so a change "
        "between the rows is the data and not the scaling — but colour scales are "
        "NOT comparable between columns.",
        AMPLITUDE_FLOOR_NOTE,
        DENSE_STITCH,
        PROXY_NOT_MODEL,
    ]
    if caption:
        caption_parts.append(caption)
    _add_caption(fig, _fold_caption(caption_parts,
                                    width=_caption_width((4.1 * n_units, 8.6))))

    logger.info("rescale before/after: %d unit(s) -> %s", n_units, out_path)
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


def plot_template_agreement(reference, candidate, out_path, coverage=None,
                            labels=("per-segment merge", "union + rescale"),
                            title=None, figsize=DEFAULT_FIGSIZE, dpi=DEFAULT_DPI,
                            caption=None):
    """Do two routes to the same template agree? Scatter plus residual.

    Written for the comparison that matters here — fusing per-segment templates
    against correcting one union pass — where the two should be identical and
    any structure in the residual is a bug worth finding.

    `labels` names the two routes; both axes are amplitudes in µV whatever the
    routes are called, and the figure says so. `caption` appends a line under
    the axes — the place to name the two routes in the reader's terms rather
    than in ours.
    """
    import numpy as np

    a = np.asarray(reference, dtype=float).ravel()
    b = np.asarray(candidate, dtype=float).ravel()
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]

    fig = _new_figure(figsize, dpi)
    left, right = fig.subplots(1, 2)

    lim = float(max(np.abs(a).max(), np.abs(b).max())) if a.size else 1.0
    left.scatter(a, b, s=1, alpha=0.15, c="0.2", rasterized=True)
    left.plot([-lim, lim], [-lim, lim], color="red", lw=0.8, ls="--")
    left.set_xlabel(f"{labels[0]} (µV)")
    left.set_ylabel(f"{labels[1]} (µV)")
    left.set_aspect("equal"); left.set_title("value by value")
    left.legend(
        handles=[
            _legend_dot("0.2", "one averaged waveform value, on both routes (µV)", size=4.0),
            _legend_line("red", "exact agreement (the line y = x)", lw=0.8, linestyle="--"),
        ],
        loc="best", fontsize=_LEGEND_FONTSIZE, framealpha=_LEGEND_FRAMEALPHA,
    )

    residual = b - a
    right.hist(residual, bins=80, color="0.3")
    right.set_yscale("log")
    right.set_xlabel(f"{labels[1]} minus {labels[0]} (µV)")
    right.set_ylabel("waveform values (count, logarithmic axis)")
    right.set_title("residual")
    right.legend(
        handles=[_legend_patch("0.3", "values whose difference falls in this bin")],
        loc="best", fontsize=_LEGEND_FONTSIZE, framealpha=_LEGEND_FRAMEALPHA,
    )

    denom = float(np.abs(a).max()) or 1.0
    stats = (f"max |difference| = {np.abs(residual).max():.3e} µV\n"
             f"as a fraction of the largest value = "
             f"{np.abs(residual).max()/denom:.2e} (no unit)")
    right.text(0.97, 0.62, stats, transform=right.transAxes,
               ha="right", va="top", fontsize=8, family="monospace")

    if title:
        fig.suptitle(title)

    caption_parts = [
        "Both routes compute the same averaged waveforms from the same recording, "
        "so every point should land on the dashed line and the residual should be "
        "a spike at zero; anything else is a difference between the two routes, "
        "not a property of the cells.",
        "Amplitudes are in microvolts (µV) on both axes.",
    ]
    if caption:
        caption_parts.append(caption)
    _add_caption(fig, _fold_caption(caption_parts, width=_caption_width(figsize)))

    logger.info("template agreement: max abs diff %.3e (relative %.2e)",
                float(np.abs(residual).max()), float(np.abs(residual).max()/denom))
    return _save_and_release(fig, out_path)


__all__ = [
    "plot_coverage_map",
    "plot_rescale_before_after",
    "plot_rescale_effect",
    "plot_footprint_gain",
    "plot_template_agreement",
    # Reader-facing wording, exported so a caller (or a test) can assert on the
    # exact string a figure prints instead of re-typing it.
    "AMPLITUDE_COLORBAR_LABEL",
    "AMPLITUDE_FLOOR_NOTE",
    "BACKBONE_LEGEND_LABEL",
    "COVERAGE_COLORBAR_LABEL",
    "COVERAGE_WEIGHT_LABEL",
    "COVERAGE_WEIGHT_SHORT_LABEL",
    "MARKER_IS_ELECTRODE_NOTE",
    "NO_ELECTRODE_LEGEND_LABEL",
    "SEGMENTS_ROUTED_COLORBAR_LABEL",
]
