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

from .channel_layout import _new_figure, _save_and_release

logger = logging.getLogger(__name__)

DEFAULT_FIGSIZE = (14.0, 9.0)
DEFAULT_DPI = 170


def plot_coverage_map(positions, n_segments_routed, out_path, title=None,
                      routing=None, segment_labels=None,
                      figsize=DEFAULT_FIGSIZE, dpi=DEFAULT_DPI):
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
    left.set_xlabel("x (um)")
    left.set_ylabel("y (um)")
    left.set_title("electrodes by segments routed")
    fig.colorbar(scatter, ax=left, label="segments routing this electrode", shrink=0.8)

    # Log y: the anchor set is a few hundred electrodes against ~13k routed once,
    # and on a linear axis the anchor bar simply disappears.
    bins = np.arange(0.5, counts.max() + 1.5)
    right.hist(counts, bins=bins, color="0.3")
    right.set_yscale("log")
    # Spelled out because the obvious short label ("segments routing an
    # electrode") reads as a segment index, and then the empty bins look like
    # segments that contributed nothing.
    right.set_xlabel("how many segments route an electrode\n(count, NOT segment index)")
    right.set_ylabel("number of electrodes (log)")
    right.set_title("electrodes per coverage count")

    once = int((counts == 1).sum())
    every = int((counts == counts.max()).sum())
    right.text(
        0.97, 0.95,
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
        bottom.set_xlabel("electrodes routed")
        bottom.set_title("per segment")
        # The floor every segment shares: the anchor set that makes
        # concatenation and sorting possible in the first place.
        anchor = int((counts == counts.max()).sum())
        bottom.axvline(anchor, color="red", ls="--", lw=1.0)
        bottom.text(anchor, per_segment.size * 0.5, f" anchor = {anchor}",
                    color="red", fontsize=8, rotation=90, va="center")
        bottom.text(0.97, 0.02,
                    f"min {per_segment.min()}  max {per_segment.max()}",
                    transform=bottom.transAxes, ha="right", va="bottom", fontsize=8)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    logger.info(
        "coverage map: %d electrodes, %d routed once, %d routed by all %d segments",
        counts.size, once, every, int(counts.max()),
    )
    return _save_and_release(fig, out_path)


def plot_rescale_effect(coverage, totals, out_path, templates=None, nbefore=None,
                        title=None, figsize=(16.0, 5.4), dpi=DEFAULT_DPI):
    """Which electrodes in a recovered footprint can be trusted, and why.

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
    """
    import numpy as np

    coverage = np.asarray(coverage, dtype=float)
    totals = np.asarray(totals, dtype=float)
    live = coverage > 0
    counts = coverage[live]

    fig = _new_figure(figsize, dpi)
    axes = fig.subplots(1, 3)

    # --- 1. how many spikes back each estimate ---------------------------
    ax = axes[0]
    bins = np.logspace(0, np.log10(max(counts.max(), 10.0)), 40)
    ax.hist(counts, bins=bins, color="0.35")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("spikes behind one unit-electrode average")
    ax.set_ylabel("unit-electrode pairs (log)")
    ax.set_title("how much data backs each estimate")
    for n, c in ((1, "red"), (10, "darkorange"), (100, "green")):
        if counts.max() >= n:
            ax.axvline(n, color=c, ls="--", lw=1.0)
            ax.text(n, ax.get_ylim()[1] * 0.5, f" n={n}", color=c, fontsize=8, rotation=90)
    ax.text(0.97, 0.95,
            f"median {int(np.median(counts))}\nmin {int(counts.min())}\n"
            f"n=1: {int((counts == 1).sum()):,}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8)

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
            ax.loglog(mids, meds, "o-", color="0.2", label="measured baseline RMS")
            # 1/sqrt(n) anchored on the first bin — the shape averaging should give.
            ax.loglog(mids, meds[0] * np.sqrt(mids[0] / mids), "--", color="red",
                      label=r"$1/\sqrt{n}$ reference")
            ax.legend(fontsize=8)
        ax.set_xlabel("spikes behind the average")
        ax.set_ylabel("residual noise (uV RMS)")
        ax.set_title("precision actually achieved")
    else:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "pass templates= for the\nmeasured-precision panel",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)

    # --- 3. what a coverage threshold costs -------------------------------
    ax = axes[2]
    thresholds = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500])
    thresholds = thresholds[thresholds <= counts.max()]
    kept = [100.0 * (counts >= t).sum() / coverage.size for t in thresholds]
    ax.plot(thresholds, kept, "o-", color="0.2")
    ax.set_xscale("log")
    ax.set_xlabel("minimum spikes required")
    ax.set_ylabel("% of unit-electrode pairs kept")
    ax.set_title("what a trust threshold costs")
    ax.grid(alpha=0.3)
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
        fig.suptitle(f"{title}\nuncorrected would have kept a median {med:.4f} "
                     f"of true amplitude", fontsize=10)
    fig.tight_layout()

    logger.info(
        "estimate reliability: coverage median %d, min %d, %d pair(s) rest on a "
        "single spike; uncorrected would have kept a median %.4f of amplitude",
        int(np.median(counts)), int(counts.min()), int((counts == 1).sum()), med,
    )
    return _save_and_release(fig, out_path)


def plot_footprint_gain(positions, template, out_path, backbone_mask=None,
                        unit_id=None, threshold_frac=0.05, title=None,
                        figsize=DEFAULT_FIGSIZE, dpi=DEFAULT_DPI):
    """One unit's footprint on the union, with the backbone marked.

    `template` is ``(n_samples, n_channels)`` for a single unit. Electrodes are
    sized and coloured by peak absolute amplitude, and `backbone_mask` outlines
    the electrodes the sorter could actually see — so the recovered extent is
    the difference between the two.
    """
    import numpy as np

    positions = np.asarray(positions, dtype=float)[:, :2]
    peak = np.abs(np.asarray(template, dtype=float)).max(axis=0)
    live = peak > 0
    strong = peak > (peak.max() * float(threshold_frac)) if peak.max() > 0 else live

    fig = _new_figure(figsize, dpi)
    ax = fig.subplots(1, 1)

    ax.scatter(positions[~live, 0], positions[~live, 1], s=2, c="0.88",
               marker="s", linewidths=0, rasterized=True)
    scatter = ax.scatter(
        positions[live, 0], positions[live, 1], c=peak[live],
        s=4 + 40 * peak[live] / max(peak.max(), 1e-12),
        cmap="magma", marker="s", linewidths=0, rasterized=True,
    )
    fig.colorbar(scatter, ax=ax, label="peak |amplitude| (uV)", shrink=0.8)

    if backbone_mask is not None:
        backbone_mask = np.asarray(backbone_mask, dtype=bool)
        ax.scatter(
            positions[backbone_mask, 0], positions[backbone_mask, 1],
            s=18, facecolors="none", edgecolors="cyan", linewidths=0.35,
            label=f"sorting backbone ({int(backbone_mask.sum())} electrodes)",
            rasterized=True,
        )
        ax.legend(loc="upper right", fontsize=8)

    ax.set_aspect("equal")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title(title or f"unit {unit_id} footprint on the union array")

    n_strong = int(strong.sum())
    ax.text(
        0.02, 0.98,
        f"electrodes above {threshold_frac:.0%} of peak: {n_strong}",
        transform=ax.transAxes, va="top", fontsize=9,
    )
    fig.tight_layout()
    logger.info("footprint unit %s: %d electrode(s) above %.0f%% of peak",
                unit_id, n_strong, 100 * threshold_frac)
    return _save_and_release(fig, out_path)


def plot_template_agreement(reference, candidate, out_path, coverage=None,
                            labels=("per-segment merge", "union + rescale"),
                            title=None, figsize=DEFAULT_FIGSIZE, dpi=DEFAULT_DPI):
    """Do two routes to the same template agree? Scatter plus residual.

    Written for the comparison that matters here — fusing per-segment templates
    against correcting one union pass — where the two should be identical and
    any structure in the residual is a bug worth finding.
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
    left.set_xlabel(labels[0]); left.set_ylabel(labels[1])
    left.set_aspect("equal"); left.set_title("value by value")

    residual = b - a
    right.hist(residual, bins=80, color="0.3")
    right.set_yscale("log")
    right.set_xlabel(f"{labels[1]} - {labels[0]}")
    right.set_ylabel("count (log)")
    right.set_title("residual")

    denom = float(np.abs(a).max()) or 1.0
    stats = (f"max |diff| = {np.abs(residual).max():.3e}\n"
             f"relative   = {np.abs(residual).max()/denom:.2e}")
    right.text(0.97, 0.95, stats, transform=right.transAxes,
               ha="right", va="top", fontsize=9, family="monospace")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    logger.info("template agreement: max abs diff %.3e (relative %.2e)",
                float(np.abs(residual).max()), float(np.abs(residual).max()/denom))
    return _save_and_release(fig, out_path)


__all__ = [
    "plot_coverage_map",
    "plot_rescale_effect",
    "plot_footprint_gain",
    "plot_template_agreement",
]
