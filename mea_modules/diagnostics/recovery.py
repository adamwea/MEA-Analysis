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
                      figsize=DEFAULT_FIGSIZE, dpi=DEFAULT_DPI):
    """Electrodes coloured by how many segments routed them; returns `out_path`."""
    import numpy as np

    positions = np.asarray(positions, dtype=float)[:, :2]
    counts = np.asarray(n_segments_routed, dtype=int)

    fig = _new_figure(figsize, dpi)
    left, right = fig.subplots(1, 2, width_ratios=[2.0, 1.0])

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
    right.set_xlabel("segments routing an electrode")
    right.set_ylabel("electrodes (log)")
    right.set_title("coverage distribution")

    once = int((counts == 1).sum())
    every = int((counts == counts.max()).sum())
    right.text(
        0.97, 0.95,
        f"routed once: {once}\nrouted by all {counts.max()}: {every}",
        transform=right.transAxes, ha="right", va="top", fontsize=9,
    )

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    logger.info(
        "coverage map: %d electrodes, %d routed once, %d routed by all %d segments",
        counts.size, once, every, int(counts.max()),
    )
    return _save_and_release(fig, out_path)


def plot_rescale_effect(coverage, totals, out_path, title=None,
                        figsize=DEFAULT_FIGSIZE, dpi=DEFAULT_DPI):
    """How far the union-recording average was from correct, per unit/electrode."""
    import numpy as np

    coverage = np.asarray(coverage, dtype=float)
    totals = np.asarray(totals, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        # The reciprocal of the correction: what fraction of the true amplitude
        # an uncorrected template would have carried.
        retained = np.where(coverage > 0, coverage / totals[:, None], np.nan)
    flat = retained[np.isfinite(retained)]

    fig = _new_figure(figsize, dpi)
    left, right = fig.subplots(1, 2)

    left.hist(flat, bins=60, color="0.3")
    left.set_yscale("log")
    left.set_xlabel("fraction of true amplitude BEFORE correction")
    left.set_ylabel("unit-electrode pairs (log)")
    left.set_title("what the uncorrected average would have kept")
    for q, style in ((0.5, "-"), (0.05, ":"), (0.95, ":")):
        v = float(np.quantile(flat, q))
        left.axvline(v, color="red", ls=style, lw=1.0)
        left.text(v, left.get_ylim()[1] * 0.6, f" {q:.0%}={v:.3f}",
                  color="red", fontsize=8, rotation=90, va="top")

    # Per-unit spread: a unit firing evenly across segments is corrected
    # uniformly; one concentrated in a few segments is not.
    per_unit = np.nanmedian(retained, axis=1)
    right.plot(np.sort(per_unit), lw=1.2, color="0.2")
    right.set_xlabel("unit (sorted)")
    right.set_ylabel("median retained fraction")
    right.set_title("per-unit median, before correction")
    right.set_ylim(0, 1.05)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    logger.info(
        "rescale effect: median retained %.4f, 5th %.4f, 95th %.4f over %d pairs",
        float(np.median(flat)), float(np.quantile(flat, 0.05)),
        float(np.quantile(flat, 0.95)), flat.size,
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
