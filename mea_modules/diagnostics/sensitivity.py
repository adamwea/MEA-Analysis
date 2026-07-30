"""Is a recovered footprint's far-field structure real, or made by the rescale?

The question
------------
Recovering the full array means averaging each unit's snippets through a
zero-padded union recording, then dividing by the number of spikes that actually
contributed at each electrode. That division is exact, but it multiplies
*whatever is there* — signal or noise. An electrode backed by one spike is
scaled by up to 2000x. So a large value can appear far from the soma with no
axon present, and on a footprint that looks exactly like arbor.

Nothing else in this pipeline tests it. The analyzer-extension audit checked
whether *derived metrics* were trustworthy; the footprints — the actual product
— were never checked.

What this measures
------------------
Three things, in order of what they establish.

**1. The variance model.** After the rescale, the residual noise of the estimate
at electrode ``c`` for unit ``u`` should be ``sigma_c / sqrt(coverage[u, c])`` —
ordinary ``1/sqrt(n)`` averaging. :func:`channel_noise_scale` measures this
directly from the templates' own pre-spike baselines and reports whether
``scale * sqrt(coverage)`` is constant. On P003454 it is, within 1.3x across
coverage from 1 to 2000, and adversarial review confirmed the model holds.

Do NOT read the ``coverage == 1`` bin landing on 9.332 uV as independent
confirmation. An earlier version of this docstring did, calling it the raw trace
noise "exactly as it must" be; review showed that agreement is a quantization
artifact which cannot distinguish any sigma between 4.7 and 13.9 uV. The model
stands on the constancy of the normalised column, not on that coincidence.

**2. Detection specificity.** Given the model, an electrode "carries signal"
when its peak exceeds a multiple of its *own* predicted noise. Poorly-supported
electrodes have higher predicted noise, so they must clear a proportionally
higher bar — which is the reliability-aware rule the coverage array exists to
enable. The threshold is swept and, at each value, the same test is applied to a
signal-free stretch of the same templates. The null is not a model: it is real
noise at real coverage carrying the real multiple-comparison burden, so it gives
an empirical false-positive count per unit.

**3. Reach, as a function of the support required.** How far from the soma
signal persists, reported separately for the fully-covered backbone and for the
single-segment electrodes where the axons must be.

A note on the sweep variable
----------------------------
An earlier version of this swept a minimum-segment-count floor ``k = 1..21``,
because segment count is a property of the electrode alone and therefore
exogenous to any unit's firing. On real AxonTracking data that sweep is
**degenerate**: 13,025 electrodes are routed by exactly one segment and 353 by
all 21, with six electrodes in between. There is no interior to sweep. So the
segment axis collapses to a two-group comparison — backbone versus
single-segment — and the informative sweep is over the detection threshold
instead. The group split is reported because it is the honest shape of the data,
not because a continuum was expected and not found.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "channel_noise_scale",
    "detection_sweep",
    "plot_sensitivity",
    "plot_variance_model",
    "plot_amplitude_vs_distance",
    "plot_footprint_threshold_ladder",
    "plot_coverage_vs_amplitude",
    "plot_enrichment_distribution",
    "propagation_test",
    "plot_propagation_test",
    "write_sensitivity_plots",
]

# Samples left between the baseline window and the spike peak, so a slightly
# early trough cannot be mistaken for baseline.
_BASELINE_GUARD = 4


def _robust_scale(values, axis=-1):
    """MAD rescaled to a Gaussian sigma."""

    median = np.median(values, axis=axis, keepdims=True)
    return 1.4826 * np.median(np.abs(values - median), axis=axis)


def _expected_max_abs_z(n_samples):
    """Expected max of ``|z|`` over ``n`` standard-normal samples.

    Needed because the real test maximises the peak over the whole post-spike
    window while the null only has the leftover pre-spike samples to work with,
    and a max-statistic grows with the number of samples it maximises over.
    Comparing a 44-sample max against an 8-sample max without correcting would
    flatter the real detections by roughly 27% for free.

    Dividing each peak by this factor puts both on a common footing: a value of
    1.0 is exactly what noise alone would produce in a window of that length.
    """

    n = max(int(n_samples), 1)
    return float(np.sqrt(2.0 * np.log(2.0 * n))) if n > 1 else 1.0


def channel_noise_scale(
    templates,
    coverage,
    *,
    nbefore,
    n_probe_units=200,
    baseline_guard=_BASELINE_GUARD,
    noise_fraction=0.5,
):
    """Per-electrode noise scale, pooled across units, plus the model check.

    A single unit's 16-sample baseline is far too short to estimate noise from —
    a MAD over 16 points is unstable enough that ``peak / noise`` produces
    thousands of spurious detections. But the noise *scale* of an electrode is a
    property of the electrode, so it can be pooled across units:

        ``scale[u, c] ~ sigma[c] / sqrt(coverage[u, c])``

    Multiplying each unit's measured baseline scale by ``sqrt(coverage[u, c])``
    therefore puts every unit on the same footing, and the median across units
    gives ``sigma[c]`` from hundreds of times more data.

    Only the first ``noise_fraction`` of the baseline window is used, leaving the
    remainder untouched so :func:`detection_sweep` has a signal-free window to
    use as a null that was not also used to fit the noise.

    Returns ``(sigma, check)`` where ``sigma`` is ``(n_channels,)`` and ``check``
    is a per-coverage-bin table of ``scale * sqrt(coverage)`` — the evidence for
    or against the model. A roughly constant column means the model holds.
    """

    coverage = np.asarray(coverage)
    n_units = templates.shape[0]
    baseline_end = max(2, int(nbefore) - int(baseline_guard))
    fit_end = max(1, int(round(baseline_end * float(noise_fraction))))

    probe = np.linspace(0, n_units - 1, min(n_probe_units, n_units)).astype(int)
    probe = np.unique(probe)

    # Pool the raw SAMPLES across units, not per-unit scale estimates. A MAD over
    # 8 points is not merely noisy, it is biased, and taking the median of 200
    # biased estimates keeps the bias while hiding it. Normalising each unit's
    # baseline by sqrt(coverage) puts every sample on a common scale first, so
    # one MAD over the pooled ~1600 samples per channel is both unbiased and
    # precise. Getting this wrong produced thousands of spurious detections.
    pooled = np.empty((probe.size, fit_end, templates.shape[2]), dtype=float)
    scales = np.empty((probe.size, templates.shape[2]), dtype=float)
    cov_probe = np.maximum(coverage[probe], 1).astype(float)
    for row, unit_index in enumerate(probe):
        window = np.asarray(templates[int(unit_index)][:fit_end, :], dtype=float)
        scales[row] = _robust_scale(window.T, axis=-1)
        covered = coverage[int(unit_index)] > 0
        pooled[row] = np.where(covered, window * np.sqrt(cov_probe[row]), np.nan)

    normalised = scales * np.sqrt(cov_probe)
    stacked = pooled.reshape(-1, templates.shape[2])
    with np.errstate(invalid="ignore"):
        centre = np.nanmedian(stacked, axis=0)
        sigma = 1.4826 * np.nanmedian(np.abs(stacked - centre), axis=0)
    fallback = float(np.nanmedian(sigma[np.isfinite(sigma) & (sigma > 0)])) \
        if np.isfinite(sigma).any() else 1.0
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, fallback)
    del pooled, stacked

    bins = [(1, 1), (2, 3), (4, 8), (9, 20), (21, 50), (51, 120), (121, 400), (401, 2000)]
    check = []
    for lo, hi in bins:
        mask = (cov_probe >= lo) & (cov_probe <= hi) & (scales > 0)
        if mask.sum() < 50:
            continue
        check.append({
            "coverage": f"{lo}-{hi}",
            "n_pairs": int(mask.sum()),
            "median_scale_uv": float(np.median(scales[mask])),
            "normalised": float(np.median(normalised[mask])),
        })

    if check:
        values = np.array([row["normalised"] for row in check])
        spread = float(values.max() / values.min()) if values.min() > 0 else np.inf
        logger.info(
            "variance model: scale*sqrt(coverage) spans %.2f-%.2f uV over coverage "
            "%s (ratio %.2fx) — %s",
            values.min(), values.max(), f"{bins[0][0]}-{bins[-1][1]}", spread,
            "consistent with 1/sqrt(n)" if spread < 2.0 else "NOT consistent with 1/sqrt(n)",
        )
    return sigma, check


def detection_sweep(
    templates,
    coverage,
    routing,
    positions,
    sigma,
    *,
    nbefore,
    thresholds=(3, 4, 5, 6, 8, 10, 15, 20, 30),
    unit_ids=None,
    max_units=None,
    baseline_guard=_BASELINE_GUARD,
    noise_fraction=0.5,
    seed=0,
):
    """Sweep the detection threshold; compare real detections against a null.

    For each threshold and each unit, counts electrodes whose peak exceeds
    ``threshold * sigma[c] / sqrt(coverage[u, c])``, split into the fully-covered
    backbone and the single-segment remainder, and measures how far from the soma
    those detections reach. The identical test is applied to the held-out half of
    the pre-spike baseline, where no spike can be, giving the false-positive
    count for that threshold on real noise at real coverage.

    The soma reference is the strongest BACKBONE electrode, so the reference
    itself cannot be a rescale artefact.
    """

    coverage = np.asarray(coverage)
    routing = np.asarray(routing)
    positions = np.asarray(positions, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)

    n_units, n_samples, n_channels = templates.shape
    per_electrode = routing.sum(axis=0).astype(int)
    n_segments = int(per_electrode.max())
    backbone = per_electrode >= n_segments
    single = per_electrode <= 1

    baseline_end = max(2, int(nbefore) - int(baseline_guard))
    fit_end = max(1, int(round(baseline_end * float(noise_fraction))))
    if baseline_end - fit_end < 2:
        raise ValueError("baseline window too short to hold both a noise fit and a null")

    if unit_ids is None:
        unit_ids = np.arange(n_units)
    unit_ids = np.asarray(unit_ids)

    order = np.arange(n_units)
    if max_units is not None and max_units < n_units:
        rng = np.random.default_rng(seed)
        order = np.sort(rng.choice(n_units, size=int(max_units), replace=False))

    shape = (order.size, thresholds.size)
    out = {
        "thresholds": thresholds,
        "unit_ids": unit_ids[order],
        "n_segments": n_segments,
        "n_backbone": int(backbone.sum()),
        "n_single": int(single.sum()),
        "n_channels": int(n_channels),
        "null_window_samples": int(baseline_end - fit_end),
        "signal_window_samples": int(n_samples - baseline_end),
    }
    for name in (
        "count_backbone", "count_single", "reach_backbone", "reach_single",
        "null_backbone", "null_single",
    ):
        out[name] = np.zeros(shape)

    predicted_scale = 1.0 / np.sqrt(np.maximum(coverage, 1).astype(float))

    # Equalise the two windows' max-statistics so real and null are comparable.
    signal_gain = _expected_max_abs_z(n_samples - baseline_end)
    null_gain = _expected_max_abs_z(baseline_end - fit_end)
    out["signal_max_gain"] = signal_gain
    out["null_max_gain"] = null_gain

    for row, unit_index in enumerate(order):
        template = np.asarray(templates[int(unit_index)], dtype=float)
        noise = sigma * predicted_scale[int(unit_index)]
        noise = np.where(noise > 0, noise, np.inf)

        peak = np.abs(template[baseline_end:, :]).max(axis=0)
        null_peak = np.abs(template[fit_end:baseline_end, :]).max(axis=0)

        # An electrode the unit never covered carries an identically-zero
        # template; it is absent, not a measurement, so exclude it outright.
        covered = coverage[int(unit_index)] > 0
        z = np.where(covered, peak / (noise * signal_gain), 0.0)
        z_null = np.where(covered, null_peak / (noise * null_gain), 0.0)

        if backbone.any():
            bb = np.flatnonzero(backbone)
            soma = int(bb[np.argmax(peak[bb])])
        else:  # pragma: no cover
            soma = int(np.argmax(peak))
        distance = np.linalg.norm(positions - positions[soma], axis=1)

        for col, threshold in enumerate(thresholds):
            hit, hit_null = z >= threshold, z_null >= threshold
            for name, group in (("backbone", backbone), ("single", single)):
                keep = hit & group
                out[f"count_{name}"][row, col] = int(keep.sum())
                out[f"reach_{name}"][row, col] = (
                    float(distance[keep].max()) if keep.any() else 0.0
                )
                out[f"null_{name}"][row, col] = int((hit_null & group).sum())

    # The honest headline: the lowest threshold at which the null is quiet.
    null_total = np.nanmedian(out["null_backbone"] + out["null_single"], axis=0)
    usable = np.flatnonzero(null_total < 1.0)
    out["threshold_null_clean"] = float(thresholds[usable[0]]) if usable.size else float("nan")

    logger.info(
        "detection sweep: %d unit(s), %d backbone / %d single-segment electrode(s); "
        "null falls below 1 false positive/unit at threshold %.0f "
        "(null window %d samples vs signal window %d — the null is a LOWER bound)",
        order.size, out["n_backbone"], out["n_single"], out["threshold_null_clean"],
        out["null_window_samples"], out["signal_window_samples"],
    )
    return out


def plot_sensitivity(sweep, check, out_path, title=None):
    """Four panels: the model check, specificity, detections, and reach."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thresholds = sweep["thresholds"]
    clean = sweep.get("threshold_null_clean", float("nan"))
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.5))

    # 1. Variance model.
    ax = axes[0, 0]
    if check:
        labels = [row["coverage"] for row in check]
        values = [row["normalised"] for row in check]
        raw = [row["median_scale_uv"] for row in check]
        x = np.arange(len(labels))
        ax.plot(x, values, marker="o", color="#2b6cb0", label=r"scale $\times\sqrt{coverage}$")
        ax.plot(x, raw, marker="s", color="#a0aec0", linestyle="--", label="raw scale")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_yscale("log")
        ax.set_ylabel("noise scale (uV)")
        ax.set_xlabel("contributing spikes at the electrode")
        ratio = max(values) / min(values) if min(values) > 0 else float("inf")
        ax.set_title(
            f"1. Variance model: flat blue = noise ~ $\\sigma/\\sqrt{{n}}$  (spread {ratio:.2f}x)",
            fontsize=10,
        )
        ax.legend(fontsize=8)
    ax.grid(alpha=0.25, linewidth=0.5)

    # 2. Specificity — the null.
    ax = axes[0, 1]
    for name, colour, label in (
        ("backbone", "#2b6cb0", f"backbone ({sweep['n_backbone']} el.)"),
        ("single", "#c05621", f"single-segment ({sweep['n_single']} el.)"),
    ):
        ax.plot(thresholds, np.nanmedian(sweep[f"count_{name}"], axis=0),
                marker="o", markersize=3, color=colour, label=f"{label} — real")
        ax.plot(thresholds, np.nanmedian(sweep[f"null_{name}"], axis=0),
                marker="x", markersize=4, color=colour, linestyle=":", label=f"{label} — null")
    ax.axhline(1.0, color="#718096", linewidth=0.8, linestyle="-")
    if np.isfinite(clean):
        ax.axvline(clean, color="#2f855a", linewidth=1.2)
        ax.annotate(f"null < 1 FP/unit\nat threshold {clean:.0f}", xy=(clean, 1.0),
                    xytext=(6, 18), textcoords="offset points", fontsize=8, color="#2f855a")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel(r"threshold (multiples of the estimate's own $\sigma$)")
    ax.set_ylabel("electrodes per unit")
    ax.set_title("2. Real detections vs false positives on signal-free data", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.25, linewidth=0.5)

    # 3. Separation: real minus null.
    ax = axes[1, 0]
    for name, colour, label in (
        ("backbone", "#2b6cb0", "backbone"),
        ("single", "#c05621", "single-segment"),
    ):
        real = np.nanmedian(sweep[f"count_{name}"], axis=0)
        null = np.nanmedian(sweep[f"null_{name}"], axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(null > 0, real / null, np.nan)
        ax.plot(thresholds, ratio, marker="o", markersize=3, color=colour, label=label)
    ax.axhline(1.0, color="#e53e3e", linewidth=1.0, linestyle="--")
    ax.annotate("at 1.0 the detections are indistinguishable from noise",
                xy=(0.03, 0.06), xycoords="axes fraction", fontsize=8, color="#e53e3e")
    ax.set_yscale("log")
    ax.set_xlabel(r"threshold (multiples of $\sigma$)")
    ax.set_ylabel("real / null  (higher is better)")
    ax.set_title("3. Do detections separate from noise?", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, linewidth=0.5)

    # 4. Reach.
    ax = axes[1, 1]
    for name, colour, label in (
        ("backbone", "#2b6cb0", "backbone"),
        ("single", "#c05621", "single-segment"),
    ):
        values = sweep[f"reach_{name}"]
        median = np.nanmedian(values, axis=0)
        ax.plot(thresholds, median, marker="o", markersize=3, color=colour, label=label)
        ax.fill_between(thresholds, np.nanpercentile(values, 25, axis=0),
                        np.nanpercentile(values, 75, axis=0), color=colour,
                        alpha=0.18, linewidth=0)
    if np.isfinite(clean):
        ax.axvline(clean, color="#2f855a", linewidth=1.2)
    ax.set_xlabel(r"threshold (multiples of $\sigma$)")
    ax.set_ylabel("reach from soma (um)")
    ax.set_title("4. How far does surviving signal extend?", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, linewidth=0.5)

    fig.suptitle(title or "Recovered footprints: is far-field structure real?", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_path)
    return out_path


# --------------------------------------------------------------------------- #
# Instructive plots
#
# The summary figure above answers the question. These are for understanding
# WHY the answer is what it is — each one makes a single step of the argument
# visible on real data, so a reader can disagree with a specific step rather
# than having to take the conclusion on trust.
# --------------------------------------------------------------------------- #


def _unit_state(template, coverage_row, sigma, nbefore, baseline_guard=_BASELINE_GUARD,
                noise_fraction=0.5):
    """Per-electrode peak, null peak, predicted noise and normalised z for one unit."""

    template = np.asarray(template, dtype=float)
    n_samples = template.shape[0]
    baseline_end = max(2, int(nbefore) - int(baseline_guard))
    fit_end = max(1, int(round(baseline_end * float(noise_fraction))))

    peak = np.abs(template[baseline_end:, :]).max(axis=0)
    null_peak = np.abs(template[fit_end:baseline_end, :]).max(axis=0)
    noise = np.asarray(sigma, dtype=float) / np.sqrt(np.maximum(coverage_row, 1))
    noise = np.where(noise > 0, noise, np.inf)

    covered = np.asarray(coverage_row) > 0
    z = np.where(covered, peak / (noise * _expected_max_abs_z(n_samples - baseline_end)), 0.0)
    z_null = np.where(covered, null_peak / (noise * _expected_max_abs_z(baseline_end - fit_end)), 0.0)
    return {
        "peak": peak, "null_peak": null_peak, "noise": noise,
        "z": z, "z_null": z_null, "covered": covered,
    }


def plot_variance_model(check, sigma, out_path, raw_noise_uv=None, title=None):
    """The foundation: does averaging n snippets suppress noise by sqrt(n)?

    Everything downstream divides by a *predicted* noise, so if this law does not
    hold the predictions are meaningless and no detection threshold means
    anything. Worth its own figure because it is the one claim that has to be
    true for the rest to be interpretable.
    """

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not check:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    coverage_mid = []
    for row in check:
        lo, hi = row["coverage"].split("-")
        coverage_mid.append(np.sqrt(float(lo) * float(hi)))  # geometric centre of the bin
    coverage_mid = np.array(coverage_mid)
    measured = np.array([row["median_scale_uv"] for row in check])
    normalised = np.array([row["normalised"] for row in check])

    ax = axes[0]
    ax.loglog(coverage_mid, measured, marker="o", color="#2b6cb0", label="measured noise scale")
    anchor = raw_noise_uv if raw_noise_uv else measured[0] * np.sqrt(coverage_mid[0])
    ax.loglog(coverage_mid, anchor / np.sqrt(coverage_mid), color="#e53e3e",
              linestyle="--", linewidth=1.2,
              label=rf"$\sigma/\sqrt{{n}}$ with $\sigma$={anchor:.2f} uV")
    ax.set_xlabel("contributing spikes at the electrode (n)")
    ax.set_ylabel("template residual noise (uV)")
    ax.set_title("Measured noise follows 1/sqrt(n) over a 2000x range", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, which="both", linewidth=0.5)

    ax = axes[1]
    ax.plot(coverage_mid, normalised, marker="o", color="#2f855a")
    ax.set_xscale("log")
    spread = normalised.max() / normalised.min() if normalised.min() > 0 else np.inf
    ax.axhspan(normalised.min(), normalised.max(), color="#2f855a", alpha=0.12, linewidth=0)
    ax.set_ylim(0, max(normalised.max() * 1.4, 1e-9))
    ax.set_xlabel("contributing spikes at the electrode (n)")
    ax.set_ylabel(r"noise $\times\sqrt{n}$  (uV)")
    ax.set_title(
        f"Flat = the law holds. Spread {spread:.2f}x across the whole range", fontsize=10,
    )
    sigma = np.asarray(sigma, dtype=float)
    finite = sigma[np.isfinite(sigma) & (sigma > 0)]
    spread_note = ""
    if finite.size:
        spread_note = (
            f"\nfitted per-electrode sigma: median {np.median(finite):.2f} uV, "
            f"p01-p99 {np.percentile(finite, 1):.2f}-{np.percentile(finite, 99):.2f}"
        )
    ax.annotate(
        "a horizontal line here is what licenses predicting each electrode's\n"
        "noise from its coverage, which is what every threshold below relies on"
        + spread_note,
        xy=(0.03, 0.08), xycoords="axes fraction", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, linewidth=0.4),
    )
    ax.grid(alpha=0.25, linewidth=0.5)

    fig.suptitle(title or "Step 1 — the variance model", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_path)
    return out_path


def plot_amplitude_vs_distance(template, coverage_row, sigma, positions, soma_channel,
                               backbone_mask, out_path, *, nbefore, threshold=6.0,
                               unit_id=None, title=None):
    """Where, in space, does signal actually rise above this estimate's own noise?

    The single most informative view: amplitude against distance from the soma,
    with the *predicted noise floor* drawn through it and the signal-free null
    scattered underneath. Real signal appears as points standing clear of the
    floor; amplified noise appears as points riding along it. A footprint plot
    cannot show this because it maps amplitude to colour without any reference to
    what amplitude would have been expected from noise alone.
    """

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    state = _unit_state(template, coverage_row, sigma, nbefore)
    positions = np.asarray(positions, dtype=float)
    distance = np.linalg.norm(positions - positions[int(soma_channel)], axis=1)
    backbone_mask = np.asarray(backbone_mask, dtype=bool)
    covered = state["covered"]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), sharey=True)

    for ax, group, label in (
        (axes[0], backbone_mask & covered, "backbone electrodes (all 21 segments)"),
        (axes[1], (~backbone_mask) & covered, "single-segment electrodes"),
    ):
        if not group.any():
            continue
        ax.scatter(distance[group], state["null_peak"][group], s=2, c="#cbd5e0",
                   label="signal-free window (null)", rasterized=True)
        detected = group & (state["z"] >= threshold)
        ax.scatter(distance[group & ~detected], state["peak"][group & ~detected], s=2,
                   c="#a0aec0", label="below threshold", rasterized=True)
        ax.scatter(distance[detected], state["peak"][detected], s=7, c="#c53030",
                   label=f"detected (z >= {threshold:g})", rasterized=True)

        # The predicted floor, binned by distance so it reads as a line.
        floor = state["noise"][group] * threshold * _expected_max_abs_z(
            np.asarray(template).shape[0] - (int(nbefore) - _BASELINE_GUARD))
        edges = np.linspace(0, distance[group].max(), 26)
        centres, values = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            band = (distance[group] >= lo) & (distance[group] < hi)
            if band.sum() >= 5:
                centres.append(0.5 * (lo + hi))
                values.append(np.median(floor[band]))
        if centres:
            ax.plot(centres, values, color="#2b6cb0", linewidth=1.8,
                    label=f"detection floor at z={threshold:g}")

        ax.set_yscale("log")
        ax.set_xlabel("distance from soma (um)")
        ax.set_title(f"{label}  —  {int(detected.sum())} detected", fontsize=10)
        ax.grid(alpha=0.25, which="both", linewidth=0.5)
        ax.legend(fontsize=7, loc="upper right", markerscale=2)

    axes[0].set_ylabel("template peak amplitude (uV)")
    unit_text = f"unit {unit_id}" if unit_id is not None else "unit"
    fig.suptitle(
        title or f"Step 2 — {unit_text}: amplitude against distance, with the noise floor",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_path)
    return out_path


def plot_footprint_threshold_ladder(template, coverage_row, sigma, positions,
                                    soma_channel, out_path, *, nbefore,
                                    thresholds=(1, 2, 4, 6, 12), unit_id=None,
                                    title=None):
    """The same footprint drawn at rising thresholds, with the null for comparison.

    This is the plot that answers the question by eye. If the far-field halo is
    real it thins gradually and keeps a connected shape radiating from the soma.
    If it is amplified noise it evaporates, and the null panel looks much like
    the mildest real panel. Put the two side by side and no statistics are
    needed to see which is happening.
    """

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    state = _unit_state(template, coverage_row, sigma, nbefore)
    positions = np.asarray(positions, dtype=float)
    soma = positions[int(soma_channel)]
    distance = np.linalg.norm(positions - soma, axis=1)
    peak, z, z_null = state["peak"], state["z"], state["z_null"]

    n_panels = len(thresholds) + 1
    fig, axes = plt.subplots(1, n_panels, figsize=(3.05 * n_panels, 3.9))
    vmax = max(float(peak.max()), 1e-3)
    vmin = max(vmax / 500.0, 0.5)
    norm = LogNorm(vmin=vmin, vmax=vmax)

    for index, threshold in enumerate(thresholds):
        ax = axes[index]
        hit = z >= threshold
        ax.scatter(positions[:, 0], positions[:, 1], s=0.5, c="#edf2f7", rasterized=True)
        if hit.any():
            ax.scatter(positions[hit, 0], positions[hit, 1], s=4, c=peak[hit],
                       cmap="inferno", norm=norm, rasterized=True)
        ax.scatter([soma[0]], [soma[1]], s=70, facecolors="none", edgecolors="#2b6cb0",
                   linewidths=1.4)
        reach = float(distance[hit].max()) if hit.any() else 0.0
        ax.set_title(f"z >= {threshold:g}\n{int(hit.sum())} el., reach {reach:.0f} um",
                     fontsize=9)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")

    # The null, at the mildest threshold shown, so the comparison is the
    # generous one rather than a flattering one.
    ax = axes[-1]
    mild = float(min(thresholds))
    hit_null = z_null >= mild
    ax.scatter(positions[:, 0], positions[:, 1], s=0.5, c="#edf2f7", rasterized=True)
    if hit_null.any():
        ax.scatter(positions[hit_null, 0], positions[hit_null, 1], s=4,
                   c=state["null_peak"][hit_null], cmap="inferno", norm=norm, rasterized=True)
    ax.scatter([soma[0]], [soma[1]], s=70, facecolors="none", edgecolors="#2b6cb0",
               linewidths=1.4)
    ax.set_title(f"NULL at z >= {mild:g}\n{int(hit_null.sum())} el. — no spike here",
                 fontsize=9, color="#c53030")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_edgecolor("#c53030")
        spine.set_linewidth(1.4)

    unit_text = f"unit {unit_id}" if unit_id is not None else "unit"
    fig.suptitle(
        title or f"Step 3 — {unit_text}: footprint as the bar rises. "
        "Circle = soma reference. Compare the last panel against the first",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_path)
    return out_path


def plot_coverage_vs_amplitude(template, coverage_row, sigma, positions, soma_channel,
                               out_path, *, nbefore, threshold=6.0, unit_id=None,
                               title=None):
    """Why a bare argmax picks the wrong electrode.

    Amplitude against how many spikes back it. The top-left region — large
    amplitude on few spikes — is where a naive `argmax` goes, and it is exactly
    where the estimate is least trustworthy. Drawing the reliability-aware
    threshold as a curve through this plane shows what the coverage array buys:
    the same amplitude means something different at n=1 than at n=2000.
    """

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    state = _unit_state(template, coverage_row, sigma, nbefore)
    positions = np.asarray(positions, dtype=float)
    distance = np.linalg.norm(positions - positions[int(soma_channel)], axis=1)
    coverage_row = np.asarray(coverage_row)
    covered = state["covered"]

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    sc = ax.scatter(np.maximum(coverage_row[covered], 1), state["peak"][covered],
                    s=3, c=distance[covered], cmap="viridis", rasterized=True)
    fig.colorbar(sc, ax=ax, label="distance from soma (um)")

    grid = np.logspace(0, np.log10(max(coverage_row.max(), 2)), 60)
    gain = _expected_max_abs_z(np.asarray(template).shape[0] - (int(nbefore) - _BASELINE_GUARD))
    ax.plot(grid, threshold * float(np.median(sigma)) / np.sqrt(grid) * gain,
            color="#c53030", linewidth=2.0,
            label=f"reliability-aware threshold (z = {threshold:g})")

    naive = np.argmax(np.where(covered, state["peak"], -np.inf))
    ax.scatter([max(coverage_row[naive], 1)], [state["peak"][naive]], s=140, marker="*",
               facecolors="none", edgecolors="#c53030", linewidths=1.6,
               label=f"bare argmax: n={int(coverage_row[naive])}, {distance[naive]:.0f} um away")

    ok = covered & (state["z"] >= threshold)
    if ok.any():
        best = np.argmax(np.where(ok, state["peak"], -np.inf))
        ax.scatter([max(coverage_row[best], 1)], [state["peak"][best]], s=140, marker="P",
                   facecolors="none", edgecolors="#2f855a", linewidths=1.6,
                   label=f"argmax among reliable: n={int(coverage_row[best])}, "
                         f"{distance[best]:.0f} um away")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("contributing spikes at the electrode (n)")
    ax.set_ylabel("template peak amplitude (uV)")
    unit_text = f"unit {unit_id}" if unit_id is not None else "unit"
    ax.set_title(title or f"Step 4 — {unit_text}: amplitude is not comparable across coverage",
                 fontsize=11)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.25, which="both", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_path)
    return out_path


def plot_enrichment_distribution(sweep, out_path, threshold=None, title=None):
    """Per-unit spread, because a median across units hides the interesting case.

    If most units carry no real far-field signal and a handful carry a lot, the
    aggregate curve looks like weak evidence everywhere when the truth is strong
    evidence somewhere. That distinction decides whether the recovery is useful
    for a subset of units or useless for all of them, so it should not be
    summarised away.
    """

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thresholds = np.asarray(sweep["thresholds"], dtype=float)
    if threshold is None:
        clean = sweep.get("threshold_null_clean", float("nan"))
        threshold = clean if np.isfinite(clean) else float(thresholds[len(thresholds) // 2])
    col = int(np.argmin(np.abs(thresholds - float(threshold))))
    used = float(thresholds[col])

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.4))

    ax = axes[0]
    for name, colour, label in (("single", "#c05621", "single-segment"),
                                ("backbone", "#2b6cb0", "backbone")):
        real = sweep[f"count_{name}"][:, col]
        ax.hist(real, bins=30, color=colour, alpha=0.55, label=f"{label} — real")
        ax.axvline(np.nanmedian(sweep[f"null_{name}"][:, col]), color=colour,
                   linestyle=":", linewidth=1.6)
    ax.set_xlabel(f"electrodes detected per unit (z >= {used:g})")
    ax.set_ylabel("units")
    ax.set_title("Detections per unit; dotted = median null", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, linewidth=0.5)

    ax = axes[1]
    real = sweep["count_single"][:, col]
    null = sweep["null_single"][:, col]
    excess = real - null
    ax.hist(excess, bins=30, color="#2f855a", alpha=0.7)
    ax.axvline(0.0, color="#c53030", linewidth=1.4)
    above = float(np.mean(excess > 0) * 100.0)
    ax.set_xlabel("real minus null, single-segment (electrodes)")
    ax.set_ylabel("units")
    ax.set_title(f"{above:.0f}% of units show excess over their own null", fontsize=10)
    ax.grid(alpha=0.25, linewidth=0.5)

    ax = axes[2]
    ax.scatter(real, sweep["reach_single"][:, col], s=10, c="#c05621", alpha=0.7)
    ax.set_xlabel(f"single-segment electrodes detected (z >= {used:g})")
    ax.set_ylabel("reported reach (um)")
    ax.set_title("Reach against how many electrodes support it", fontsize=10)
    ax.annotate(
        "points at large reach with few electrodes are\nsingle distant detections, not arbors",
        xy=(0.03, 0.9), xycoords="axes fraction", fontsize=8, va="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, linewidth=0.4),
    )
    ax.grid(alpha=0.25, linewidth=0.5)

    fig.suptitle(title or f"Step 5 — per-unit spread at z >= {used:g}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_path)
    return out_path


def propagation_test(
    templates, coverage, routing, positions, sigma, *, nbefore,
    thresholds=(1.0, 2.0, 3.0, 6.0), coverage_floor=25, max_units=120,
    min_distance_um=30.0, sampling_frequency=20000.0, seed=0,
):
    """Does the far-field signal PROPAGATE, or just appear? The decisive test.

    Amplitude tests cannot separate a real axon from noise the rescale amplified,
    because the rescale acts on amplitude. Timing is untouched by it, so a
    propagation test is immune to the transformation that confounds everything
    else — and a single spike multiplied 2000x cannot manufacture a coherent
    delay gradient.

    Two statistics, the first of which is the one to trust:

    **Fraction of positive delays.** Under noise, peak delays are symmetric about
    the median, so ~50% are positive. Under conduction away from the soma they
    must be predominantly positive. This needs no fit, no velocity estimate and
    no assumption about the delay-distance relationship being linear — which is
    why it survived when everything else was ambiguous.

    **Apparent velocity and its r-squared**, from a Theil-Sen fit of delay against
    distance. Report both together and never the velocity alone: fitting a slope
    to uncorrelated data returns a number regardless, and on real data here it
    returned anything from 0.8 to 7.5 m/s purely as a function of the threshold
    chosen. A genuine conduction velocity is threshold-invariant; one that tracks
    the threshold is an artefact. Ronchi et al. 2021 discard velocity branches
    below r-squared 0.9.

    The null repeats both on the pre-spike window. Note it is a contaminated
    upper bound rather than a clean null — detection rates there fall
    monotonically with distance from the spike — so treat a real-versus-null gap
    as conservative.
    """

    from scipy import stats

    coverage = np.asarray(coverage)
    routing = np.asarray(routing)
    positions = np.asarray(positions, dtype=float)
    per_electrode = routing.sum(axis=0).astype(int)
    backbone = per_electrode >= int(per_electrode.max())
    baseline_end = max(2, int(nbefore) - _BASELINE_GUARD)

    n_units = templates.shape[0]
    rng = np.random.default_rng(seed)
    units = np.sort(rng.choice(n_units, size=min(int(max_units), n_units), replace=False))

    rows = []
    for threshold in np.asarray(thresholds, dtype=float):
        for label, use_null in (("real", False), ("null", True)):
            fractions, velocities, r2s, counts = [], [], [], []
            for unit_index in units:
                template = np.asarray(templates[int(unit_index)], dtype=float)
                state = _unit_state(template, coverage[int(unit_index)], sigma, nbefore)
                bb = np.flatnonzero(backbone)
                soma = int(bb[np.argmax(state["peak"][bb])]) if bb.size else int(np.argmax(state["peak"]))
                distance = np.linalg.norm(positions - positions[soma], axis=1)

                window = template[:baseline_end, :] if use_null else template[baseline_end:, :]
                z = state["z_null"] if use_null else state["z"]
                delay_ms = np.argmin(window, axis=0).astype(float) / sampling_frequency * 1000.0

                keep = (
                    (z >= threshold)
                    & (coverage[int(unit_index)] >= coverage_floor)
                    & (distance > float(min_distance_um))
                    & (~backbone)
                )
                if keep.sum() < 8:
                    continue
                d = distance[keep]
                centred = delay_ms[keep] - np.median(delay_ms[keep])
                counts.append(int(keep.sum()))
                fractions.append(float((centred > 0).mean()))
                try:
                    slope = stats.theilslopes(centred, d)[0]  # ms per um
                    if abs(slope) > 1e-12:
                        velocities.append(abs(1e-3 / slope))  # um/ms -> m/s
                    r2s.append(float(stats.pearsonr(d, centred)[0] ** 2))
                except Exception:  # pragma: no cover - degenerate fits
                    pass

            if not counts:
                continue
            rows.append({
                "threshold": float(threshold),
                "kind": label,
                "n_units": len(counts),
                "median_electrodes": int(np.median(counts)),
                "positive_delay_fraction": float(np.median(fractions)),
                "median_velocity_m_s": float(np.median(velocities)) if velocities else float("nan"),
                "median_r2": float(np.median(r2s)) if r2s else float("nan"),
                "frac_r2_above_0p5": float(np.mean(np.asarray(r2s) > 0.5)) if r2s else float("nan"),
            })

    real = [r for r in rows if r["kind"] == "real"]
    if real:
        worst = max(abs(r["positive_delay_fraction"] - 0.5) for r in real)
        logger.info(
            "propagation test: positive-delay fraction departs from 0.5 by at most "
            "%.3f across thresholds %s — %s. Median r2 %.3f (Ronchi's branch "
            "criterion is 0.9)",
            worst, [r["threshold"] for r in real],
            "NO evidence of outward propagation" if worst < 0.05
            else "some directional asymmetry, investigate",
            float(np.nanmedian([r["median_r2"] for r in real])),
        )
    return rows


def plot_propagation_test(rows, out_path, title=None):
    """Three panels: delay symmetry, apparent velocity, and fit quality."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return None
    real = [r for r in rows if r["kind"] == "real"]
    null = [r for r in rows if r["kind"] == "null"]
    if not real:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.3))

    ax = axes[0]
    for series, colour, label in ((real, "#2b6cb0", "real"), (null, "#a0aec0", "null (pre-spike)")):
        if series:
            ax.plot([r["threshold"] for r in series],
                    [100 * r["positive_delay_fraction"] for r in series],
                    marker="o", color=colour, label=label)
    ax.axhline(50.0, color="#c53030", linestyle="--", linewidth=1.4)
    ax.annotate("50% = symmetric = noise", xy=(0.04, 0.9), xycoords="axes fraction",
                fontsize=8, color="#c53030")
    ax.set_ylim(35, 75)
    ax.set_xlabel("detection threshold (z)")
    ax.set_ylabel("% of delays positive")
    ax.set_title("Outward propagation would push this ABOVE 50%", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, linewidth=0.5)

    ax = axes[1]
    for series, colour, label in ((real, "#2b6cb0", "real"), (null, "#a0aec0", "null")):
        if series:
            ax.plot([r["threshold"] for r in series],
                    [r["median_velocity_m_s"] for r in series],
                    marker="o", color=colour, label=label)
    ax.axhspan(0.3, 2.0, color="#2f855a", alpha=0.15, linewidth=0)
    ax.annotate("unmyelinated axon\n0.3-2 m/s", xy=(0.5, 1.0), xycoords=("axes fraction", "data"),
                fontsize=8, color="#2f855a", va="center")
    ax.set_yscale("log")
    ax.set_xlabel("detection threshold (z)")
    ax.set_ylabel("apparent velocity (m/s)")
    ax.set_title("A real velocity would be FLAT across thresholds", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, which="both", linewidth=0.5)

    ax = axes[2]
    for series, colour, label in ((real, "#2b6cb0", "real"), (null, "#a0aec0", "null")):
        if series:
            ax.plot([r["threshold"] for r in series], [r["median_r2"] for r in series],
                    marker="o", color=colour, label=label)
    ax.axhline(0.9, color="#c53030", linestyle="--", linewidth=1.4)
    ax.annotate("Ronchi 2021 branch criterion", xy=(0.04, 0.9), xycoords="axes fraction",
                fontsize=8, color="#c53030")
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("detection threshold (z)")
    ax.set_ylabel("median r$^2$ of delay vs distance")
    ax.set_title("Without this, the velocity is a fit to noise", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, linewidth=0.5)

    fig.suptitle(title or "Does the far-field signal propagate?", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_path)
    return out_path


def write_sensitivity_plots(
    templates, coverage, routing, positions, sigma, check, sweep, out_dir, *,
    nbefore, unit_ids=None, example_units=3, threshold=6.0, prefix="footprint_validity",
):
    """Emit the whole explanatory set. Returns the paths written.

    Example units are chosen to span the range rather than taken from the front:
    the strongest, the median, and a weak one by backbone peak amplitude, so the
    reader sees a convincing case, a typical case, and a marginal one.
    """

    from pathlib import Path

    out_dir = Path(out_dir)
    coverage = np.asarray(coverage)
    routing = np.asarray(routing)
    positions = np.asarray(positions, dtype=float)
    per_electrode = routing.sum(axis=0).astype(int)
    backbone = per_electrode >= int(per_electrode.max())
    if unit_ids is None:
        unit_ids = np.arange(templates.shape[0])
    unit_ids = np.asarray(unit_ids)

    written = []
    raw_noise = None
    if check:
        first = check[0]
        if first["coverage"].startswith("1-1"):
            raw_noise = first["median_scale_uv"]

    written.append(plot_variance_model(
        check, sigma, out_dir / f"{prefix}_1_variance_model.png", raw_noise_uv=raw_noise,
    ))
    written.append(plot_enrichment_distribution(
        sweep, out_dir / f"{prefix}_5_per_unit_spread.png", threshold=threshold,
    ))

    # Rank units by backbone peak so the examples are chosen on well-supported
    # evidence rather than on whichever electrode the rescale happened to inflate.
    bb = np.flatnonzero(backbone)
    strength = np.empty(templates.shape[0])
    for index in range(templates.shape[0]):
        strength[index] = np.abs(np.asarray(templates[index])[:, bb]).max()
    ranked = np.argsort(strength)[::-1]
    picks = []
    if ranked.size:
        picks = [ranked[0], ranked[ranked.size // 2], ranked[max(0, ranked.size - 2)]]
    picks = list(dict.fromkeys(int(p) for p in picks))[:max(1, int(example_units))]

    for rank, unit_index in enumerate(picks):
        template = np.asarray(templates[unit_index], dtype=float)
        state = _unit_state(template, coverage[unit_index], sigma, nbefore)
        soma = int(bb[np.argmax(state["peak"][bb])]) if bb.size else int(np.argmax(state["peak"]))
        tag = ["strongest", "median", "weak"][min(rank, 2)]
        unit_id = unit_ids[unit_index]

        written.append(plot_footprint_threshold_ladder(
            template, coverage[unit_index], sigma, positions, soma,
            out_dir / f"{prefix}_3_ladder_{tag}_unit{unit_id}.png",
            nbefore=nbefore, unit_id=unit_id,
        ))
        written.append(plot_amplitude_vs_distance(
            template, coverage[unit_index], sigma, positions, soma, backbone,
            out_dir / f"{prefix}_2_amplitude_distance_{tag}_unit{unit_id}.png",
            nbefore=nbefore, threshold=threshold, unit_id=unit_id,
        ))
        written.append(plot_coverage_vs_amplitude(
            template, coverage[unit_index], sigma, positions, soma,
            out_dir / f"{prefix}_4_coverage_amplitude_{tag}_unit{unit_id}.png",
            nbefore=nbefore, threshold=threshold, unit_id=unit_id,
        ))

    written = [path for path in written if path is not None]
    logger.info("wrote %d explanatory plot(s) to %s", len(written), out_dir)
    return written
