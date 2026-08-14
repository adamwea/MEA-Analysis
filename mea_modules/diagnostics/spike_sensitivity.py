"""How many spikes a per-segment template needs, and which units are spike-starved.

This is a PRE-BUILD calibration: it runs per-well on the post-sort sort, BEFORE
`build_segment_analyzers` and `stitch_templates` exist, and answers two
questions, two costs.

  1. CENSUS (cheap, every unit). Tabulate, per unit, how many spikes each
     segment actually caught — a ``(n_units, n_segments)`` count matrix. The
     capsule bins the post-sort sort's spike times against the concatenate
     manifest's segment boundaries (``spike_accounting.per_unit_segment_matrix``)
     and hands the matrix to :func:`spike_census_from_counts`; no recording is
     touched, so the census always runs and covers every unit. Its product is
     the spike-starved roster: units whose per-segment spike counts sit below
     the calibrated "enough" threshold, ranked worst-first for curation.

  2. CONVERGENCE SWEEP (expensive, sampled). The capsule builds its OWN small
     sample of dense per-segment analyzers (via `build_segment_analyzers`, which
     `stitch_templates` later reuses — so no compute is wasted) and hands their
     dirs to :func:`convergence_sweep`. For a grid of subsample sizes N it
     recomputes each unit's average template from N randomly chosen spikes and
     correlates (Pearson) the N-spike template against the unit's fullest
     available template. That correlation rising toward 1 as N grows is the
     convergence curve; where the MEDIAN unit's curve plateaus is "enough
     spikes", which is the recommended `max_spikes_per_unit` default and the
     census's starved threshold.

Adaptive climb. The bare dense analyzer already holds ALL of each unit's spikes;
the sweep only SUBSAMPLES it at each grid N, so pushing N higher never rebuilds
anything — it just draws more of the spikes already there. When a sweep still
reports unstable units (units that HAD >= the top grid point's spikes yet whose
template kept moving), the caller EXTENDS the grid (:func:`next_climb_grid`) and
re-runs, up to a cap, rather than rebuilding. On spike-starved data no unit
reaches the top grid point, so there are no unstable units and the climb never
fires (correct: a 44-spike unit cannot be pushed to 500).

Why the sweep must sample, and why it cannot crash the node. Recomputing
templates at every grid point for hundreds of units on ONE segment takes
minutes and holds one segment's recording plus template arrays in memory. So the
sweep walks its segment sample STRICTLY SEQUENTIALLY, drops each analyzer before
the next, and computes templates with `operators=["average"]` only — an in-place
running mean, never a materialised waveform buffer — so peak memory is one
segment's traces, not `n_units x n_spikes x n_channels x n_samples`. The number
of segments and units swept is bounded by the caller; the default is a few
segments, enough for a stable curve.

Finding on P005843 (2026-08-14, 3 segments): the median unit's template reaches
corr 0.99 to its asymptote by ~75 spikes and 0.986 by 200, but only ~2% of
unit-segments even HAVE 500 spikes to draw from (median ~44). So the 500 cap
almost never binds — the units are data-limited by the AxonTracking protocol,
not the extraction cap — and ~87% of unit-segments are spike-starved. That is
the direct cause of the stitched-peak wander `stitch_consistency` reports.

Pure library: no argparse, no printing, no ``__main__``.
"""

import json
import logging
from pathlib import Path

import numpy as np

from .channel_layout import _new_figure, _save_and_release

logger = logging.getLogger(__name__)

# The subsample grid. Spans the sparse regime (a Maxwell AxonTracking segment
# median is ~44 spikes) through the 500 cap, dense where the curve is steep.
DEFAULT_GRID = (12, 25, 50, 100, 200, 350, 500)

# A template correlating this well with its own asymptote is visually and
# quantitatively stable: the remaining motion is below the per-spike noise a
# reviewer would call "the same template". The N at which the MEDIAN unit first
# clears this bar is the recommended `max_spikes_per_unit` and the census's
# starved threshold. 0.99 is reported alongside as the strict bar.
DEFAULT_CORR_TARGET = 0.95
STRICT_CORR_TARGET = 0.99

SPIKE_COUNTS_FILENAME = "spike_counts.npy"


# ---------------------------------------------------------------------------
# 1. Census — cheap, every unit. A plain count-matrix read, no recording.
# ---------------------------------------------------------------------------

def spike_census_from_counts(counts, unit_ids=None, starved_threshold=None):
    """Per-unit spike inventory from a ``(n_units, n_segments)`` count matrix.

    ``counts[u, s]`` is how many spikes unit ``u`` fired in segment ``s``. This
    is the numeric core of the census: it touches no files, so it serves equally
    the capsule's live source (the post-sort sort binned against the concat
    manifest — ``spike_accounting.per_unit_segment_matrix``'s ``matrix_counts``)
    and the file-backed :func:`spike_census` wrapper below.

    ``starved_threshold`` (spikes per active segment) is applied if given —
    normally the sweep's calibrated soft plateau — to flag and rank
    spike-starved units. Returns a JSON-safe dict; the ranked ``starved`` roster
    is worst-first.
    """
    counts = np.asarray(counts, dtype=np.int64)
    if counts.ndim != 2:
        raise ValueError(
            f"spike_census_from_counts expects a 2-D (n_units, n_segments) "
            f"count matrix; got shape {counts.shape!r}"
        )
    n_units, n_segments = counts.shape

    if n_units == 0:
        # A retention set with segments but zero units — the percentile calls
        # below would raise on empty input. Return an empty-but-valid census.
        return {
            "n_units": 0, "n_segments": int(n_segments),
            "total_spikes_deciles": [], "per_active_median_spikes_deciles": [],
            "n_units_below_median_spikes": {}, "_total_spikes": np.array([]),
            "_per_active_median": np.array([]), "_per_active_max": np.array([]),
            "_n_active_segments": np.array([]), "_unit_ids": [],
        }

    active = counts > 0
    n_active = active.sum(axis=1)
    total = counts.astype(np.int64).sum(axis=1)
    per_active_median = np.array(
        [float(np.median(counts[u, active[u]])) if active[u].any() else 0.0
         for u in range(n_units)]
    )
    per_active_max = np.array(
        [int(counts[u, active[u]].max()) if active[u].any() else 0
         for u in range(n_units)]
    )

    if unit_ids is None or len(unit_ids) != n_units:
        unit_ids = list(range(n_units))

    # Fraction below a grid of standard bars — the shape of the starvation, so
    # a reviewer can see how a chosen cut moves the retained-unit count.
    band_counts = {
        str(thr): int((per_active_median < thr).sum())
        for thr in (25, 50, 100, 200, 350, 500)
    }

    census = {
        "n_units": int(n_units),
        "n_segments": int(n_segments),
        "total_spikes_deciles": [
            int(x) for x in np.percentile(total, [10, 25, 50, 75, 90]).round()
        ],
        "per_active_median_spikes_deciles": [
            int(x) for x in np.percentile(per_active_median, [10, 25, 50, 75, 90]).round()
        ],
        "n_units_below_median_spikes": band_counts,
        # arrays kept for the plot + the capsule; small (one float/int per unit)
        "_total_spikes": total,
        "_per_active_median": per_active_median,
        "_per_active_max": per_active_max,
        "_n_active_segments": n_active,
        "_unit_ids": list(unit_ids),
    }

    if starved_threshold is not None:
        census.update(_flag_starved(census, float(starved_threshold)))
    return census


def spike_census(segment_contributions_dir, unit_ids=None, starved_threshold=None):
    """Per-unit spike inventory over a retained ``segment_contributions`` tree.

    Reads every `segment_*/spike_counts.npy` under `segment_contributions_dir`
    (each a ``(n_units,)`` int array in the shared retained-unit row order),
    stacks them to ``(n_units, n_segments)``, and delegates to
    :func:`spike_census_from_counts`. Retained as the file-backed entry point
    for older on-disk runs and the pure tests; the capsule itself now sources
    its counts from the sort + concat manifest instead of a retained tree.
    """
    seg_dirs = sorted(Path(segment_contributions_dir).glob("segment_*"))
    if not seg_dirs:
        raise FileNotFoundError(
            f"no segment_*/ under {segment_contributions_dir}; a retained "
            "segment_contributions tree is required for the file-backed census"
        )
    counts = np.stack(
        [np.load(d / SPIKE_COUNTS_FILENAME) for d in seg_dirs], axis=1
    ).astype(np.int64)  # (units, segments)
    return spike_census_from_counts(
        counts, unit_ids=unit_ids, starved_threshold=starved_threshold,
    )


def _flag_starved(census, threshold):
    """Rank units with per-active-segment median below `threshold`, worst-first."""
    per_active = census["_per_active_median"]
    total = census["_total_spikes"]
    n_active = census["_n_active_segments"]
    unit_ids = census["_unit_ids"]
    starved_mask = per_active < threshold
    order = np.argsort(per_active)  # most starved first
    roster = [
        {
            "unit_id": unit_ids[u],
            "unit_index": int(u),
            "median_spikes_per_active_segment": float(per_active[u]),
            "total_spikes": int(total[u]),
            "n_active_segments": int(n_active[u]),
        }
        for u in order if starved_mask[u]
    ]
    return {
        "starved_threshold_spikes": float(threshold),
        "n_spike_starved_units": int(starved_mask.sum()),
        "spike_starved_fraction": float(starved_mask.mean()),
        "starved": roster,
    }


def plot_spike_census(census, out_path, title=None, starved_threshold=None,
                      figsize=(11, 4.0), dpi=140):
    """Three panels: per-active-segment spike histogram, per-unit total-spike
    distribution, and the cumulative "units below N spikes/active-seg" curve
    with the starved threshold marked."""
    per_active = np.asarray(census["_per_active_median"])
    total = np.asarray(census["_total_spikes"])
    thr = starved_threshold if starved_threshold is not None else census.get(
        "starved_threshold_spikes"
    )

    fig = _new_figure(figsize, dpi)
    ax1, ax2, ax3 = (fig.add_subplot(1, 3, i) for i in (1, 2, 3))

    ax1.hist(per_active, bins=40, color="#4C72B0")
    if thr is not None:
        ax1.axvline(thr, color="#C44E52", lw=1.5, ls="--",
                    label=f"starved < {thr:g}")
        ax1.legend(fontsize=8, loc="upper right")
    ax1.set_xlabel("median spikes / active segment")
    ax1.set_ylabel("units")
    ax1.set_title("per-segment sampling", fontsize=10)

    ax2.hist(np.log10(np.maximum(total, 1)), bins=40, color="#55A868")
    ax2.set_xlabel("log10(total spikes across segments)")
    ax2.set_ylabel("units")
    ax2.set_title("pooled sampling", fontsize=10)

    xs = np.sort(per_active)
    frac = np.arange(1, xs.size + 1) / xs.size
    ax3.plot(xs, frac * 100.0, color="#8172B3")
    if thr is not None:
        below = float((per_active < thr).mean()) * 100.0
        ax3.axvline(thr, color="#C44E52", lw=1.5, ls="--")
        ax3.annotate(f"{below:.0f}% below", (thr, below),
                     textcoords="offset points", xytext=(6, -12), fontsize=8,
                     color="#C44E52")
    ax3.set_xlabel("median spikes / active segment")
    ax3.set_ylabel("% of units at or below")
    ax3.set_title("starvation CDF", fontsize=10)

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95 if title else 1.0))
    return _save_and_release(fig, out_path)


# ---------------------------------------------------------------------------
# 2. Convergence sweep — expensive, sampled, memory-disciplined.
# ---------------------------------------------------------------------------

def convergence_sweep(segment_analyzer_dirs, *, grid=DEFAULT_GRID, max_units=None,
                      seed=0, ms_before=None, ms_after=None,
                      corr_target=DEFAULT_CORR_TARGET, progress=None):
    """Subsample-convergence of per-segment average templates.

    For each analyzer dir in `segment_analyzer_dirs` (a capsule 08
    `.../analyzer`, walked STRICTLY SEQUENTIALLY so only one recording is ever
    resident), copy it to an in-memory analyzer and, for each N in `grid`,
    recompute every unit's average template from N uniformly-chosen spikes.
    Correlate each unit's N-spike template against its fullest template
    (``grid[-1]``, which for a unit with fewer than that many spikes is simply
    its full template). ``operators=["average"]`` keeps this an in-place mean —
    no waveform buffer — so the sweep is memory-safe on a headless node.

    `max_units` caps the per-segment unit count (None = all). `progress(msg)`,
    if given, is called once per segment for a heartbeat.

    Returns a JSON-safe dict: the median-correlation curve and converged
    fraction per N, the per-(unit,segment) "enough N", and — via
    :func:`recommend_default` — the recommended `max_spikes_per_unit` and the
    strict/soft crossing points.
    """
    from mea_modules.postprocess.analyzer import (
        DEFAULT_MS_AFTER, DEFAULT_MS_BEFORE, load_analyzer,
    )

    grid = [int(n) for n in grid]
    ms_before = DEFAULT_MS_BEFORE if ms_before is None else ms_before
    ms_after = DEFAULT_MS_AFTER if ms_after is None else ms_after
    nmax = grid[-1]

    per_unit_seg_conv = []   # one row per (unit, seg): corr to asymptote at each N
    enough_N = []            # first N reaching corr_target, per (unit, seg)
    strict_N = []            # first N reaching STRICT_CORR_TARGET
    ceilings = []            # each (unit, seg)'s own spike count (its data ceiling)
    n_segments_swept = 0
    swept_dirs = []

    for seg_dir in segment_analyzer_dirs:
        seg_dir = Path(seg_dir)
        try:
            az = load_analyzer(seg_dir)
        except Exception as exc:  # noqa: BLE001 - a bad segment must not sink the sweep
            logger.warning("sweep: cannot load analyzer %s: %s", seg_dir, exc)
            continue
        if az.is_sparse():
            logger.warning("sweep: %s is sparse; skipping (need dense templates)", seg_dir)
            continue
        az = az.save_as(format="memory")

        tmpl = {}
        for n in grid:
            az.compute("random_spikes", method="uniform", max_spikes_per_unit=n, seed=seed)
            az.compute("templates", ms_before=ms_before, ms_after=ms_after,
                       operators=["average"])
            tmpl[n] = np.asarray(az.get_extension("templates").get_data(), dtype=np.float32)
        ref = tmpl[nmax]
        nspikes = np.asarray(list(az.sorting.count_num_spikes_per_unit().values()))
        n_units = ref.shape[0]
        unit_iter = range(n_units if max_units is None else min(n_units, max_units))
        for u in unit_iter:
            r = ref[u].ravel()
            if not np.any(r):
                continue
            ceiling = int(nspikes[u])
            corrs = np.full(len(grid), np.nan)
            for i, n in enumerate(grid):
                # Subsampling to N > the unit's own spike count returns ALL its
                # spikes, so the N-template IS the asymptote (corr == 1.0,
                # trivially). Left NaN on purpose: the calibration MEDIAN below
                # must stay conditional on actually HAVING N spikes, and the
                # ceiling is tracked separately so a data-capped unit is never
                # misread as genuinely unstable.
                if n > ceiling:
                    continue
                a = tmpl[n][u].ravel()
                if not np.any(a):
                    continue
                corrs[i] = float(np.corrcoef(a, r)[0, 1])
            per_unit_seg_conv.append(corrs)
            ceilings.append(ceiling)
            reached = [grid[i] for i, c in enumerate(corrs)
                       if np.isfinite(c) and c >= corr_target]
            enough_N.append(reached[0] if reached else np.nan)
            reached_strict = [grid[i] for i, c in enumerate(corrs)
                              if np.isfinite(c) and c >= STRICT_CORR_TARGET]
            strict_N.append(reached_strict[0] if reached_strict else np.nan)

        n_segments_swept += 1
        swept_dirs.append(seg_dir.parent.name if seg_dir.name == "analyzer" else seg_dir.name)
        if progress is not None:
            progress(f"swept {seg_dir.parent.name}: {n_units} units")
        del az, tmpl, ref
        import gc
        gc.collect()

    conv = np.array(per_unit_seg_conv) if per_unit_seg_conv else np.empty((0, len(grid)))
    curve = []
    for i, n in enumerate(grid):
        col = conv[:, i] if conv.size else np.array([])
        col = col[np.isfinite(col)]
        curve.append({
            "n_spikes": int(n),
            "median_corr": float(np.median(col)) if col.size else None,
            "frac_converged": float((col >= STRICT_CORR_TARGET).mean()) if col.size else None,
            "n_unit_segments": int(col.size),
        })

    enough = np.array(enough_N, dtype=float)
    strict = np.array(strict_N, dtype=float)
    ceil_arr = np.array(ceilings, dtype=float)
    top = grid[-1]
    # A unit that never reaches the soft target within its AVAILABLE spikes is
    # either DATA-CAPPED (too few spikes to even test the high-N grid — the
    # spike-limited case, expected to dominate on sparse AxonTracking data) or
    # genuinely UNSTABLE (it had >= the top grid point's spikes and the template
    # still moved). Splitting them keeps the headline honest: "X% never
    # converged" alone conflates a data property with a template-quality one.
    nan_mask = np.isnan(enough)
    n_data_capped = int(np.sum(nan_mask & (ceil_arr < top))) if ceil_arr.size else 0
    n_unstable = int(np.sum(nan_mask & (ceil_arr >= top))) if ceil_arr.size else 0
    result = {
        "grid": grid,
        "corr_target": float(corr_target),
        "strict_corr_target": float(STRICT_CORR_TARGET),
        "n_segments_swept": int(n_segments_swept),
        "segments_swept": swept_dirs,
        "n_unit_segments": int(conv.shape[0]),
        "curve": curve,
        "enough_N": {
            "median": _pct(enough, 50), "p75": _pct(enough, 75), "p90": _pct(enough, 90),
        },
        "strict_N": {
            "median": _pct(strict, 50), "p75": _pct(strict, 75), "p90": _pct(strict, 90),
        },
        "n_data_capped": n_data_capped,
        "n_unstable": n_unstable,
        "frac_data_capped": float(n_data_capped / enough.size) if enough.size else None,
        "frac_unstable": float(n_unstable / enough.size) if enough.size else None,
        "_conv": conv,       # (unit_seg, grid) — kept for the plot
    }
    result.update(recommend_default(result))
    return result


def _pct(arr, q):
    finite = arr[np.isfinite(arr)]
    return int(np.percentile(finite, q)) if finite.size else None


def recommend_default(sweep, corr_target=None):
    """Turn a convergence curve into a recommended `max_spikes_per_unit`.

    The recommendation is the smallest grid N whose MEDIAN template correlation
    clears `corr_target` (default: the sweep's own). Also reports the strict
    (0.99) crossing and the enough-N p90, so the caller can choose a coverage
    stance rather than a single number.
    """
    grid = sweep["grid"]
    target = sweep.get("corr_target", DEFAULT_CORR_TARGET) if corr_target is None else corr_target

    if not sweep.get("n_unit_segments"):
        # Nothing was actually swept (every analyzer failed to load, or none
        # discovered) — refuse to hand back a confident cap from no data.
        return {
            "recommended_max_spikes_per_unit": None, "median_soft_plateau_N": None,
            "median_strict_plateau_N": None, "enough_N_p90": None,
        }

    def _cross(bar):
        for pt in sweep["curve"]:
            if pt["median_corr"] is not None and pt["median_corr"] >= bar:
                return pt["n_spikes"]
        return None

    soft = _cross(target)
    strict = _cross(sweep.get("strict_corr_target", STRICT_CORR_TARGET))
    p90 = (sweep.get("enough_N") or {}).get("p90")
    # The default should cover the median template's soft plateau AND not fall
    # below the p90 unit's strict need where that is known; grid-capped.
    candidates = [n for n in (soft, strict, p90) if n is not None]
    recommended = max(candidates) if candidates else grid[-1]
    return {
        "recommended_max_spikes_per_unit": int(recommended),
        "median_soft_plateau_N": soft,
        "median_strict_plateau_N": strict,
        "enough_N_p90": p90,
    }


def next_climb_grid(grid, n_unstable, max_cap):
    """One step of the adaptive climb: an EXTENDED grid, or ``None`` to stop.

    The bare dense analyzer already holds every one of a unit's spikes, so a
    larger grid point only subsamples MORE of them — the "climb" extends the
    grid, it never rebuilds. Given the current `grid`, the sweep's `n_unstable`
    count (units that had >= the top grid point's spikes yet whose template was
    still moving), and the `max_cap` ceiling, return ``grid`` with one point
    appended — ``min(grid[-1] * 2, max_cap)`` — when the climb should continue,
    or ``None`` when it should stop:

      * no unstable units (``n_unstable <= 0``) — the sweep already converged, so
        on spike-starved data (where nothing reaches the top grid point) the
        climb never fires;
      * the top grid point already sits at/above `max_cap`;
      * the doubled point would not actually advance the grid.
    """
    if n_unstable is None or n_unstable <= 0:
        return None
    grid = [int(n) for n in grid]
    if not grid:
        return None
    top = grid[-1]
    max_cap = int(max_cap)
    if top >= max_cap:
        return None
    nxt = min(top * 2, max_cap)
    if nxt <= top:
        return None
    return grid + [nxt]


def plot_convergence(sweep, out_path, configured_default=None, title=None,
                     figsize=(14, 4.2), dpi=140):
    """Three panels that make the calibration legible.

    1. MEDIAN template-correlation vs N, with the soft/strict target bars, the
       recommended cap, the configured default, and — as a text box — the
       data-capped vs unstable split and the adaptive climb (when one happened).
    2. a sample of PER-UNIT correlation-vs-N traces (from the sweep's retained
       ``_conv``), so "the template stops changing as N grows" is literally
       visible: each faint line is one (unit, segment) rising toward its own
       asymptote; the bold line is the median.
    3. the converged-fraction vs N curve.
    """
    grid = np.asarray(sweep["grid"])
    med = np.array([pt["median_corr"] if pt["median_corr"] is not None else np.nan
                    for pt in sweep["curve"]])
    frac = np.array([pt["frac_converged"] if pt["frac_converged"] is not None else np.nan
                     for pt in sweep["curve"]])
    n = np.array([pt["n_unit_segments"] for pt in sweep["curve"]])
    target = sweep.get("corr_target", DEFAULT_CORR_TARGET)
    strict = sweep.get("strict_corr_target", STRICT_CORR_TARGET)
    rec = sweep.get("recommended_max_spikes_per_unit")
    conv = sweep.get("_conv")
    conv = np.asarray(conv) if conv is not None else np.empty((0, grid.size))

    fig = _new_figure(figsize, dpi)
    ax1, ax2, ax3 = (fig.add_subplot(1, 3, i) for i in (1, 2, 3))

    # --- panel 1: median convergence curve + calibration annotations ---
    ax1.plot(grid, med, "-o", color="#4C72B0")
    ax1.axhline(target, color="#DD8452", ls="--", lw=1, label=f"soft {target:g}")
    ax1.axhline(strict, color="#C44E52", ls=":", lw=1, label=f"strict {strict:g}")
    if rec is not None:
        ax1.axvline(rec, color="#55A868", lw=1.5, label=f"recommend {rec}")
    if configured_default is not None:
        ax1.axvline(configured_default, color="#8172B3", lw=1.2, ls="-.",
                    label=f"configured {configured_default}")
    for x, y, cnt in zip(grid, med, n):
        if np.isfinite(y):
            ax1.annotate(f"n={cnt}", (x, y), textcoords="offset points",
                         xytext=(0, 6), fontsize=7, ha="center", color="#555")
    split_lines = []
    n_dc, n_un = sweep.get("n_data_capped"), sweep.get("n_unstable")
    if n_dc is not None or n_un is not None:
        split_lines.append(f"data-capped {n_dc}, unstable {n_un}")
    climb = sweep.get("climb") or {}
    caps_tried = climb.get("caps_tried") or []
    if len(caps_tried) > 1:
        split_lines.append(f"climb {caps_tried} → {climb.get('stopped_because')}")
    if split_lines:
        ax1.text(0.03, 0.03, "\n".join(split_lines), transform=ax1.transAxes,
                 fontsize=7, va="bottom", ha="left", color="#555",
                 bbox=dict(boxstyle="round", fc="white", ec="#cccccc", alpha=0.85))
    ax1.set_xscale("log")
    ax1.set_xlabel("spikes subsampled (N)")
    ax1.set_ylabel("median corr to asymptote")
    ax1.set_title("template convergence", fontsize=10)
    ax1.legend(fontsize=7, loc="lower right")

    # --- panel 2: a sample of per-unit corr-vs-N traces ---
    if conv.size and conv.shape[1] == grid.size:
        rng = np.random.default_rng(0)
        n_show = min(40, conv.shape[0])
        rows = (rng.choice(conv.shape[0], size=n_show, replace=False)
                if conv.shape[0] > n_show else range(conv.shape[0]))
        for i in rows:
            row = conv[i]
            mask = np.isfinite(row)
            if mask.any():
                ax2.plot(grid[mask], row[mask], "-", color="#4C72B0",
                         alpha=0.15, lw=0.8)
        ax2.plot(grid, med, "-o", color="#C44E52", lw=1.5, label="median")
        ax2.axhline(target, color="#DD8452", ls="--", lw=1)
        ax2.legend(fontsize=7, loc="lower right")
    else:
        ax2.text(0.5, 0.5, "no per-unit traces\n(no unit reached the grid)",
                 transform=ax2.transAxes, ha="center", va="center",
                 fontsize=8, color="#999")
    ax2.set_xscale("log")
    ax2.set_xlabel("spikes subsampled (N)")
    ax2.set_ylabel("corr to asymptote")
    ax2.set_title("per-unit convergence (sample)", fontsize=10)

    # --- panel 3: converged fraction ---
    ax3.plot(grid, frac * 100.0, "-o", color="#C44E52")
    ax3.set_xscale("log")
    ax3.set_xlabel("spikes subsampled (N)")
    ax3.set_ylabel(f"% of unit-segments converged (>= {strict:g})")
    ax3.set_title("converged fraction", fontsize=10)

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95 if title else 1.0))
    return _save_and_release(fig, out_path)


def census_json_safe(census):
    """Drop the leading-underscore array fields so a census dict is dumpable."""
    return {k: v for k, v in census.items() if not k.startswith("_")}


def sweep_json_safe(sweep):
    """Drop the leading-underscore array fields so a sweep dict is dumpable."""
    return {k: v for k, v in sweep.items() if not k.startswith("_")}
