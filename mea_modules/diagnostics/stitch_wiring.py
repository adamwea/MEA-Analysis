"""Point the union-route diagnostics at `stitch_templates`' outputs instead.

`recovery.py` (490 lines) and `sensitivity.py` (1067 lines) were written for
the abandoned union+rescale route and orphaned when it was retired — but the
maths never depended on the route, only on five inputs. This module is the
input mapping from the diagnostics catalog (`docs/diagnostics-catalog.md`
§3.5, on the MEA-recon-pipeline `docs/diagnostics-catalog` branch), made
executable:

    union-route input          round-2 equivalent (capsule 10 stitch_templates)
    -------------------------  ------------------------------------------------
    coverage[u, c]             contributing_weight.npy  (n_units, n_ch_union)
    templates[u, c-as-LAST]    merged_templates.npy — BUT the axis order moved:
                               the union route used SpikeInterface's
                               (n_units, n_samples, n_channels); the stitch
                               writes (n_units, n_channels, n_samples).
                               :func:`sensitivity_templates` does the transpose.
    positions                  channel_locations_xy.npy (n_ch_union, 2)
    routing[s, c]              segment_contributions/<seg>/channel_ids.json
                               when capsule 10 ran with --retain-segments;
                               otherwise a column-sum-faithful surrogate — see
                               :func:`resolve_routing`
    backbone_mask              routing column sums at their max (retention), or
                               contributing_weight at its max under `uniform`
    nbefore                    ms_before x sampling_frequency_hz / 1000, both
                               from merge_manifest.json
                               (:func:`nbefore_from_manifest`)

Two representation differences the mapping has to absorb, both handled here so
`recovery`/`sensitivity` stay byte-for-byte untouched (re-wire, not rewrite):

* **NaN vs zero.** The union route zero-filled electrodes a unit never
  covered; the stitch writes `NaN` there (with `contributing_weight` exactly
  0.0 — E1's R14① validation confirmed the two masks agree cell-for-cell).
  The orphaned code was written against the zero convention, and a bare
  `argmax` over a NaN-carrying row returns the first NaN index, so
  :func:`sensitivity_templates` restores the zero convention. The NaN mask
  itself is NOT lost — :func:`nan_coverage_summary` reports it as its own
  diagnostic (it feeds `15 post_bombcell_diagnostics` later).

* **What `coverage` counts.** Under `averaging_method="spike_count"` (the
  Round-2 default) `contributing_weight` is the summed selected-spike count —
  exactly the union route's `coverage` semantics, so the `1/sqrt(n)` variance
  model reads as before. Under `"uniform"` it is the number of contributing
  SEGMENTS; the model then describes averaging n per-segment estimates, whose
  own precision varies with each segment's spike count — expect a noisier
  variance-model panel on `uniform` stitches (the KCNT1 reference run) and
  read it accordingly.

Pure library: no argparse, no printing, no ``__main__``.
"""

import json
import logging
from pathlib import Path

import numpy as np

from .channel_layout import _new_figure, _save_and_release

logger = logging.getLogger(__name__)

RETENTION_INDEX_FILENAME = "segments_index.json"
RETENTION_UNIT_IDS_FILENAME = "unit_ids.json"


def nbefore_from_manifest(manifest):
    """Samples before the spike peak, from `merge_manifest.json`'s own fields.

    The catalog mapping: `nbefore = ms_before x sampling_frequency_hz / 1000`.
    Uses the same int() truncation SpikeInterface applies when it builds the
    window (`int(ms_before * fs / 1000)`), so the baseline split lands on the
    same sample the templates were actually aligned to.
    """
    ms_before = float(manifest["ms_before"])
    fs = float(manifest["sampling_frequency_hz"])
    return int(ms_before * fs / 1000.0)


def sensitivity_templates(templates, contributing_weight=None, dtype=np.float32):
    """Stitched templates re-shaped for `recovery`/`sensitivity`: transpose + zero-fill.

    Takes the stitch's `(n_units, n_channels, n_samples)` array (NaN where no
    segment measured a unit on a channel) and returns
    `(n_units, n_samples, n_channels)` with the NaNs replaced by 0.0 — the two
    conventions the union-route diagnostics were written against (see the
    module docstring). float32 by default: this is a diagnostic input, the
    halved footprint matters at (780, 13384, 60), and every consumer casts
    per-unit slices to float anyway.

    `contributing_weight` is accepted only for a consistency check: entries
    that are NaN in the templates must carry weight 0.0 and vice versa. A
    mismatch means the stitch output is internally inconsistent and every
    coverage-keyed diagnostic downstream would silently lie, so it raises.
    """
    templates = np.asarray(templates)
    if templates.ndim != 3:
        raise ValueError(f"expected (n_units, n_channels, n_samples), got {templates.shape}")

    if contributing_weight is not None:
        weight = np.asarray(contributing_weight)
        nan_mask = np.isnan(templates).any(axis=2)
        covered = weight > 0
        mismatch = int((nan_mask & covered).sum() + (~nan_mask & ~covered).sum())
        if mismatch:
            raise ValueError(
                f"{mismatch} (unit, channel) cell(s) disagree between the "
                "templates' NaN mask and contributing_weight's zero mask -- "
                "the stitch output is internally inconsistent"
            )

    out = np.transpose(templates, (0, 2, 1)).astype(dtype, copy=True)
    np.nan_to_num(out, copy=False)
    return out


def routing_from_retention(retention_dir, channel_ids):
    """The exact `(n_segments, n_channels_union)` routing table, from retention.

    `retention_dir` is capsule 10's `segment_contributions/` (written under
    `--retain-segments`; layout defined by
    `mea_modules.templates.merge_segment_templates`). Each segment's
    `channel_ids.json` lists the channels that segment routed; membership
    against the well's union `channel_ids` IS the routing table — no geometry
    matching, no tolerance, because on Maxwell the channel id is the
    electrode id (`mea_modules.registration.coverage`'s own note).

    Returns `(routing, index)` — the bool table plus the parsed
    `segments_index.json` entries, in index order.
    """
    retention_dir = Path(retention_dir)
    index_path = retention_dir / RETENTION_INDEX_FILENAME
    if not index_path.is_file():
        raise FileNotFoundError(
            f"no {RETENTION_INDEX_FILENAME} at {retention_dir}; either capsule 10 "
            "ran without --retain-segments or the retention write did not finish "
            "(the index is written last, as the completeness marker)"
        )
    index = sorted(json.loads(index_path.read_text()), key=lambda e: int(e["index"]))

    slot = {str(c): i for i, c in enumerate(channel_ids)}
    routing = np.zeros((len(index), len(slot)), dtype=bool)
    for row, entry in enumerate(index):
        seg_channels = json.loads(
            (retention_dir / entry["dirname"] / "channel_ids.json").read_text()
        )
        unmatched = 0
        for channel_id in seg_channels:
            i = slot.get(str(channel_id))
            if i is None:
                unmatched += 1
                continue
            routing[row, i] = True
        if unmatched:
            logger.warning(
                "retention segment %s: %d channel id(s) not in the union set -- "
                "retention and merge output disagree", entry["dirname"], unmatched,
            )
    logger.info(
        "routing from retention: %d segment(s) x %d electrode(s); %d routed once, "
        "%d routed by every segment",
        routing.shape[0], routing.shape[1],
        int((routing.sum(axis=0) == 1).sum()),
        int((routing.sum(axis=0) == routing.shape[0]).sum()),
    )
    return routing, index


def routing_surrogate_from_weight(contributing_weight, n_segments):
    """A column-sum-faithful routing table when retention is absent (`uniform` only).

    Under `averaging_method="uniform"` a (unit, channel) weight IS the number
    of segments that measured the unit there, so the per-electrode routed
    count is bounded below by the max over units — and with hundreds of units,
    any electrode's most-active unit almost surely fired in every segment
    routing it, making the bound tight. The per-SEGMENT membership is not
    recoverable this way, so the result is a surrogate matrix whose COLUMN
    SUMS are right (`routing[k, c] = k < count[c]`) while its rows are not
    real segments. Everything `sensitivity` does with routing is
    `routing.sum(axis=0)` (the backbone/single split), which is exactly what
    survives; anything that needs true per-segment rows (the coverage map's
    per-segment panel, cross-segment consistency) must come from retention.

    NEVER valid under `spike_count` weighting, where the weight counts spikes,
    not segments — callers go through :func:`resolve_routing`, which enforces
    that.
    """
    weight = np.asarray(contributing_weight)
    counts = weight.max(axis=0)
    counts = np.clip(np.round(counts).astype(int), 0, int(n_segments))
    routing = np.arange(1, int(n_segments) + 1)[:, None] <= counts[None, :]
    logger.info(
        "routing surrogate from uniform weights: %d electrode(s), %d at full "
        "coverage %d (column sums exact, rows synthetic)",
        counts.size, int((counts == int(n_segments)).sum()), int(n_segments),
    )
    return routing


def resolve_routing(well_dir, manifest, contributing_weight):
    """`(routing, source)` for a stitch well dir, most trustworthy source first.

    1. `segment_contributions/` retention (exact, any weighting) — present
       when capsule 10 ran `--retain-segments`;
    2. the `uniform`-weighting surrogate (column sums exact, rows synthetic);
    3. `(None, "unavailable")` — a `spike_count` stitch without retention has
       no honest routing reconstruction, and callers degrade (skip the
       routing-dependent diagnostics, loudly) rather than guess.
    """
    well_dir = Path(well_dir)
    retention_dirname = manifest.get("segment_contributions_dir") or "segment_contributions"
    retention_dir = well_dir / retention_dirname
    if (retention_dir / RETENTION_INDEX_FILENAME).is_file():
        channel_ids = json.loads((well_dir / "channel_ids.json").read_text())
        routing, _index = routing_from_retention(retention_dir, channel_ids)
        return routing, "retention"

    method = manifest.get("averaging_method", manifest.get("weighting"))
    if method == "uniform":
        routing = routing_surrogate_from_weight(
            contributing_weight, int(manifest["n_segments"])
        )
        return routing, "uniform-weight surrogate"

    logger.warning(
        "no routing available: stitch used averaging_method=%r and kept no "
        "segment retention -- routing-dependent diagnostics will be skipped "
        "(re-run capsule 10 with --retain-segments to enable them)", method,
    )
    return None, "unavailable"


# --------------------------------------------------------------------------- #
# The NaN / coverage mask as a diagnostic of its own
# --------------------------------------------------------------------------- #


def nan_coverage_summary(contributing_weight, unit_ids=None, n_segments=None):
    """The stitched NaN/coverage mask, summarised as JSON-able numbers.

    `contributing_weight == 0` and template-NaN are the same mask by
    construction (checked in :func:`sensitivity_templates`), so the weight
    array alone — (n_units, n_channels), a few tens of MB — carries the whole
    story without touching the multi-GB template file.

    Reported per unit (covered channels, zero-coverage units) and per channel
    (units covered), plus the whole-well fractions. Downstream: `15
    post_bombcell_diagnostics`' label x footprint-extent cross-tab reads the
    same mask, which is why this stays a first-class output instead of a log
    line.
    """
    weight = np.asarray(contributing_weight)
    n_units, n_channels = weight.shape
    covered = weight > 0

    per_unit = covered.sum(axis=1)
    per_channel = covered.sum(axis=0)
    zero_units = np.flatnonzero(per_unit == 0)
    if unit_ids is None:
        unit_ids = [str(i) for i in range(n_units)]

    summary = {
        "n_units": int(n_units),
        "n_channels_union": int(n_channels),
        "n_segments": int(n_segments) if n_segments is not None else None,
        "covered_cells": int(covered.sum()),
        "nan_fraction_unit_channel_cells": float(1.0 - covered.mean()),
        "per_unit_covered_channels": {
            "min": int(per_unit.min()), "median": float(np.median(per_unit)),
            "max": int(per_unit.max()),
        },
        "per_channel_covered_units": {
            "min": int(per_channel.min()), "median": float(np.median(per_channel)),
            "max": int(per_channel.max()),
        },
        "n_zero_coverage_units": int(zero_units.size),
        "zero_coverage_unit_ids": [str(unit_ids[i]) for i in zero_units],
        "n_channels_covered_by_no_unit": int((per_channel == 0).sum()),
    }
    logger.info(
        "nan/coverage mask: %.1f%% of unit-channel cells uncovered; %d zero-"
        "coverage unit(s); per-unit covered channels median %d of %d",
        100.0 * summary["nan_fraction_unit_channel_cells"], zero_units.size,
        int(np.median(per_unit)), n_channels,
    )
    return summary


def plot_nan_coverage_summary(contributing_weight, positions, out_path,
                              title=None, figsize=(15.0, 5.2), dpi=170):
    """The coverage mask drawn: where units were measured, and how many channels each got.

    Left — every electrode coloured by how many UNITS carry a measurement
    there (the per-channel view; the backbone lights up at n_units). Right —
    the per-unit covered-channel histogram (the per-unit view; a unit at the
    left edge is a candidate for the zero/low-coverage report). Together they
    are the honest picture of what the stitch did NOT measure — the mask
    capsule 15's label x footprint-extent cross-tab later reads.
    """
    weight = np.asarray(contributing_weight)
    positions = np.asarray(positions, dtype=float)[:, :2]
    covered = weight > 0
    per_channel = covered.sum(axis=0)
    per_unit = covered.sum(axis=1)

    fig = _new_figure(figsize, dpi)
    left, right = fig.subplots(1, 2, width_ratios=[1.6, 1.0])

    scatter = left.scatter(
        positions[:, 0], positions[:, 1], c=per_channel, s=4, marker="s",
        cmap="viridis", linewidths=0, rasterized=True,
    )
    left.set_aspect("equal")
    left.set_xlabel("x (um)")
    left.set_ylabel("y (um)")
    left.set_title("units with a measurement, per electrode")
    fig.colorbar(scatter, ax=left, label="units covered", shrink=0.85)

    right.hist(per_unit, bins=40, color="0.35")
    right.set_yscale("log")
    right.set_xlabel("covered channels per unit")
    right.set_ylabel("units (log)")
    right.set_title("per-unit covered-channel distribution")
    right.text(
        0.03, 0.95,
        f"median {int(np.median(per_unit))}\nmin {int(per_unit.min())}\n"
        f"zero-coverage units: {int((per_unit == 0).sum())}",
        transform=right.transAxes, ha="left", va="top", fontsize=8,
    )

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return _save_and_release(fig, out_path)


# --------------------------------------------------------------------------- #
# Two stitches of the same well, compared (uniform vs spike_count)
# --------------------------------------------------------------------------- #


def compare_stitches(well_inputs_a, well_inputs_b, labels=("a", "b")):
    """Cell-level comparison of two stitch passes over the same well.

    Both arguments are `mea_modules.templates.load_well_inputs` dicts. The
    interesting split is by coverage multiplicity: on cells only ONE segment
    measured, every averaging method must agree exactly (nothing to weight),
    so any difference there is a bug; on multi-segment cells the methods are
    ALLOWED to differ, and the size of that difference is the empirical answer
    to "does the weighting choice matter on this data".

    Multiplicity is taken from whichever input carries a `uniform` weight
    (where weight IS the segment count); when neither does, cells with equal
    weight in both are used as the agree-exactly proxy. Streams per unit so
    peak memory stays one unit's pair of dense templates.

    Returns a JSON-able dict; raises when the two wells' unit or channel axes
    disagree (they are then not two stitches of the same thing).
    """
    a, b = well_inputs_a, well_inputs_b
    if a["unit_ids"] != b["unit_ids"]:
        raise ValueError("unit_ids differ between the two stitch dirs")
    if a["templates"].shape != b["templates"].shape:
        raise ValueError(
            f"template shapes differ: {a['templates'].shape} vs {b['templates'].shape}"
        )

    wa = np.asarray(a["contributing_weight"])
    wb = np.asarray(b["contributing_weight"])
    covered = (wa > 0) & (wb > 0)
    only_a = int(((wa > 0) & ~(wb > 0)).sum())
    only_b = int((~(wa > 0) & (wb > 0)).sum())

    # Segment multiplicity: a uniform weighting stores it directly.
    single = None
    for weight in (wa, wb):
        rounded = np.round(weight)
        if np.allclose(weight[weight > 0], rounded[weight > 0]) and weight.max() <= 64:
            single = covered & (rounded == 1)
            break
    if single is None:
        single = covered & (wa == wb)

    n_units = len(a["unit_ids"])
    max_abs = np.zeros(a["contributing_weight"].shape, dtype=np.float64)
    peak_a = np.zeros(a["contributing_weight"].shape, dtype=np.float64)
    for u in range(n_units):
        ta = np.nan_to_num(np.asarray(a["templates"][u], dtype=np.float64))
        tb = np.nan_to_num(np.asarray(b["templates"][u], dtype=np.float64))
        max_abs[u] = np.abs(tb - ta).max(axis=1)
        peak_a[u] = np.abs(ta).max(axis=1)

    multi = covered & ~single
    diffs_multi = max_abs[multi]
    diffs_single = max_abs[single]

    def _pct(x, q):
        return float(np.percentile(x, q)) if x.size else 0.0

    summary = {
        "labels": list(labels),
        "n_units": n_units,
        "covered_cells_both": int(covered.sum()),
        "covered_only_first": only_a,
        "covered_only_second": only_b,
        "single_coverage_cells": int(single.sum()),
        "single_coverage_max_abs_diff_uv": float(diffs_single.max()) if diffs_single.size else 0.0,
        "multi_coverage_cells": int(multi.sum()),
        "multi_coverage_cells_changed_gt_0p01uv": int((diffs_multi > 0.01).sum()),
        "multi_coverage_abs_diff_uv": {
            "p50": _pct(diffs_multi, 50), "p90": _pct(diffs_multi, 90),
            "p99": _pct(diffs_multi, 99),
            "max": float(diffs_multi.max()) if diffs_multi.size else 0.0,
        },
    }
    logger.info(
        "stitch comparison %s vs %s: %d multi-coverage cell(s), %.1f%% changed "
        "> 0.01 uV (p50 %.3g uV, max %.3g uV); single-coverage max diff %.3g uV "
        "(must be ~0)",
        labels[0], labels[1], summary["multi_coverage_cells"],
        100.0 * summary["multi_coverage_cells_changed_gt_0p01uv"]
        / max(summary["multi_coverage_cells"], 1),
        summary["multi_coverage_abs_diff_uv"]["p50"],
        summary["multi_coverage_abs_diff_uv"]["max"],
        summary["single_coverage_max_abs_diff_uv"],
    )
    return summary, {"max_abs": max_abs, "peak_first": peak_a,
                     "covered": covered, "single": single}


def plot_stitch_comparison(summary, arrays, weight_first, out_path, title=None,
                           figsize=(15.0, 5.2), dpi=170):
    """Where and how much two stitch passes disagree.

    Left — |diff| distribution, single- vs multi-coverage cells (single must
    hug zero; multi is the real effect of the weighting choice). Right —
    per-cell |diff| against the first stitch's weight, the direct view of
    "the more segments disagree about a cell, the more the weighting rule
    decides", with the diff as a fraction of that cell's own peak on colour.
    """
    max_abs = arrays["max_abs"]
    covered = arrays["covered"]
    single = arrays["single"]
    multi = covered & ~single
    weight = np.asarray(weight_first, dtype=float)

    fig = _new_figure(figsize, dpi)
    left, right = fig.subplots(1, 2)

    bins = np.logspace(-4, np.log10(max(max_abs[covered].max(), 1e-3)), 60)
    left.hist(np.maximum(max_abs[multi], 1e-4), bins=bins, color="#c05621",
              alpha=0.7, label=f"multi-coverage ({int(multi.sum()):,} cells)")
    left.hist(np.maximum(max_abs[single], 1e-4), bins=bins, color="#2b6cb0",
              alpha=0.7, label=f"single-coverage ({int(single.sum()):,} cells)")
    left.set_xscale("log")
    left.set_yscale("log")
    left.set_xlabel("|difference| per unit-channel cell (uV, floored at 1e-4)")
    left.set_ylabel("cells (log)")
    left.set_title("single-coverage must agree; multi-coverage is the method's effect")
    left.legend(fontsize=8)

    if multi.any():
        # Subsample for the scatter; the histogram already carries the totals.
        idx = np.flatnonzero(multi.ravel())
        if idx.size > 200_000:
            rng = np.random.default_rng(0)
            idx = rng.choice(idx, size=200_000, replace=False)
        flat_diff = max_abs.ravel()[idx]
        flat_w = weight.ravel()[idx]
        flat_peak = arrays["peak_first"].ravel()[idx]
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(flat_peak > 0, flat_diff / flat_peak, 0.0)
        sc = right.scatter(flat_w, np.maximum(flat_diff, 1e-4), s=2,
                           c=np.clip(rel, 0, 2), cmap="magma", linewidths=0,
                           rasterized=True)
        fig.colorbar(sc, ax=right, label="|diff| / cell's own peak", shrink=0.85)
        right.set_xscale("log")
        right.set_yscale("log")
        right.set_xlabel("contributing weight (first stitch)")
        right.set_ylabel("|difference| (uV)")
        right.set_title("difference against how much data backs the cell")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return _save_and_release(fig, out_path)


__all__ = [
    "nbefore_from_manifest",
    "sensitivity_templates",
    "routing_from_retention",
    "routing_surrogate_from_weight",
    "resolve_routing",
    "nan_coverage_summary",
    "plot_nan_coverage_summary",
    "compare_stitches",
    "plot_stitch_comparison",
    "RETENTION_INDEX_FILENAME",
    "RETENTION_UNIT_IDS_FILENAME",
]
