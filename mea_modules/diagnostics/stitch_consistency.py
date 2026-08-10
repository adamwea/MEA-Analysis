"""Cross-segment consistency of stitched templates — do the segments agree?

The stitch (`mea_modules.templates.merge_segment_templates`) folds each
segment's per-unit template into one full-array average. Averaging is only
meaningful when the things averaged agree; these diagnostics measure that
agreement on the one place it is measurable — the BACKBONE electrodes every
segment routed — plus the intra-unit peak-consistency check the planning note
flags as the candidate signal for merges we are missing (plan §1b, deferred
risks: "units that are identical but show different local peaks across
segments => merges we're missing").

Everything here reads capsule 10's `segment_contributions/` retention set
(written under `--retain-segments`; layout defined in
`mea_modules/templates/merge.py` — per segment: `templates.npy`
`(n_units, n_ch_seg, n_samples)` float32 with all-NaN rows for units with no
spikes there, `spike_counts.npy`, `channel_ids.json`, `locations_xy.npy`).
Without retention these inputs die inside the stitch's streaming loop, which
is exactly why the flag exists (plan §6 R14②). Segments stream one at a time,
mirroring the merge's own memory discipline: peak state is one segment's
float32 buffer plus (n_units x n_segments)-sized accumulators.

A deliberate framing note on peak consistency: a segment's NATIVE peak
(argmax over its own ~1k routed channels) is allowed to move between
segments, because the routed subsets differ — that motion is geometry, not
biology. The apples-to-apples check is the BACKBONE peak (same electrodes in
every segment); motion THERE means the segments genuinely disagree about
where the unit is loudest, the stitched peak is a smear of the two, and every
peak-centric metric downstream reads the smear. Both are computed; the
backbone one is the headline.

Pure library: no argparse, no printing, no ``__main__``.
"""

import json
import logging
from pathlib import Path

import numpy as np

from .channel_layout import _new_figure, _save_and_release
from .stitch_wiring import RETENTION_INDEX_FILENAME

logger = logging.getLogger(__name__)


def load_retention_index(retention_dir):
    """Parsed `segments_index.json` entries (index order) + the retained unit ids."""
    retention_dir = Path(retention_dir)
    index_path = retention_dir / RETENTION_INDEX_FILENAME
    if not index_path.is_file():
        raise FileNotFoundError(
            f"no {RETENTION_INDEX_FILENAME} at {retention_dir}; capsule 10 must run "
            "with --retain-segments before cross-segment diagnostics can read it"
        )
    index = sorted(json.loads(index_path.read_text()), key=lambda e: int(e["index"]))
    unit_ids = [
        str(u) for u in json.loads((retention_dir / "unit_ids.json").read_text())
    ]
    return index, unit_ids


def _segment_channels(retention_dir, entry):
    return [str(c) for c in json.loads(
        (Path(retention_dir) / entry["dirname"] / "channel_ids.json").read_text()
    )]


def backbone_channel_ids(retention_dir, index=None):
    """Channel ids routed by EVERY retained segment — the shared electrodes.

    Order follows the first segment's own channel order, so downstream
    per-segment extraction can column-index consistently.
    """
    retention_dir = Path(retention_dir)
    if index is None:
        index, _units = load_retention_index(retention_dir)
    shared = None
    first_order = None
    for entry in index:
        channels = _segment_channels(retention_dir, entry)
        if shared is None:
            shared = set(channels)
            first_order = channels
        else:
            shared &= set(channels)
    backbone = [c for c in first_order if c in shared]
    logger.info(
        "backbone: %d electrode(s) shared by all %d retained segment(s)",
        len(backbone), len(index),
    )
    return backbone


def segment_activity(retention_dir, index=None):
    """`(n_units, n_segments)` selected-spike counts — which unit fired where.

    Read straight from each segment's retained `spike_counts.npy` (the exact
    spike_count weights the stitch used). This is both a diagnostic in its own
    right (a unit alive in one segment only is a different animal) and the
    activity profile :func:`missed_merge_candidates` keys on.
    """
    retention_dir = Path(retention_dir)
    if index is None:
        index, _units = load_retention_index(retention_dir)
    columns = [
        np.load(retention_dir / entry["dirname"] / "spike_counts.npy")
        for entry in index
    ]
    return np.stack(columns, axis=1)


def backbone_agreement(retention_dir, stitched_templates, channel_ids,
                       *, index=None, min_amplitude_uv=0.0):
    """Per (unit, segment) agreement with the stitched template, on the backbone.

    For every unit and every segment the unit fired in, the segment's backbone
    sub-template is compared against the stitched template's backbone part:

    * `r[u, s]` — Pearson correlation across (backbone channels x samples);
      shape agreement, amplitude-scale free;
    * `rms[u, s]` — RMS difference in uV; the absolute disagreement;
    * `active[u, s]` — whether the unit had any selected spikes there
      (`NaN` rows in the retained templates mark the inactive cells, mirroring
      the merge's own skip).

    Segments stream one at a time. The stitched reference is NaN-filled to 0
    on its backbone part only (backbone cells are covered for every active
    unit by construction, so this is a no-op guard, not a data change).

    A LOW r for a (unit, segment) the unit genuinely fired in means the stitch
    averaged different things together at that cell — the believability
    question this capsule exists to answer. A systematically low COLUMN means
    one segment disagrees with the consensus (drift, a bad configuration); a
    systematically low ROW means one unit is incoherent across the scan
    (unstable unit, or two neurons sharing a label).
    """
    retention_dir = Path(retention_dir)
    if index is None:
        index, _units = load_retention_index(retention_dir)

    backbone = backbone_channel_ids(retention_dir, index)
    union_slot = {str(c): i for i, c in enumerate(channel_ids)}
    bb_union = np.asarray([union_slot[c] for c in backbone], dtype=int)

    stitched = np.asarray(stitched_templates)  # (n_units, n_channels_union, n_samples)
    n_units = stitched.shape[0]
    reference = np.nan_to_num(stitched[:, bb_union, :].astype(np.float32))

    n_segments = len(index)
    r = np.full((n_units, n_segments), np.nan, dtype=np.float64)
    rms = np.full((n_units, n_segments), np.nan, dtype=np.float64)
    active = np.zeros((n_units, n_segments), dtype=bool)

    ref_flat = reference.reshape(n_units, -1)
    ref_centred = ref_flat - ref_flat.mean(axis=1, keepdims=True)
    ref_norm = np.sqrt((ref_centred ** 2).sum(axis=1))

    for s, entry in enumerate(index):
        seg_dir = retention_dir / entry["dirname"]
        seg_channels = _segment_channels(retention_dir, entry)
        seg_slot = {c: i for i, c in enumerate(seg_channels)}
        bb_local = np.asarray([seg_slot[c] for c in backbone], dtype=int)

        templates = np.load(seg_dir / "templates.npy", mmap_mode="r")
        counts = np.load(seg_dir / "spike_counts.npy")
        seg_bb = np.asarray(templates[:, bb_local, :], dtype=np.float32)
        del templates

        fired = counts > 0
        active[:, s] = fired
        seg_flat = seg_bb.reshape(n_units, -1)
        # The all-NaN rows of units with no spikes here would emit
        # "Mean of empty slice" warnings from the nan-reductions; those rows
        # are discarded below (only `fired` rows are assigned), so the
        # warnings are noise — suppressed, not papered over.
        import warnings

        with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            seg_centred = seg_flat - np.nanmean(seg_flat, axis=1, keepdims=True)
            seg_norm = np.sqrt(np.nansum(seg_centred ** 2, axis=1))
            dot = np.nansum(seg_centred * ref_centred, axis=1)
            r_col = dot / (seg_norm * ref_norm)
            rms_col = np.sqrt(np.nanmean((seg_flat - ref_flat) ** 2, axis=1))
        r[fired, s] = r_col[fired]
        rms[fired, s] = rms_col[fired]

    if min_amplitude_uv > 0:
        weak = np.abs(ref_flat).max(axis=1) < float(min_amplitude_uv)
        r[weak, :] = np.nan
        rms[weak, :] = np.nan

    per_segment_median_r = np.nanmedian(r, axis=0)
    logger.info(
        "backbone agreement: %d unit(s) x %d segment(s) on %d shared electrode(s); "
        "per-segment median r spans %.3f-%.3f; %.1f%% of active cells at r >= 0.9",
        n_units, n_segments, bb_union.size,
        float(np.nanmin(per_segment_median_r)), float(np.nanmax(per_segment_median_r)),
        100.0 * float(np.nanmean((r >= 0.9)[active & np.isfinite(r)]))
        if np.isfinite(r).any() else 0.0,
    )
    return {
        "r": r, "rms": rms, "active": active,
        "backbone_channel_ids": backbone,
        "backbone_union_index": bb_union,
        "segment_dirnames": [entry["dirname"] for entry in index],
    }


def plot_backbone_agreement(agreement, out_path, title=None,
                            figsize=(15.0, 6.4), dpi=170):
    """The (units x segments) agreement matrix, plus its per-segment summary.

    Left — r heatmap (rows: units ordered by their own median r, worst at the
    top, so the eye lands on trouble; grey = unit inactive in that segment).
    Right — per-segment median r with the per-segment active-unit count, the
    view that catches ONE segment disagreeing with the consensus.
    """
    import warnings

    r = np.asarray(agreement["r"])
    active = np.asarray(agreement["active"])
    n_units, n_segments = r.shape

    with warnings.catch_warnings():
        # a zero-coverage unit's row is all-NaN by construction; its median is
        # honestly NaN and the ordering below sends it to the bottom
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        unit_median = np.nanmedian(r, axis=1)
    order = np.argsort(np.nan_to_num(unit_median, nan=2.0))  # worst first, all-NaN last

    fig = _new_figure(figsize, dpi)
    left, right = fig.subplots(1, 2, width_ratios=[1.8, 1.0])

    import matplotlib as mpl

    # with_extremes returns a copy, so the registered colormap is never mutated
    cmap = mpl.colormaps["RdYlGn"].with_extremes(bad="0.85")
    shown = r[order, :]
    masked = np.ma.masked_invalid(shown)
    img = left.imshow(masked, aspect="auto", cmap=cmap, vmin=-1.0, vmax=1.0,
                      interpolation="nearest")
    left.set_xlabel("segment index")
    left.set_ylabel(f"unit (ordered worst median r first, n={n_units})")
    left.set_title("backbone waveform agreement r(segment, stitched)")
    fig.colorbar(img, ax=left, label="Pearson r", shrink=0.85)

    x = np.arange(n_segments)
    right.plot(x, np.nanmedian(r, axis=0), "o-", color="#2b6cb0", label="median r")
    right.plot(x, np.nanpercentile(r, 10, axis=0), "--", color="#c05621",
               linewidth=1.0, label="p10 r")
    right2 = right.twinx()
    right2.bar(x, active.sum(axis=0), color="0.8", zorder=0, label="active units")
    right2.set_ylabel("active units (grey bars)")
    right.set_ylim(-0.1, 1.05)
    right.set_xlabel("segment index")
    right.set_ylabel("r")
    right.set_title("per-segment consensus")
    right.legend(fontsize=8, loc="lower left")
    right.set_zorder(right2.get_zorder() + 1)
    right.patch.set_visible(False)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return _save_and_release(fig, out_path)


def peak_consistency(retention_dir, *, index=None):
    """Does a unit's peak electrode move between segments? The smeared-soma check.

    Per unit, over the segments it fired in:

    * `backbone_peak_channel[u, s]` / `backbone_peak_uv[u, s]` — argmax |amp|
      restricted to the shared backbone electrodes (apples-to-apples: same
      candidate set in every segment);
    * `native_peak_uv[u, s]` — the segment's own full-set peak amplitude, kept
      because a native peak far above the backbone peak means the unit's true
      maximum lives OFF the backbone in that segment's tile;
    * summary per unit: number of distinct backbone peak electrodes, max
      pairwise distance between them (um), and the amplitude spread.

    Backbone-peak MOTION is the signal (plan §1b): if segments disagree where
    the unit is loudest on the very same electrodes, the stitched template's
    peak is an average of disagreeing shapes, every peak-centric metric
    downstream reads a smear — and, if two apparent units are one neuron seen
    in different segments, this is where it first shows.
    """
    retention_dir = Path(retention_dir)
    if index is None:
        index, _units = load_retention_index(retention_dir)

    backbone = backbone_channel_ids(retention_dir, index)
    n_segments = len(index)

    first = index[0]
    counts0 = np.load(retention_dir / first["dirname"] / "spike_counts.npy")
    n_units = counts0.shape[0]

    peak_channel = np.full((n_units, n_segments), -1, dtype=int)
    peak_uv = np.full((n_units, n_segments), np.nan)
    native_uv = np.full((n_units, n_segments), np.nan)
    backbone_xy = None

    for s, entry in enumerate(index):
        seg_dir = retention_dir / entry["dirname"]
        seg_channels = _segment_channels(retention_dir, entry)
        seg_slot = {c: i for i, c in enumerate(seg_channels)}
        bb_local = np.asarray([seg_slot[c] for c in backbone], dtype=int)
        locations = np.load(seg_dir / "locations_xy.npy")
        if backbone_xy is None:
            backbone_xy = np.asarray(locations[bb_local, :2], dtype=float)

        templates = np.load(seg_dir / "templates.npy", mmap_mode="r")
        counts = np.load(seg_dir / "spike_counts.npy")
        fired = counts > 0

        seg_bb_amp = np.abs(np.asarray(templates[:, bb_local, :], dtype=np.float32)).max(axis=2)
        native_amp = np.abs(np.asarray(templates, dtype=np.float32)).max(axis=2)
        del templates

        peak_channel[fired, s] = np.argmax(seg_bb_amp[fired], axis=1)
        peak_uv[fired, s] = seg_bb_amp[fired, peak_channel[fired, s]]
        native_uv[fired, s] = native_amp[fired].max(axis=1)

    n_active = (peak_channel >= 0).sum(axis=1)
    n_distinct = np.zeros(n_units, dtype=int)
    max_spread_um = np.zeros(n_units)
    for u in range(n_units):
        used = peak_channel[u][peak_channel[u] >= 0]
        if used.size == 0:
            continue
        distinct = np.unique(used)
        n_distinct[u] = distinct.size
        if distinct.size > 1:
            xy = backbone_xy[distinct]
            diff = xy[:, None, :] - xy[None, :, :]
            max_spread_um[u] = float(np.sqrt((diff ** 2).sum(axis=2)).max())

    logger.info(
        "peak consistency: %d unit(s); %d with a single backbone peak electrode "
        "across every active segment, %d moving > 50 um (max spread %.0f um)",
        n_units, int((n_distinct == 1).sum()), int((max_spread_um > 50).sum()),
        float(max_spread_um.max()) if n_units else 0.0,
    )
    return {
        "backbone_channel_ids": backbone,
        "backbone_xy": backbone_xy,
        "peak_channel": peak_channel,
        "peak_uv": peak_uv,
        "native_peak_uv": native_uv,
        "n_active_segments": n_active,
        "n_distinct_peaks": n_distinct,
        "max_spread_um": max_spread_um,
    }


def plot_peak_consistency(result, out_path, unit_ids=None, spread_flag_um=50.0,
                          title=None, figsize=(15.0, 5.4), dpi=170):
    """Population view of backbone-peak motion, with the flagged tail named.

    Left — max peak-motion distance per unit (log-y histogram) with the flag
    threshold drawn. Right — motion against how many segments the unit fired
    in (a unit "moving" across 2 segments is one disagreement; across 20 it is
    a pattern), coloured by distinct peak-electrode count, worst units
    annotated by id so the per-unit follow-up is one glance away.
    """
    spread = np.asarray(result["max_spread_um"])
    n_active = np.asarray(result["n_active_segments"])
    n_distinct = np.asarray(result["n_distinct_peaks"])
    if unit_ids is None:
        unit_ids = [str(i) for i in range(spread.size)]

    fig = _new_figure(figsize, dpi)
    left, right = fig.subplots(1, 2)

    left.hist(spread, bins=50, color="0.35")
    left.set_yscale("log")
    left.axvline(spread_flag_um, color="red", ls="--", lw=1.2)
    flagged = int((spread > spread_flag_um).sum())
    left.text(spread_flag_um, left.get_ylim()[1] * 0.5,
              f"  {flagged} unit(s) > {spread_flag_um:.0f} um", color="red", fontsize=9)
    left.set_xlabel("max distance between a unit's backbone peak electrodes (um)")
    left.set_ylabel("units (log)")
    left.set_title("how far each unit's peak moves across segments")

    sc = right.scatter(n_active, spread, s=14, c=n_distinct, cmap="viridis",
                       linewidths=0, alpha=0.8)
    fig.colorbar(sc, ax=right, label="distinct peak electrodes", shrink=0.85)
    right.axhline(spread_flag_um, color="red", ls="--", lw=1.0)
    worst = np.argsort(spread)[::-1][:8]
    for u in worst:
        if spread[u] > spread_flag_um:
            right.annotate(str(unit_ids[u]), (n_active[u], spread[u]),
                           textcoords="offset points", xytext=(4, 3), fontsize=7)
    right.set_xlabel("segments the unit fired in")
    right.set_ylabel("max peak motion (um)")
    right.set_title("motion vs how much of the scan saw the unit")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return _save_and_release(fig, out_path)


def plot_unit_peak_map(result, unit_index, out_path, unit_id=None,
                       title=None, figsize=(7.5, 7.0), dpi=170):
    """One flagged unit's per-segment backbone peaks on the array.

    Backbone electrodes in grey; each active segment's peak electrode drawn
    at its position, coloured by segment index and sized by that segment's
    peak amplitude — the direct picture of WHERE the disagreement lives.
    """
    xy = np.asarray(result["backbone_xy"])
    peaks = np.asarray(result["peak_channel"][unit_index])
    amps = np.asarray(result["peak_uv"][unit_index])
    active = peaks >= 0

    fig = _new_figure(figsize, dpi)
    ax = fig.subplots(1, 1)
    ax.scatter(xy[:, 0], xy[:, 1], s=6, c="0.85", marker="s", linewidths=0,
               rasterized=True, label=f"backbone ({xy.shape[0]})")
    if active.any():
        seg_idx = np.flatnonzero(active)
        amp = amps[seg_idx]
        size = 30 + 120 * (amp / max(float(amp.max()), 1e-9))
        sc = ax.scatter(xy[peaks[seg_idx], 0], xy[peaks[seg_idx], 1], s=size,
                        c=seg_idx, cmap="plasma", edgecolors="k", linewidths=0.4)
        fig.colorbar(sc, ax=ax, label="segment index", shrink=0.8)
    ax.set_aspect("equal")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    label = unit_id if unit_id is not None else unit_index
    ax.set_title(title or
                 f"unit {label}: backbone peak electrode per segment "
                 f"(size = peak uV, spread {result['max_spread_um'][unit_index]:.0f} um)")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return _save_and_release(fig, out_path)


def missed_merge_candidates(stitched_templates, contributing_weight, positions,
                            activity, unit_ids=None, *, peak_distance_um=150.0,
                            min_footprint_cosine=0.8, max_activity_jaccard=0.5,
                            min_active_segments=1, min_shared_channels=8):
    """Pairs of units that look like ONE neuron seen in DIFFERENT segments.

    The §1b missed-merge hypothesis: kilosort splits one neuron into two units
    whose spikes come from (largely) different segments; each unit's stitched
    template then covers a different slice of the scan, their local peaks
    differ, and no within-sort merge ever saw them side by side. Such a pair
    shows three signatures at once, which is what this ranks on:

    1. **complementary activity** — low Jaccard overlap of their active-segment
       sets (from the retained per-segment spike counts);
    2. **compatible geometry** — stitched peak electrodes within
       `peak_distance_um` (the prefilter that keeps this O(pairs-that-matter));
    3. **matching footprints where both were measured** — cosine similarity of
       peak-amplitude-per-channel over the channels BOTH units cover, at least
       `min_footprint_cosine`.

    Emits a ranked candidate table for a human (or a later curation capsule)
    to inspect — this diagnostic proposes, it never merges. Cost: peak
    extraction is one pass over units; the pair loop touches only
    geometry-prefiltered pairs.
    """
    weight = np.asarray(contributing_weight)
    positions = np.asarray(positions, dtype=float)[:, :2]
    activity = np.asarray(activity)
    n_units = weight.shape[0]
    if unit_ids is None:
        unit_ids = [str(i) for i in range(n_units)]

    stitched = stitched_templates  # (n_units, n_channels, n_samples), NaN-carrying
    amp = np.empty(weight.shape, dtype=np.float32)  # peak |amplitude| per channel
    for u in range(n_units):
        amp[u] = np.nan_to_num(
            np.abs(np.asarray(stitched[u], dtype=np.float32)).max(axis=1)
        )

    covered = weight > 0
    active_sets = activity > 0
    peak_channel = amp.argmax(axis=1)
    peak_xy = positions[peak_channel]
    alive = covered.any(axis=1) & (active_sets.sum(axis=1) >= int(min_active_segments))

    candidates = []
    n_geometry = 0
    for u in range(n_units):
        if not alive[u]:
            continue
        for v in range(u + 1, n_units):
            if not alive[v]:
                continue
            d = float(np.hypot(*(peak_xy[u] - peak_xy[v])))
            if d > float(peak_distance_um):
                continue
            n_geometry += 1

            both = active_sets[u] | active_sets[v]
            inter = int((active_sets[u] & active_sets[v]).sum())
            jaccard = inter / max(int(both.sum()), 1)

            shared = covered[u] & covered[v]
            if int(shared.sum()) < int(min_shared_channels):
                continue
            a, b = amp[u][shared], amp[v][shared]
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            cosine = float(a @ b / denom) if denom > 0 else 0.0

            if cosine >= float(min_footprint_cosine) and jaccard <= float(max_activity_jaccard):
                candidates.append({
                    "unit_a": str(unit_ids[u]), "unit_b": str(unit_ids[v]),
                    "peak_distance_um": round(d, 1),
                    "footprint_cosine": round(cosine, 4),
                    "activity_jaccard": round(jaccard, 4),
                    "n_shared_channels": int(shared.sum()),
                    "active_segments_a": int(active_sets[u].sum()),
                    "active_segments_b": int(active_sets[v].sum()),
                    "score": round(cosine * (1.0 - jaccard), 4),
                })

    candidates.sort(key=lambda c: c["score"], reverse=True)
    logger.info(
        "missed-merge candidates: %d geometry-close pair(s) examined, %d candidate(s) "
        "at cosine >= %.2f and activity-jaccard <= %.2f",
        n_geometry, len(candidates), min_footprint_cosine, max_activity_jaccard,
    )
    return candidates


def plot_missed_merge_candidates(candidates, stitched_templates, contributing_weight,
                                 positions, unit_ids, out_path, max_pairs=6,
                                 title=None, dpi=170):
    """Top candidate pairs, footprint beside footprint.

    One row per pair: unit A's and unit B's stitched peak-amplitude maps on a
    shared log colour scale, so "same neuron, different segments" is judged by
    eye the way it will eventually be judged in curation review.
    """
    from matplotlib.colors import LogNorm

    if not candidates:
        return None
    shown = candidates[: int(max_pairs)]
    positions = np.asarray(positions, dtype=float)[:, :2]
    id_index = {str(u): i for i, u in enumerate(unit_ids)}

    fig = _new_figure((9.5, 4.0 * len(shown)), dpi)
    axes = np.atleast_2d(fig.subplots(len(shown), 2))

    for row, pair in enumerate(shown):
        peaks = []
        for col, key in enumerate(("unit_a", "unit_b")):
            u = id_index[pair[key]]
            amp = np.nan_to_num(
                np.abs(np.asarray(stitched_templates[u], dtype=np.float32)).max(axis=1)
            )
            peaks.append(amp)
        vmax = max(float(peaks[0].max()), float(peaks[1].max()), 1e-3)
        norm = LogNorm(vmin=max(vmax * 1e-3, 1e-3), vmax=vmax)
        for col, key in enumerate(("unit_a", "unit_b")):
            ax = axes[row, col]
            u = id_index[pair[key]]
            covered = np.asarray(contributing_weight[u]) > 0
            ax.scatter(positions[:, 0], positions[:, 1], s=1, c="0.9",
                       linewidths=0, rasterized=True)
            ax.scatter(positions[covered, 0], positions[covered, 1], s=4,
                       c=np.maximum(peaks[col][covered], 1e-3), cmap="magma",
                       norm=norm, marker="s", linewidths=0, rasterized=True)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(
                f"unit {pair[key]} ({pair['active_segments_' + key[-1]]} seg)",
                fontsize=9,
            )
        axes[row, 0].set_ylabel(
            f"cos {pair['footprint_cosine']:.2f} | jac {pair['activity_jaccard']:.2f}\n"
            f"peaks {pair['peak_distance_um']:.0f} um apart", fontsize=8,
        )

    fig.suptitle(title or "missed-merge candidates: same footprint, different segments",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save_and_release(fig, out_path)


__all__ = [
    "load_retention_index",
    "backbone_channel_ids",
    "segment_activity",
    "backbone_agreement",
    "plot_backbone_agreement",
    "peak_consistency",
    "plot_peak_consistency",
    "plot_unit_peak_map",
    "missed_merge_candidates",
    "plot_missed_merge_candidates",
]
