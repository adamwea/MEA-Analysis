"""Where did every spike go? — post-sort spike accounting.

Four questions about the spikes a sort produced, cheapest first:

  1. DISTRIBUTION (cheap, from the sort). How many spikes did each unit get?
     A dense HD-MEA sort with a long tail of tiny units is the classic
     over-split signature — the thing bombcell's quality gates and SLAy's
     merges exist to repair. This reads the KiloSort output directly (no
     recording needed), so it survives the concat-cache cleanup.

  2. PER-UNIT x PER-SEGMENT (cheap, from the sort + the concat manifest). How
     do each unit's spikes distribute across the scan's segments? `spike_times`
     index the CONCATENATED timeline; the concatenate manifest records each
     segment's [start_frame, end_frame). Binning one against the other gives a
     units x segments matrix — the direct over-split view: a real neuron fires
     across most segments; a fragment lights up in only one or two.

  3. SORTER ACCOUNTING (cheap, from sorter_output). A template-matcher like
     KS4 records, per detected spike, whether it survived detection
     (`kept_spikes.npy`) and which cluster it landed in (`spike_clusters.npy`).
     So the sorter itself answers "were spikes dropped between detection and
     assignment?" — no independent pass required. When the sorter exposes this
     (KS4 does), it is the AUTHORITATIVE loss number; use it in preference to
     any threshold estimate (the caller's stated rule).

  4. INDEPENDENT THRESHOLD DETECTION (expensive, opt-in). The one thing the
     sorter's own numbers cannot tell us: did its DETECTION THRESHOLD miss real
     spikes that never entered the sort? A locally-exclusive MAD threshold pass
     is the independent check — and (Adam, 2026-08-14) it runs on the
     PREPROCESSED recording the sorter actually consumed, not the raw file:
     each segment's preprocessing chain is rebuilt from its descriptor (the
     same `preprocess_segment(**kwargs)` call `concatenate` replays) and sliced
     to the concatenation's common electrodes, so the census sees the identical
     highpass + local-CMR, common-channel signal KiloSort sorted. HONEST
     CAVEAT, baked into the report: a threshold count is still crossings, not
     curated spikes — it includes multi-unit hash and (on stimulation datasets)
     residual stim artifacts, so it OVER-counts. But because the signal now
     matches the sorter's, the ratio of crossings to sorted spikes is a fair
     completeness check. KiloSort's template matching recovers overlapping and
     sub-threshold spikes a naive crossing count misses, so the sorted rate >=
     the census (ratio <= 1) is the HEALTHY expectation — it confirms the sorter
     is not missing threshold-obvious spikes. A census that FAR EXCEEDS the
     sorted rate is usually hash + residual artifacts, but it is the one place a
     detection-threshold miss could hide.

Pure library: no argparse, no printing, no ``__main__``.
"""

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np

from .channel_layout import _new_figure, _save_and_release

logger = logging.getLogger(__name__)

# Independent-detection defaults. A locally-exclusive MAD pass, the same
# primitive a sorter's own detection stage uses; bounded to a few segments
# because a dense scan is many channels x 20 kHz.
DEFAULT_DETECT_THRESHOLD = 5.0     # in MAD units of the per-channel noise
DEFAULT_MAX_DETECT_SEGMENTS = 3


# ---------------------------------------------------------------------------
# 1. Distribution — cheap, no recording.
# ---------------------------------------------------------------------------

def sort_spike_distribution(sorter_output_dir):
    """Per-unit spike-count distribution for the FINAL sorting units.

    Reads the KiloSort output directly (`read_kilosort`, no recording attached,
    so it works after the concat cache is gone). Returns full order statistics,
    the standard low-count-tail bands (the over-split signal), the good/MUA
    label split, and the per-unit counts for the histogram.
    """
    import spikeinterface.extractors as se

    sorter_output_dir = Path(sorter_output_dir)
    sorting = se.read_kilosort(sorter_output_dir, keep_good_only=False)
    counts = np.array(
        list(sorting.count_num_spikes_per_unit().values()), dtype=np.int64
    )
    n_units = counts.size
    fs = float(sorting.get_sampling_frequency())

    labels = _cluster_labels(sorter_output_dir)
    pctl = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    stats = {int(p): int(v) for p, v in zip(pctl, np.percentile(counts, pctl))} if n_units else {}
    bands = {str(t): int((counts < t).sum()) for t in (50, 100, 300, 500, 1000, 2000)}

    return {
        "n_units": int(n_units),
        "sampling_frequency": fs,
        "total_assigned_spikes": int(counts.sum()),
        "min": int(counts.min()) if n_units else None,
        "max": int(counts.max()) if n_units else None,
        "mean": float(counts.mean()) if n_units else None,
        "median": float(np.median(counts)) if n_units else None,
        "std": float(counts.std()) if n_units else None,
        "percentiles": stats,
        "n_units_below": bands,
        "label_counts": labels,
        "_counts": counts,
    }


def _cluster_label_map(sorter_output_dir):
    """cluster id -> label (lower-cased) from cluster_group.tsv (fallback KSLabel).

    Case is normalised so downstream 'good'/'mua' lookups are robust to variant
    TSVs (some tools capitalise the label column).
    """
    for name in ("cluster_group.tsv", "cluster_KSLabel.tsv"):
        path = Path(sorter_output_dir) / name
        if path.is_file():
            out = {}
            for row in path.read_text().splitlines()[1:]:
                parts = row.split("\t")
                if len(parts) >= 2:
                    try:
                        out[int(parts[0])] = parts[-1].strip().lower()
                    except ValueError:
                        continue
            if out:
                return out
    return {}


def _cluster_labels(sorter_output_dir):
    """good/MUA/noise counts (lower-cased) from the cluster label file."""
    label_map = _cluster_label_map(sorter_output_dir)
    return dict(Counter(label_map.values())) if label_map else {}


# ---------------------------------------------------------------------------
# 2. Per-unit x per-segment spike counts — cheap, from the sort + manifest.
# ---------------------------------------------------------------------------

def per_unit_segment_matrix_from_si(sorting_dir, concat_manifest):
    """Like :func:`per_unit_segment_matrix`, but for an SI NumpySorting folder.

    `dense_merge_apply` persists its merged sorting as a SpikeInterface
    NumpySorting (`spikes.npy` structured array with sample_index on the
    concatenated timeline + `numpysorting_info.json` carrying unit_ids), not a
    KiloSort `sorter_output`. Bin the same way so the post-merge re-census is
    directly comparable to the pre-merge one. Returns None on missing/empty
    inputs, like the KiloSort reader.
    """
    d = Path(sorting_dir)
    info_path = d / "numpysorting_info.json"
    spikes_path = d / "spikes.npy"
    if not info_path.is_file() or not spikes_path.is_file():
        return None
    try:
        info = json.loads(info_path.read_text())
        spikes = np.load(spikes_path)
    except (OSError, ValueError) as exc:
        logger.warning("cannot read SI sorting at %s: %s", d, exc)
        return None
    if spikes.size == 0 or "sample_index" not in (spikes.dtype.names or ()):
        return None
    st = np.asarray(spikes["sample_index"], dtype=np.int64)
    # unit_index rows index info["unit_ids"] — map through so the census talks
    # in the sorting's REAL unit ids (post-merge these are surviving member ids)
    unit_id_list = [int(u) for u in info.get("unit_ids", [])]
    if not unit_id_list:
        return None
    sc = np.asarray([unit_id_list[i] for i in spikes["unit_index"]], dtype=np.int64)

    manifest = (
        concat_manifest if isinstance(concat_manifest, dict)
        else json.loads(Path(concat_manifest).read_text())
    )
    segs = manifest.get("segments") or []
    if not segs or any(s.get("start_frame") is None or s.get("end_frame") is None
                       for s in segs):
        return None
    fs = float(manifest.get("fs_hz") or info.get("sampling_frequency") or 20000.0)
    starts = np.array([int(s["start_frame"]) for s in segs], dtype=np.int64)
    ends = np.array([int(s["end_frame"]) for s in segs], dtype=np.int64)
    seg_labels = [str(s.get("rec")) for s in segs]
    order = np.argsort(starts)
    starts, ends = starts[order], ends[order]
    seg_labels = [seg_labels[i] for i in order]
    durations = np.maximum((ends - starts), 1) / fs
    n_seg = len(segs)

    seg_idx = np.clip(np.searchsorted(starts, st, side="right") - 1, 0, n_seg - 1)
    unit_ids = np.unique(sc)
    rows = np.searchsorted(unit_ids, sc)
    matrix = np.zeros((unit_ids.size, n_seg), dtype=np.int64)
    np.add.at(matrix, (rows, seg_idx), 1)
    totals = matrix.sum(axis=1)
    active = (matrix > 0).sum(axis=1)
    n_units = unit_ids.size
    return {
        "n_units": int(n_units),
        "n_segments": int(n_seg),
        "fs_hz": fs,
        "segment_labels": seg_labels,
        "segment_durations_s": [round(float(x), 3) for x in durations],
        "unit_ids": [int(u) for u in unit_ids],
        "unit_labels": [None] * int(n_units),
        "unit_total_spikes": totals.tolist(),
        "unit_active_segments": active.tolist(),
        "median_active_segments": float(np.median(active)) if n_units else None,
        "mean_active_segments": float(active.mean()) if n_units else None,
        "n_units_1_segment": int((active == 1).sum()),
        "n_units_le2_segments": int((active <= 2).sum()),
        "frac_le2_segments": float((active <= 2).mean()) if n_units else None,
        "n_units_all_segments": int((active == n_seg).sum()),
        "matrix_counts": matrix.tolist(),
        "matrix_rate_hz": (matrix / durations[None, :]).tolist(),
    }


def per_unit_segment_matrix(sorter_output_dir, concat_manifest):
    """A units x segments matrix of per-unit spike counts across the scan.

    `spike_times.npy` index the concatenated timeline; the concatenate manifest
    records each segment's `[start_frame, end_frame)`. `searchsorted` places
    every spike in its segment, then a scatter-add builds the matrix. Segment
    lengths differ, so a firing-RATE matrix (Hz) is derived alongside the raw
    counts for a fair per-segment comparison. Returns the matrix, the per-unit
    active-segment count (the over-split signal — a fragment fires in only one
    or two segments), and the good/MUA label per unit.

    `concat_manifest` is the manifest dict or a path to `concat_manifest.json`.
    Returns None when the sort or the manifest is missing/empty.
    """
    d = Path(sorter_output_dir)
    st = _load_npy(d / "spike_times.npy")
    sc = _load_npy(d / "spike_clusters.npy")
    if st is None or sc is None or st.size == 0 or st.size != sc.size:
        return None

    manifest = (
        concat_manifest if isinstance(concat_manifest, dict)
        else json.loads(Path(concat_manifest).read_text())
    )
    segs = manifest.get("segments") or []
    if not segs:
        return None
    # A malformed manifest (segment missing its frame bounds) can't be binned;
    # return None like the other bad-input guards rather than raising KeyError.
    if any(s.get("start_frame") is None or s.get("end_frame") is None for s in segs):
        logger.warning("concat manifest segment missing start/end_frame; "
                       "skipping per-unit x per-segment matrix")
        return None

    fs = float(manifest.get("fs_hz") or 20000.0)
    starts = np.array([int(s["start_frame"]) for s in segs], dtype=np.int64)
    ends = np.array([int(s["end_frame"]) for s in segs], dtype=np.int64)
    seg_labels = [str(s.get("rec")) for s in segs]
    order = np.argsort(starts)  # timeline order, defensive
    starts, ends = starts[order], ends[order]
    seg_labels = [seg_labels[i] for i in order]
    durations = np.maximum((ends - starts), 1) / fs
    n_seg = len(segs)

    seg_idx = np.clip(np.searchsorted(starts, st, side="right") - 1, 0, n_seg - 1)
    unit_ids = np.unique(sc)
    n_units = unit_ids.size
    rows = np.searchsorted(unit_ids, sc)

    matrix = np.zeros((n_units, n_seg), dtype=np.int64)
    np.add.at(matrix, (rows, seg_idx), 1)
    rate = matrix / durations[None, :]

    totals = matrix.sum(axis=1)
    active = (matrix > 0).sum(axis=1)
    label_map = _cluster_label_map(sorter_output_dir)
    unit_labels = [label_map.get(int(u)) for u in unit_ids]

    return {
        "n_units": int(n_units),
        "n_segments": int(n_seg),
        "fs_hz": fs,
        "segment_labels": seg_labels,
        "segment_durations_s": [round(float(x), 3) for x in durations],
        "unit_ids": [int(u) for u in unit_ids],
        "unit_labels": unit_labels,
        "unit_total_spikes": totals.tolist(),
        "unit_active_segments": active.tolist(),
        "median_active_segments": float(np.median(active)) if n_units else None,
        "mean_active_segments": float(active.mean()) if n_units else None,
        "n_units_1_segment": int((active == 1).sum()),
        "n_units_le2_segments": int((active <= 2).sum()),
        "frac_le2_segments": float((active <= 2).mean()) if n_units else None,
        "n_units_all_segments": int((active == n_seg).sum()),
        "matrix_counts": matrix.tolist(),
        "_matrix": matrix,
        "_rate": rate,
        "_totals": totals,
        "_active": active,
    }


def plot_per_unit_segment(mat, out_path, title=None, figsize=(12, 6.0), dpi=140):
    """Left: units x segments firing-rate heatmap (log colour, units sorted by
    total spikes). Right: the histogram of how many segments each unit fires in
    — the over-split money plot (a spike of units at 1-2 segments is fragments).
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    matrix = np.asarray(mat["_matrix"])
    rate = np.asarray(mat["_rate"])
    totals = np.asarray(mat["_totals"])
    active = np.asarray(mat["_active"])
    n_units, n_seg = matrix.shape

    fig = _new_figure(figsize, dpi)
    gs = fig.add_gridspec(1, 2, width_ratios=(2.8, 1.7), wspace=0.38)
    ax_h = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    order = np.argsort(totals)[::-1]  # most-active unit at the top
    shown = rate[order]
    positive = shown[shown > 0]
    vmin = float(positive.min()) if positive.size else 0.1
    vmax = float(shown.max()) if shown.size else 1.0
    masked = np.ma.masked_where(shown <= 0, shown)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#ededed")
    im = ax_h.imshow(
        masked, aspect="auto", cmap=cmap,
        norm=LogNorm(vmin=max(vmin, 1e-3), vmax=max(vmax, vmin * 10)),
        interpolation="nearest",
    )
    ax_h.set_xlabel(f"scan segment #  (1–{n_seg}, in acquisition order)")
    ax_h.set_ylabel(f"unit — one row each, sorted by total spikes  (n={n_units})")
    ax_h.set_title(f"Firing rate of every unit across the {n_seg} scan segments",
                   fontsize=10)
    if n_seg <= 30:
        ax_h.set_xticks(range(n_seg))
        ax_h.set_xticklabels(range(1, n_seg + 1), fontsize=7)  # 1-indexed segments
    cbar = fig.colorbar(im, ax=ax_h, fraction=0.046, pad=0.02)
    cbar.set_label("spikes/s", fontsize=8)

    bins = np.arange(0.5, n_seg + 1.5, 1.0)
    ax_b.hist(active, bins=bins, orientation="horizontal", color="#4C72B0")
    med = float(np.median(active)) if active.size else 0.0
    n_all = int((active == n_seg).sum())
    ax_b.axhline(med, color="#C44E52", ls="--", lw=1.4,
                 label=f"median = {med:.0f} of {n_seg}")
    ax_b.axhline(2.5, color="#DD8452", ls=":", lw=1.1, label="≤2 = localized / fragment")
    ax_b.set_ylim(0.5, n_seg + 0.5)
    ax_b.set_xlabel("number of units")
    # Spell the axis out as a COUNT so its top value (n_seg) is never misread as
    # "segment #n_seg": this is how MANY of the segments a unit fires in.
    ax_b.set_ylabel(f"# of the {n_seg} segments a unit fires in  (0–{n_seg})")
    ax_b.set_title(f"In how many of the {n_seg} segments\nis each unit active?",
                   fontsize=10)
    if n_units:
        ax_b.annotate(
            f"{n_all} of {n_units} units\nfire in ALL {n_seg} segments",
            xy=(0.85, n_seg), xycoords=("axes fraction", "data"),
            xytext=(0.5, 0.5), textcoords="axes fraction",
            ha="center", va="center", fontsize=7.5, color="#333333",
            arrowprops=dict(arrowstyle="->", color="#999999", lw=0.8),
        )
    ax_b.legend(fontsize=7, loc="lower right")

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95 if title else 1.0))
    return _save_and_release(fig, out_path)


# ---------------------------------------------------------------------------
# 3. Sorter accounting — cheap, from the sorter's own arrays.
# ---------------------------------------------------------------------------

def ks_detection_accounting(sorter_output_dir):
    """Detected -> kept -> assigned, from the sorter's own arrays.

    `spike_times.npy` / `spike_clusters.npy` are every spike the sorter carried
    to assignment; `kept_spikes.npy` (KS4) is the survival mask from its
    detection/dedup stage, so `~kept` is what the sorter dropped BEFORE
    assignment. The mask may arrive as bool or as a 0/1 integer array (numpy /
    matlab round-trips vary); both are honoured, and a non-mask array is logged
    and ignored rather than silently discarded. Returns the counts plus a
    `lost_fraction` the report reads. When NO source array loads at all,
    `lost_fraction` is None ("unknown"), not 0.0 ("confirmed no loss").
    """
    d = Path(sorter_output_dir)
    st = _load_npy(d / "spike_times.npy")
    kept = _load_npy(d / "kept_spikes.npy")
    sc = _load_npy(d / "spike_clusters.npy")

    mask = _as_bool_mask(kept, d / "kept_spikes.npy")

    n_assigned = int(st.size) if st is not None else None
    n_detected = None
    n_dropped = None
    if mask is not None:
        n_detected = int(mask.size)
        n_dropped = int((~mask).sum())
    elif st is not None:
        # No survival mask exposed: detected == what reached assignment.
        n_detected = int(st.size)
        n_dropped = None

    if n_dropped is not None and n_detected:
        lost_fraction = float(n_dropped / n_detected)
    elif n_detected is None and n_assigned is None:
        lost_fraction = None  # nothing loaded — unknown, not zero
    else:
        lost_fraction = 0.0
    n_clusters = int(np.unique(sc).size) if sc is not None else None
    return {
        "source": "kilosort_kept_spikes" if mask is not None else "kilosort_spike_times",
        "n_detected": n_detected,
        "n_assigned_to_units": n_assigned,
        "n_dropped_pre_assign": n_dropped,
        "lost_fraction": lost_fraction,
        "n_clusters_in_spike_clusters": n_clusters,
    }


def _as_bool_mask(arr, path):
    """Coerce a survival array to a boolean mask, or None if it is not one.

    Accepts a real bool array, or a 0/1 integer array (a common save-format
    variant). Anything else (floats, values outside {0,1}) is not a mask: it is
    warned about and ignored, so a real mask is never dropped in silence.
    """
    if arr is None:
        return None
    if arr.dtype == bool:
        return arr
    if arr.dtype.kind in "iu":
        # An empty int array is a (degenerate) mask; a non-empty one is a mask
        # only if every value is in {0, 1}. Guard the min/max against size 0.
        if arr.size == 0 or (int(arr.min()) >= 0 and int(arr.max()) <= 1):
            return arr.astype(bool)
    logger.warning(
        "%s loaded but is not a boolean/0-1 survival mask (dtype=%s); ignoring it",
        path, arr.dtype,
    )
    return None


def _load_npy(path):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return np.load(path).ravel()
    except Exception as exc:  # noqa: BLE001 - a missing/odd file must not sink the audit
        logger.warning("could not load %s: %s", path, exc)
        return None


def plot_spike_distribution(dist, out_path, title=None, figsize=(11, 4.0), dpi=140):
    """Two panels: the per-unit spike-count histogram (log x) with the low-count
    bands marked, and the good/MUA/noise label split."""
    counts = np.asarray(dist["_counts"])
    fig = _new_figure(figsize, dpi)
    ax1, ax2 = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)

    if counts.size:
        ax1.hist(np.log10(np.maximum(counts, 1)), bins=40, color="#4C72B0")
        med = float(np.median(counts))
        ax1.axvline(np.log10(max(med, 1)), color="#C44E52", ls="--", lw=1.5,
                    label=f"median {med:.0f}")
        ax1.axvline(np.log10(500), color="#DD8452", ls=":", lw=1.2, label="500")
        ax1.legend(fontsize=8)
    ax1.set_xlabel("log10(spikes per unit)")
    ax1.set_ylabel("units")
    ax1.set_title(f"per-unit spike counts (n={dist['n_units']})", fontsize=10)

    labels = dist.get("label_counts") or {}
    if labels:
        keys = list(labels.keys())
        ax2.bar(keys, [labels[k] for k in keys], color="#55A868")
        for i, k in enumerate(keys):
            ax2.annotate(str(labels[k]), (i, labels[k]), ha="center",
                         va="bottom", fontsize=9)
    ax2.set_ylabel("units")
    ax2.set_title("sorter labels", fontsize=10)

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95 if title else 1.0))
    return _save_and_release(fig, out_path)


# ---------------------------------------------------------------------------
# 4. Independent threshold detection — on the PREPROCESSED signal, opt-in.
# ---------------------------------------------------------------------------

def threshold_detection_census(h5_path, manifest, *,
                               detect_threshold=DEFAULT_DETECT_THRESHOLD,
                               max_segments=DEFAULT_MAX_DETECT_SEGMENTS,
                               n_jobs=4, progress=None):
    """Independent MAD threshold-crossing census on the PREPROCESSED recording.

    For up to `max_segments` of the concatenation's segments (from the concat
    manifest), rebuild the preprocessing chain from that segment's descriptor
    (`preprocess_segment(**kwargs)` — the same call `concatenate` replays),
    slice to the concatenation's common electrodes, and run a locally-exclusive
    MAD `detect_peaks`. The census therefore sees the same highpass +
    local-CMR, common-channel signal KiloSort sorted — not the raw file. For a
    concatenation written before the chain's filters read a settling margin
    (2026-09-21) the replay differs from what was sorted by up to a few
    microvolts within ~45 ms of each of that binary's chunk joins: the margin
    is part of the code, not of the recorded settings.

    Returns per-segment crossing counts + the pooled rate. This is a COARSE
    completeness context, not a curated spike count (see the module docstring):
    crossings include multi-unit hash and residual stim artifacts and exceed the
    sorted rate. Bounded on purpose — `max_segments` caps the cost and segments
    are processed one at a time.
    """
    from spikeinterface.sortingcomponents.peak_detection import detect_peaks

    from mea_modules.io import load_segment
    from mea_modules.preprocessing import preprocess_segment

    segments = list(manifest.get("segments") or [])[:max_segments]
    common = [int(e) for e in (manifest.get("common_electrodes") or [])]
    well = manifest.get("well")

    per_segment = []
    total_crossings = 0
    total_seconds = 0.0
    total_channels = 0
    for seg in segments:
        rec_name = seg.get("rec")
        try:
            descriptor = json.loads(Path(seg["descriptor"]).read_text())
            rec = load_segment(
                h5_path, stream_id=well, rec_name=rec_name,
                hdf5_plugin_path=descriptor.get("hdf5_plugin_path"),
            )
            rec = preprocess_segment(rec, **dict(descriptor["preprocessing"]))
            rec = _select_common_channels(rec, common)
            peaks = detect_peaks(
                rec, method="locally_exclusive", peak_sign="neg",
                detect_threshold=detect_threshold, exclude_sweep_ms=0.5,
                n_jobs=n_jobs, chunk_duration="1s", progress_bar=False,
            )
        except Exception as exc:  # noqa: BLE001 - a bad segment must not sink the census
            logger.warning("detection: segment %s failed: %s", rec_name, exc)
            continue
        dur = float(rec.get_total_duration())
        nch = int(rec.get_num_channels())
        n = int(len(peaks))
        per_segment.append({
            "rec": rec_name, "n_crossings": n, "duration_s": round(dur, 2),
            "n_channels": nch,
            "hz_per_channel": round(n / dur / nch, 3) if dur and nch else None,
        })
        total_crossings += n
        total_seconds += dur
        total_channels = max(total_channels, nch)
        if progress is not None:
            progress(f"detected {n} crossings in {rec_name} ({dur:.1f}s)")

    rate = (total_crossings / total_seconds) if total_seconds else None
    return {
        "signal": "preprocessed_recording",
        "detect_threshold_mad": float(detect_threshold),
        "n_common_channels": len(common),
        "n_segments_detected": len(per_segment),
        "total_crossings": int(total_crossings),
        "total_seconds": round(total_seconds, 2),
        "crossings_per_second": round(rate, 1) if rate is not None else None,
        "hz_per_channel": round(total_crossings / total_seconds / total_channels, 3)
                          if (total_seconds and total_channels) else None,
        "per_segment": per_segment,
    }


def _select_common_channels(rec, common):
    """Slice a rebuilt segment to the concatenation's common electrodes.

    `preprocess_segment` re-keys channels to electrode ids, so the common set
    (electrode numbers) selects directly. Mirrors `concatenate`'s own resolve so
    the census channel set matches the sorted recording's exactly.
    """
    if not common:
        return rec
    by_value = {int(cid): cid for cid in rec.get_channel_ids()}
    want = [by_value[e] for e in common if int(e) in by_value]
    if not want:
        # No common electrode is in this segment's channel set — detecting on
        # the FULL channel set would inflate the crossing count and break
        # comparability with the sorted (common-channel) recording. Flag it
        # loudly rather than silently census a different channel set.
        logger.warning(
            "detection: none of the %d common electrodes are in this segment's "
            "channel set; censusing its full %d-channel set instead",
            len(common), rec.get_num_channels(),
        )
        return rec
    return rec.select_channels(want)


def compare_detection_to_sort(detection, dist, accounting, *, duration_s=None):
    """Fold the sources into one comparison the report can render.

    Preference order for "spikes lost", per the caller's rule: the sorter's own
    pre-assignment loss (KS4 kept_spikes) is AUTHORITATIVE; the independent
    threshold census on the preprocessed signal is a completeness check. When
    the recording duration is known, the sorted spikes/s and the crossings/s are
    both whole-population per-second rates, so their ratio is comparable.
    """
    detection = detection or {}
    crossings_per_s = detection.get("crossings_per_second")
    total_assigned = (dist or {}).get("total_assigned_spikes")
    sorted_per_s = (
        float(total_assigned) / duration_s
        if (total_assigned and duration_s) else None
    )
    ratio = (
        crossings_per_s / sorted_per_s
        if (crossings_per_s and sorted_per_s) else None
    )
    return {
        "authoritative_lost_fraction": accounting.get("lost_fraction"),
        "authoritative_source": accounting.get("source"),
        "n_dropped_pre_assign": accounting.get("n_dropped_pre_assign"),
        "signal": detection.get("signal"),
        "independent_crossings_per_second": crossings_per_s,
        "independent_hz_per_channel": detection.get("hz_per_channel"),
        "sorted_spikes_per_second": round(sorted_per_s, 1) if sorted_per_s else None,
        "crossings_to_sorted_ratio": round(ratio, 2) if ratio else None,
        "note": (
            "Authoritative loss = the sorter's own detected-but-dropped count. "
            "The independent census runs on the PREPROCESSED signal the sorter "
            "saw (highpass + local-CMR, common electrodes). KiloSort's template "
            "matching recovers overlapping and sub-threshold spikes a naive "
            "crossing count misses, so the sorted rate >= the census (ratio "
            "<= 1) is the healthy expectation and confirms the sorter is not "
            "missing threshold-obvious spikes. A census far ABOVE the sorted "
            "rate is usually hash + residual artifacts, but is the one place a "
            "detection-threshold miss could hide."
        ),
    }


def accounting_json_safe(dist):
    """Drop the leading-underscore array fields so a dict is JSON-dumpable."""
    return {k: v for k, v in dist.items() if not k.startswith("_")}
