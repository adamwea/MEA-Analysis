"""Toy-data tests for mea_modules.diagnostics.spike_accounting.

Covers the pure logic the post_sort_spike_audit capsule reads — the sorter
accounting (detected -> kept -> assigned, including the mask-dtype variants),
the good/MUA label counting, the per-unit x per-segment matrix, and the
detection/sort comparison — with synthetic `.npy`/`.tsv` fixtures. The two
integration points (`sort_spike_distribution` via `read_kilosort`,
`threshold_detection_census` via spikeinterface) are exercised by the capsule
run itself.
"""
import json

import numpy as np
import pytest

from mea_modules.diagnostics import spike_accounting as sa


def _write_sort(tmp_path, times, clusters, *, kept=None, labels=None):
    """Write a minimal sorter_output tree the pure functions read."""
    d = tmp_path / "sorter_output"
    d.mkdir()
    np.save(d / "spike_times.npy", np.asarray(times, dtype=np.int64))
    np.save(d / "spike_clusters.npy", np.asarray(clusters, dtype=np.int32))
    if kept is not None:
        np.save(d / "kept_spikes.npy", np.asarray(kept))
    if labels is not None:
        rows = ["cluster_id\tgroup"] + [f"{cid}\t{grp}" for cid, grp in labels]
        (d / "cluster_group.tsv").write_text("\n".join(rows) + "\n")
    return d


# --- sorter accounting: the mask-dtype matrix ------------------------------

def test_accounting_bool_mask_is_authoritative(tmp_path):
    d = _write_sort(tmp_path, times=[1, 2, 3, 4],
                    clusters=[0, 0, 1, 1],
                    kept=np.array([True, True, False, True, True]))
    acc = sa.ks_detection_accounting(d)
    assert acc["source"] == "kilosort_kept_spikes"
    assert acc["n_detected"] == 5
    assert acc["n_assigned_to_units"] == 4
    assert acc["n_dropped_pre_assign"] == 1
    assert acc["lost_fraction"] == pytest.approx(1 / 5)


def test_accounting_int_0_1_mask_is_coerced(tmp_path):
    # A 0/1 uint8 mask (a common save-format variant) must NOT be silently
    # dropped into the "no mask" branch — MEDIUM-2 from the 2026-08-14 verifier.
    d = _write_sort(tmp_path, times=[1, 2, 3, 4],
                    clusters=[0, 0, 1, 1],
                    kept=np.array([1, 1, 0, 1, 1], dtype=np.uint8))
    acc = sa.ks_detection_accounting(d)
    assert acc["source"] == "kilosort_kept_spikes"
    assert acc["n_detected"] == 5
    assert acc["n_dropped_pre_assign"] == 1
    assert acc["lost_fraction"] == pytest.approx(1 / 5)


def test_accounting_non_mask_array_is_ignored(tmp_path):
    # Values outside {0,1} are not a survival mask: fall back to spike_times,
    # do not misread it as a mask.
    d = _write_sort(tmp_path, times=[1, 2, 3, 4],
                    clusters=[0, 0, 1, 1],
                    kept=np.array([2, 3, 0, 1, 1], dtype=np.int64))
    acc = sa.ks_detection_accounting(d)
    assert acc["source"] == "kilosort_spike_times"
    assert acc["n_detected"] == 4
    assert acc["n_dropped_pre_assign"] is None
    assert acc["lost_fraction"] == 0.0


def test_accounting_nothing_loaded_is_unknown_not_zero(tmp_path):
    # LOW-1: a sorter_output with no spike arrays reads as UNKNOWN loss, not
    # "confirmed zero loss".
    d = tmp_path / "sorter_output"
    d.mkdir()
    acc = sa.ks_detection_accounting(d)
    assert acc["n_detected"] is None
    assert acc["n_assigned_to_units"] is None
    assert acc["lost_fraction"] is None


def test_as_bool_mask_dtypes():
    assert sa._as_bool_mask(np.array([True, False]), "p").tolist() == [True, False]
    assert sa._as_bool_mask(np.array([1, 0, 1], dtype=np.uint8), "p").tolist() == [True, False, True]
    assert sa._as_bool_mask(np.array([0.0, 1.0]), "p") is None      # float: not a mask
    assert sa._as_bool_mask(np.array([2, 0], dtype=np.int64), "p") is None  # out of {0,1}
    assert sa._as_bool_mask(None, "p") is None
    # An empty int array must not call min()/max() (ValueError); treat as a mask.
    empty = sa._as_bool_mask(np.array([], dtype=np.int64), "p")
    assert empty is not None and empty.dtype == bool and empty.size == 0


# --- label counting: case normalisation ------------------------------------

def test_cluster_labels_lowercased(tmp_path):
    # LOW-2: variant TSVs may capitalise the label column.
    d = _write_sort(tmp_path, times=[1, 2, 3], clusters=[0, 1, 2],
                    labels=[(0, "Good"), (1, "MUA"), (2, "good")])
    assert sa._cluster_labels(d) == {"good": 2, "mua": 1}
    assert sa._cluster_label_map(d) == {0: "good", 1: "mua", 2: "good"}


# --- per-unit x per-segment matrix -----------------------------------------

def _manifest(segments, fs=10.0, well="well000"):
    return {
        "well": well, "fs_hz": fs,
        "duration_s": max(s["end_frame"] for s in segments) / fs,
        "segments": segments,
        "common_electrodes": [754, 755],
    }


def test_per_unit_segment_matrix_bins_and_flags_fragments(tmp_path):
    # unit 0 fires in both segments (times 10,20 in seg0; 150 in seg1),
    # unit 1 fires only in seg0 (time 30) -> a 1-segment fragment.
    d = _write_sort(tmp_path, times=[10, 20, 150, 30], clusters=[0, 0, 0, 1])
    manifest = _manifest([
        {"rec": "rec0000", "start_frame": 0, "end_frame": 100},
        {"rec": "rec0001", "start_frame": 100, "end_frame": 200},
    ])
    mat = sa.per_unit_segment_matrix(d, manifest)
    assert mat["n_units"] == 2 and mat["n_segments"] == 2
    assert mat["matrix_counts"] == [[2, 1], [1, 0]]
    assert mat["unit_active_segments"] == [2, 1]
    assert mat["n_units_1_segment"] == 1
    assert mat["n_units_le2_segments"] == 2
    assert mat["frac_le2_segments"] == pytest.approx(1.0)
    assert mat["median_active_segments"] == pytest.approx(1.5)
    # firing-RATE array respects the per-segment duration (100 frames / 10 Hz = 10 s)
    assert np.asarray(mat["_rate"])[0, 0] == pytest.approx(0.2)


def test_per_unit_segment_matrix_accepts_manifest_path(tmp_path):
    d = _write_sort(tmp_path, times=[5, 105], clusters=[0, 0])
    manifest = _manifest([
        {"rec": "rec0000", "start_frame": 0, "end_frame": 100},
        {"rec": "rec0001", "start_frame": 100, "end_frame": 200},
    ])
    mpath = tmp_path / "concat_manifest.json"
    mpath.write_text(json.dumps(manifest))
    mat = sa.per_unit_segment_matrix(d, mpath)
    assert mat["matrix_counts"] == [[1, 1]]


def test_per_unit_segment_matrix_none_on_empty(tmp_path):
    d = _write_sort(tmp_path, times=[], clusters=[])
    assert sa.per_unit_segment_matrix(d, _manifest([
        {"rec": "r", "start_frame": 0, "end_frame": 100}])) is None


def test_per_unit_segment_matrix_none_on_malformed_manifest(tmp_path):
    # A segment missing its frame bounds must return None, not raise KeyError.
    d = _write_sort(tmp_path, times=[5, 105], clusters=[0, 0])
    bad = {"well": "well000", "fs_hz": 10.0,
           "segments": [{"rec": "r0", "start_frame": 0}]}  # no end_frame
    assert sa.per_unit_segment_matrix(d, bad) is None


# --- common-electrode channel selection (detection census) ------------------

class _StubRec:
    def __init__(self, channel_ids):
        self._ids = list(channel_ids)
    def get_channel_ids(self):
        return self._ids
    def get_num_channels(self):
        return len(self._ids)
    def select_channels(self, ids):
        return ("selected", list(ids))


def test_select_common_channels_slices_to_common():
    rec = _StubRec([754, 755, 800])
    assert sa._select_common_channels(rec, [755, 754]) == ("selected", [755, 754])


def test_select_common_channels_empty_intersection_returns_rec():
    # No common electrode present -> fall back to the full set (warned), not a crash.
    rec = _StubRec([1, 2, 3])
    assert sa._select_common_channels(rec, [754, 755]) is rec


# --- comparison + json safety ----------------------------------------------

def test_compare_detection_to_sort_computes_ratio():
    detection = {"signal": "preprocessed_recording",
                 "crossings_per_second": 1000.0, "hz_per_channel": 2.0}
    dist = {"total_assigned_spikes": 2800}
    accounting = {"lost_fraction": 0.01, "source": "kilosort_kept_spikes",
                  "n_dropped_pre_assign": 28}
    cmp = sa.compare_detection_to_sort(detection, dist, accounting, duration_s=280.0)
    assert cmp["sorted_spikes_per_second"] == pytest.approx(10.0)
    assert cmp["crossings_to_sorted_ratio"] == pytest.approx(100.0)
    assert cmp["authoritative_lost_fraction"] == 0.01
    assert cmp["signal"] == "preprocessed_recording"


def test_compare_detection_to_sort_no_duration_no_ratio():
    cmp = sa.compare_detection_to_sort(
        {"crossings_per_second": 500.0}, {"total_assigned_spikes": 2800},
        {"lost_fraction": 0.0}, duration_s=None)
    assert cmp["sorted_spikes_per_second"] is None
    assert cmp["crossings_to_sorted_ratio"] is None


def test_accounting_json_safe_strips_underscore_keys():
    d = {"n_units": 3, "_counts": np.array([1, 2, 3]), "median": 2.0}
    safe = sa.accounting_json_safe(d)
    assert safe == {"n_units": 3, "median": 2.0}
    json.dumps(safe)  # must not raise


def test_load_npy_missing_is_none(tmp_path):
    assert sa._load_npy(tmp_path / "nope.npy") is None
    p = tmp_path / "a.npy"
    np.save(p, np.array([[1, 2], [3, 4]]))
    assert sa._load_npy(p).tolist() == [1, 2, 3, 4]  # ravelled
