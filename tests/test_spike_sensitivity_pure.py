"""Toy-data tests for mea_modules.diagnostics.spike_sensitivity.

Covers the pure numeric logic that produces the capsule's production numbers —
the spike census + starved roster, and recommend_default's crossing/candidate
math — without needing a real SortingAnalyzer. The convergence_sweep integration
(which loads analyzers) is exercised by the capsule run itself.
"""
import numpy as np
import pytest

from mea_modules.diagnostics import spike_sensitivity as ss


def _write_contributions(tmp_path, counts):
    """Write a synthetic `segment_contributions/segment_*/spike_counts.npy` tree.

    `counts` is (n_units, n_segments); each column becomes one segment's file.
    """
    root = tmp_path / "segment_contributions"
    root.mkdir()
    for s in range(counts.shape[1]):
        d = root / f"segment_{s:03d}"
        d.mkdir()
        np.save(d / "spike_counts.npy", counts[:, s].astype(np.int64))
    return root


def test_spike_census_ranks_starved_worst_first(tmp_path):
    counts = np.array([
        [500, 500, 500, 500],   # unit 10: rich, per-active median 500
        [100, 0, 120, 80],      # unit 11: per-active median 100
        [2, 3, 0, 0],           # unit 12: per-active median 2.5 (starved)
    ])
    contrib = _write_contributions(tmp_path, counts)
    census = ss.spike_census(contrib, unit_ids=[10, 11, 12], starved_threshold=100)

    assert census["n_units"] == 3 and census["n_segments"] == 4
    assert int(census["_total_spikes"][0]) == 2000
    # threshold is strict-less-than: unit 11's median (100) is NOT below 100.
    starved_ids = [r["unit_id"] for r in census["starved"]]
    assert starved_ids == [12]            # only the truly-starved unit, and it ranks first
    assert census["n_spike_starved_units"] == 1


def test_spike_census_empty_does_not_raise(tmp_path):
    contrib = _write_contributions(tmp_path, np.zeros((0, 3), dtype=np.int64))
    census = ss.spike_census(contrib)     # np.percentile on empty would raise without the guard
    assert census["n_units"] == 0
    assert census["total_spikes_deciles"] == []


def test_recommend_default_takes_max_of_candidates():
    curve = [
        {"n_spikes": 12, "median_corr": 0.70, "frac_converged": 0.0, "n_unit_segments": 10},
        {"n_spikes": 100, "median_corr": 0.96, "frac_converged": 0.2, "n_unit_segments": 8},
        {"n_spikes": 350, "median_corr": 0.995, "frac_converged": 0.8, "n_unit_segments": 4},
    ]
    sweep = {"grid": [12, 100, 350], "corr_target": 0.95, "strict_corr_target": 0.99,
             "n_unit_segments": 10, "curve": curve, "enough_N": {"p90": 200}}
    rec = ss.recommend_default(sweep)
    assert rec["median_soft_plateau_N"] == 100    # first median >= 0.95
    assert rec["median_strict_plateau_N"] == 350  # first median >= 0.99
    assert rec["recommended_max_spikes_per_unit"] == 350  # max(100, 350, 200)


def test_recommend_default_refuses_a_number_from_no_data():
    sweep = {"grid": [12, 100], "n_unit_segments": 0, "curve": []}
    rec = ss.recommend_default(sweep)
    assert rec["recommended_max_spikes_per_unit"] is None


def test_json_safe_helpers_drop_underscore_arrays():
    d = {"keep": 1, "_arr": np.zeros(3)}
    assert ss.census_json_safe(d) == {"keep": 1}
    assert ss.sweep_json_safe(d) == {"keep": 1}


# ---------------------------------------------------------------------------
# spike_census_from_counts — the numeric core, fed a raw count matrix (the way
# the capsule now feeds it, from per_unit_segment_matrix's matrix_counts).
# ---------------------------------------------------------------------------


def test_spike_census_from_counts_ranks_starved_worst_first():
    counts = np.array([
        [500, 500, 500, 500],   # unit 10: rich, per-active median 500
        [100, 0, 120, 80],      # unit 11: per-active median 100
        [2, 3, 0, 0],           # unit 12: per-active median 2.5 (starved)
    ])
    census = ss.spike_census_from_counts(counts, unit_ids=[10, 11, 12], starved_threshold=100)

    assert census["n_units"] == 3 and census["n_segments"] == 4
    assert int(census["_total_spikes"][0]) == 2000
    # threshold is strict-less-than: unit 11's median (100) is NOT below 100.
    starved_ids = [r["unit_id"] for r in census["starved"]]
    assert starved_ids == [12]
    assert census["n_spike_starved_units"] == 1


def test_spike_census_from_counts_accepts_a_list_matrix():
    # matrix_counts arrives as a list-of-lists (JSON-safe) from
    # per_unit_segment_matrix; the core np.asarray's it internally.
    counts = [[10, 20, 30], [1, 0, 2]]
    census = ss.spike_census_from_counts(counts, starved_threshold=5)
    assert census["n_units"] == 2 and census["n_segments"] == 3
    assert census["n_spike_starved_units"] == 1  # only the [1,0,2] unit


def test_spike_census_from_counts_empty_does_not_raise():
    census = ss.spike_census_from_counts(np.zeros((0, 3), dtype=np.int64))
    assert census["n_units"] == 0
    assert census["total_spikes_deciles"] == []


def test_spike_census_from_counts_rejects_non_2d():
    with pytest.raises(ValueError):
        ss.spike_census_from_counts(np.zeros(5, dtype=np.int64))


def test_spike_census_wrapper_matches_core(tmp_path):
    # The file-backed spike_census stacks segment_*/spike_counts.npy then
    # delegates to spike_census_from_counts; the two paths must agree.
    counts = np.array([
        [500, 500, 500, 500],
        [100, 0, 120, 80],
        [2, 3, 0, 0],
    ])
    contrib = _write_contributions(tmp_path, counts)
    from_files = ss.spike_census(contrib, unit_ids=[10, 11, 12], starved_threshold=100)
    from_core = ss.spike_census_from_counts(counts, unit_ids=[10, 11, 12], starved_threshold=100)
    assert from_files["n_spike_starved_units"] == from_core["n_spike_starved_units"]
    assert ([r["unit_id"] for r in from_files["starved"]]
            == [r["unit_id"] for r in from_core["starved"]])


# ---------------------------------------------------------------------------
# next_climb_grid — the adaptive-climb step helper (extend the grid, no rebuild).
# ---------------------------------------------------------------------------


def test_next_climb_grid_extends_when_unstable_under_cap():
    assert ss.next_climb_grid([12, 100, 500], n_unstable=3, max_cap=2000) == [12, 100, 500, 1000]


def test_next_climb_grid_clamps_the_appended_point_to_the_cap():
    # doubling 1500 would be 3000 > 2000, so the appended point clamps to the cap.
    assert ss.next_climb_grid([500, 1500], n_unstable=2, max_cap=2000) == [500, 1500, 2000]


def test_next_climb_grid_stops_when_no_unstable():
    # spike-starved data: nothing reached the top grid point, so n_unstable == 0
    # and the climb never fires.
    assert ss.next_climb_grid([12, 100, 500], n_unstable=0, max_cap=2000) is None


def test_next_climb_grid_stops_at_cap():
    assert ss.next_climb_grid([1000, 2000], n_unstable=5, max_cap=2000) is None


def test_next_climb_grid_handles_none_unstable():
    assert ss.next_climb_grid([12, 100], n_unstable=None, max_cap=2000) is None
