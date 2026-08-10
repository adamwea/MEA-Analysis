"""Pure-arithmetic tests for mea_modules.curation.dense_merge.

The load-bearing one is the exactness theorem: under spike_count averaging,
the closed-form per-channel merge of stitched templates equals a direct
re-derivation from the pooled per-(unit, segment) contributions -- the
property capsule 18 relies on to merge dense templates WITHOUT re-stitching
(round-2 planning, TOPOLOGY-RESOLVED block; revision R14 flipped capsule
10's default to spike_count so this holds on real runs).
"""

import numpy as np
import pytest

from mea_modules.curation import (
    dedup_coincident_spikes,
    merge_group_dense,
    plan_merge_output,
)

# --------------------------------------------------------------------------- #
# Synthetic stitch fixture
# --------------------------------------------------------------------------- #
# Two units, four channels, two samples, three "segments" with different
# channel routings and spike counts. NaN/0-weight where a segment never
# measured a (unit, channel).
#
#   channel:        c0      c1      c2      c3
#   unit A seg1 (n=4):  routed  routed  --      --
#   unit A seg2 (n=1):  --      routed  routed  --
#   unit B seg2 (n=3):  --      routed  routed  --
#   unit B seg3 (n=2):  --      --      routed  routed
#
# So after a spike_count stitch: A covers c0..c2, B covers c1..c3; the merge
# of {A, B} covers all four with divergent per-channel weights.

SEGMENTS = [
    # (unit, segment, spike_count, {channel: waveform (2 samples)})
    ("A", "s1", 4, {0: [1.0, 2.0], 1: [3.0, 1.0]}),
    ("A", "s2", 1, {1: [5.0, 3.0], 2: [2.0, 4.0]}),
    ("B", "s2", 3, {1: [7.0, 5.0], 2: [6.0, 2.0]}),
    ("B", "s3", 2, {2: [1.0, 1.0], 3: [4.0, 8.0]}),
]
UNITS = ["A", "B"]
N_CHANNELS = 4
N_SAMPLES = 2


def spike_count_stitch():
    """The stitch capsule's arithmetic, applied to the fixture directly."""
    templates = np.full((len(UNITS), N_CHANNELS, N_SAMPLES), np.nan)
    weights = np.zeros((len(UNITS), N_CHANNELS))
    for u_index, unit in enumerate(UNITS):
        for channel in range(N_CHANNELS):
            num = np.zeros(N_SAMPLES)
            den = 0.0
            for seg_unit, _seg, count, waves in SEGMENTS:
                if seg_unit == unit and channel in waves:
                    num += count * np.asarray(waves[channel])
                    den += count
            if den > 0:
                templates[u_index, channel] = num / den
                weights[u_index, channel] = den
    return templates, weights


def pooled_rederivation(channel):
    """Ground truth: spike-count-weighted mean over ALL contributions of both
    units that measured `channel` -- what re-stitching the combined unit
    would compute."""
    num = np.zeros(N_SAMPLES)
    den = 0.0
    for _unit, _seg, count, waves in SEGMENTS:
        if channel in waves:
            num += count * np.asarray(waves[channel])
            den += count
    return (num / den if den > 0 else np.full(N_SAMPLES, np.nan)), den


def test_merge_is_exact_under_spike_count_weighting():
    templates, weights = spike_count_stitch()
    merged, merged_weight = merge_group_dense(templates, weights, [0, 1])
    for channel in range(N_CHANNELS):
        expected, expected_weight = pooled_rederivation(channel)
        np.testing.assert_allclose(
            merged[channel], expected, rtol=1e-15, atol=0,
            err_msg=f"channel {channel}",
        )
        assert merged_weight[channel] == expected_weight


def test_divergent_coverage_channel_is_passthrough_bitwise():
    """A channel only one member covers must come out as that member's value
    EXACTLY -- the soma-axon footprint case SLAy's global average diluted."""
    templates, weights = spike_count_stitch()
    merged, merged_weight = merge_group_dense(templates, weights, [0, 1])
    # c0 is unit A only; c3 is unit B only.
    assert np.array_equal(merged[0], templates[0, 0])
    assert merged_weight[0] == weights[0, 0]
    assert np.array_equal(merged[3], templates[1, 3])
    assert merged_weight[3] == weights[1, 3]


def test_uncovered_channel_stays_nan_with_zero_weight():
    templates, weights = spike_count_stitch()
    # Add a channel neither unit covers.
    templates = np.concatenate(
        [templates, np.full((2, 1, N_SAMPLES), np.nan)], axis=1
    )
    weights = np.concatenate([weights, np.zeros((2, 1))], axis=1)
    merged, merged_weight = merge_group_dense(templates, weights, [0, 1])
    assert np.isnan(merged[-1]).all()
    assert merged_weight[-1] == 0.0


def test_shared_channel_is_weighted_average():
    templates, weights = spike_count_stitch()
    merged, _ = merge_group_dense(templates, weights, [0, 1])
    # c1: A = (4*[3,1] + 1*[5,3])/5 = [3.4, 1.4] with w=5; B = [7,5] w=3.
    np.testing.assert_allclose(templates[0, 1], [3.4, 1.4], rtol=1e-15)
    expected = (5 * np.array([3.4, 1.4]) + 3 * np.array([7.0, 5.0])) / 8
    np.testing.assert_allclose(merged[1], expected, rtol=1e-15)


def test_multiway_group():
    """Three synthetic units; associativity of the closed form."""
    rng = np.random.default_rng(7)
    weights = rng.integers(0, 5, size=(3, 6)).astype(float)
    templates = rng.normal(size=(3, 6, 4))
    templates[weights == 0] = np.nan
    merged, merged_weight = merge_group_dense(templates, weights, [0, 1, 2])
    for channel in range(6):
        w = weights[:, channel]
        if w.sum() == 0:
            assert np.isnan(merged[channel]).all()
            continue
        expected = (
            np.nansum(np.nan_to_num(templates[:, channel]) * w[:, None], axis=0)
            / w.sum()
        )
        np.testing.assert_allclose(merged[channel], expected, rtol=1e-14)
        assert merged_weight[channel] == w.sum()


def test_singleton_group_is_identity():
    templates, weights = spike_count_stitch()
    merged, merged_weight = merge_group_dense(templates, weights, [1])
    covered = weights[1] > 0
    assert np.array_equal(merged[covered], templates[1, covered])
    assert np.isnan(merged[~covered]).all()
    assert np.array_equal(merged_weight, weights[1])


def test_mismatched_nan_weight_raises():
    templates, weights = spike_count_stitch()
    bad = weights.copy()
    bad[0, 0] = 0.0  # template finite there -> contradiction
    with pytest.raises(ValueError, match="matched pair"):
        merge_group_dense(templates, bad, [0, 1])
    bad2 = templates.copy()
    bad2[0, 0] = np.nan  # weight positive there -> contradiction
    with pytest.raises(ValueError, match="matched pair"):
        merge_group_dense(bad2, weights, [0, 1])


def test_empty_group_raises():
    templates, weights = spike_count_stitch()
    with pytest.raises(ValueError, match="empty"):
        merge_group_dense(templates, weights, [])


# --------------------------------------------------------------------------- #
# plan_merge_output
# --------------------------------------------------------------------------- #


def test_plan_orders_and_survivor_ids():
    unit_ids = [10, 20, 30, 40, 50]
    plan = plan_merge_output(unit_ids, [[40, 20]])
    assert [entry["unit_id"] for entry in plan] == [10, 20, 30, 50]
    merged_entry = plan[1]
    assert merged_entry["merged"] is True
    assert merged_entry["member_ids"] == [40, 20]
    assert merged_entry["member_indices"] == [3, 1]
    assert merged_entry["unit_id"] == 20  # lowest member id survives
    for entry in (plan[0], plan[2], plan[3]):
        assert entry["merged"] is False
        assert entry["member_ids"] == [entry["unit_id"]]


def test_plan_str_int_id_tolerance():
    """json round-trips turn ids into strings; the join must not care."""
    plan = plan_merge_output(["10", "20", "30"], [[30, 10]])
    assert len(plan) == 2
    assert plan[0]["merged"] is True
    assert plan[0]["member_indices"] == [2, 0]
    assert plan[0]["unit_id"] == 10
    assert plan[1]["merged"] is False


def test_plan_rejects_bad_maps():
    unit_ids = [1, 2, 3]
    with pytest.raises(ValueError, match="not in the unit-id list"):
        plan_merge_output(unit_ids, [[1, 99]])
    with pytest.raises(ValueError, match="disjoint"):
        plan_merge_output(unit_ids, [[1, 2], [2, 3]])
    with pytest.raises(ValueError, match="at least two"):
        plan_merge_output(unit_ids, [[1]])
    with pytest.raises(ValueError, match="repeats"):
        plan_merge_output(unit_ids, [[2, 2]])


# --------------------------------------------------------------------------- #
# dedup_coincident_spikes
# --------------------------------------------------------------------------- #


def test_dedup_drops_cross_unit_duplicates_keep_first():
    times = np.array([100, 103, 200, 300])
    sources = np.array([1, 2, 1, 2])
    keep = dedup_coincident_spikes(times, sources, censor_samples=5)
    assert keep.tolist() == [True, False, True, True]


def test_dedup_never_drops_same_unit_spikes():
    times = np.array([100, 102, 104])
    sources = np.array([1, 1, 1])
    keep = dedup_coincident_spikes(times, sources, censor_samples=5)
    assert keep.all()


def test_dedup_outside_censor_kept():
    times = np.array([100, 106])
    sources = np.array([1, 2])
    keep = dedup_coincident_spikes(times, sources, censor_samples=5)
    assert keep.all()
    # boundary: dt == censor is INSIDE the window (inclusive)
    keep = dedup_coincident_spikes(np.array([100, 105]), sources, censor_samples=5)
    assert keep.tolist() == [True, False]


def test_dedup_chain_measures_from_last_kept():
    """A dropped spike must not shield a later one: 100(A) 103(B dropped)
    108(B) -- 108 is within 5 of 103 but 8 from the last KEPT spike."""
    times = np.array([100, 103, 108])
    sources = np.array([1, 2, 2])
    keep = dedup_coincident_spikes(times, sources, censor_samples=5)
    assert keep.tolist() == [True, False, True]


def test_dedup_handles_unsorted_input():
    times = np.array([300, 100, 103])
    sources = np.array([2, 1, 2])
    keep = dedup_coincident_spikes(times, sources, censor_samples=5)
    assert keep.tolist() == [True, True, False]


def test_dedup_shape_mismatch_raises():
    with pytest.raises(ValueError):
        dedup_coincident_spikes(np.array([1, 2]), np.array([1]))
