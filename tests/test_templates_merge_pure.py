"""Pure-arithmetic tests for `mea_modules.templates.merge`'s bucket layer.

Deliberately separate from `tests/test_merge_templates_toy.py` in the pipeline
repo (`MEA-recon-pipeline`), which exercises the streaming orchestrator against
real synthetic `SortingAnalyzer`s and the capsule wrapped around it. THIS file
covers only `new_unit_accumulator` / `accumulate_channel_contributions` /
`finalize_unit_accumulator`: plain-numpy bucket arithmetic with no
SpikeInterface dependency, so it belongs beside `mea_modules` rather than the
pipeline, and runs in milliseconds with no fixtures or tmp_path needed. Modeled
on the old (pre-rebuild) build's own
`templates/tests/test_merge_core.py::test_merge_sources_per_channel_*`, which
hand-computed its expected merged values the same way.
"""

import numpy as np
import pytest

from mea_modules.templates import (
    accumulate_channel_contributions,
    finalize_unit_accumulator,
    new_unit_accumulator,
)
from mea_modules.templates.merge import _channel_sort_key


def test_single_source_channel_passes_through_unchanged():
    """A channel routed by exactly one segment merges to that segment's own
    value exactly -- its weight cancels top and bottom, whatever it is."""
    accumulator = new_unit_accumulator()
    accumulate_channel_contributions(
        accumulator,
        channel_ids=["7"],
        locations_xy=[[10.0, 20.0]],
        template_ch_by_t=np.asarray([[1.0, -2.0, 3.0]]),
        weight=41.0,  # an arbitrary, deliberately "odd" weight
    )
    channel_ids, template, locations, weights = finalize_unit_accumulator(accumulator)
    assert channel_ids == ["7"]
    np.testing.assert_allclose(template, [[1.0, -2.0, 3.0]])
    np.testing.assert_allclose(locations, [[10.0, 20.0]])
    np.testing.assert_allclose(weights, [41.0])


def test_two_sources_average_by_construction_not_by_dividing_by_source_count():
    """Three channels: one seen by A only, one by B only, one by both -- the
    shared channel must land at the WEIGHTED average of the two contributors,
    never diluted by a channel count neither of them actually has.
    """
    accumulator = new_unit_accumulator()
    # Segment A routes channels 0 and 1.
    accumulate_channel_contributions(
        accumulator,
        channel_ids=[0, 1],
        locations_xy=[[0.0, 0.0], [1.0, 0.0]],
        template_ch_by_t=np.asarray([[1.0, 1.0], [3.0, 3.0]]),
        weight=2.0,
    )
    # Segment B routes channels 1 and 2 (channel 1 overlaps A).
    accumulate_channel_contributions(
        accumulator,
        channel_ids=[1, 2],
        locations_xy=[[1.0, 0.0], [2.0, 0.0]],
        template_ch_by_t=np.asarray([[5.0, 5.0], [7.0, 7.0]]),
        weight=4.0,
    )
    channel_ids, template, _locations, weights = finalize_unit_accumulator(accumulator)

    assert channel_ids == [0, 1, 2]
    # channel 0: A only -> [1, 1]
    np.testing.assert_allclose(template[0], [1.0, 1.0])
    # channel 1: (3*2 + 5*4) / (2+4) = 26/6, NOT (3+5)/2 = 4.0
    np.testing.assert_allclose(template[1], [26.0 / 6.0, 26.0 / 6.0])
    assert not np.allclose(template[1], 4.0), "landed on the plain mean, not the weighted one"
    # channel 2: B only -> [7, 7]
    np.testing.assert_allclose(template[2], [7.0, 7.0])
    np.testing.assert_allclose(weights, [2.0, 6.0, 4.0])


def test_uniform_weight_one_per_segment_reduces_to_plain_mean():
    """weight=1.0 from every contributing segment (the
    `averaging_method="uniform"` case in merge_segment_templates) is just an
    unweighted mean."""
    accumulator = new_unit_accumulator()
    accumulate_channel_contributions(
        accumulator, channel_ids=["a"], locations_xy=[[0.0, 0.0]],
        template_ch_by_t=np.asarray([[2.0, 4.0]]), weight=1.0,
    )
    accumulate_channel_contributions(
        accumulator, channel_ids=["a"], locations_xy=[[0.0, 0.0]],
        template_ch_by_t=np.asarray([[6.0, 8.0]]), weight=1.0,
    )
    _channel_ids, template, _locations, weights = finalize_unit_accumulator(accumulator)
    np.testing.assert_allclose(template, [[4.0, 6.0]])
    np.testing.assert_allclose(weights, [2.0])


def test_nonpositive_weight_is_a_no_op():
    """A zero (or negative) weight must never enter a bucket -- this is the
    mechanism `merge_segment_templates` relies on to skip a unit that had no
    spikes in a segment without a separate branch in the orchestrator."""
    accumulator = new_unit_accumulator()
    accumulate_channel_contributions(
        accumulator, channel_ids=["x"], locations_xy=[[0.0, 0.0]],
        template_ch_by_t=np.asarray([[99.0]]), weight=0.0,
    )
    accumulate_channel_contributions(
        accumulator, channel_ids=["x"], locations_xy=[[0.0, 0.0]],
        template_ch_by_t=np.asarray([[99.0]]), weight=-5.0,
    )
    assert accumulator == {}
    channel_ids, template, _locations, _weights = finalize_unit_accumulator(accumulator)
    assert channel_ids == []
    assert template.shape == (0, 0)


def test_mismatched_sample_count_raises_rather_than_silently_truncating():
    accumulator = new_unit_accumulator()
    accumulate_channel_contributions(
        accumulator, channel_ids=["c"], locations_xy=[[0.0, 0.0]],
        template_ch_by_t=np.asarray([[1.0, 2.0, 3.0]]), weight=1.0,
    )
    with pytest.raises(ValueError, match="disagree on the waveform window"):
        accumulate_channel_contributions(
            accumulator, channel_ids=["c"], locations_xy=[[0.0, 0.0]],
            template_ch_by_t=np.asarray([[1.0, 2.0]]), weight=1.0,
        )


def test_empty_accumulator_finalizes_to_empty_arrays_not_an_error():
    channel_ids, template, locations, weights = finalize_unit_accumulator(new_unit_accumulator())
    assert channel_ids == []
    assert template.shape == (0, 0)
    assert locations.shape == (0, 2)
    assert weights.shape == (0,)


def test_channel_sort_key_orders_numeric_ids_numerically():
    # A plain string sort would put '10' before '2'; Maxwell channel ids are
    # unpadded digit strings, so this ordering is load-bearing, not cosmetic.
    ids = ["10", "2", "1", "20"]
    assert sorted(ids, key=_channel_sort_key) == ["1", "2", "10", "20"]
    # Falls back to lexical ordering for anything non-numeric instead of raising.
    mixed = ["3", "beta", "1", "alpha"]
    assert sorted(mixed, key=_channel_sort_key) == ["1", "3", "alpha", "beta"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
