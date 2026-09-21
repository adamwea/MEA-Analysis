"""A gap table re-addressed onto a decimated series places every kept sample
at the same real elapsed time as the native frame it came from.

The concatenated well's whole-timeline traces are cached every `step`-th
native frame. Their gap table has to be renumbered to match, or the shading
and the join marks land on the wrong stretch. The only property worth
asserting is the end-to-end one: native time of frame ``j * step`` equals the
decimated time of sample ``j``.
"""
import numpy as np
import pytest

from mea_modules.diagnostics.timebase import rescale_time_gaps, sample_times

FS = 10_000.0

GAPS = {
    "gaps": {
        # Counter steps: 1 + missing frames. Breaks on, just after, and just
        # before a multiple of the step, so ceil() is exercised every way.
        "break_sample_indices": [1_000, 1_005, 2_019, 7_777],
        "break_gap_frames": [41, 2, 101, 7],
    },
    "segment_gaps": [
        {"start_sample": 4_000, "gap_before_s": 30.0},
        {"start_sample": 8_013, "gap_before_s": 12.5},
    ],
}


def _times(frames, fs, structure):
    return sample_times(frames, fs, gaps=structure["gaps"], segment_gaps=structure["segment_gaps"])


@pytest.mark.parametrize("step", [1, 3, 20, 64])
def test_a_kept_sample_sits_where_its_native_frame_sits(step):
    rescaled = rescale_time_gaps(GAPS, step)
    kept = np.arange(0, 10_000 // step)
    native = _times(kept * step, FS, GAPS)
    decimated = _times(kept, FS / step, rescaled)
    assert np.allclose(decimated, native, rtol=0, atol=1e-9)


def test_the_arithmetic_of_one_step():
    rescaled = rescale_time_gaps(GAPS, 20)
    assert rescaled["frame_step"] == 20
    assert rescaled["gaps"]["break_sample_indices"] == [50, 51, 101, 389]
    # (steps - 1) / 20 + 1: the same seconds at the decimated rate.
    assert rescaled["gaps"]["break_gap_frames"] == pytest.approx([3.0, 1.05, 6.0, 1.3])
    assert [s["start_sample"] for s in rescaled["segment_gaps"]] == [200, 401]
    assert [s["gap_before_s"] for s in rescaled["segment_gaps"]] == [30.0, 12.5]


def test_step_one_hands_the_same_numbers_back():
    rescaled = rescale_time_gaps(GAPS, 1)
    assert rescaled["gaps"]["break_sample_indices"] == GAPS["gaps"]["break_sample_indices"]
    assert rescaled["gaps"]["break_gap_frames"] == pytest.approx(GAPS["gaps"]["break_gap_frames"])


def test_the_total_missing_time_is_unchanged():
    for step in (1, 7, 20):
        rescaled = rescale_time_gaps(GAPS, step)
        native_end = _times([10_000], FS, GAPS)[0] - 10_000 / FS
        decimated_end = _times([10_000 // step + 1], FS / step, rescaled)[0] - (10_000 // step + 1) * step / FS
        assert decimated_end == pytest.approx(native_end)


def test_a_concatenated_gaps_summary_is_dropped_from_the_rescaled_table():
    structure = dict(GAPS, summary={"well": "well000"})
    rescaled = rescale_time_gaps(structure, 20)
    assert set(rescaled) == {"gaps", "segment_gaps", "frame_step"}
