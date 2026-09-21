"""What one well's recordings share, and the gaps of the ones actually joined.

Two readers of the same source file. `electrode_coverage` is the per-electrode
view behind `find_common_electrodes`, so the first property is that they agree
on every count. `concatenated_gaps(recs=...)` must lay the gap table out the way
the concatenation laid the samples out: only the segments joined, in the order
joined.
"""
import numpy as np
import pytest

from mea_modules.diagnostics import plot_electrode_coverage, plot_segment_electrode_counts
from mea_modules.io.gaps import concatenated_gaps
from mea_modules.io.metadata import electrode_coverage, find_common_electrodes

FS = 10_000.0

RECORDINGS = [
    {"rec": "rec0000", "n_samples": 1_000, "electrodes": list(range(0, 40)),
     "breaks": [(250, 4)], "start_s": 0.0, "stop_s": 10.0},
    {"rec": "rec0001", "n_samples": 800, "electrodes": list(range(20, 70)),
     "breaks": [], "start_s": 40.0, "stop_s": 48.0},
    {"rec": "rec0002", "n_samples": 1_200, "electrodes": list(range(10, 35)) + [500, 501],
     "breaks": [(600, 9), (900, 2)], "start_s": 70.0, "stop_s": 82.0},
]


# --------------------------------------------------------------------------
# coverage
# --------------------------------------------------------------------------


@pytest.mark.parametrize("recs", [None, ["rec0000", "rec0002"]])
def test_coverage_agrees_with_the_shared_set_on_every_count(maxtwo_file, recs):
    path = maxtwo_file(RECORDINGS)
    coverage = electrode_coverage(path, "well000", rec_names=recs)
    common = find_common_electrodes(path, "well000", rec_names=recs)

    kept = [e for e, keep in zip(coverage["electrode_ids"], coverage["kept"]) if keep]
    assert kept == common["common_electrodes"]
    assert coverage["common_count"] == common["common_count"]
    assert coverage["union_count"] == common["union_count"]
    assert {row["rec"]: row["n_electrodes"] for row in coverage["segments"]} == common["per_rec_counts"]


def test_coverage_counts_each_electrode_once_per_recording_and_skips_unrouted_rows(maxtwo_file):
    coverage = electrode_coverage(maxtwo_file(RECORDINGS), "well000")
    counts = dict(zip(coverage["electrode_ids"], coverage["n_segments_routed"]))
    assert 999_999 not in counts  # channel -1: never routed
    assert counts[0] == 1  # listed twice in rec0000's mapping, still one recording
    assert counts[20] == 3 and counts[34] == 3 and counts[35] == 2
    assert counts[500] == 1
    assert [row["n_electrodes"] for row in coverage["segments"]] == [40, 50, 27]
    assert coverage["common_count"] == len(range(20, 35))


def test_coverage_positions_come_from_the_mapping(maxtwo_file):
    coverage = electrode_coverage(maxtwo_file(RECORDINGS), "well000")
    by_id = {e: (x, y) for e, x, y in zip(coverage["electrode_ids"], coverage["x_um"], coverage["y_um"])}
    assert by_id[501] == pytest.approx(((501 % 220) * 17.5, (501 // 220) * 17.5))


def test_both_coverage_figures_draw_from_the_dict_alone(maxtwo_file, tmp_path):
    coverage = electrode_coverage(maxtwo_file(RECORDINGS), "well000")
    for plot, name in ((plot_electrode_coverage, "map.png"), (plot_segment_electrode_counts, "bars.png")):
        out = plot(coverage, tmp_path / name)
        assert out.is_file() and out.stat().st_size > 5_000


def test_the_coverage_figures_refuse_nothing_to_draw(tmp_path):
    empty = {"electrode_ids": [], "x_um": [], "y_um": [], "n_segments_routed": [], "kept": [],
             "n_segments": 0, "segments": [], "union_count": 0, "common_count": 0}
    with pytest.raises(ValueError):
        plot_electrode_coverage(empty, tmp_path / "map.png")
    with pytest.raises(ValueError):
        plot_segment_electrode_counts(empty, tmp_path / "bars.png")


# --------------------------------------------------------------------------
# the gap table of the segments joined
# --------------------------------------------------------------------------


def test_all_recordings_in_file_order_by_default(maxtwo_file):
    gaps = concatenated_gaps(maxtwo_file(RECORDINGS), "well000", fs_hz=FS)
    starts = [s["start_sample"] for s in gaps["segment_gaps"]]
    assert starts == [0, 1_000, 1_800]
    assert gaps["gaps"]["break_sample_indices"] == [250, 1_800 + 600, 1_800 + 900]
    assert gaps["gaps"]["break_gap_frames"] == [5, 10, 3]
    assert [s["gap_before_s"] for s in gaps["segment_gaps"]] == [None, pytest.approx(30.0),
                                                                  pytest.approx(22.0)]


def test_a_subset_is_laid_out_as_joined_not_as_filed(maxtwo_file):
    """Two of three joined: the second segment starts where the first ends,
    and its gap is to the previous JOINED segment."""
    gaps = concatenated_gaps(maxtwo_file(RECORDINGS), "well000", fs_hz=FS,
                             recs=["rec0000", "rec0002"])
    assert [s["rec"] for s in gaps["segment_gaps"]] == ["rec0000", "rec0002"]
    assert [s["start_sample"] for s in gaps["segment_gaps"]] == [0, 1_000]
    assert gaps["gaps"]["break_sample_indices"] == [250, 1_600, 1_900]
    assert gaps["segment_gaps"][1]["gap_before_s"] == pytest.approx(60.0)


def test_an_unknown_recording_is_refused(maxtwo_file):
    with pytest.raises(ValueError, match="rec0009"):
        concatenated_gaps(maxtwo_file(RECORDINGS), "well000", fs_hz=FS, recs=["rec0000", "rec0009"])


def test_the_summary_carries_each_segments_own_frame_count(maxtwo_file):
    """The independent number a continuity check compares a binary against."""
    gaps = concatenated_gaps(maxtwo_file(RECORDINGS), "well000", fs_hz=FS)
    rows = {row["rec"]: row for row in gaps["summary"]["segments"]}
    assert {rec: row["n_samples"] for rec, row in rows.items()} == {
        "rec0000": 1_000, "rec0001": 800, "rec0002": 1_200,
    }
    assert np.isclose(gaps["summary"]["n_samples"], 3_000)
