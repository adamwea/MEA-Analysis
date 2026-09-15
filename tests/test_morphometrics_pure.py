"""Pure tests for `mea_modules.morphometrics` — the canonical per-unit
branch-length morphometrics over the `reconstruction_geometry` v1 schema — and
for `split_branch_polylines`, the canonical reader of that schema's split
convention.

No `axon_velocity` needed: the metrics consume the portable npz arrays (or their
in-memory equivalents), so a duck-typed `_FakeGtr` fed through
`export_reconstruction_geometry` produces a real artifact to measure. Lengths are
checked to exact values on a known 17.5 µm grid.
"""

import numpy as np
import pytest

from mea_modules.reconstruction import (
    RECON_GEOMETRY_NPZ_FILENAME,
    export_reconstruction_geometry,
    split_branch_polylines,
)
from mea_modules.morphometrics import (
    segment_lengths_um,
    polyline_length_um,
    unit_morphometrics,
    unit_morphometrics_from_npz,
)

PITCH = 17.5


class _FakeGtr:
    def __init__(self, locations, init_channel, branches):
        self.locations = locations
        self.init_channel = init_channel
        self.branches = branches


def _grid(n_rows=8, n_cols=8, pitch=PITCH):
    rows, cols = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")
    return np.stack([cols.ravel() * pitch, rows.ravel() * pitch], axis=1).astype(float)


def _branch(channels):
    return {"channels": np.asarray(channels, dtype=np.int64), "velocity": 100.0,
            "r2": 0.99, "pval": 0.01, "offset": 1.0, "raw_path_idx": 0}


# The three branches below are the same ones the geometry round-trip test uses;
# on the 17.5 um grid their Euclidean polyline lengths are 52.5, 35.0, 70.0 um.
_BRANCHES = [[0, 1, 2, 3], [0, 8, 16], [20, 21, 22, 23, 31]]
_EXPECT_PER_BRANCH = [52.5, 35.0, 70.0]


def test_segment_lengths_um_and_length_is_their_sum():
    # 3 hops of 17.5 um along a row -> three equal segments.
    p = _grid()[[0, 1, 2, 3], :2]
    segs = segment_lengths_um(p)
    assert segs.shape == (3,)
    assert segs == pytest.approx([17.5, 17.5, 17.5])
    # polyline_length_um is exactly the sum of the segment lengths (single primitive).
    assert polyline_length_um(p) == pytest.approx(float(segs.sum()))
    # Degenerate polylines have no segments.
    assert segment_lengths_um(np.empty((0, 2))).shape == (0,)
    assert segment_lengths_um(np.array([[1.0, 2.0]])).shape == (0,)


def test_polyline_length_um_basic():
    # 3 hops of 17.5 um along a row.
    p = _grid()[[0, 1, 2, 3], :2]
    assert polyline_length_um(p) == pytest.approx(52.5)
    # Degenerate polylines have no length to measure.
    assert polyline_length_um(np.empty((0, 2))) == 0.0
    assert polyline_length_um(np.array([[1.0, 2.0]])) == 0.0


def test_unit_morphometrics_exact_values():
    loc = _grid()
    branch_points = np.vstack([loc[b, :2] for b in _BRANCHES]).astype(float)
    node_counts = np.array([len(b) for b in _BRANCHES], dtype=np.int64)
    m = unit_morphometrics(branch_points, node_counts)
    assert m["n_branches"] == 3
    assert m["per_branch_length_um"] == pytest.approx(_EXPECT_PER_BRANCH)
    assert m["mean_branch_length_um"] == pytest.approx(52.5)
    assert m["longest_branch_um"] == pytest.approx(70.0)
    # PROVISIONAL total = sum of per-branch lengths (double-counts shared path).
    assert m["total_axon_length_um"] == pytest.approx(157.5)


def test_unit_morphometrics_zero_branches():
    m = unit_morphometrics(np.empty((0, 2)), np.empty((0,), dtype=np.int64))
    assert m["n_branches"] == 0
    assert m["per_branch_length_um"] == []
    assert np.isnan(m["mean_branch_length_um"])
    assert np.isnan(m["longest_branch_um"])
    assert m["total_axon_length_um"] == 0.0


def test_from_npz_roundtrip(tmp_path):
    loc = _grid()
    gtr = _FakeGtr(loc, np.int64(0), [_branch(b) for b in _BRANCHES])
    export_reconstruction_geometry(gtr, tmp_path / "u", unit_id=1)
    # Accepts either the directory or the npz path itself.
    m_dir = unit_morphometrics_from_npz(tmp_path / "u")
    m_npz = unit_morphometrics_from_npz(tmp_path / "u" / RECON_GEOMETRY_NPZ_FILENAME)
    assert m_dir == m_npz
    assert m_dir["per_branch_length_um"] == pytest.approx(_EXPECT_PER_BRANCH)
    assert m_dir["longest_branch_um"] == pytest.approx(70.0)


def test_split_branch_polylines_inverts_concatenation():
    loc = _grid()
    branch_points = np.vstack([loc[b, :2] for b in _BRANCHES]).astype(float)
    node_counts = np.array([len(b) for b in _BRANCHES], dtype=np.int64)
    polylines = split_branch_polylines(branch_points, node_counts)
    assert len(polylines) == 3
    for pl, b in zip(polylines, _BRANCHES):
        assert np.array_equal(pl, loc[b, :2])


def test_split_branch_polylines_zero_branches_returns_empty():
    # np.split(empty, []) would return [empty] (one chunk); the guard must give [].
    assert split_branch_polylines(np.empty((0, 2)), np.empty((0,), dtype=np.int64)) == []


def test_split_branch_polylines_mismatch_raises():
    with pytest.raises(ValueError):
        split_branch_polylines(np.zeros((5, 2)), np.array([2, 2], dtype=np.int64))  # sum 4 != 5


def test_comparison_total_length_delegates_to_canonical():
    """`ReconUnit.total_length_um` must equal the canonical per-branch sum, proving
    the single-definition guard holds (no divergent local length formula)."""
    from mea_modules.comparison.pipeline import ReconUnit

    loc = _grid()
    branches = [{"xy": loc[b, :2], "velocity": 100.0} for b in _BRANCHES]
    unit = ReconUnit(unit_id=1, init_channel=0, init_xy=loc[0, :2], branches=branches)
    canonical = sum(polyline_length_um(b["xy"]) for b in branches)
    assert unit.total_length_um() == pytest.approx(canonical)
    assert unit.total_length_um() == pytest.approx(157.5)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
