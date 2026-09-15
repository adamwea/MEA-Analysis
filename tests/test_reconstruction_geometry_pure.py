"""Pure tests for `mea_modules.reconstruction.geometry.export_reconstruction_geometry`
— the portable branch-polyline (axon reconstruction) export.

No `axon_velocity` needed: the serializer reads only plain attributes
(`locations` / `init_channel` / `branches` with ordered `channels`), so a
duck-typed stand-in exercises the same path an unpickled `GraphAxonTracking`
does. The pipeline repo's toy test covers the capsule-on-real-`gtr` wiring; THIS
file covers the geometry contract: ragged concat + split round-trips the
polylines, points sit in the same µm frame as `channel_locations`, more branches
give more polylines, and the empty / degenerate cases behave.
"""

import json

import numpy as np
import pytest

from mea_modules.reconstruction import (
    RECON_GEOMETRY_JSON_FILENAME,
    RECON_GEOMETRY_NPZ_FILENAME,
    export_reconstruction_geometry,
)


class _FakeGtr:
    def __init__(self, locations, init_channel, branches):
        self.locations = locations
        self.init_channel = init_channel
        self.branches = branches


def _grid(n_rows=8, n_cols=8, pitch=17.5):
    rows, cols = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")
    return np.stack([cols.ravel() * pitch, rows.ravel() * pitch], axis=1).astype(float)


def _branch(channels, velocity=100.0, r2=0.99, pval=0.01, offset=1.0, raw_path_idx=0):
    ch = np.asarray(channels, dtype=np.int64)
    return {
        "channels": ch,
        "velocity": np.float64(velocity),
        "r2": np.float64(r2),
        "pval": np.float64(pval),
        "offset": np.float64(offset),
        "raw_path_idx": np.int64(raw_path_idx),
        "distances": np.arange(len(ch) - 1, dtype=float) * pitch_default(),
        "peak_times": np.arange(len(ch) - 1, dtype=float) * 0.05,
    }


def pitch_default():
    return 17.5


def test_polylines_roundtrip_and_share_channel_frame(tmp_path):
    loc = _grid()
    branches = [
        _branch([0, 1, 2, 3], velocity=120.0, raw_path_idx=0),
        _branch([0, 8, 16], velocity=90.0, raw_path_idx=1),
        _branch([20, 21, 22, 23, 31], velocity=np.float64(-50.0), raw_path_idx=3),
    ]
    gtr = _FakeGtr(loc, init_channel=np.int64(0), branches=branches)
    meta = export_reconstruction_geometry(gtr, tmp_path / "u", unit_id=7)

    with np.load(tmp_path / "u" / RECON_GEOMETRY_NPZ_FILENAME) as z:
        assert z["branch_node_counts"].tolist() == [4, 3, 5]
        assert int(z["branch_points_xy"].shape[0]) == 12 and z["branch_points_xy"].shape[1] == 2
        # split by cumsum reproduces each branch's polyline == locations[channels]
        offsets = np.cumsum(z["branch_node_counts"])[:-1]
        polylines = np.split(z["branch_points_xy"], offsets)
        chans = np.split(z["branch_channels"], offsets)
        for br, pl, ch in zip(branches, polylines, chans):
            assert np.array_equal(ch, np.asarray(br["channels"]))
            assert np.array_equal(pl, loc[np.asarray(br["channels"]), :2])  # same µm frame
        assert z["branch_velocity"].tolist() == [120.0, 90.0, -50.0]  # sign preserved
        assert int(z["init_channel"]) == 0
        assert z["init_xy"].tolist() == loc[0, :2].tolist()

    doc = json.loads((tmp_path / "u" / RECON_GEOMETRY_JSON_FILENAME).read_text())
    assert doc["schema"] == "mea_recon.reconstruction_geometry"
    assert doc["n_branches"] == 3 and doc["n_points_total"] == 12
    assert doc["position_units"] == "um"
    # JSON is self-sufficient (stdlib-only): inline points reproduce the polyline.
    b0 = doc["branches"][0]
    assert b0["channels"] == [0, 1, 2, 3]
    assert b0["points_xy"] == loc[[0, 1, 2, 3], :2].tolist()
    assert b0["velocity"] == 120.0
    # plain JSON types throughout
    json.dumps(doc)


def test_more_branches_more_polylines(tmp_path):
    loc = _grid()
    one = _FakeGtr(loc, np.int64(0), [_branch([0, 1, 2])])
    six = _FakeGtr(loc, np.int64(0), [_branch([i, i + 1, i + 2]) for i in range(6)])
    m1 = export_reconstruction_geometry(one, tmp_path / "one", unit_id=1)
    m6 = export_reconstruction_geometry(six, tmp_path / "six", unit_id=6)
    assert m1["n_branches"] == 1 and m6["n_branches"] == 6
    with np.load(tmp_path / "one" / RECON_GEOMETRY_NPZ_FILENAME) as z1, \
         np.load(tmp_path / "six" / RECON_GEOMETRY_NPZ_FILENAME) as z6:
        assert len(z1["branch_node_counts"]) == 1
        assert len(z6["branch_node_counts"]) == 6


def test_zero_branches_writes_empty_but_valid(tmp_path):
    loc = _grid()
    gtr = _FakeGtr(loc, np.int64(5), branches=[])
    meta = export_reconstruction_geometry(gtr, tmp_path / "none", unit_id=2)
    assert meta["n_branches"] == 0 and meta["n_points_total"] == 0
    assert meta["branches"] == []
    with np.load(tmp_path / "none" / RECON_GEOMETRY_NPZ_FILENAME) as z:
        assert z["branch_points_xy"].shape == (0, 2)
        assert z["branch_channels"].shape == (0,)
        assert z["branch_node_counts"].shape == (0,)
    # init still recorded
    assert meta["init_channel"] == 5


def test_out_of_range_channels_filtered(tmp_path):
    loc = _grid(n_rows=4, n_cols=4)  # 16 channels
    gtr = _FakeGtr(loc, np.int64(0), [_branch([0, 1, 99, 2])])  # 99 out of range
    meta = export_reconstruction_geometry(gtr, tmp_path / "u", unit_id=1)
    # the out-of-range vertex is dropped; the rest still form the polyline
    assert meta["branches"][0]["channels"] == [0, 1, 2]
    assert meta["branches"][0]["n_nodes"] == 3


def test_bad_init_channel_gives_null_init(tmp_path):
    loc = _grid(n_rows=4, n_cols=4)
    gtr = _FakeGtr(loc, np.int64(999), [_branch([0, 1, 2])])
    meta = export_reconstruction_geometry(gtr, tmp_path / "u", unit_id=1)
    assert meta["init_channel"] == 999 and meta["init_xy"] is None
    with np.load(tmp_path / "u" / RECON_GEOMETRY_NPZ_FILENAME) as z:
        assert np.isnan(z["init_xy"]).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
