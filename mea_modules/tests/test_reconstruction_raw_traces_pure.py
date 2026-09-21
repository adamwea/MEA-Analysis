"""Pure tests for `mea_modules.reconstruction.raw_traces.export_raw_traces` —
the portable, tool-independent re-emit of a reconstructed unit's raw
per-channel template traces.

No `axon_velocity` needed: `export_raw_traces` reads only plain attributes
(`template` / `locations` / `fs` / `selected_channels` / `init_channel` /
`branches`), so a duck-typed stand-in exercises the exact same code path an
unpickled `GraphAxonTracking` does — the same approach
`test_reconstruction_overlay_pure.py` uses. The pipeline repo's
`tests/test_export_recon_traces_toy.py` covers the real capsule-on-real-`gtr`
wiring; THIS file covers the serialization contract itself: raw-first (every
channel kept, verbatim), the npz array set, the json schema/provenance, that the
artifact loads with numpy + stdlib json alone, and JSON-safety on numpy scalars.
"""

import json

import numpy as np
import pytest

from mea_modules.reconstruction import (
    RAW_TRACES_JSON_FILENAME,
    RAW_TRACES_NPZ_FILENAME,
    export_raw_traces,
)


class _FakeGtr:
    """The attributes `export_raw_traces` documents itself to read."""

    def __init__(self, template, locations, fs, selected_channels, init_channel, branches):
        self.template = template
        self.locations = locations
        self.fs = fs
        self.selected_channels = selected_channels
        self.init_channel = init_channel
        self.branches = branches


def _fake_gtr(n_channels=40, n_samples=60, seed=0):
    """A synthetic unit: a signal channel plus a low-amplitude noise blanket, so
    the raw-first "keep every channel" contract has something to prove.
    """
    rng = np.random.default_rng(seed)
    template = rng.normal(0.0, 0.5, size=(n_channels, n_samples))  # noise blanket
    template[5] += -80.0 * np.exp(-0.5 * ((np.arange(n_samples) - 30) / 6.0) ** 2)
    locations = np.stack(
        [np.arange(n_channels) * 17.5, np.zeros(n_channels)], axis=1
    ).astype(float)
    # A STRICT SUBSET the tracker "selected" — export must NOT restrict to this.
    selected_channels = np.array([5, 6, 7, 8], dtype=np.int64)
    branches = [{"channels": np.array([5, 6, 7])}, {"channels": np.array([5, 8])}]
    return _FakeGtr(
        template=template,
        locations=locations,
        fs=np.float64(20000.0),  # numpy scalar — must survive json.dumps
        selected_channels=selected_channels,
        init_channel=np.int64(5),  # numpy scalar — must survive json.dumps
        branches=branches,
    )


def test_export_writes_loadable_npz_and_json(tmp_path):
    gtr = _fake_gtr()
    meta = export_raw_traces(gtr, tmp_path / "unit5", unit_id=5)

    npz_path = tmp_path / "unit5" / RAW_TRACES_NPZ_FILENAME
    json_path = tmp_path / "unit5" / RAW_TRACES_JSON_FILENAME
    assert npz_path.is_file() and json_path.is_file()

    # Loadable with numpy alone.
    with np.load(npz_path) as npz:
        assert set(npz.files) == {"template", "channel_locations", "channel_indices"}
        # RAW-FIRST: every channel kept, verbatim (bit-for-bit), incl. the noise
        # blanket — NOT restricted to the 4 selected channels.
        assert npz["template"].shape == (40, 60)
        assert np.array_equal(npz["template"], gtr.template)
        assert np.array_equal(npz["channel_locations"], gtr.locations)
        assert np.array_equal(npz["channel_indices"], np.arange(40))
        assert npz["channel_indices"].dtype == np.int64

    # Loadable with stdlib json alone.
    doc = json.loads(json_path.read_text())
    assert doc == meta  # returned dict IS what was written
    assert doc["schema"] == "mea_recon.raw_traces"
    assert doc["n_channels"] == 40 and doc["n_samples"] == 60
    assert doc["sampling_rate_hz"] == 20000.0
    assert doc["trace_units"] == "uV" and doc["location_units"] == "um"
    assert doc["raw_first"] is True
    assert doc["unit_id"] == "5"  # stringified


def test_provenance_records_selection_without_filtering(tmp_path):
    gtr = _fake_gtr()
    meta = export_raw_traces(gtr, tmp_path / "unit5", unit_id="5")
    prov = meta["provenance"]

    # selected_channels captured as provenance...
    assert prov["selected_channels"] == [5, 6, 7, 8]
    assert prov["n_selected_channels"] == 4
    assert prov["init_channel"] == 5
    assert prov["n_branches"] == 2
    # ...but the exported template still carries ALL channels (raw-first).
    assert meta["n_channels"] == 40


def test_json_is_plain_types_no_numpy(tmp_path):
    """The whole point of the sibling json is portability — every value must be
    a plain Python/JSON type, not a numpy scalar (which json.dumps chokes on).
    """
    gtr = _fake_gtr()
    meta = export_raw_traces(gtr, tmp_path / "u", unit_id=7)
    # A successful round-trip through json text proves nothing numpy leaked in.
    reloaded = json.loads(json.dumps(meta))
    assert reloaded["sampling_rate_hz"] == 20000.0
    assert isinstance(reloaded["provenance"]["init_channel"], int)
    assert all(isinstance(c, int) for c in reloaded["provenance"]["selected_channels"])


def test_missing_optional_attrs_still_exports(tmp_path):
    """A gtr lacking selected_channels/init_channel/branches still exports its
    raw traces, with those provenance fields null/empty."""
    class _Bare:
        template = np.zeros((10, 20))
        locations = np.zeros((10, 2))
        fs = 20000.0

    meta = export_raw_traces(_Bare(), tmp_path / "bare", unit_id=1)
    assert meta["n_channels"] == 10
    assert meta["provenance"]["selected_channels"] == []
    assert meta["provenance"]["n_selected_channels"] == 0
    assert meta["provenance"]["init_channel"] is None
    assert meta["provenance"]["n_branches"] == 0


def test_channel_axis_mismatch_raises(tmp_path):
    gtr = _fake_gtr()
    gtr.locations = gtr.locations[:-1]  # drop one channel → axes disagree
    with pytest.raises(ValueError):
        export_raw_traces(gtr, tmp_path / "bad", unit_id=5)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
