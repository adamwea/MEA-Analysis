"""The compute-once contract between a capsule and a plot tool.

The cache is what makes "diagnostics are computed inside the capsule, plots are
drawn from cache" enforceable rather than aspirational, so these tests are about
the contract and not about any one figure: a cache round-trips, a missing cache
names the capsule that writes it instead of falling back to recomputation, and
the geometry stand-in refuses to be mistaken for a recording.
"""

import json

import numpy as np
import pytest

from mea_modules.diagnostics.cache import (
    ARRAYS_NAME,
    CACHE_VERSION,
    RECORD_NAME,
    CachedProbe,
    CacheMissing,
    CacheVersionMismatch,
    cache_dir,
    cache_ready,
    read_cache,
    write_cache,
)

CHANNEL_IDS = ["e1", "e2", "e3", "e4"]
LOCATIONS = [[0.0, 0.0], [17.5, 0.0], [0.0, 17.5], [17.5, 17.5]]
FS_HZ = 10_000.0


def _probe():
    return CachedProbe(CHANNEL_IDS, LOCATIONS, sampling_frequency=FS_HZ)


def _write(tmp_path, **kwargs):
    payload = dict(
        probe=_probe(),
        metrics={"noise": {"channel_ids": CHANNEL_IDS, "noise": [1.0, 1.1, 0.9, 1.2]}},
        traces={
            "preprocessed": {
                "channel_ids": ["e1", "e2"],
                "time_s": np.linspace(0.0, 1.0, 50),
                "traces": np.zeros((50, 2)),
                "unit": "uV",
            }
        },
        events={"raster": {"times_s": [0.1, 0.5, 0.9], "labels": [0, 2, 2]}},
        spectra={
            "raw": {
                "frequencies": np.linspace(1.0, 5000.0, 32),
                "psd": np.ones((32, 2)),
                "channel_ids": ["e1", "e2"],
                "unit": "uV^2/Hz",
            }
        },
        meta={"well": "well005", "rec": "rec0001"},
    )
    payload.update(kwargs)
    return write_cache(tmp_path, **payload)


# --------------------------------------------------------------------------
# round trip
# --------------------------------------------------------------------------


def test_a_cache_round_trips_everything_a_figure_needs(tmp_path):
    _write(tmp_path)
    cache = read_cache(tmp_path, capsule="preprocess_segment")

    assert cache.probe.get_channel_ids() == CHANNEL_IDS
    assert np.allclose(cache.probe.get_channel_locations(), LOCATIONS)
    assert cache.probe.get_sampling_frequency() == FS_HZ

    assert cache.metric("noise")["noise"] == [1.0, 1.1, 0.9, 1.2]
    assert cache.metric("nothing_computed_this_run") is None

    ids, time_s, traces = cache.traces("preprocessed")
    assert ids == ["e1", "e2"]
    assert time_s.shape == (50,)
    assert traces.shape == (50, 2)

    times, labels = cache.events("raster")
    assert times.tolist() == [0.1, 0.5, 0.9]
    assert labels.tolist() == [0, 2, 2]

    spectra = cache.all_spectra()
    assert list(spectra) == ["raw"]
    assert spectra["raw"]["frequencies"].shape == (32,)
    assert spectra["raw"]["psd"].shape == (32, 2)
    # Non-array keys survive, so the panel emitter still knows its unit.
    assert spectra["raw"]["unit"] == "uV^2/Hz"


def test_the_cache_lands_inside_the_capsule_output_as_two_readable_files(tmp_path):
    out = _write(tmp_path)

    assert out == cache_dir(tmp_path)
    assert out.parent == tmp_path  # inside the capsule's own output, not beside it
    assert (out / RECORD_NAME).is_file()
    assert (out / ARRAYS_NAME).is_file()

    # The dicts stay human-readable: most of why they are cached at all is that
    # someone opens the file to check a number.
    record = json.loads((out / RECORD_NAME).read_text())
    assert record["version"] == CACHE_VERSION
    assert record["metrics"]["noise"]["noise"] == [1.0, 1.1, 0.9, 1.2]
    assert record["meta"] == {"well": "well005", "rec": "rec0001"}


def test_an_empty_cache_is_still_a_valid_cache(tmp_path):
    """A capsule whose diagnostics were all switched off still writes geometry."""
    write_cache(tmp_path, probe=_probe())
    cache = read_cache(tmp_path)

    assert cache.trace_names() == []
    assert cache.event_names() == []
    assert cache.spectra_names() == []
    assert cache.probe.get_num_channels() == 4


# --------------------------------------------------------------------------
# the fallback that must not exist
# --------------------------------------------------------------------------


def test_a_missing_cache_names_the_capsule_rather_than_recomputing(tmp_path):
    assert not cache_ready(tmp_path)

    with pytest.raises(CacheMissing) as raised:
        read_cache(tmp_path, capsule="preprocess_segment")

    message = str(raised.value)
    assert "preprocess_segment" in message
    assert str(tmp_path) in message


def test_a_newer_cache_is_refused_rather_than_half_read(tmp_path):
    _write(tmp_path)
    record_path = cache_dir(tmp_path) / RECORD_NAME
    record = json.loads(record_path.read_text())
    record["version"] = CACHE_VERSION + 1
    record_path.write_text(json.dumps(record))

    with pytest.raises(CacheVersionMismatch):
        read_cache(tmp_path, capsule="preprocess_segment")


# --------------------------------------------------------------------------
# the stand-in
# --------------------------------------------------------------------------


def test_the_probe_is_not_a_recording_and_says_so(tmp_path):
    probe = _probe()

    with pytest.raises(TypeError, match="geometry only"):
        probe.get_traces(start_frame=0, end_frame=10)


def test_the_probe_matches_what_the_layout_emitters_ask_a_recording_for():
    from mea_modules.diagnostics.channel_layout import _channel_xy

    channel_ids, x, y = _channel_xy(_probe())

    assert channel_ids == CHANNEL_IDS
    assert x.tolist() == [0.0, 17.5, 0.0, 17.5]
    assert y.tolist() == [0.0, 0.0, 17.5, 17.5]


def test_a_probe_snapshot_survives_a_recording_with_no_usable_geometry():
    class NoProbe:
        def get_channel_ids(self):
            return ["a", "b"]

        def get_channel_locations(self):
            raise RuntimeError("no probe attached")

        def get_sampling_frequency(self):
            return 1000.0

    probe = CachedProbe.from_recording(NoProbe())

    assert probe.get_channel_ids() == ["a", "b"]
    assert np.isnan(probe.get_channel_locations()).all()
    # And the layout emitters' own degrade path still recognises it.
    from mea_modules.diagnostics.channel_layout import _channel_xy

    assert _channel_xy(probe)[0] == ["a", "b"]


def test_locations_that_do_not_match_the_channels_are_refused_at_construction():
    with pytest.raises(ValueError, match="n_channels"):
        CachedProbe(CHANNEL_IDS, [[0.0, 0.0]])
