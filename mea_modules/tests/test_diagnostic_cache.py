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
    CachedTraces,
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
N_FRAMES = 100_000
TRACE_OFFSET = 2_000
TRACE_FRAMES = 500


def _probe():
    return CachedProbe(
        CHANNEL_IDS, LOCATIONS, sampling_frequency=FS_HZ, num_frames=N_FRAMES
    )


def _window():
    return np.arange(TRACE_FRAMES * 2, dtype=np.float32).reshape(TRACE_FRAMES, 2)


def _welch_result():
    """A REAL welch_spectra result, never a hand-built stand-in.

    Hand-building it is how this file previously agreed with a cache that used
    the wrong key names: the cache stored `frequencies`/`psd`, `welch_spectra`
    returns `freqs`/`power`, and `plot_spectra_panels` reads `freqs`/`power` --
    so the round trip passed and the figure would have raised KeyError. Taking
    the dict from the producer is what makes this test load-bearing.
    """
    from spikeinterface.core import NumpyRecording

    from mea_modules.diagnostics.spectra import welch_spectra

    rng = np.random.default_rng(0)
    traces = rng.normal(0.0, 1.0, (4096, 2)).astype(np.float32)
    recording = NumpyRecording([traces], sampling_frequency=FS_HZ)
    return welch_spectra(
        recording, list(recording.get_channel_ids()), duration_s=None, return_in_uV=False
    )


def _write(tmp_path, **kwargs):
    payload = dict(
        probe=_probe(),
        metrics={"noise": {"channel_ids": CHANNEL_IDS, "noise": [1.0, 1.1, 0.9, 1.2]}},
        traces={
            "preprocessed": {
                "channel_ids": ["e1", "e2"],
                "traces": _window(),
                "frame_offset": TRACE_OFFSET,
                "sampling_frequency": FS_HZ,
                "unit": "uV",
            }
        },
        events={"raster": {"times_s": [0.1, 0.5, 0.9], "labels": [0, 2, 2]}},
        spectra={"raw": _welch_result()},
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

    view = cache.traces("preprocessed")
    assert view.get_channel_ids() == ["e1", "e2"]
    assert view.get_sampling_frequency() == FS_HZ
    assert view.unit == "uV"
    # Geometry rides along, restricted to the cached channels, so a layout
    # inset beside the traces draws without a second lookup.
    assert view.get_channel_locations().tolist() == LOCATIONS[:2]

    times, labels = cache.events("raster")
    assert times.tolist() == [0.1, 0.5, 0.9]
    assert labels.tolist() == [0, 2, 2]

    spectra = cache.all_spectra()
    assert list(spectra) == ["raw"]
    live = _welch_result()
    assert np.allclose(spectra["raw"]["freqs"], live["freqs"])
    assert np.allclose(spectra["raw"]["power"], live["power"])
    # Non-array keys survive, so the panel emitter still knows its unit.
    assert spectra["raw"]["unit"] == live["unit"]
    assert spectra["raw"]["nperseg"] == live["nperseg"]


def test_cached_spectra_actually_draw(tmp_path):
    """The round trip is worthless if the panel emitter cannot read the result.

    `welch_spectra` returns `freqs`/`power` and `plot_spectra_panels` reads
    those same names. A cache that renamed them would pass every round-trip
    assertion above and then raise KeyError on the one call that matters, so
    that call is made here.
    """
    from mea_modules.diagnostics import plot_spectra_panels

    _write(tmp_path)
    cache = read_cache(tmp_path)

    out = plot_spectra_panels(cache.all_spectra(), tmp_path / "psd.png", annotate=False)

    assert out.is_file()
    assert out.stat().st_size > 0


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


def test_the_probe_carries_the_span_that_was_analysed():
    """A raster sets its x limits from the ANALYSED window, not from where the
    last event happened, so a window where firing stops halfway still shows the
    silence. That span has to survive into the cache."""
    assert _probe().get_num_frames() == N_FRAMES


# --------------------------------------------------------------------------
# the trace view is frame-faithful
# --------------------------------------------------------------------------


def _view():
    return CachedTraces(
        ["e1", "e2"], _window(), sampling_frequency=FS_HZ, frame_offset=TRACE_OFFSET
    )


def test_cached_traces_are_addressed_in_the_original_frame_numbers():
    """Renumbering the samples would put the real-elapsed gap shading in the
    wrong place, because the emitters compute it from frame numbers."""
    view = _view()

    assert view.get_num_frames() == TRACE_OFFSET + TRACE_FRAMES
    assert view.start_time_s == TRACE_OFFSET / FS_HZ
    assert view.duration_s == TRACE_FRAMES / FS_HZ

    block = view.get_traces(start_frame=TRACE_OFFSET, end_frame=TRACE_OFFSET + 10)
    assert block.shape == (10, 2)
    assert np.array_equal(block, _window()[:10, :])


def test_a_cached_window_refuses_frames_it_does_not_hold():
    view = _view()

    with pytest.raises(ValueError, match="outside the cached window"):
        view.get_traces(start_frame=0, end_frame=10)
    with pytest.raises(ValueError, match="outside the cached window"):
        view.get_traces(
            start_frame=TRACE_OFFSET, end_frame=TRACE_OFFSET + TRACE_FRAMES + 1
        )


def test_a_cached_window_can_be_read_one_channel_at_a_time():
    view = _view()

    both = view.get_traces(start_frame=TRACE_OFFSET, end_frame=TRACE_OFFSET + 5)
    second = view.get_traces(
        start_frame=TRACE_OFFSET, end_frame=TRACE_OFFSET + 5, channel_ids=["e2"]
    )

    assert second.shape == (5, 1)
    assert np.array_equal(second[:, 0], both[:, 1])


def test_a_cached_window_round_trips_through_the_cache(tmp_path):
    _write(tmp_path)
    view = read_cache(tmp_path).traces("preprocessed")

    assert np.array_equal(
        view.get_traces(start_frame=TRACE_OFFSET, end_frame=TRACE_OFFSET + 20),
        _window()[:20, :],
    )


# --------------------------------------------------------------------------
# the unit travels with the window, or the figure lies about it
# --------------------------------------------------------------------------


def test_a_window_cached_in_microvolts_says_so():
    """The emitters label their axes from `has_scaleable_traces`.

    A µV window that answered False here would draw real microvolts under a
    "device counts (ADC)" label: right numbers, wrong unit, and nothing on the
    figure to catch it.
    """
    from mea_modules.diagnostics.traces import _effective_uv

    view = CachedTraces(
        ["e1", "e2"], _window(), sampling_frequency=FS_HZ, unit="uV"
    )

    assert view.has_scaleable_traces() is True
    assert _effective_uv(view, True) is True


def test_a_window_cached_in_device_counts_says_that_instead():
    from mea_modules.diagnostics.traces import _effective_uv

    view = CachedTraces(
        ["e1", "e2"], _window(), sampling_frequency=FS_HZ, unit="adc"
    )

    assert view.has_scaleable_traces() is False
    assert _effective_uv(view, False) is False


def test_a_cached_window_refuses_to_be_served_in_the_other_unit():
    """It cannot convert -- the gain never travelled with the samples -- so it
    refuses rather than returning the wrong unit under the right name."""
    view = CachedTraces(["e1", "e2"], _window(), sampling_frequency=FS_HZ, unit="uV")

    with pytest.raises(ValueError, match="cannot be served"):
        view.get_traces(start_frame=0, end_frame=10, return_in_uV=False)


def test_a_window_with_no_recorded_unit_serves_either_request():
    """Older caches predate the unit field; they must still draw."""
    view = CachedTraces(["e1", "e2"], _window(), sampling_frequency=FS_HZ)

    assert view.get_traces(start_frame=0, end_frame=5, return_in_uV=True).shape == (5, 2)
    assert view.get_traces(start_frame=0, end_frame=5, return_in_uV=False).shape == (5, 2)


# --------------------------------------------------------------------------
# version 2: gap tables travel with the cache, and a stale cache is not ready
# --------------------------------------------------------------------------

_NATIVE_GAPS = {
    "gaps": {"break_sample_indices": [300, 4_500], "break_gap_frames": [3, 11]},
    "segment_gaps": [{"start_sample": 4_000, "gap_before_s": 30.0}],
    "n_samples": N_FRAMES,
}


def test_a_gap_table_round_trips_in_the_shape_an_emitter_takes(tmp_path):
    from mea_modules.diagnostics.timebase import sample_times

    _write(tmp_path, time_gaps={"native": _NATIVE_GAPS, "absent": None})
    cached = read_cache(tmp_path, capsule="preprocess_segment")

    assert cached.time_gap_names() == ["native"]
    assert cached.time_gaps("absent") is None
    structure = cached.time_gaps("native")
    assert list(structure["gaps"]["break_sample_indices"]) == [300, 4_500]
    assert list(structure["gaps"]["break_gap_frames"]) == pytest.approx([3.0, 11.0])
    assert structure["segment_gaps"] == _NATIVE_GAPS["segment_gaps"]
    assert structure["n_samples"] == N_FRAMES

    frames = np.arange(0, N_FRAMES, 997)
    assert np.allclose(
        sample_times(frames, FS_HZ, gaps=structure["gaps"], segment_gaps=structure["segment_gaps"]),
        sample_times(frames, FS_HZ, gaps=_NATIVE_GAPS["gaps"],
                     segment_gaps=_NATIVE_GAPS["segment_gaps"]),
    )


def test_a_decimated_trace_block_records_its_step_and_its_gap_table(tmp_path):
    traces = {
        "preprocessed": {
            "channel_ids": ["e1", "e2"],
            "traces": _window(),
            "frame_offset": 0,
            "sampling_frequency": FS_HZ / 20,
            "unit": "uV",
            "frame_step": 20,
            "time_gaps": "traces",
        }
    }
    _write(tmp_path, traces=traces)
    block = read_cache(tmp_path).trace_block("preprocessed")
    assert block["frame_step"] == 20
    assert block["time_gaps"] == "traces"
    assert block["sampling_frequency"] == FS_HZ / 20


def test_a_full_rate_block_defaults_to_step_one(tmp_path):
    _write(tmp_path)
    block = read_cache(tmp_path).trace_block("preprocessed")
    assert block["frame_step"] == 1
    assert block["time_gaps"] is None


def test_an_older_cache_is_refused_and_says_which_capsule_to_rerun(tmp_path):
    _write(tmp_path)
    record_path = cache_dir(tmp_path) / RECORD_NAME
    record = json.loads(record_path.read_text())
    record["version"] = CACHE_VERSION - 1
    record_path.write_text(json.dumps(record))

    with pytest.raises(CacheVersionMismatch, match="Re-run the `preprocess_segment` capsule"):
        read_cache(tmp_path, capsule="preprocess_segment")


def test_cache_ready_means_ready_for_this_reader(tmp_path):
    """A resume check that accepted any cache would keep one no suite can draw."""
    from mea_modules.diagnostics.cache import cache_version

    assert cache_version(tmp_path) is None and cache_ready(tmp_path) is False
    _write(tmp_path)
    assert cache_version(tmp_path) == CACHE_VERSION and cache_ready(tmp_path) is True

    record_path = cache_dir(tmp_path) / RECORD_NAME
    record = json.loads(record_path.read_text())
    record["version"] = CACHE_VERSION - 1
    record_path.write_text(json.dumps(record))
    assert cache_version(tmp_path) == CACHE_VERSION - 1
    assert cache_ready(tmp_path) is False


def test_traces_drawn_from_cache_on_the_real_elapsed_axis_are_byte_identical(tmp_path):
    """The cached gap table and the cached window together redraw the live
    real-elapsed figure exactly: the shading and the time axis both come out of
    the round trip."""
    import hashlib

    from probeinterface import Probe
    from spikeinterface.core import NumpyRecording

    from mea_modules.diagnostics import plot_traces

    rng = np.random.default_rng(3)
    samples = rng.normal(0.0, 5.0, (N_FRAMES, 4)).astype(np.float32)
    recording = NumpyRecording([samples], sampling_frequency=FS_HZ, channel_ids=CHANNEL_IDS)
    probe = Probe(ndim=2)
    probe.set_contacts(positions=np.asarray(LOCATIONS), shapes="square", shape_params={"width": 5})
    probe.set_device_channel_indices(np.arange(4))
    recording.set_probe(probe, in_place=True)

    window = int(0.6 * FS_HZ)
    traces = {
        "preprocessed": {
            "channel_ids": ["e1", "e3"],
            "traces": recording.get_traces(start_frame=0, end_frame=window, channel_ids=["e1", "e3"]),
            "frame_offset": 0,
            "sampling_frequency": FS_HZ,
            "unit": None,
            "time_gaps": "native",
        }
    }
    _write(tmp_path, probe=CachedProbe.from_recording(recording), traces=traces,
           time_gaps={"native": _NATIVE_GAPS})
    cached = read_cache(tmp_path)

    kwargs = dict(channel_ids=["e1", "e3"], start_time_s=0.0, duration_s=0.6,
                  return_in_uV=False, annotate=False)
    live = plot_traces(recording, tmp_path / "live.png", time_gaps=_NATIVE_GAPS, **kwargs)
    drawn = plot_traces(cached.traces("preprocessed"), tmp_path / "cached.png",
                        time_gaps=cached.time_gaps("native"), **kwargs)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()  # noqa: E731
    assert digest(live) == digest(drawn)
