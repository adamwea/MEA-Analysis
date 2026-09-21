"""The producer half: what a capsule collects is what the tool can draw.

`collect_segment_diagnostics` is the one computation both sides agree on. These
tests hold it to the two things the split actually needs:

* everything a figure reads comes back from ONE call, survives a round trip
  through the cache files, and lands under the key the real emitter reads;
* a figure drawn from that cache is byte-for-byte the figure drawn live.

Fixtures come from the real producer and go to the real consumer on purpose. Two
defects shipped on 2026-09-20 because a test hand-built the dict it was checking:
the cache stored spectra under key names `welch_spectra` does not use (the round
trip passed, the figure would have raised), and the figure-identity proof used
the one unit setting where a unit bug is invisible. A test that builds its own
input agrees with the bug.
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from mea_modules.diagnostics import (
    collect_segment_diagnostics,
    plot_raster_threshold,
    plot_spectra_panels,
    plot_traces,
    read_cache,
    write_cache,
)

FS_HZ = 10_000.0
N_CHANNELS = 12
N_FRAMES = 60_000
PITCH_UM = 17.5
WINDOW_S = 2.0
MAD_THRESHOLD = 5.0


def _locations():
    return np.array(
        [[(i % 4) * PITCH_UM, (i // 4) * PITCH_UM] for i in range(N_CHANNELS)],
        dtype=float,
    )


def _recording():
    from probeinterface import Probe
    from spikeinterface.core import NumpyRecording

    rng = np.random.default_rng(7)
    traces = rng.normal(0.0, 8.0, (N_FRAMES, N_CHANNELS)).astype(np.float32)
    # Sparse large negative excursions, so there is something to detect and the
    # two paths have something to disagree about.
    traces[500:40_000:287, :] = -180.0

    recording = NumpyRecording([traces], sampling_frequency=FS_HZ)
    probe = Probe(ndim=2)
    probe.set_contacts(positions=_locations(), shapes="square", shape_params={"width": 5})
    probe.set_device_channel_indices(np.arange(N_CHANNELS))
    return recording.set_probe(probe)


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def recording():
    return _recording()


@pytest.fixture(scope="module")
def collected(recording):
    """One collection pass, exactly as a capsule would make it.

    raw_view and qc_rec are the same object here -- this synthetic recording is
    already float and unreferenced -- which keeps the test about the collection
    contract rather than about the filter chain. The chain distinction is R10's
    business and is asserted where the capsule builds it.
    """
    return collect_segment_diagnostics(
        recording,
        recording,
        source="preprocessed",
        duration_s=1.0,
        num_chunks=2,
        seed=17,
        mad_threshold=MAD_THRESHOLD,
        dead_noise_ratio=0.1,
        artifacts_duration_s=1.0,
        window_s=WINDOW_S,
        trace_channels=4,
        raster_max_channels=64,
    )


# --------------------------------------------------------------------------
# what one call has to produce
# --------------------------------------------------------------------------


def test_one_call_produces_everything_the_figures_need(collected):
    assert set(collected) == {
        "probe", "metrics", "traces", "events", "spectra", "time_gaps", "meta",
    }
    metrics = collected["metrics"]
    for name in ("noise", "rms", "activity", "flags", "flagged", "clipping", "artifacts"):
        assert metrics[name] is not None, f"{name} was not collected"
    assert set(collected["traces"]) == {"preprocessed", "raw"}
    assert "raster" in collected["events"]
    assert set(collected["spectra"]) == {"raw", "preprocessed"}


def test_nothing_failed_quietly(collected):
    """`errors` is the record of what did not compute. On a clean input it is
    empty -- and when it is not, the cache says so rather than hiding the gap."""
    assert collected["metrics"]["errors"] == {}


def test_the_rates_were_cut_at_the_noise_reported_beside_them(recording, collected):
    """The detector is handed the noise dict, so the thresholds the rates were
    counted against are k times the very MAD-sigma the noise map shows -- not a
    second estimate SpikeInterface would otherwise make for itself."""
    from mea_modules.quality import detect_events

    noise = collected["metrics"]["noise"]
    activity = collected["metrics"]["activity"]
    assert activity["detect_threshold"] == MAD_THRESHOLD
    assert "mad_noise" in activity["noise_levels"]
    assert len(activity["rate_hz"]) == len(noise["noise"])
    assert noise["seed"] == 17

    again = detect_events(recording, noise, detect_threshold=MAD_THRESHOLD)
    assert np.allclose(again["thresholds"], MAD_THRESHOLD * np.asarray(noise["noise"]))
    assert sum(activity["n_events"]) == again["frames"].size


def test_the_native_raster_is_a_view_of_the_same_detection(collected):
    """One detection serves both: the raster's events are the activity's events
    inside the raster window, so the two figures cannot disagree about what an
    event is."""
    raster = collected["metrics"]["raster"]
    assert raster["decimation_factor"] == 1
    assert raster["noise_source"] == "the segment's MAD noise"
    times = collected["events"]["raster"]["times_s"]
    assert times.size == raster["n_events"] > 0
    assert float(times.max()) < WINDOW_S
    # Every excursion planted inside the window, on every channel.
    planted = np.arange(500, int(WINDOW_S * FS_HZ), 287)
    assert raster["n_events"] == planted.size * N_CHANNELS


def test_the_rms_sits_beside_the_mad_with_its_ratio(collected):
    rms = collected["metrics"]["rms"]
    assert len(rms["rms"]) == len(rms["rms_over_mad"]) == N_CHANNELS
    assert rms["seed"] == collected["metrics"]["noise"]["seed"]
    # Measured on the same windows as the MAD, so the ratio is exactly the
    # quotient of the two numbers the maps show.
    noise = collected["metrics"]["noise"]["noise"]
    assert rms["rms_over_mad"] == pytest.approx(
        [value / mad for value, mad in zip(rms["rms"], noise)]
    )


def test_every_step_is_timed_with_its_start_and_end(collected):
    timings = collected["metrics"]["timings"]
    for name in ("noise", "rms", "detection", "activity", "raster"):
        entry = timings[name]
        assert entry["seconds"] >= 0.0
        assert entry["started"] <= entry["ended"]


def test_the_seed_travels_into_the_record(collected):
    assert collected["meta"]["sampling"]["seed"] == 17


def test_both_channel_counts_are_recorded(collected):
    """A figure's name carries the QC and the layout counts separately, and the
    renderer has no recording left to ask."""
    assert collected["meta"]["n_channels"] == N_CHANNELS
    assert collected["meta"]["n_layout_channels"] == N_CHANNELS


def test_the_trace_windows_carry_their_unit_and_their_frames(collected):
    for name in ("preprocessed", "raw"):
        block = collected["traces"][name]
        assert block["frame_offset"] == 0
        assert block["sampling_frequency"] == FS_HZ
        assert block["unit"] in ("uV", "ADC")
        assert block["traces"].shape[0] == int(WINDOW_S * FS_HZ)


# --------------------------------------------------------------------------
# the round trip, and the figures on the far side of it
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cached(tmp_path_factory, collected):
    out = tmp_path_factory.mktemp("capsule_out")
    write_cache(out, **collected)
    return read_cache(out, capsule="preprocess_segment")


def test_the_metrics_survive_the_round_trip(cached, collected):
    assert cached.metric("noise")["median_noise"] == pytest.approx(
        collected["metrics"]["noise"]["median_noise"]
    )
    assert cached.meta["mad_threshold"] == MAD_THRESHOLD


def test_the_cached_spectra_draw(cached, tmp_path):
    """Taken from the real producer and handed to the real consumer -- the pair
    that would have caught the key-name defect of 2026-09-20."""
    out = plot_spectra_panels(cached.all_spectra(), tmp_path / "psd.png", annotate=False)
    assert Path(out).is_file()


def test_a_raster_from_the_collected_events_is_byte_identical(
    tmp_path, recording, cached, collected
):
    block = collected["events"]["raster"]
    live = plot_raster_threshold(
        recording, tmp_path / "live.png", max_channels=64,
        start_time_s=0.0, duration_s=WINDOW_S,
        threshold_factor=MAD_THRESHOLD, annotate=False,
        events=(block["times_s"], block["labels"]),
    )
    drawn = plot_raster_threshold(
        cached.probe, tmp_path / "cached.png", max_channels=64,
        start_time_s=0.0, duration_s=WINDOW_S,
        threshold_factor=MAD_THRESHOLD, annotate=False,
        events=cached.events("raster"),
    )
    assert _digest(live) == _digest(drawn)


def test_traces_from_the_collected_window_are_byte_identical(
    tmp_path, recording, cached, collected
):
    channel_ids = collected["metrics"]["representative_channels"]
    view = cached.traces("preprocessed")
    want_uv = view.has_scaleable_traces()

    live = plot_traces(
        recording, tmp_path / "live_tr.png", channel_ids=channel_ids,
        max_channels=4, start_time_s=0.0, duration_s=WINDOW_S,
        return_in_uV=want_uv, annotate=False,
    )
    drawn = plot_traces(
        view, tmp_path / "cached_tr.png", channel_ids=channel_ids,
        max_channels=4, start_time_s=0.0, duration_s=WINDOW_S,
        return_in_uV=want_uv, annotate=False,
    )
    assert _digest(live) == _digest(drawn)


def test_the_cached_probe_refuses_to_serve_samples(cached):
    """The proof that a redraw genuinely recomputes nothing: any path that still
    reached for traces would fail here instead of quietly re-deriving them."""
    with pytest.raises(TypeError, match="geometry only"):
        cached.probe.get_traces(start_frame=0, end_frame=1)
