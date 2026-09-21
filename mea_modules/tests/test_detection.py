"""The one event detector: SpikeInterface's, called one fixed way.

`mea_modules.quality.detection.detect_events` is a thin wrapper, so the tests
that matter are the ones that would catch it drifting from what it wraps: the
same call made by hand finds the same peaks, the thresholds are OUR noise
estimate in the recording's native units, and a span or channel subset changes
which events come back but never where they are.
"""
import logging

import numpy as np
import pytest
from spikeinterface.core import NumpyRecording

from mea_modules.quality import mad_noise
from mea_modules.quality.detection import (
    DEFAULT_EXCLUDE_SWEEP_MS,
    SPIKEINTERFACE_VERSION,
    channel_labels,
    check_spikeinterface_version,
    detect_events,
    event_rates,
    noise_in_native_units,
)

FS = 10_000.0
N_CHANNELS = 6
N_FRAMES = 30_000


def _traces(seed=0):
    """Unit noise with 1 ms troughs planted on a known grid, channel-staggered."""
    rng = np.random.default_rng(seed)
    traces = rng.normal(0.0, 1.0, size=(N_FRAMES, N_CHANNELS)).astype(np.float32)
    width = int(0.001 * FS)
    shape = (-25.0 * np.sin(np.linspace(0.0, np.pi, width))).astype(np.float32)
    for channel in range(N_CHANNELS):
        for start in range(200 + 37 * channel, N_FRAMES - width, 700):
            traces[start:start + width, channel] += shape
    return traces


def _recording(traces=None, gains=None):
    recording = NumpyRecording(
        [_traces() if traces is None else traces], sampling_frequency=FS,
        channel_ids=[str(100 + c) for c in range(N_CHANNELS)],
    )
    if gains is not None:
        recording.set_channel_gains(gains)
        recording.set_channel_offsets(np.zeros(N_CHANNELS))
    return recording


def _noise(recording, return_in_uV=False):
    return mad_noise(
        recording, duration_s=1.0, num_chunks=3, seed=0,
        highpass_hz=None, return_in_uV=return_in_uV,
    )


def test_the_wrapper_finds_exactly_what_the_same_call_by_hand_finds():
    from spikeinterface.sortingcomponents.peak_detection import detect_peaks

    recording = _recording()
    noise = _noise(recording)
    ours = detect_events(recording, noise, detect_threshold=5.0)

    levels = np.asarray(noise["noise"], dtype=float)
    peaks = detect_peaks(
        recording, method="by_channel",
        method_kwargs={
            "peak_sign": "neg", "detect_threshold": 5.0,
            "exclude_sweep_ms": DEFAULT_EXCLUDE_SWEEP_MS, "noise_levels": levels,
        },
        job_kwargs={"n_jobs": 1, "chunk_duration": "1s", "progress_bar": False},
    )
    order = np.lexsort((peaks["channel_index"], peaks["sample_index"]))
    assert np.array_equal(ours["frames"], peaks["sample_index"][order])
    assert np.array_equal(ours["channel_index"], peaks["channel_index"][order])
    assert np.allclose(ours["thresholds"], 5.0 * levels)


def test_each_planted_trough_is_one_event_on_its_own_channel():
    recording = _recording()
    events = detect_events(recording, _noise(recording), detect_threshold=5.0)
    per_channel = np.bincount(events["channel_index"], minlength=N_CHANNELS)
    expected = [len(range(200 + 37 * c, N_FRAMES - 10, 700)) for c in range(N_CHANNELS)]
    assert per_channel.tolist() == expected
    # Sorted in time, labelled with the real electrode ids.
    assert np.all(np.diff(events["frames"]) >= 0)
    assert set(events["labels"].tolist()) == {100 + c for c in range(N_CHANNELS)}


def test_microvolt_noise_becomes_native_thresholds_through_the_gains():
    """Detection runs on native samples; a MAD measured in microvolts is
    divided back through each channel's gain, so the same peaks are found as
    on a recording whose samples ARE microvolts."""
    gains = np.linspace(0.5, 3.0, N_CHANNELS)
    native = _traces()
    scaled = _recording(native, gains=gains)
    noise_uv = _noise(scaled, return_in_uV=True)
    assert noise_uv["unit"] in ("uV", "µV")

    as_native = noise_in_native_units(scaled, noise_uv)
    assert np.allclose(as_native, np.asarray(noise_uv["noise"]) / gains)

    in_uv = _recording((native * gains).astype(np.float32))
    a = detect_events(scaled, noise_uv, detect_threshold=5.0)
    b = detect_events(in_uv, np.asarray(noise_uv["noise"]), detect_threshold=5.0)
    assert np.array_equal(a["frames"], b["frames"])
    assert np.array_equal(a["channel_index"], b["channel_index"])


def test_noise_that_came_back_through_json_still_lines_up():
    recording = _recording()
    noise = _noise(recording)
    shuffled = {
        "channel_ids": [str(cid) for cid in reversed(noise["channel_ids"])],
        "noise": list(reversed(noise["noise"])),
        "unit": noise["unit"],
    }
    assert np.allclose(
        noise_in_native_units(recording, shuffled), noise_in_native_units(recording, noise)
    )


def test_noise_missing_a_channel_is_refused():
    recording = _recording()
    noise = _noise(recording)
    partial = {"channel_ids": noise["channel_ids"][:-1], "noise": noise["noise"][:-1],
               "unit": noise["unit"]}
    with pytest.raises(ValueError, match="does not cover"):
        detect_events(recording, partial)


def test_a_span_returns_the_same_events_on_the_recordings_own_numbering():
    recording = _recording()
    noise = _noise(recording)
    whole = detect_events(recording, noise)
    start, end = 5_000, 17_000
    span = detect_events(recording, noise, start_frame=start, end_frame=end)
    assert span["start_frame"] == start and span["n_frames"] == end - start
    # Away from the span's edges (one exclusion sweep), identical.
    margin = int(DEFAULT_EXCLUDE_SWEEP_MS * FS / 1000.0) + 1
    inner = (whole["frames"] >= start + margin) & (whole["frames"] < end - margin)
    got = (span["frames"] >= start + margin) & (span["frames"] < end - margin)
    assert np.array_equal(span["frames"][got], whole["frames"][inner])
    assert np.array_equal(span["channel_index"][got], whole["channel_index"][inner])


def test_a_channel_subset_indexes_and_labels_the_subset():
    recording = _recording()
    noise = _noise(recording)
    subset = ["102", "105"]
    events = detect_events(recording, noise, channel_ids=subset)
    assert events["channel_ids"] == subset
    assert set(events["channel_index"].tolist()) == {0, 1}
    assert set(events["labels"].tolist()) == {102, 105}
    whole = detect_events(recording, noise)
    for label in (102, 105):
        assert np.array_equal(
            events["frames"][events["labels"] == label],
            whole["frames"][whole["labels"] == label],
        )


def test_a_flat_channel_fires_on_nothing_rather_than_on_everything():
    traces = _traces()
    traces[:, 2] = 0.0
    recording = _recording(traces)
    events = detect_events(recording, _noise(recording))
    assert not np.any(events["channel_index"] == 2)
    assert events["thresholds"][2] > 0.0


def test_an_empty_span_is_a_valid_empty_result():
    recording = _recording()
    events = detect_events(recording, _noise(recording), start_frame=100, end_frame=100)
    assert events["frames"].size == 0 and events["n_frames"] == 0
    assert event_rates(events)["rate_hz"] == [0.0] * N_CHANNELS


def test_the_result_carries_its_own_provenance():
    recording = _recording()
    events = detect_events(recording, _noise(recording), detect_threshold=4.0)
    assert events["detect_threshold"] == 4.0
    assert events["peak_sign"] == "neg"
    assert events["exclude_sweep_ms"] == DEFAULT_EXCLUDE_SWEEP_MS
    assert "by_channel" in events["detector"]
    assert "mad_noise" in events["noise_levels"]


def test_non_numeric_ids_fall_back_to_row_positions():
    assert channel_labels(["7", 9, "11"]).tolist() == [7, 9, 11]
    assert channel_labels(["a", "b", "c"]).tolist() == [0, 1, 2]


def test_a_different_spikeinterface_is_named_in_the_log(monkeypatch, caplog):
    import spikeinterface

    monkeypatch.setattr(spikeinterface, "__version__", "9.9.9")
    with caplog.at_level(logging.WARNING, logger="mea_modules.quality.detection"):
        assert check_spikeinterface_version() == "9.9.9"
    assert SPIKEINTERFACE_VERSION in caplog.text and "9.9.9" in caplog.text
