"""Downsampled threshold detection.

The native-rate path is the contract: the detection loop was rewritten to count
in detection frames so that one implementation serves both rates, and the proof
that the rewrite was faithful is that factor 1 still finds exactly what it found
before — on real data, against a cache written by the old code, and here on a
synthetic signal whose answer is known by construction.
"""
import numpy as np
import pytest
from spikeinterface.core import NumpyRecording

from mea_modules.diagnostics.raster import (
    decimation_for,
    detect_threshold_crossings,
    estimate_channel_thresholds,
)

FS = 10_000.0


def _recording(n_channels=4, seconds=2.0, spike_every_s=0.05, seed=0):
    """Noise with a negative spike planted on a known grid.

    The spike is a 1 ms trough, which is what a real one is: at 10 kHz that is
    ten samples, and it survives decimation down to about 2.5 kHz.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    traces = rng.normal(0.0, 1.0, size=(n, n_channels)).astype(np.float32)
    width = int(0.001 * FS)
    planted = np.arange(int(0.01 * FS), n - width, int(spike_every_s * FS))
    shape = -30.0 * np.sin(np.linspace(0.0, np.pi, width))
    for start in planted:
        traces[start:start + width, :] += shape[:, None].astype(np.float32)
    return (
        NumpyRecording([traces], sampling_frequency=FS, channel_ids=list(range(n_channels))),
        planted,
    )


def test_decimation_for_reports_what_was_achieved_not_what_was_asked():
    """10 kHz asked down to 3 kHz detects at 3333 Hz, and says so.

    A whole number of native samples per detection sample is what keeps an
    event's frame exactly recoverable; the price is that the rate you get is
    rarely the rate you typed, and a cache recording the request would describe
    a run that did not happen."""
    assert decimation_for(FS, 3000.0) == (3, pytest.approx(FS / 3))
    assert decimation_for(FS, 5000.0) == (2, 5000.0)


@pytest.mark.parametrize("target", [None, 0, 0.0, FS, FS * 2])
def test_no_target_or_a_target_at_or_above_the_native_rate_is_the_native_path(target):
    """Factor 1 is not a special case to be handled, it is the default."""
    assert decimation_for(FS, target) == (1, FS)


def test_native_rate_detection_is_unchanged_by_the_detection_frame_rewrite():
    """The loop now counts in detection frames; at factor 1 that IS native."""
    recording, planted = _recording()
    times, labels = detect_threshold_crossings(
        recording, threshold_factor=5.0, start_time_s=0.0, duration_s=2.0
    )
    # One event per planted spike per channel, and no more: the refractory
    # period is what stops a ten-sample trough counting ten times.
    assert times.size == planted.size * 4
    assert set(np.unique(labels).tolist()) == {0, 1, 2, 3}


def test_downsampled_detection_finds_the_same_spikes_on_a_coarser_grid():
    """Decimating to 5 kHz still resolves a 1 ms trough; the times quantize."""
    recording, planted = _recording()
    times, _ = detect_threshold_crossings(
        recording, threshold_factor=5.0, start_time_s=0.0, duration_s=2.0,
        downsample_to_hz=5000.0,
    )
    assert times.size == planted.size * 4
    # Every event lands on the decimated grid: two native samples apart.
    frames = np.rint(times * FS).astype(int)
    assert set((frames % 2).tolist()) == {0}


def test_an_event_frame_is_exactly_recoverable_at_any_factor():
    """Times are seconds on the recording's own clock either way, so nothing
    downstream has to know which rate ran. What must hold is that a decimated
    event sits on a native sample, never between two."""
    recording, _ = _recording()
    for target in (None, 5000.0, 2000.0):
        times, _ = detect_threshold_crossings(
            recording, threshold_factor=5.0, start_time_s=0.0, duration_s=2.0,
            downsample_to_hz=target,
        )
        frames = times * FS
        assert np.allclose(frames, np.rint(frames))


def test_thresholds_are_estimated_on_the_band_the_detector_will_see():
    """The anti-alias filter removes real power, so the decimated MAD is
    genuinely smaller. Estimating at native rate and detecting at a lower one
    would apply a threshold calibrated to a band the detector never sees."""
    recording, _ = _recording()
    native = estimate_channel_thresholds(recording, threshold_factor=5.0, duration_s=2.0)
    decimated = estimate_channel_thresholds(
        recording, threshold_factor=5.0, duration_s=2.0, downsample_to_hz=2000.0
    )
    assert native.size == decimated.size == 4
    assert np.all(decimated < native)


def test_decimation_does_not_alias_the_noise_floor_upward():
    """A plain stride would fold everything above the new Nyquist back into the
    band and INFLATE the measured noise. Anti-aliased decimation must not."""
    rng = np.random.default_rng(1)
    n = int(2.0 * FS)
    traces = rng.normal(0.0, 1.0, size=(n, 2)).astype(np.float32)
    recording = NumpyRecording([traces], sampling_frequency=FS, channel_ids=[0, 1])

    decimated = estimate_channel_thresholds(
        recording, threshold_factor=1.0, duration_s=2.0, downsample_to_hz=2000.0
    )
    strided = np.median(np.abs(traces[::5] - np.median(traces[::5], axis=0)), axis=0) / 0.6745

    # White noise decimated with an anti-alias filter keeps roughly 1/5 of its
    # power; a naive stride keeps all of it. The filtered estimate must be the
    # smaller of the two, by a clear margin rather than by rounding.
    assert np.all(decimated < strided * 0.8)


def test_a_window_shorter_than_one_detection_frame_yields_nothing_rather_than_raising():
    recording, _ = _recording(seconds=0.2)
    times, labels = detect_threshold_crossings(
        recording, threshold_factor=5.0, start_time_s=0.0, duration_s=0.0,
        downsample_to_hz=2000.0,
    )
    assert times.size == 0 and labels.size == 0
