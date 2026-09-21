"""Detecting on a downsampled signal.

The raster may detect on an anti-aliased decimation of the segment instead of
the native signal. Two things have to hold for that to be honest: the rate it
reports is the rate it actually ran at, and the decimation filters before it
strides -- a plain stride folds the full-band noise above the new Nyquist back
into the spike band, raises the threshold and changes the count, in a figure
that still renders.
"""
import numpy as np
import pytest
from spikeinterface.core import NumpyRecording

from mea_modules.quality import mad_noise
from mea_modules.quality.detection import (
    detect_events,
    resample_rate_for,
    resampled_for_detection,
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


def _noise(recording):
    return mad_noise(
        recording, duration_s=2.0, num_chunks=1, seed=0,
        highpass_hz=None, return_in_uV=False,
    )


def test_the_rate_reported_is_the_rate_achieved_not_the_rate_asked():
    """10 kHz asked down to 3 kHz detects at 3333.3 Hz? No: SpikeInterface's
    resampler takes its anti-aliased decimation path only for a whole-hertz
    rate that divides the native one, so the factor steps up to the next one
    that does, and the result says so."""
    assert resample_rate_for(FS, 5000.0) == (2, 5000.0)
    assert resample_rate_for(FS, 3000.0) == (4, 2500.0)
    factor, effective = resample_rate_for(FS, 2000.0)
    assert (factor, effective) == (5, 2000.0)
    assert FS / factor == int(FS / factor)


@pytest.mark.parametrize("target", [None, 0, 0.0, FS, FS * 2])
def test_no_target_or_a_target_at_or_above_the_native_rate_is_the_native_path(target):
    """Factor 1 is not a special case to be handled, it is the default."""
    assert resample_rate_for(FS, target) == (1, FS)
    recording, _ = _recording(seconds=0.2)
    same, factor, effective = resampled_for_detection(recording, target)
    assert same is recording and factor == 1 and effective == FS


def test_a_native_rate_that_is_not_whole_hertz_stays_native():
    """No whole-hertz rate divides 20000.5 Hz, so the search for one used to
    run forever; the native rate is the only honest answer."""
    assert resample_rate_for(20000.5, 3000.0) == (1, 20000.5)


def test_native_detection_finds_each_planted_spike_once_per_channel():
    """The 1 ms exclusion sweep is what stops a ten-sample trough counting ten times."""
    recording, planted = _recording()
    events = detect_events(recording, _noise(recording), detect_threshold=5.0)
    assert events["frames"].size == planted.size * 4
    assert set(np.unique(events["labels"]).tolist()) == {0, 1, 2, 3}


def test_downsampled_detection_finds_the_same_spikes_on_a_coarser_grid():
    recording, planted = _recording()
    decimated, factor, effective = resampled_for_detection(recording, 5000.0)
    assert (factor, effective) == (2, 5000.0)
    assert decimated.get_sampling_frequency() == pytest.approx(5000.0)
    events = detect_events(decimated, _noise(decimated), detect_threshold=5.0)
    assert events["frames"].size == planted.size * 4
    # Mapped back to native frames, every event sits within one decimated
    # sample of a planted trough's centre.
    native = events["frames"] * factor
    centres = planted + int(0.0005 * FS)
    nearest = np.abs(native[:, None] - centres[None, :]).min(axis=1)
    assert nearest.max() <= 2 * factor


def test_thresholds_are_estimated_on_the_band_the_detector_will_see():
    """The anti-alias filter removes real power, so the decimated MAD is
    genuinely smaller. Estimating at native rate and detecting at a lower one
    would apply a threshold calibrated to a band the detector never sees."""
    recording, _ = _recording()
    decimated, _, _ = resampled_for_detection(recording, 2000.0)
    native = np.asarray(_noise(recording)["noise"])
    band = np.asarray(_noise(decimated)["noise"])
    assert native.size == band.size == 4
    assert np.all(band < native)


def test_decimation_does_not_alias_the_noise_floor_upward():
    """White noise decimated with an anti-alias filter keeps roughly 1/5 of its
    power; a naive stride keeps all of it. The filtered estimate must be the
    smaller of the two, by a clear margin rather than by rounding."""
    rng = np.random.default_rng(1)
    traces = rng.normal(0.0, 1.0, size=(int(2.0 * FS), 2)).astype(np.float32)
    recording = NumpyRecording([traces], sampling_frequency=FS, channel_ids=[0, 1])

    decimated, _, _ = resampled_for_detection(recording, 2000.0)
    filtered = np.asarray(_noise(decimated)["noise"])
    strided = np.median(np.abs(traces[::5] - np.median(traces[::5], axis=0)), axis=0) / 0.6745

    assert np.all(filtered < strided * 0.8)
