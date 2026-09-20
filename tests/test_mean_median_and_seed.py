"""Both centres get reported, and every sampled window answers to a seed.

Two properties, tested together because they are the same property seen twice:
a sampled diagnostic is only reproducible if the windows are fixed, and only
interpretable if you can see whether the robust summary was doing any work.

The synthetic well: 1000 Hz, four channels, ten seconds, unit-sigma Gaussian
noise. Two faults are injected separately, into frames 4000..5000 of channel 1:

* a NOISIER stretch (sigma 40) -- what the noise estimate is supposed to reject,
* SPARSE large spikes -- what the crossing count is supposed to find.

They cannot share a recording: a stretch noisy enough to move the MAD also
raises that window's own detection threshold, which is exactly the behaviour
being relied on, so the two faults would cancel.

Sampling is `placement="strided"` for the aggregation cases, so the windows are
evenly spaced and it is decidable which one the fault lands in: ten windows over
ten seconds start at frames 0, 1055, 2111, 3166, 4222 ... -- window 4 sits
wholly inside the fault and the other nine sit wholly outside it. The seed cases
use `placement="random"`, which is the path a seed actually governs.
"""

import numpy as np
import pytest

from mea_modules.diagnostics.traces import channel_activity_rms
from mea_modules.quality import activity_rate, mad_noise

FS_HZ = 1000.0
N_CHANNELS = 4
N_SAMPLES = 10_000
SIGMA = 1.0
FAULT = slice(4000, 5000)
FAULT_CHANNEL = 1

STRIDED = {
    "duration_s": 5.0,
    "num_chunks": 10,
    "placement": "strided",
    "highpass_hz": None,
    "return_in_uV": False,
}


def _traces(seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, SIGMA, (N_SAMPLES, N_CHANNELS)).astype(np.float32)


def _wrap(traces):
    from spikeinterface.core import NumpyRecording

    return NumpyRecording([traces], sampling_frequency=FS_HZ)


def _clean_recording():
    return _wrap(_traces())


def _noisy_window_recording():
    """One window's worth of much louder noise, on one channel."""
    traces = _traces()
    rng = np.random.default_rng(1)
    traces[FAULT, FAULT_CHANNEL] += rng.normal(
        0.0, 40.0 * SIGMA, FAULT.stop - FAULT.start
    ).astype(np.float32)
    return _wrap(traces)


def _sparse_spike_recording():
    """Forty large negative excursions inside one window, on one channel.

    Sparse on purpose: too few to move that window's MAD, which is what lets
    them clear a threshold derived from it.
    """
    traces = _traces()
    traces[FAULT.start : FAULT.stop : 25, FAULT_CHANNEL] = -20.0 * SIGMA
    return _wrap(traces)


# --------------------------------------------------------------------------
# both centres
# --------------------------------------------------------------------------


def test_mad_noise_reports_the_mean_beside_the_median():
    result = mad_noise(_noisy_window_recording(), **STRIDED)

    median_estimate = result["noise"][FAULT_CHANNEL]
    mean_estimate = result["noise_mean"][FAULT_CHANNEL]

    # The median is the estimate, and the loud window did not move it.
    assert median_estimate == pytest.approx(SIGMA, rel=0.25)
    # The mean did move -- which is the whole reason it is worth reporting.
    assert mean_estimate > 2.0 * median_estimate
    assert result["window_aggregation"] == "median"


def test_the_two_centres_agree_when_there_is_nothing_to_reject():
    result = mad_noise(_clean_recording(), **STRIDED)

    for median_estimate, mean_estimate in zip(result["noise"], result["noise_mean"]):
        assert mean_estimate == pytest.approx(median_estimate, rel=0.2)


def test_only_the_faulty_channel_shows_the_two_centres_apart():
    result = mad_noise(_noisy_window_recording(), **STRIDED)

    for channel in range(N_CHANNELS):
        if channel == FAULT_CHANNEL:
            continue
        assert result["noise_mean"][channel] == pytest.approx(
            result["noise"][channel], rel=0.2
        )


def test_mad_noise_summarises_across_channels_both_ways():
    result = mad_noise(_noisy_window_recording(), **STRIDED)

    assert result["median_noise"] == pytest.approx(float(np.median(result["noise"])))
    assert result["mean_noise"] == pytest.approx(float(np.mean(result["noise"])))


def test_activity_rate_reports_the_pooled_rate_and_the_window_median():
    result = activity_rate(
        _sparse_spike_recording(), threshold_sd=5.0, polarity="negative", **STRIDED
    )

    pooled = result["rate_hz"][FAULT_CHANNEL]
    per_window_median = result["rate_hz_median_window"][FAULT_CHANNEL]

    # The spikes sit in one window of ten, so the pooled rate (a
    # duration-weighted mean) carries them and the window median does not. A
    # channel where those two disagree fired in a burst, not steadily.
    assert pooled > 0.0
    assert per_window_median == 0.0
    assert result["window_aggregation"] == "duration-weighted mean"
    assert result["mean_rate_hz"] == pytest.approx(float(np.mean(result["rate_hz"])))
    assert result["median_rate_hz"] == pytest.approx(float(np.median(result["rate_hz"])))


def test_per_segment_rates_report_both_centres():
    from mea_modules.diagnostics.segment_event_rates import _per_channel_row

    # Activity concentrated in one electrode out of five: the shape these
    # arrays actually have, and the shape a mean misreports on its own.
    row = _per_channel_row([0, 0, 1, 1, 200], recorded_s=10.0)

    assert row["median_events_per_s_per_channel"] == pytest.approx(0.1)
    assert row["mean_events_per_s_per_channel"] > row["median_events_per_s_per_channel"]
    assert row["n_channels_measured"] == 5


def test_an_unmeasured_segment_reports_neither_centre():
    from mea_modules.diagnostics.segment_event_rates import _per_channel_row

    row = _per_channel_row([0, 0, 0], recorded_s=0.0)

    assert row["mean_events_per_s_per_channel"] is None
    assert row["median_events_per_s_per_channel"] is None


# --------------------------------------------------------------------------
# the seed
# --------------------------------------------------------------------------


def _random_noise(recording, seed):
    return mad_noise(
        recording, duration_s=2.0, num_chunks=8, seed=seed,
        placement="random", highpass_hz=None, return_in_uV=False,
    )


def test_random_window_placement_is_reproducible_under_one_seed():
    recording = _noisy_window_recording()

    first = _random_noise(recording, 7)
    again = _random_noise(recording, 7)
    other = _random_noise(recording, 8)

    assert first["noise"] == again["noise"]
    assert first["seed"] == 7
    # A seed that changes nothing is a seed that is not plumbed. The faulty
    # channel is the one whose value depends on which windows were drawn.
    assert first["noise"][FAULT_CHANNEL] != other["noise"][FAULT_CHANNEL]


def test_channel_activity_scoring_follows_its_seed():
    recording = _noisy_window_recording()

    same = [channel_activity_rms(recording, seed=3, return_in_uV=False) for _ in range(2)]
    other = channel_activity_rms(recording, seed=4, return_in_uV=False)

    assert np.array_equal(same[0], same[1])
    assert not np.array_equal(same[0], other)


def test_motion_estimation_seeds_the_chunks_its_noise_is_drawn_from(monkeypatch):
    """SpikeInterface defaults that picker to ``seed=None`` -- OS entropy, and
    not the process-wide numpy seed, so an unseeded call is genuinely
    irreproducible. Everything downstream is deterministic once the noise levels
    are fixed, so this one kwarg is what makes the figure repeat.
    """
    import spikeinterface.core as si_core
    import spikeinterface.sortingcomponents.motion as si_motion
    import spikeinterface.sortingcomponents.peak_detection as si_detect
    import spikeinterface.sortingcomponents.peak_localization as si_localize

    from mea_modules.diagnostics.motion import estimate_motion_over_recording

    seen = {}

    def fake_noise(recording, return_in_uV=True, **kwargs):
        seen.update(kwargs)
        return np.ones(N_CHANNELS, dtype=float)

    class FakeMotion:
        displacement = [np.zeros(3)]
        temporal_bins_s = [np.arange(3, dtype=float)]

    monkeypatch.setattr(si_core, "get_noise_levels", fake_noise)
    monkeypatch.setattr(si_detect, "detect_peaks", lambda *a, **k: np.zeros(0))
    monkeypatch.setattr(si_localize, "localize_peaks", lambda *a, **k: np.zeros(0))
    monkeypatch.setattr(si_motion, "estimate_motion", lambda *a, **k: FakeMotion())

    summary = estimate_motion_over_recording(_clean_recording(), well="well000", seed=11)

    assert seen["random_slices_kwargs"] == {"seed": 11}
    assert summary["seed"] == 11
