"""Numeric tests for the per-segment diagnostic mechanics, without a real file.

Covers the arithmetic half of `mea_modules.diagnostics`'s per-recording set:
`flag_channels` (which ids are unusable, under which rule), `robust_color_limits`
(the percentile clip behind every metric map), `clipping_census` (per-channel
rail hits) and `welch_spectra` (per-channel power). Every expected value below
is hand-computed from the synthetic input rather than recorded from a previous
run, so a silent change of rule fails the test rather than being blessed by it.

The two functions that read traces take a recording, not a path, so a
duck-typed stand-in exercises them end to end with no SpikeInterface objects,
no HDF5 file and no fixtures -- which is why these live here with the package
rather than in the pipeline repo's capsule tests. The plot emitters those
numbers feed are deliberately NOT covered here: a PNG is reviewed by eye, and
the arithmetic that decides what it shows is already pinned below.
"""

import numpy as np
import pytest

from mea_modules.diagnostics import (
    clipping_census,
    flag_channels,
    robust_color_limits,
    welch_spectra,
)


class FakeRecording:
    """The smallest object the trace-reading diagnostics actually call.

    Samples are (n_samples, n_channels), matching SpikeInterface's orientation.
    """

    def __init__(self, samples, fs_hz=10_000.0, channel_ids=None, scaleable=False):
        self._samples = np.asarray(samples)
        self._fs = float(fs_hz)
        n_channels = self._samples.shape[1]
        self._channel_ids = list(channel_ids if channel_ids is not None else range(n_channels))
        self._scaleable = bool(scaleable)

    def get_sampling_frequency(self):
        return self._fs

    def get_num_samples(self):
        return int(self._samples.shape[0])

    def get_num_channels(self):
        return int(self._samples.shape[1])

    def get_channel_ids(self):
        return list(self._channel_ids)

    def get_dtype(self):
        return self._samples.dtype

    def has_scaleable_traces(self):
        return self._scaleable

    def has_time_vector(self):
        return False

    def get_traces(self, start_frame=0, end_frame=None, channel_ids=None, return_in_uV=False):
        end_frame = self.get_num_samples() if end_frame is None else int(end_frame)
        block = self._samples[int(start_frame):end_frame]
        if channel_ids is None:
            return block
        index = [self._channel_ids.index(cid) for cid in channel_ids]
        return block[:, index]


# --------------------------------------------------------------------------
# flag_channels
# --------------------------------------------------------------------------


def test_flag_channels_separates_the_three_rules_around_the_median():
    """Median noise is 10, so at the defaults the dead line sits at 1.0 (0.1x)
    and the noisy line at 50.0 (5x). One channel is put on each side of each,
    and one is flagged only by the detector."""
    noise = {
        "channel_ids": ["a", "b", "c", "d", "e"],
        "noise": [0.5, 10.0, 10.0, 60.0, 10.0],
    }
    bad = {"bad_channel_ids": ["e"]}
    result = flag_channels(noise, bad_channels=bad)

    assert result["by_rule"]["dead"] == ["a"]        # 0.5 <= 1.0
    assert result["by_rule"]["noisy"] == ["d"]       # 60.0 >= 50.0
    assert result["by_rule"]["detector_bad"] == ["e"]
    assert result["flagged_channel_ids"] == ["a", "d", "e"]  # recording order
    assert result["n_flagged"] == 3
    assert result["flagged_fraction"] == pytest.approx(3 / 5)


def test_flag_channels_counts_nan_noise_as_dead_not_as_missing():
    """A channel whose noise could not be measured is unusable, and the
    negated comparison is the mechanism -- `values <= cut` would drop a NaN."""
    noise = {"channel_ids": [0, 1, 2], "noise": [np.nan, 10.0, 10.0]}
    result = flag_channels(noise)
    assert result["by_rule"]["dead"] == [0]
    assert result["n_flagged"] == 1


def test_flag_channels_calls_everything_dead_when_the_median_is_not_positive():
    """With a zero median there is nothing to be relative to, so no channel can
    be trusted -- and nothing may be called `noisy`, which is a ratio to a
    median that does not exist."""
    noise = {"channel_ids": ["x", "y"], "noise": [0.0, 0.0]}
    result = flag_channels(noise)
    assert result["by_rule"]["dead"] == ["x", "y"]
    assert result["by_rule"]["noisy"] == []
    assert result["flagged_fraction"] == pytest.approx(1.0)


def test_flag_channels_joins_detector_ids_across_int_and_str_id_spaces():
    """SpikeInterface hands back numpy ids; the noise dict may carry plain
    strings. The join is on str(), so a 7 matches a "7" rather than silently
    flagging nothing."""
    noise = {"channel_ids": ["7", "8"], "noise": [10.0, 10.0]}
    result = flag_channels(noise, bad_channels={"bad_channel_ids": [np.int64(7)]})
    assert result["by_rule"]["detector_bad"] == ["7"]


def test_flag_channels_ratios_are_thresholds_not_constants():
    """Both rules move with their argument: raising the dead ratio to 1.1x the
    median sweeps in the median channels themselves."""
    noise = {"channel_ids": [0, 1, 2], "noise": [1.0, 10.0, 10.0]}
    assert flag_channels(noise, dead_noise_ratio=0.1)["n_flagged"] == 1
    assert flag_channels(noise, dead_noise_ratio=1.1)["n_flagged"] == 3
    assert flag_channels(noise, noisy_noise_ratio=0.5)["by_rule"]["noisy"] == [1, 2]


# --------------------------------------------------------------------------
# robust_color_limits
# --------------------------------------------------------------------------


def test_robust_color_limits_clips_one_outlier_out_of_the_colour_range():
    """101 values, 0..99 plus a 10000: the 2nd and 98th percentiles must land
    inside the bulk, so the outlier cannot flatten the map."""
    values = list(range(100)) + [10_000.0]
    low, high = robust_color_limits(values)
    assert low == pytest.approx(2.0)
    assert high == pytest.approx(98.0)


def test_robust_color_limits_autoscales_on_a_degenerate_or_empty_range():
    """vmin == vmax renders one flat colour, so a constant array and an
    all-NaN array both hand the range back to matplotlib."""
    assert robust_color_limits([5.0, 5.0, 5.0]) == (None, None)
    assert robust_color_limits([np.nan, np.nan]) == (None, None)
    assert robust_color_limits([]) == (None, None)


# --------------------------------------------------------------------------
# clipping_census
# --------------------------------------------------------------------------


def _railed_recording():
    """Four channels over 1000 samples of int16: channel 0 sits on the positive
    dtype rail for a tenth of them, channel 1 on the negative rail for a
    twentieth, channels 2 and 3 never rail."""
    samples = np.zeros((1000, 4), dtype=np.int16)
    samples[:100, 0] = np.iinfo(np.int16).max
    samples[:50, 1] = np.iinfo(np.int16).min
    samples[:, 2] = 7
    samples[:, 3] = -7
    return FakeRecording(samples, fs_hz=1000.0)


def test_clipping_census_counts_dtype_rail_hits_per_channel():
    """One window covering the whole recording: the fractions are the hit
    counts over 1000 samples, exactly."""
    census = clipping_census(_railed_recording(), duration_s=1.0, num_chunks=1)

    assert census["sampled_s"] == pytest.approx(1.0)
    assert census["dtype"] == "int16"
    assert census["dtype_rails"] == [-32768, 32767]
    assert census["dtype_rail_fraction"] == pytest.approx([0.1, 0.05, 0.0, 0.0])
    assert census["summary"]["channels_at_dtype_rail"] == 2
    assert census["summary"]["max_dtype_rail_fraction"] == pytest.approx(0.1)


def test_clipping_census_reports_per_channel_observed_extremes():
    """The observed pair is per channel and must survive a channel that never
    leaves a constant value."""
    census = clipping_census(_railed_recording(), duration_s=1.0, num_chunks=1)
    assert census["per_channel_observed_min"] == pytest.approx([0.0, -32768.0, 7.0, -7.0])
    assert census["per_channel_observed_max"] == pytest.approx([32767.0, 0.0, 7.0, -7.0])
    # The observed-extreme fractions are counted against the GLOBAL extremes,
    # which here are the dtype rails themselves -- so they agree with the rail
    # fractions, and the constant channels register nothing.
    assert census["observed_extremes"] == {"min": -32768.0, "max": 32767.0}
    assert census["observed_extreme_fraction"] == pytest.approx([0.1, 0.05, 0.0, 0.0])


def test_clipping_census_reports_no_dtype_rails_on_a_float_recording():
    """A float recording has no iinfo bounds. This is the failure mode of
    handing the census the filtered chain instead of the raw view: it does not
    raise, it quietly reports nothing."""
    census = clipping_census(
        FakeRecording(np.zeros((100, 2), dtype=np.float32), fs_hz=1000.0),
        duration_s=0.1,
        num_chunks=1,
    )
    assert census["dtype_rails"] == [None, None]
    assert census["dtype_rail_fraction"] == [None, None]
    assert census["summary"]["channels_at_dtype_rail"] is None


def test_clipping_census_spreads_its_budget_over_strided_windows():
    """Three windows of a fifth of a second each out of a one-second recording:
    the reported windows are evenly strided and only the budget is read."""
    census = clipping_census(
        FakeRecording(np.zeros((1000, 2), dtype=np.int16), fs_hz=1000.0),
        duration_s=0.6,
        num_chunks=3,
    )
    assert census["windows"] == [[0, 200], [400, 600], [800, 1000]]
    assert census["sampled_s"] == pytest.approx(0.6)


# --------------------------------------------------------------------------
# welch_spectra
# --------------------------------------------------------------------------


def test_welch_spectra_puts_a_pure_tone_in_its_own_frequency_bin():
    """A 250 Hz sine at 10 kHz: the peak of the density must land on the 250 Hz
    bin, and the flat channel must sit at the floor everywhere."""
    fs = 10_000.0
    t = np.arange(4096) / fs
    samples = np.column_stack([np.sin(2 * np.pi * 250.0 * t), np.zeros_like(t)])
    recording = FakeRecording(samples, fs_hz=fs, channel_ids=["tone", "flat"])

    result = welch_spectra(recording, ["tone", "flat"], duration_s=0.4096, nperseg=1024)

    assert result["fs_hz"] == pytest.approx(fs)
    assert result["nperseg"] == 1024
    assert result["power"].shape == (result["freqs"].size, 2)
    peak_hz = float(result["freqs"][np.argmax(result["power"][:, 0])])
    assert peak_hz == pytest.approx(250.0, abs=fs / 1024)
    assert np.all(result["power"][:, 1] == pytest.approx(1e-12))


def test_welch_spectra_floors_zero_power_so_a_flat_channel_stays_on_a_log_axis():
    """An exactly-zero bin has no place on the shared log axis the two panels
    use; the floor is what keeps a dead channel from taking that axis with it."""
    recording = FakeRecording(np.zeros((2048, 1)), fs_hz=1000.0)
    result = welch_spectra(recording, [0], duration_s=2.048, nperseg=256, power_floor=1e-9)
    assert np.all(result["power"] >= 1e-9)


def test_welch_spectra_clamps_nperseg_and_the_window_to_a_short_recording():
    """Asking for more window or more segment than exists must shorten the
    read rather than raise -- a short segment still deserves a PSD."""
    recording = FakeRecording(np.random.default_rng(0).normal(size=(500, 2)), fs_hz=1000.0)
    result = welch_spectra(recording, [0, 1], duration_s=30.0, nperseg=4096)
    assert result["n_samples"] == 500
    assert result["nperseg"] == 500


def test_welch_spectra_names_the_unit_it_actually_measured_in():
    """A recording carrying no gain/offset cannot report microvolts, and the
    label has to say so -- the figure's y axis is built from this string."""
    counts = FakeRecording(np.zeros((1024, 1), dtype=np.int16), fs_hz=1000.0, scaleable=False)
    scaled = FakeRecording(np.zeros((1024, 1), dtype=np.int16), fs_hz=1000.0, scaleable=True)
    assert welch_spectra(counts, [0], duration_s=1.024)["unit"] == "adc^2/Hz"
    assert welch_spectra(scaled, [0], duration_s=1.024)["unit"] == "uV^2/Hz"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
