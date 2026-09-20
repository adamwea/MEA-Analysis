"""Numeric tests for `mea_modules.diagnostics.artifact_census`.

Separate from `test_segment_diagnostics_pure.py`, which deliberately keeps to
duck-typed recordings: the census calls `frame_slice` and hands the result to
`detect_artifacts`, so it needs a real SpikeInterface recording to exercise the
scan bound at all. An in-memory `NumpyRecording` over a few seconds of
synthetic noise is enough -- no file, no fixtures.

The artifacts are planted at known frames, so both halves of what this function
adds around the detector are checked against hand-computed values: the scan
budget (does an artifact past the window stay out of the count) and the rate
arithmetic (is the per-minute figure the count over the window actually
scanned, not over the recording).
"""

import numpy as np
import pytest
import spikeinterface.core as sc

from mea_modules.diagnostics import artifact_census

FS_HZ = 10_000.0
N_CHANNELS = 16
N_SECONDS = 5.0

# Frames where every channel jumps at once. The last one sits past the 3 s scan
# budget used below, which is the point of planting three rather than two.
ONSET_FRAMES = (5_000, 21_000, 33_000)
ARTIFACT_WIDTH_FRAMES = 40
ARTIFACT_AMPLITUDE = 400.0
NOISE_SCALE = 5.0


def _recording_with_planted_artifacts():
    """Gaussian noise with a synchronous excursion on every channel at each
    planted onset -- 80x the noise scale, so detection is not the thing under
    test here."""
    rng = np.random.default_rng(3)
    traces = rng.normal(
        scale=NOISE_SCALE, size=(int(N_SECONDS * FS_HZ), N_CHANNELS)
    ).astype("float32")
    for onset in ONSET_FRAMES:
        traces[onset:onset + ARTIFACT_WIDTH_FRAMES, :] += ARTIFACT_AMPLITUDE
    return sc.NumpyRecording([traces], sampling_frequency=FS_HZ)


def test_artifact_census_finds_the_planted_onsets_inside_the_scan_window():
    """Two of the three onsets are inside a 3 s scan; each is reported once, at
    the second it was planted at."""
    census = artifact_census(_recording_with_planted_artifacts(), duration_s=3.0)

    assert census["n_artifacts"] == 2
    assert census["scanned_frames"] == 30_000
    assert census["scanned_s"] == pytest.approx(3.0)
    assert census["onset_times_s"] == pytest.approx([0.5, 2.1], abs=1e-3)


def test_artifact_census_rate_is_per_minute_of_the_window_actually_scanned():
    """2 onsets in 3 s is 40 per minute -- the divisor is the scan, not the 5 s
    recording, which would give 24."""
    census = artifact_census(_recording_with_planted_artifacts(), duration_s=3.0)
    assert census["artifact_rate_per_min"] == pytest.approx(40.0)
    assert census["artifact_rate_per_min"] != pytest.approx(24.0)


def test_artifact_census_scans_the_whole_recording_on_a_non_positive_budget():
    """Zero is the documented opt-out: the third onset comes back, and the rate
    falls to 3 in 5 s = 36 per minute."""
    census = artifact_census(_recording_with_planted_artifacts(), duration_s=0.0)

    assert census["scanned_frames"] == int(N_SECONDS * FS_HZ)
    assert census["n_artifacts"] == 3
    assert census["onset_times_s"] == pytest.approx([0.5, 2.1, 3.3], abs=1e-3)
    assert census["artifact_rate_per_min"] == pytest.approx(36.0)


def test_artifact_census_reports_no_onsets_on_plain_noise():
    """Nothing synchronous means nothing to report -- and a zero count must
    still produce a well-formed census rather than an empty dict."""
    rng = np.random.default_rng(11)
    traces = rng.normal(scale=NOISE_SCALE, size=(int(FS_HZ), N_CHANNELS)).astype("float32")
    census = artifact_census(sc.NumpyRecording([traces], sampling_frequency=FS_HZ))

    assert census["n_artifacts"] == 0
    assert census["onset_frames"] == []
    assert census["onset_times_s"] == []
    assert census["artifact_rate_per_min"] == pytest.approx(0.0)
    assert census["fs_hz"] == pytest.approx(FS_HZ)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
