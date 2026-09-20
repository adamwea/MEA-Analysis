"""Array-wide artifact onsets over a bounded scan, as a reportable census.

:func:`mea_modules.preprocessing.detect_artifacts` answers the detection
question — which samples are moments when at least half the channels exceed
their own noise threshold at once, which is stimulation, saturation recovery or
a bumped rig, and which amplitude alone cannot see. This module wraps that in
the two things a diagnostic report needs around it: a scan budget, so the cost
is a number of seconds rather than the length of the recording, and the rate
arithmetic that turns a list of frames into something comparable between
recordings of different lengths.

A rate per minute is the comparable quantity: onset counts from a 60 s scan and
a 600 s scan say nothing to each other, and ``scanned_s`` is reported alongside
so a reader can always see which window the rate came from.

Returns a JSON-serializable dict; writes nothing.
"""

import logging

logger = logging.getLogger(__name__)

# The leading window scanned by default. Long enough that a recurring artifact
# shows up, short enough that the scan is a fixed cost per recording.
DEFAULT_SCAN_DURATION_S = 60.0


def artifact_census(recording, duration_s=DEFAULT_SCAN_DURATION_S, noise_levels=None):
    """Detect array-wide artifacts in a bounded leading window of `recording`.

    Parameters
    ----------
    recording
        The recording to scan. Pass the same chain the noise metrics were
        measured on: per-channel thresholds are derived from it, so scanning a
        raw view and reporting against a filtered one would compare different
        noise floors.
    duration_s : float
        Leading seconds scanned. Zero or negative scans the whole recording,
        which is the opt-out rather than the default.
    noise_levels : array-like or None
        A :func:`mea_modules.preprocessing.estimate_noise_levels` result to
        reuse; None estimates it on the scanned window.

    Returns
    -------
    dict
        ``scanned_s``, ``scanned_frames``, ``fs_hz``, ``n_artifacts``,
        ``onset_frames``, ``onset_times_s`` and ``artifact_rate_per_min``.
        Onset times are offsets from the first sample of the recording, which
        is also the first sample of the scan.
    """
    from mea_modules.preprocessing import detect_artifacts, estimate_noise_levels

    fs = float(recording.get_sampling_frequency())
    n_samples = int(recording.get_num_samples())
    limit_s = float(duration_s)
    if limit_s > 0:
        end = min(n_samples, int(round(limit_s * fs)))
        scan = recording.frame_slice(start_frame=0, end_frame=end)
    else:
        scan = recording
        end = n_samples

    if noise_levels is None:
        noise_levels = estimate_noise_levels(scan)
    onsets = detect_artifacts(scan, noise_levels=noise_levels)

    census = {
        "scanned_s": end / fs if fs else 0.0,
        "scanned_frames": int(end),
        "fs_hz": fs,
        "n_artifacts": int(len(onsets)),
        "onset_frames": [int(v) for v in onsets],
        "onset_times_s": [float(v / fs) for v in onsets] if fs else [],
        "artifact_rate_per_min": (
            60.0 * len(onsets) / (end / fs) if end and fs else 0.0
        ),
    }
    logger.info(
        "artifact census: %d onset(s) in %.1f s", census["n_artifacts"], census["scanned_s"]
    )
    return census
