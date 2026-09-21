"""Threshold-crossing events: one detector for every caller.

Two home-grown detectors used to live in this package: a rising-edge count with
a 1.0 ms dead time behind the activity rate, and a local-minimum rule with a
0.8 ms refractory period behind the rasters. Neither number had a recorded
derivation, and the pair disagreed with each other by 0.2 ms for no reason.
Both are replaced by SpikeInterface's own peak detector, which the rest of the
stack already depends on and which the spike-sorting literature describes.

The call, fixed here so every consumer makes the same one::

    detect_peaks(recording, method="by_channel",
                 method_kwargs=dict(peak_sign="neg", detect_threshold=5,
                                    exclude_sweep_ms=1.0, noise_levels=<ours>))

* ``by_channel``, not ``locally_exclusive``. The latter credits a spike seen on
  five neighbouring electrodes to one of them, which is the right answer to
  "is anything alive" and the wrong one to "how active is each electrode" --
  every figure here is a per-electrode figure.
* ``exclude_sweep_ms=1.0`` matches the old activity dead time. SpikeInterface's
  default of 0.1 ms would raise every count ~5% as an unannounced side effect.
  Measured on a real 266-channel slice: 48,358 events against the old
  detector's 48,210 (0.3%).
* ``noise_levels`` is OUR noise estimate, :func:`mea_modules.quality.mad_noise`
  -- the same MAD-sigma, the same seeded windows, the same median across them.
  Left to itself, SpikeInterface would estimate its own (a mean across 20
  random chunks), and the noise map a reader looks at and the thresholds the
  detector applied would be two different numbers.

SpikeInterface's ``sortingcomponents`` API is under revision (the installed
0.103.2 already carries a compatibility shim due to be removed in 0.105.0), so
the version is pinned in this package's dependencies and checked here.

Detection runs in the recording's NATIVE units: the thresholds are the noise
converted back through each channel's gain, not the traces converted to
microvolts. MAD scales exactly with the gain and ignores the offset, so the two
are the same threshold; converting the noise is one division per channel,
converting the traces is one multiplication per sample.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

SPIKEINTERFACE_VERSION = "0.103.2"

DETECT_METHOD = "by_channel"
DEFAULT_DETECT_THRESHOLD = 5.0
DEFAULT_PEAK_SIGN = "neg"
DEFAULT_EXCLUDE_SWEEP_MS = 1.0

# In-process and quiet. SpikeInterface's global default shows a progress bar,
# which in a pipeline log is a line of carriage returns per chunk.
_JOB_KWARGS = {"n_jobs": 1, "chunk_duration": "1s", "progress_bar": False}

# Microvolt spellings, as the metric dicts write them.
_UV_UNITS = frozenset({"uV", "µV", "microvolts"})


def detector_description(detect_threshold, peak_sign, exclude_sweep_ms):
    """The provenance block every event-derived result carries."""
    import spikeinterface

    return {
        "detector": f"spikeinterface.sortingcomponents.peak_detection.detect_peaks({DETECT_METHOD})",
        "spikeinterface_version": str(spikeinterface.__version__),
        "detect_threshold": float(detect_threshold),
        "peak_sign": str(peak_sign),
        "exclude_sweep_ms": float(exclude_sweep_ms),
        "noise_levels": "mea_modules.quality.mad_noise (median of seeded windows)",
    }


def check_spikeinterface_version():
    """Warn when the installed SpikeInterface is not the pinned one.

    A warning rather than a failure: an analysis repo running a newer release
    may well be fine, but a count that moved because a kwarg was renamed should
    be traceable to the version that moved it.
    """
    import spikeinterface

    installed = str(spikeinterface.__version__)
    if installed != SPIKEINTERFACE_VERSION:
        logger.warning(
            "SpikeInterface %s is installed; the detector was validated on %s "
            "(its sortingcomponents API changes between minor versions)",
            installed, SPIKEINTERFACE_VERSION,
        )
    return installed


def channel_labels(channel_ids):
    """Integer electrode labels for event rows, or row positions.

    Maxwell channel ids are integer-like, and labelling an event with its real
    electrode id keeps a raster comparable with the layout figure. When the ids
    are not numeric at all, every channel falls back to its row position so the
    axis stays consistent.
    """
    try:
        return np.asarray([int(channel_id) for channel_id in channel_ids], dtype=np.int64)
    except (TypeError, ValueError):
        return np.arange(len(channel_ids), dtype=np.int64)


def noise_in_native_units(recording, noise, channel_ids=None):
    """Per-channel noise in the recording's native units, aligned to `channel_ids`.

    `noise` is a :func:`mea_modules.quality.mad_noise` dict or a bare
    per-channel sequence in native units. A dict measured in microvolts is
    divided back through each channel's gain; its offset plays no part, because
    a MAD is centred. Channels are matched on their string form, so a noise
    dict that came back through JSON still lines up.
    """
    ids = list(recording.get_channel_ids()) if channel_ids is None else list(channel_ids)
    if isinstance(noise, dict):
        values = np.asarray(noise["noise"], dtype=np.float64)
        keyed = {str(cid): value for cid, value in zip(noise["channel_ids"], values)}
        missing = [cid for cid in ids if str(cid) not in keyed]
        if missing:
            raise ValueError(
                f"the noise estimate does not cover {len(missing)} of the "
                f"{len(ids)} channels to detect on (e.g. {missing[:5]})"
            )
        aligned = np.asarray([keyed[str(cid)] for cid in ids], dtype=np.float64)
        if noise.get("unit") in _UV_UNITS:
            gains = recording.get_channel_gains(channel_ids=ids)
            if gains is None:
                raise ValueError("noise is in microvolts but the recording carries no gains")
            aligned = aligned / np.asarray(gains, dtype=np.float64)
        return aligned
    aligned = np.asarray(noise, dtype=np.float64)
    if aligned.size != len(ids):
        raise ValueError(f"{aligned.size} noise values for {len(ids)} channels")
    return aligned


def detect_events(
    recording,
    noise,
    *,
    detect_threshold=DEFAULT_DETECT_THRESHOLD,
    peak_sign=DEFAULT_PEAK_SIGN,
    exclude_sweep_ms=DEFAULT_EXCLUDE_SWEEP_MS,
    channel_ids=None,
    start_frame=None,
    end_frame=None,
    job_kwargs=None,
):
    """Detect threshold-crossing events; return a dict of aligned arrays.

    Runs SpikeInterface's ``detect_peaks`` with the call fixed in this module's
    docstring, over `channel_ids` (default: every channel) and the frame span
    ``[start_frame, end_frame)`` (default: the whole recording), with the
    thresholds set at `detect_threshold` times `noise`.

    Returns::

        {"frames", "channel_index", "labels", "channel_ids", "thresholds",
         "start_frame", "end_frame", "n_frames", "sampling_frequency",
         **detector_description(...)}

    `frames` are on the recording's OWN numbering (the span's start added
    back), sorted in time; `channel_index` indexes `channel_ids`; `labels` are
    the matching integer electrode ids (:func:`channel_labels`). `thresholds`
    are in native units, one per channel. Seconds are left to the caller,
    because only the caller knows which timeline it is drawing.
    """
    from spikeinterface.sortingcomponents.peak_detection import detect_peaks

    check_spikeinterface_version()
    fs = float(recording.get_sampling_frequency())
    n_total = int(recording.get_num_samples())
    start = 0 if start_frame is None else max(0, int(start_frame))
    end = n_total if end_frame is None else min(n_total, int(end_frame))

    ids = list(recording.get_channel_ids()) if channel_ids is None else list(channel_ids)
    levels = noise_in_native_units(recording, noise, ids)
    # A perfectly flat channel has zero noise, and a zero threshold fires on
    # every sample; the same floor the old detector applied.
    levels = np.clip(levels, 1e-6, None)

    result = {
        "channel_ids": ids,
        "thresholds": levels * float(detect_threshold),
        "start_frame": start,
        "end_frame": end,
        "n_frames": max(0, end - start),
        "sampling_frequency": fs,
        **detector_description(detect_threshold, peak_sign, exclude_sweep_ms),
    }
    if end <= start or not ids:
        result.update(
            frames=np.asarray([], dtype=np.int64),
            channel_index=np.asarray([], dtype=np.int64),
            labels=np.asarray([], dtype=np.int64),
        )
        return result

    target = recording
    if channel_ids is not None:
        target = target.select_channels(ids)
    if start != 0 or end != n_total:
        target = target.frame_slice(start_frame=start, end_frame=end)

    peaks = detect_peaks(
        target,
        method=DETECT_METHOD,
        method_kwargs={
            "peak_sign": str(peak_sign),
            "detect_threshold": float(detect_threshold),
            "exclude_sweep_ms": float(exclude_sweep_ms),
            "noise_levels": levels,
        },
        job_kwargs={**_JOB_KWARGS, **(job_kwargs or {})},
    )
    frames = np.asarray(peaks["sample_index"], dtype=np.int64) + start
    channel_index = np.asarray(peaks["channel_index"], dtype=np.int64)
    order = np.lexsort((channel_index, frames))
    frames = frames[order]
    channel_index = channel_index[order]
    result.update(
        frames=frames,
        channel_index=channel_index,
        labels=channel_labels(ids)[channel_index] if channel_index.size else channel_index,
    )
    logger.info(
        "detected %d event(s) on %d channel(s) over %.1f s (%g x MAD-sigma, %s, %g ms)",
        int(frames.size), len(ids), result["n_frames"] / fs if fs else 0.0,
        float(detect_threshold), peak_sign, float(exclude_sweep_ms),
    )
    return result


def event_rates(events):
    """Per-channel event rate from :func:`detect_events`'s result, in events/s.

    Every channel detection covered is reported, a silent one as a real zero:
    dropping it would raise the mean and hide exactly the electrodes a reviewer
    most wants counted. The rate divides by the span detection actually read.

    Returns a dict with ``channel_ids``, ``rate_hz``, ``n_events``,
    ``mean_rate_hz``, ``median_rate_hz``, ``duration_s`` and the detector's
    provenance.
    """
    ids = list(events["channel_ids"])
    fs = float(events["sampling_frequency"])
    duration_s = events["n_frames"] / fs if fs else 0.0
    counts = np.bincount(np.asarray(events["channel_index"], dtype=np.int64), minlength=len(ids))
    rates = counts / duration_s if duration_s else np.zeros(len(ids), dtype=float)
    return {
        "channel_ids": [cid.item() if hasattr(cid, "item") else cid for cid in ids],
        "rate_hz": [float(value) for value in rates],
        "n_events": [int(value) for value in counts],
        "mean_rate_hz": float(np.mean(rates)) if len(ids) else 0.0,
        "median_rate_hz": float(np.median(rates)) if len(ids) else 0.0,
        "duration_s": float(duration_s),
        "window_aggregation": "whole span, one detection",
        **{key: events[key] for key in (
            "detector", "spikeinterface_version", "detect_threshold", "peak_sign",
            "exclude_sweep_ms", "noise_levels",
        )},
    }


def resample_rate_for(sampling_frequency, target_hz):
    """`(factor, effective_hz)` for detecting on a downsampled signal, or `(1, fs)`.

    The effective rate is a whole number of hertz that divides the native rate
    exactly, which is what lets SpikeInterface's resampler take its anti-aliased
    decimation path (``scipy.signal.decimate``) rather than an FFT resample, and
    what keeps every detection sample on a native one. It is rarely the rate
    asked for, which is why it is returned: a cache that stored the request
    would describe a run that did not happen.
    """
    fs = float(sampling_frequency)
    if not target_hz or float(target_hz) <= 0 or fs <= 0:
        return 1, fs
    factor = int(fs // float(target_hz))
    while factor > 1 and (fs / factor) != int(fs / factor):
        factor += 1
    if factor <= 1:
        return 1, fs
    return factor, fs / factor


def resampled_for_detection(recording, target_hz):
    """`(recording, factor, effective_hz)` -- anti-aliased, or untouched at factor 1.

    Plain striding is wrong for this signal and quietly so: the preprocessing
    chain is a high-pass with no low-pass, so the signal is full-band to
    Nyquist, and taking every k-th sample folds everything above the new
    Nyquist into the spike band. That inflates the noise, raises the threshold
    and changes the count, in a figure that still renders. SpikeInterface's own
    decimation (``DecimateRecording``) is plain striding, so this goes through
    its resampler, which filters first.
    """
    factor, effective_hz = resample_rate_for(recording.get_sampling_frequency(), target_hz)
    if factor == 1:
        return recording, 1, effective_hz
    from spikeinterface.preprocessing import resample

    return resample(recording, resample_rate=int(effective_hz)), factor, effective_hz


__all__ = [
    "DEFAULT_DETECT_THRESHOLD",
    "DEFAULT_EXCLUDE_SWEEP_MS",
    "DEFAULT_PEAK_SIGN",
    "SPIKEINTERFACE_VERSION",
    "channel_labels",
    "check_spikeinterface_version",
    "detect_events",
    "detector_description",
    "event_rates",
    "noise_in_native_units",
    "resample_rate_for",
    "resampled_for_detection",
]
