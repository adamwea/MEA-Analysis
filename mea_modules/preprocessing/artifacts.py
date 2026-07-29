"""Find and blank array-wide artifacts.

An artifact on a dense MEA looks nothing like a spike. A spike is *local* — a
handful of neighbouring electrodes deflect while the rest of the array sits in
noise. Stimulation, saturation and mechanical knocks hit a large fraction of the
array within a sample or two of each other. That difference in spatial extent,
not amplitude, is what :func:`detect_artifacts` keys on: amplitude alone cannot
separate a big spike from a small artifact, but "half the chip moved at once"
can.

Detection has to read traces — that is its job, it returns frame indices rather
than a recording. :func:`blank_artifacts` is the lazy half: it wraps the
recording and substitutes samples only when something downstream pulls them.

The noise estimator here is ported from the older build's threshold-crossing
raster code, which used the same sampled-window MAD approach against real
Maxwell scans. Note that it assumes a zero-centred input (i.e. run this *after*
the high-pass in :mod:`mea_modules.preprocessing.filters`), because it takes the
median of ``|x|`` rather than a median absolute deviation about the mean.
"""

import logging

logger = logging.getLogger(__name__)

# Consistency factor turning a median-absolute value into a Gaussian sigma:
# scipy.stats.norm.ppf(0.75). Carried over verbatim from the older build.
_MAD_TO_SIGMA = 0.6744897501960817

# Sampling the noise beats scanning the whole recording: a handful of windows
# spread across the file tracks drift without reading gigabytes.
DEFAULT_NOISE_WINDOW_FRAMES = 20_000
DEFAULT_NOISE_WINDOWS = 4
_MIN_NOISE_WINDOW = 256

DEFAULT_DETECTION_CHUNK_FRAMES = 50_000
_MIN_DETECTION_CHUNK = 1024

DEFAULT_THRESHOLD_FACTOR = 5.0
DEFAULT_CHANNEL_FRACTION = 0.5
DEFAULT_MIN_SEPARATION_MS = 10.0

DEFAULT_MS_BEFORE = 0.5
DEFAULT_MS_AFTER = 3.0


def _resolve_channel_ids(recording, channel_ids=None):
    """Channel ids as a list, left in the extractor's own id dtype."""
    if channel_ids is not None:
        return list(channel_ids)
    return list(recording.get_channel_ids())


def _as_2d(traces):
    """Traces as (samples, channels), promoting the single-channel case."""
    import numpy as np

    array = np.asarray(traces, dtype=float)
    if array.ndim == 1:
        return array[:, None]
    return array


def estimate_noise_levels(
    recording,
    channel_ids=None,
    window_frames=DEFAULT_NOISE_WINDOW_FRAMES,
    n_windows=DEFAULT_NOISE_WINDOWS,
    segment_index=0,
):
    """Per-channel noise sigma, estimated from windows spread across the segment.

    Reads traces — a few short windows, not the whole recording. Windows are
    placed evenly from the start to the last full window, and the per-window
    estimates are combined with a median so one window landing on a burst (or on
    an artifact) cannot inflate the result.

    Returns a float array, one entry per channel, in the same order as
    `channel_ids`.
    """
    import numpy as np

    ids = _resolve_channel_ids(recording, channel_ids)
    total_samples = int(recording.get_num_samples(segment_index=segment_index))
    if total_samples <= 0 or not ids:
        return np.asarray([], dtype=float)

    window = max(_MIN_NOISE_WINDOW, min(int(window_frames), total_samples))
    max_start = max(0, total_samples - window)
    if max_start <= 0:
        window_starts = [0]
    else:
        window_starts = np.unique(
            np.linspace(0, max_start, num=max(1, int(n_windows)), dtype=np.int64)
        ).tolist()

    estimates = []
    for start_frame in window_starts:
        end_frame = min(total_samples, int(start_frame) + window)
        traces = _as_2d(
            recording.get_traces(
                start_frame=int(start_frame),
                end_frame=int(end_frame),
                channel_ids=ids,
                segment_index=segment_index,
            )
        )
        if traces.size == 0:
            continue
        estimates.append(np.median(np.abs(traces), axis=0) / _MAD_TO_SIGMA)

    if not estimates:
        return np.zeros(len(ids), dtype=float)

    # Clip away exact zeros: a dead channel would otherwise make every sample on
    # it count as a threshold crossing.
    return np.clip(np.median(np.vstack(estimates), axis=0).astype(float), 1e-6, None)


def detect_artifacts(
    recording,
    threshold_factor=DEFAULT_THRESHOLD_FACTOR,
    channel_fraction=DEFAULT_CHANNEL_FRACTION,
    min_separation_ms=DEFAULT_MIN_SEPARATION_MS,
    channel_ids=None,
    noise_levels=None,
    window_frames=DEFAULT_NOISE_WINDOW_FRAMES,
    n_windows=DEFAULT_NOISE_WINDOWS,
    chunk_frames=DEFAULT_DETECTION_CHUNK_FRAMES,
    segment_index=0,
):
    """Return the onset frames of array-wide artifacts.

    A sample is flagged when at least `channel_fraction` of the channels exceed
    `threshold_factor` times their own noise sigma at the same moment. Per-channel
    thresholds matter here — a shared absolute threshold would flag every sample
    on the noisiest electrodes and none on the quietest.

    Consecutive flagged samples are one artifact, so only the leading edge of
    each run is reported, and runs closer together than `min_separation_ms` are
    merged into the first. Runs are tracked across chunk boundaries, so an
    artifact straddling a chunk edge still yields exactly one onset.

    Pass `noise_levels` to reuse an estimate from :func:`estimate_noise_levels`
    instead of recomputing it.

    Returns an int64 array of frame indices, suitable as the trigger list for
    :func:`blank_artifacts`. This reads traces; it is the one function in this
    package that does.
    """
    import numpy as np

    ids = _resolve_channel_ids(recording, channel_ids)
    total_samples = int(recording.get_num_samples(segment_index=segment_index))
    if total_samples <= 0 or not ids:
        return np.asarray([], dtype=np.int64)

    if noise_levels is None:
        noise_levels = estimate_noise_levels(
            recording,
            channel_ids=ids,
            window_frames=window_frames,
            n_windows=n_windows,
            segment_index=segment_index,
        )
    thresholds = np.asarray(noise_levels, dtype=float) * float(threshold_factor)
    if thresholds.size != len(ids):
        raise ValueError(
            f"noise_levels has {thresholds.size} entries but {len(ids)} channels were selected"
        )

    fs_hz = float(recording.get_sampling_frequency())
    min_separation_frames = max(1, int(round(fs_hz * (float(min_separation_ms) / 1000.0))))
    chunk = max(_MIN_DETECTION_CHUNK, int(chunk_frames))
    min_channels = max(1, int(np.ceil(float(channel_fraction) * len(ids))))

    onsets = []
    last_onset = None
    # Carries the flag state over a chunk boundary so a run spanning two chunks
    # is not reported twice.
    previous_flag = False

    for chunk_start in range(0, total_samples, chunk):
        chunk_stop = min(total_samples, chunk_start + chunk)
        traces = _as_2d(
            recording.get_traces(
                start_frame=int(chunk_start),
                end_frame=int(chunk_stop),
                channel_ids=ids,
                segment_index=segment_index,
            )
        )
        if traces.size == 0:
            continue

        over = np.abs(traces) >= thresholds[None, :]
        flagged = over.sum(axis=1) >= min_channels
        if not bool(flagged.any()):
            previous_flag = False
            continue

        # Rising edges only: an artifact is the start of a run, not every sample in it.
        padded = np.concatenate(([previous_flag], flagged))
        rising = np.flatnonzero(flagged & ~padded[:-1]).astype(np.int64) + int(chunk_start)
        previous_flag = bool(flagged[-1])

        for frame in rising.tolist():
            if last_onset is not None and (frame - last_onset) < min_separation_frames:
                continue
            onsets.append(int(frame))
            last_onset = int(frame)

    if onsets:
        logger.debug(
            "Detected %d artifact onset(s) over %d channels (>=%d channels above %.1f x sigma)",
            len(onsets),
            len(ids),
            min_channels,
            float(threshold_factor),
        )
    return np.asarray(onsets, dtype=np.int64)


def blank_artifacts(
    recording,
    triggers,
    ms_before=DEFAULT_MS_BEFORE,
    ms_after=DEFAULT_MS_AFTER,
    mode="zeros",
    **remove_kwargs,
):
    """Blank a window around each trigger frame, returned lazily.

    `triggers` is either a flat sequence of frames (single-segment recordings) or
    one sequence per segment, which is what SpikeInterface itself wants. An empty
    trigger list returns the recording untouched rather than an identity wrapper,
    keeping the provenance chain clean when nothing was detected.

    The default ``"zeros"`` mode only makes sense on traces already centred
    around zero, so run this after the high-pass. Use ``"linear"`` or ``"cubic"``
    on unfiltered traces.
    """
    import numpy as np
    import spikeinterface.preprocessing as spre

    n_segments = int(recording.get_num_segments())

    # Nothing detected: hand back the recording itself rather than wrap it in an
    # identity filter that would show up in provenance for no reason.
    if len(triggers) == 0:
        return recording

    # A flat list of frames is the common case; SpikeInterface needs it nested.
    # ndim rather than isscalar, so numpy integer frames are recognised too.
    if np.ndim(triggers[0]) == 0:
        if n_segments != 1:
            raise ValueError(
                f"recording has {n_segments} segments; pass one trigger list per segment"
            )
        list_triggers = [list(triggers)]
    else:
        list_triggers = [list(item) for item in triggers]

    if len(list_triggers) != n_segments:
        raise ValueError(
            f"got {len(list_triggers)} trigger lists for a recording with {n_segments} segments"
        )

    if not any(len(item) for item in list_triggers):
        return recording

    return spre.remove_artifacts(
        recording,
        list_triggers=list_triggers,
        ms_before=ms_before,
        ms_after=ms_after,
        mode=mode,
        **remove_kwargs,
    )


def blank_detected_artifacts(
    recording,
    threshold_factor=DEFAULT_THRESHOLD_FACTOR,
    channel_fraction=DEFAULT_CHANNEL_FRACTION,
    min_separation_ms=DEFAULT_MIN_SEPARATION_MS,
    ms_before=DEFAULT_MS_BEFORE,
    ms_after=DEFAULT_MS_AFTER,
    mode="zeros",
    channel_ids=None,
    segment_index=0,
):
    """Detect then blank, in one call — returns a lazy recording.

    Only the detection pass reads traces; the blanking it feeds is lazy, so the
    returned recording still costs nothing until sampled.
    """
    triggers = detect_artifacts(
        recording,
        threshold_factor=threshold_factor,
        channel_fraction=channel_fraction,
        min_separation_ms=min_separation_ms,
        channel_ids=channel_ids,
        segment_index=segment_index,
    )
    return blank_artifacts(
        recording,
        triggers,
        ms_before=ms_before,
        ms_after=ms_after,
        mode=mode,
    )
