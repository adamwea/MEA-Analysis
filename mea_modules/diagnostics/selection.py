"""Choosing and reading the channels and frames a diagnostic caches.

Channel selection (representative channels, the raster's channel budget) and the
trace-reading helpers the collectors and the Welch estimate share. They lived in
the drawing modules (:mod:`.traces`, :mod:`.raster`), which import them back.
"""

import logging

from .electrodes import _channel_xy, cluster_center_channels

logger = logging.getLogger(__name__)


_DEFAULT_ACTIVITY_CHUNKS = 4


_DEFAULT_ACTIVITY_CHUNK_FRAMES = 4_000


# A trace plot is only readable at single-digit channel counts. Lowered from 8
# to 6 (Adam, 2026-08-11): fewer stacked panels means each one is taller, and
# detail in an individual trace is what these figures are reviewed for. This is
# a knob everywhere it matters — nothing may hardcode the count, least of all
# legend text, since `channel_layout.png`'s highlighted set is exactly this many.
_DEFAULT_MAX_CHANNELS = 6


def _effective_uv(recording, return_in_uV):
    """True when traces can actually be returned in microvolts.

    Honours the request but never fails on it: asking for µV from a recording
    that carries no gain/offset (``has_scaleable_traces()`` False) downgrades
    to device units with a warning, because a review plot in ADC counts beats
    no plot. Callers label axes from THIS value, so a figure always states the
    units it actually drew rather than the units that were requested.
    """
    if not return_in_uV:
        return False
    try:
        scaleable = bool(recording.has_scaleable_traces())
    except Exception:  # pragma: no cover - defensive; exotic recording objects
        scaleable = False
    if not scaleable:
        logger.warning(
            "return_in_uV requested but the recording carries no gain/offset; "
            "falling back to device units (ADC counts)"
        )
    return scaleable


def _resolve_frame_window(recording, start_time_s=0.0, duration_s=None):
    """Clamp a (start, duration) request to a valid [start_frame, end_frame).

    Times are offsets from the first sample, not positions on the recording's
    own time vector, so this stays meaningful whether or not wall-clock times
    have been attached.
    """
    fs = float(recording.get_sampling_frequency())
    total = int(recording.get_num_samples())
    if total <= 0 or fs <= 0.0:
        return 0, 0, fs

    start_frame = max(0, int(round(float(start_time_s or 0.0) * fs)))
    start_frame = min(start_frame, max(0, total - 1))
    if duration_s is None:
        end_frame = total
    else:
        span = max(1, int(round(float(duration_s) * fs)))
        end_frame = min(total, start_frame + span)
    return start_frame, max(start_frame, end_frame), fs


def _has_time_vector(recording):
    """True when the recording carries explicit sample times."""
    try:
        return bool(recording.has_time_vector())
    except Exception:
        # Not every recording object implements it; absence just means "no".
        return False


def _frames_to_seconds(recording, frames, fs, has_times=None):
    """Convert sample indices to seconds, honouring an attached time vector."""
    import numpy as np

    frames = np.asarray(frames, dtype=np.int64)
    if has_times is None:
        has_times = _has_time_vector(recording)
    if has_times:
        try:
            return np.asarray(recording.sample_index_to_time(frames), dtype=float)
        except Exception:
            pass
    return frames.astype(float) / float(fs) if fs else frames.astype(float)


def _read_traces(recording, start_frame, end_frame, channel_ids, return_in_uV):
    """get_traces with a 2-D result guaranteed, even for a single channel."""
    import numpy as np

    traces = recording.get_traces(
        start_frame=int(start_frame),
        end_frame=int(end_frame),
        channel_ids=list(channel_ids),
        return_in_uV=bool(return_in_uV),
    )
    traces = np.asarray(traces)
    if traces.ndim == 1:
        traces = traces[:, None]
    return traces


def channel_activity_rms(
    recording,
    channel_ids=None,
    num_chunks=_DEFAULT_ACTIVITY_CHUNKS,
    chunk_frames=_DEFAULT_ACTIVITY_CHUNK_FRAMES,
    seed=0,
    start_time_s=0.0,
    duration_s=None,
    return_in_uV=True,
):
    """RMS amplitude per channel, sampled from a few random chunks.

    Returns a float array aligned with `channel_ids` (all channels by default).
    Chunk starts are drawn from a seeded generator and shared by every channel,
    so the scores are comparable to each other and stable across runs — that is
    what makes them usable as a ranking key.

    Scores are in microvolts by default (device units on explicit opt-out, or
    automatically when the recording cannot scale). With a uniform gain the
    RANKING is identical either way; the µV default exists so the scores read
    in the same unit the trace figures draw.

    All channels are read together per chunk rather than one at a time: the
    samples, and therefore the scores, are identical, but a recording backed by
    a memmap is touched once per chunk instead of once per channel.
    """
    import numpy as np

    if channel_ids is None:
        channel_ids = list(recording.get_channel_ids())
    channel_ids = list(channel_ids)
    if not channel_ids:
        return np.asarray([], dtype=float)

    return_in_uV = _effective_uv(recording, return_in_uV)

    window_start, window_end, _fs = _resolve_frame_window(recording, start_time_s, duration_s)
    span = int(window_end - window_start)
    if span <= 0:
        return np.zeros(len(channel_ids), dtype=float)

    chunk_frames = max(100, int(chunk_frames))
    num_chunks = max(1, int(num_chunks))

    if span <= chunk_frames:
        starts = [window_start]
        chunk_frames = span
    else:
        rng = np.random.default_rng(int(seed))
        starts = (
            window_start + rng.integers(0, span - chunk_frames, size=num_chunks, endpoint=False)
        ).tolist()

    sum_squares = np.zeros(len(channel_ids), dtype=float)
    count = 0
    for start in starts:
        traces = _read_traces(recording, int(start), int(start) + chunk_frames, channel_ids, return_in_uV)
        if traces.size == 0:
            continue
        traces = traces.astype(float, copy=False)
        sum_squares += np.sum(traces * traces, axis=0)
        count += int(traces.shape[0])

    return np.sqrt(sum_squares / max(count, 1))


def select_representative_channels(
    recording,
    n_channels=_DEFAULT_MAX_CHANNELS,
    eps=None,
    num_chunks=_DEFAULT_ACTIVITY_CHUNKS,
    chunk_frames=_DEFAULT_ACTIVITY_CHUNK_FRAMES,
    seed=0,
    start_time_s=0.0,
    duration_s=None,
    return_in_uV=True,
    restrict_to=None,
):
    """Pick the `n_channels` most active cluster representatives.

    Activity RMS is scored in microvolts by default (see
    :func:`channel_activity_rms`); with a uniform gain the selection is
    identical in either unit, so flipping `return_in_uV` never changes which
    channels a paired trace figure draws.

    One candidate is taken from each electrode cluster — the member nearest the
    cluster centroid, via
    :func:`mea_modules.diagnostics.channel_layout.cluster_center_channels` —
    and the candidates are then ordered by activity RMS, loudest first. Pass
    `n_channels` <= 0 to get every candidate in that order.

    `restrict_to` narrows the electrode pool BEFORE clustering. Pass the shared
    electrode set (:func:`.stitch_consistency.backbone_channel_ids`) to select over
    only the electrodes every segment kept, which is what a figure comparing
    segments must do.

    Falls back to the recording's own channel order when the probe carries no
    usable locations, so this never fails on a probe-less recording.

    Returns channel ids in the recording's own id type, ready to hand back to
    ``get_traces``.
    """
    channel_ids, xs, _ys = _channel_xy(recording)
    if not channel_ids:
        return []

    if xs is None:
        logger.warning("no usable channel locations; falling back to recording channel order")
        pool = list(channel_ids)
        if restrict_to is not None:
            wanted = {str(cid) for cid in restrict_to}
            pool = [cid for cid in pool if str(cid) in wanted] or pool
        return pool if n_channels <= 0 else pool[: max(1, int(n_channels))]

    candidates = cluster_center_channels(recording, channel_ids=restrict_to, eps=eps)
    if not candidates:
        candidates = list(channel_ids)

    scores = channel_activity_rms(
        recording,
        channel_ids=candidates,
        num_chunks=num_chunks,
        chunk_frames=chunk_frames,
        seed=seed,
        start_time_s=start_time_s,
        duration_s=duration_s,
        return_in_uV=return_in_uV,
    )
    ranked = [
        channel_id
        for channel_id, _score in sorted(zip(candidates, list(scores)), key=lambda item: item[1], reverse=True)
    ]

    logger.info(
        "selected %d representative channels from %d clusters over %d channels",
        len(ranked) if n_channels <= 0 else min(len(ranked), max(1, int(n_channels))),
        len(candidates),
        len(channel_ids) if restrict_to is None else len(list(restrict_to)),
    )
    if n_channels <= 0:
        return ranked
    return ranked[: max(1, int(n_channels))]


def _select_channels(recording, channel_ids, max_channels):
    """Channel ids to threshold, thinned evenly across the array if needed.

    Thinning takes every k-th channel rather than the first N: channel order
    tracks position on the array, so a prefix would raster one corner of the
    chip and report the rest as silent.
    """
    import numpy as np

    if channel_ids is None:
        channel_ids = list(recording.get_channel_ids())
    channel_ids = list(channel_ids)
    if max_channels is None or max_channels <= 0 or len(channel_ids) <= int(max_channels):
        return channel_ids

    keep = np.linspace(0, len(channel_ids) - 1, num=int(max_channels)).astype(int)
    logger.warning(
        "rastering %d of %d channels (evenly spaced); raise max_channels to cover all",
        int(keep.size),
        len(channel_ids),
    )
    return [channel_ids[index] for index in np.unique(keep)]
