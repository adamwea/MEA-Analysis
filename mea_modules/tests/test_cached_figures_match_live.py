"""A figure drawn from cache is the same figure, not an approximation of it.

This is the property the compute-in-capsule / plot-from-cache split stands on.
If a cached redraw differed from the live one in any visible way, every review
figure would carry a silent asterisk and the operator would be right to re-run
the computation -- which is the cost the split exists to remove.

So each case here draws the SAME figure twice, once against a real recording and
once against the cache, and compares the rendered PNG byte for byte. An
identical hash is the only claim worth making.
"""

import hashlib

import numpy as np
import pytest

from mea_modules.diagnostics import plot_raster_threshold, plot_traces
from mea_modules.diagnostics.cache import CachedProbe, CachedTraces
from mea_modules.quality import detect_events, mad_noise

FS_HZ = 10_000.0
N_CHANNELS = 8
N_FRAMES = 40_000
PITCH_UM = 17.5
WINDOW_S = 2.0
THRESHOLD_FACTOR = 5.0


def _locations():
    return np.array(
        [[(i % 4) * PITCH_UM, (i // 4) * PITCH_UM] for i in range(N_CHANNELS)],
        dtype=float,
    )


def _recording():
    from probeinterface import Probe
    from spikeinterface.core import NumpyRecording

    rng = np.random.default_rng(0)
    traces = rng.normal(0.0, 1.0, (N_FRAMES, N_CHANNELS)).astype(np.float32)
    # Sparse large negative excursions, so the raster has something to draw and
    # the two paths have something to disagree about.
    traces[1000:20_000:311, :] = -25.0

    recording = NumpyRecording([traces], sampling_frequency=FS_HZ)
    probe = Probe(ndim=2)
    probe.set_contacts(positions=_locations(), shapes="square", shape_params={"width": 5})
    probe.set_device_channel_indices(np.arange(N_CHANNELS))
    return recording.set_probe(probe)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def recording():
    return _recording()


# --------------------------------------------------------------------------
# raster: cached EVENTS, geometry only
# --------------------------------------------------------------------------


def _window_events(recording):
    """What the capsule caches: one detection, as seconds and electrode labels."""
    noise = mad_noise(recording, duration_s=WINDOW_S, num_chunks=1, seed=0,
                      highpass_hz=None, return_in_uV=False)
    detected = detect_events(
        recording, noise, detect_threshold=THRESHOLD_FACTOR,
        end_frame=int(WINDOW_S * FS_HZ),
    )
    return detected["frames"] / FS_HZ, detected["labels"]


def test_a_raster_drawn_from_cached_events_is_byte_identical(tmp_path, recording):
    events = _window_events(recording)
    assert events[0].size, "the planted excursions must be detected"
    live = plot_raster_threshold(
        recording,
        tmp_path / "live.png",
        duration_s=WINDOW_S,
        threshold_factor=THRESHOLD_FACTOR,
        annotate=False,
        events=events,
    )

    # The cached path: the same events plus geometry and the span that was
    # analysed. No samples.
    probe = CachedProbe.from_recording(recording)

    cached = plot_raster_threshold(
        probe,
        tmp_path / "cached.png",
        duration_s=WINDOW_S,
        threshold_factor=THRESHOLD_FACTOR,
        annotate=False,
        events=events,
    )

    assert _digest(live) == _digest(cached)


def test_the_cached_raster_needs_no_samples_at_all(tmp_path, recording):
    """The proof that detection is genuinely not re-run: the probe raises on
    any read, so a path that still reached for traces would fail here."""
    probe = CachedProbe.from_recording(recording)

    with pytest.raises(TypeError, match="geometry only"):
        probe.get_traces(start_frame=0, end_frame=1)

    plot_raster_threshold(
        probe,
        tmp_path / "cached.png",
        duration_s=WINDOW_S,
        annotate=False,
        events=(np.asarray([0.1, 0.2]), np.asarray([0, 3])),
    )


def test_an_empty_event_cache_still_draws_the_analysed_window(tmp_path, recording):
    """A segment where nothing fired is a result, not a missing figure -- and
    the x axis must still span what was looked at."""
    out = plot_raster_threshold(
        CachedProbe.from_recording(recording),
        tmp_path / "silent.png",
        duration_s=WINDOW_S,
        annotate=False,
        events=(np.asarray([]), np.asarray([], dtype=np.int64)),
    )

    assert out.is_file()


# --------------------------------------------------------------------------
# traces: a cached window, addressed in the original frames
# --------------------------------------------------------------------------


def test_traces_drawn_from_a_cached_window_are_byte_identical(tmp_path, recording):
    channel_ids = list(recording.get_channel_ids())[:4]
    start_s, duration_s = 0.5, 1.0

    live = plot_traces(
        recording,
        tmp_path / "live.png",
        channel_ids=channel_ids,
        start_time_s=start_s,
        duration_s=duration_s,
        return_in_uV=False,
        annotate=False,
    )

    # What the capsule would cache: the window it plotted, at full rate, keyed
    # by its own frame numbers.
    offset = int(round(start_s * FS_HZ))
    frames = int(round(duration_s * FS_HZ))
    view = CachedTraces(
        channel_ids,
        recording.get_traces(
            start_frame=offset, end_frame=offset + frames, channel_ids=channel_ids
        ),
        sampling_frequency=FS_HZ,
        frame_offset=offset,
        locations=_locations()[: len(channel_ids)],
    )

    cached = plot_traces(
        view,
        tmp_path / "cached.png",
        channel_ids=channel_ids,
        start_time_s=start_s,
        duration_s=duration_s,
        return_in_uV=False,
        annotate=False,
    )

    assert _digest(live) == _digest(cached)


def test_a_cached_window_drawn_on_the_real_elapsed_axis_matches_too(tmp_path, recording):
    """The frame-faithful part earning its keep: the gap shading is computed
    from frame numbers, so a renumbered window would shade the wrong stretch."""
    channel_ids = list(recording.get_channel_ids())[:3]
    start_s, duration_s = 0.0, 1.0
    # One dropped stretch inside the plotted window.
    time_gaps = {"gaps": [{"start_frame": 4000, "n_missing": 2500}]}

    live = plot_traces(
        recording,
        tmp_path / "live_rt.png",
        channel_ids=channel_ids,
        start_time_s=start_s,
        duration_s=duration_s,
        return_in_uV=False,
        annotate=False,
        time_gaps=time_gaps,
    )

    frames = int(round(duration_s * FS_HZ))
    view = CachedTraces(
        channel_ids,
        recording.get_traces(start_frame=0, end_frame=frames, channel_ids=channel_ids),
        sampling_frequency=FS_HZ,
        frame_offset=0,
        locations=_locations()[: len(channel_ids)],
    )

    cached = plot_traces(
        view,
        tmp_path / "cached_rt.png",
        channel_ids=channel_ids,
        start_time_s=start_s,
        duration_s=duration_s,
        return_in_uV=False,
        annotate=False,
        time_gaps=time_gaps,
    )

    assert _digest(live) == _digest(cached)
