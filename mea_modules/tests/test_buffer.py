"""A buffered signal is the same signal, read once.

What must survive the copy is everything a diagnostic reads off the recording
besides the samples: the channel ids, the gains that turn counts into
microvolts, and the ``is_filtered`` flag -- the noise estimators high-pass any
recording that does not report itself filtered, so a buffer that dropped the
flag would be filtered twice and every noise number would move.
"""
import numpy as np
import pytest
from spikeinterface.core import NumpyRecording
from spikeinterface.preprocessing.basepreprocessor import BasePreprocessor, BasePreprocessorSegment

from mea_modules.diagnostics.buffer import SIGNAL_BUFFER_MODES, buffered_signal, signal_bytes
from mea_modules.quality import mad_noise

FS = 10_000.0


def _recording():
    rng = np.random.default_rng(0)
    traces = rng.normal(0.0, 4.0, (20_000, 5)).astype(np.float32)
    recording = NumpyRecording([traces], sampling_frequency=FS, channel_ids=["a", "b", "c", "d", "e"])
    recording.set_channel_gains(np.asarray([0.5, 1.0, 1.5, 2.0, 2.5]))
    recording.set_channel_offsets(np.zeros(5))
    recording.annotate(is_filtered=True)
    return recording


@pytest.mark.parametrize("mode", ["memory", "disk"])
def test_the_buffer_keeps_ids_gains_and_the_filtered_flag(mode, tmp_path):
    source = _recording()
    with buffered_signal(source, mode, scratch_dir=tmp_path) as (buffered, info):
        assert list(buffered.get_channel_ids()) == list(source.get_channel_ids())
        assert np.allclose(buffered.get_channel_gains(), source.get_channel_gains())
        assert buffered.is_filtered() is True
        assert np.array_equal(buffered.get_traces(), source.get_traces())
        assert info["mode"] == mode and info["bytes"] == signal_bytes(source)
        assert info["seconds"] >= 0.0 and info["started"] <= info["ended"]


@pytest.mark.parametrize("mode", ["memory", "disk"])
def test_noise_measured_on_the_buffer_is_the_noise_of_the_source(mode, tmp_path):
    """The flag matters because this number would move without it."""
    source = _recording()
    kwargs = dict(duration_s=0.5, num_chunks=4, seed=1)
    with buffered_signal(source, mode, scratch_dir=tmp_path) as (buffered, _):
        assert mad_noise(buffered, **kwargs)["noise"] == pytest.approx(mad_noise(source, **kwargs)["noise"])


def test_lazy_is_the_recording_itself():
    source = _recording()
    with buffered_signal(source, "lazy") as (buffered, info):
        assert buffered is source
        assert info == {"mode": "lazy", "bytes": signal_bytes(source), "seconds": 0.0}


def test_a_disk_buffer_is_removed_even_when_the_work_raises(tmp_path):
    with pytest.raises(RuntimeError):
        with buffered_signal(_recording(), "disk", scratch_dir=tmp_path):
            assert any(tmp_path.glob("signal_buffer_*"))
            raise RuntimeError("a diagnostic failed")
    assert not any(tmp_path.glob("signal_buffer_*"))


class _FailingSegment(BasePreprocessorSegment):
    def get_traces(self, start_frame, end_frame, channel_indices):
        if start_frame and start_frame >= int(FS):
            raise OSError("no space left on device")
        return self.parent_recording_segment.get_traces(start_frame, end_frame, channel_indices)


class _FailsAfterOneSecond(BasePreprocessor):
    """Writes its first chunk, then fails -- a copy that dies partway."""

    def __init__(self, recording):
        BasePreprocessor.__init__(self, recording)
        for segment in recording._recording_segments:
            self.add_recording_segment(_FailingSegment(segment))


def test_a_disk_buffer_whose_copy_fails_partway_is_removed(tmp_path):
    with pytest.raises(OSError, match="no space"):
        with buffered_signal(_FailsAfterOneSecond(_recording()), "disk", scratch_dir=tmp_path):
            pass
    assert not any(tmp_path.glob("signal_buffer_*"))


def test_a_disk_buffer_needs_somewhere_to_live():
    with pytest.raises(ValueError, match="scratch_dir"):
        with buffered_signal(_recording(), "disk"):
            pass


def test_auto_is_not_a_mode_here():
    """`auto` is resolved by the caller, who knows the machine; this module
    takes only the resolved mode."""
    assert "auto" not in SIGNAL_BUFFER_MODES
    with pytest.raises(ValueError, match="signal buffer"):
        with buffered_signal(_recording(), "auto"):
            pass


def test_the_traces_with_layout_figure_draws_the_array_it_is_handed(tmp_path):
    """A cached window holds only the channels it traces; handed the full
    array's geometry, the layout panel shows where those few sit on it."""
    import matplotlib

    from mea_modules.diagnostics import plot_traces_with_layout
    from mea_modules.diagnostics import traces as tr
    from mea_modules.diagnostics.cache import CachedProbe, CachedTraces

    locations = np.asarray([[i * 17.5, (i % 3) * 17.5] for i in range(12)])
    probe = CachedProbe([str(i) for i in range(12)], locations, sampling_frequency=FS,
                        num_frames=20_000)
    window = CachedTraces(["2", "7"], np.zeros((5_000, 2), dtype=np.float32),
                          sampling_frequency=FS, frame_offset=0, locations=locations[[2, 7]])

    counts = {}
    real = tr._save_and_release

    def capture(fig, out_path):
        scatters = [
            collection for axis in fig.get_axes() for collection in axis.collections
            if isinstance(collection, matplotlib.collections.PathCollection)
        ]
        counts["points"] = sum(len(c.get_offsets()) for c in scatters)  # traced + the rest
        return real(fig, out_path)

    tr._save_and_release = capture
    try:
        manifest = plot_traces_with_layout(window, tmp_path / "with_array.png",
                                           channel_ids=["2", "7"], duration_s=0.5,
                                           return_in_uV=False, probe=probe)
        with_array = counts["points"]
        plot_traces_with_layout(window, tmp_path / "alone.png", channel_ids=["2", "7"],
                                duration_s=0.5, return_in_uV=False)
        alone = counts["points"]
    finally:
        tr._save_and_release = real

    assert manifest["channel_ids"] == ["2", "7"]
    assert with_array == 12 and alone == 2
