"""`save_concatenated`'s worker count decides how fast the binary is written,
never what is in it.

The concatenated binary is what a sort reads, so a pipeline treats the worker
count as a machine knob -- outside the result's fingerprint -- only on this
proof: the same filtered, concatenated chain written with one worker and with
several gives byte-identical files. Chunks are small, so the write is many
chunks spread across the workers, each filtering across its own chunk edges.
"""

import numpy as np
import pytest

from mea_modules.concatenation import concatenate_segments, save_concatenated

sc = pytest.importorskip("spikeinterface.core")
spre = pytest.importorskip("spikeinterface.preprocessing")

FS = 20_000.0
N_CHANNELS = 6


def _segment(seed, n_frames):
    rng = np.random.default_rng(seed)
    traces = rng.normal(0.0, 10.0, (n_frames, N_CHANNELS)).astype(np.float32)
    traces[::997, :] -= 150.0
    recording = sc.NumpyRecording([traces], sampling_frequency=FS)
    return spre.highpass_filter(recording, freq_min=300.0)


def _bytes(folder):
    return b"".join(path.read_bytes() for path in sorted(folder.glob("*.raw")))


def test_the_worker_count_never_changes_the_concatenated_bytes(tmp_path):
    joined = concatenate_segments([_segment(1, 9_000), _segment(2, 7_000)])
    one = save_concatenated(joined, tmp_path / "one", n_jobs=1, chunk_duration="0.05s",
                            progress_bar=False)
    four = save_concatenated(joined, tmp_path / "four", n_jobs=4, chunk_duration="0.05s",
                             progress_bar=False)
    assert one.get_num_samples() == four.get_num_samples() == 16_000
    written = _bytes(tmp_path / "one")
    assert written and written == _bytes(tmp_path / "four")
