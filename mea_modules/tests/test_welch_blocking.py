"""The spectra estimate holds a bounded amount of memory, and says the same thing.

`welch` is `csd(x, x)`, which ends at one line in ShortTimeFFT.spectrogram --
``Sx.real**2 + Sx.imag**2`` -- holding four full-size arrays of
(channels, frequencies, segments) at once. Estimating every channel in one call
made that the peak of the whole capsule. These pin the two properties the fix
depends on: the block is derived from the shape rather than fixed, and blocking
does not move a single bit of the answer.
"""
import numpy as np
import pytest
from scipy.signal import welch as scipy_welch

from mea_modules.diagnostics import welch as welch_module
from mea_modules.diagnostics.welch import (
    _BLOCK_TARGET_BYTES,
    _BYTES_PER_CELL,
    _channel_block,
    welch_spectra,
)

# The shape the pipeline actually runs: a 30 s window of a 354-channel
# AxonTracking segment at 20 kHz. The arithmetic is free to check at full size;
# only the equality test needs real arrays, and it shrinks the target instead.
PRODUCTION_SAMPLES = 600_000
PRODUCTION_CHANNELS = 354
PRODUCTION_NPERSEG = 4096


class _Recording:
    """The narrow slice of the reader interface `welch_spectra` touches."""

    def __init__(self, traces, fs=20000.0):
        self._traces = traces
        self._fs = fs

    def get_num_samples(self):
        return self._traces.shape[0]

    def get_sampling_frequency(self):
        return self._fs

    def get_channel_ids(self):
        return list(range(self._traces.shape[1]))

    def has_scaleable_traces(self):
        return True

    def get_traces(self, start_frame=None, end_frame=None, channel_ids=None,
                   return_scaled=False, **_):
        columns = ([self.get_channel_ids().index(c) for c in channel_ids]
                   if channel_ids is not None else slice(None))
        return self._traces[start_frame:end_frame, columns]


def _signal(n_samples, n_channels, seed=7):
    rng = np.random.default_rng(seed)
    # Per-channel scale, so a column that landed in the wrong place shows up in
    # the answer rather than hiding behind identical noise.
    scale = np.linspace(1.0, 4.0, n_channels, dtype=np.float32)
    return (rng.standard_normal((n_samples, n_channels), dtype=np.float32) * scale)


def _cells(n_samples, nperseg):
    noverlap = nperseg // 2
    n_freqs = nperseg // 2 + 1
    n_segments = (n_samples - noverlap) // (nperseg - noverlap) + 1
    return n_freqs * n_segments


def test_the_production_shape_is_split_into_blocks_that_fit_the_target():
    """At the shape that caused this, one block must not be the whole pool."""
    block = _channel_block(PRODUCTION_SAMPLES, PRODUCTION_NPERSEG)
    held = block * _cells(PRODUCTION_SAMPLES, PRODUCTION_NPERSEG) * _BYTES_PER_CELL

    assert held <= _BLOCK_TARGET_BYTES, (
        f"a block of {block} channels holds {held / 1e9:.2f} GB, over the "
        f"{_BLOCK_TARGET_BYTES / 1e9:.2f} GB target"
    )
    assert block < PRODUCTION_CHANNELS, (
        f"a block of {block} covers the whole {PRODUCTION_CHANNELS}-channel "
        "pool, so nothing would be split and the fix would not apply"
    )


def test_a_longer_read_shrinks_the_block_rather_than_costing_more():
    """Derived from the shape, so a bigger read cannot quietly cost more."""
    assert _channel_block(600_000, 4096) < _channel_block(60_000, 4096)


def test_nperseg_does_not_change_what_a_block_costs():
    """Not an oversight -- the cell count is set by the read, not the window.

    Doubling `nperseg` doubles the frequency bins and halves the segments, so
    an STFT holds about as many cells either way. Pinned because it looks like
    a knob that should matter, and a future reader may otherwise `fix` the
    block derivation to react to it.
    """
    assert (_channel_block(600_000, 8192)
            == _channel_block(600_000, 4096)
            == _channel_block(600_000, 2048))


def test_the_block_never_collapses_to_zero():
    """A shape whose single channel exceeds the target still estimates it."""
    assert _channel_block(50_000_000, 65536) >= 1


def test_blocking_does_not_move_the_answer(monkeypatch):
    """Channels are independent under `axis=0`, so this must be exact.

    The target is shrunk rather than the shape grown: the property under test
    is the split, and forcing it at a small shape keeps the test honest without
    allocating the gigabytes the production shape would.
    """
    n_samples, n_channels = 20_000, 24
    traces = _signal(n_samples, n_channels)
    recording = _Recording(traces)
    channel_ids = recording.get_channel_ids()
    nperseg = min(PRODUCTION_NPERSEG, n_samples)

    monkeypatch.setattr(
        welch_module, "_BLOCK_TARGET_BYTES",
        _cells(n_samples, nperseg) * _BYTES_PER_CELL * 5,   # a block of 5
    )
    assert welch_module._channel_block(n_samples, nperseg) < n_channels, (
        "this pool still fits in one block, so the comparison proves nothing"
    )

    blocked = welch_spectra(recording, channel_ids, duration_s=n_samples / 20000.0)

    # The estimate the old code made: every channel in a single call.
    _, whole = scipy_welch(np.asarray(traces, dtype=np.float64), fs=20000.0,
                           nperseg=nperseg, axis=0)
    whole = np.maximum(whole, 1e-12)

    assert blocked["power"].shape == whole.shape
    assert blocked["power"].tobytes() == whole.tobytes(), "blocking changed the answer"


def test_a_pool_wider_than_one_block_is_estimated_in_several_calls(monkeypatch):
    """The guard for the fix itself, and the only one that can catch a revert.

    Blocking is bit-identical by construction, so every other test here passes
    just as well against a single whole-pool call -- they check that the fix is
    harmless, not that it is present. This one counts the calls, which is the
    thing that changes when someone puts the one-shot estimate back.
    """
    n_samples, n_channels = 20_000, 24
    recording = _Recording(_signal(n_samples, n_channels))
    nperseg = min(PRODUCTION_NPERSEG, n_samples)

    monkeypatch.setattr(
        welch_module, "_BLOCK_TARGET_BYTES",
        _cells(n_samples, nperseg) * _BYTES_PER_CELL * 5,   # a block of 5
    )
    block = welch_module._channel_block(n_samples, nperseg)
    assert block < n_channels, "nothing would be split; the count proves nothing"

    calls = []
    real = scipy_welch

    def counted(x, *args, **kwargs):
        calls.append(np.shape(x)[1] if np.ndim(x) > 1 else 1)
        return real(x, *args, **kwargs)

    monkeypatch.setattr("scipy.signal.welch", counted)
    welch_spectra(recording, recording.get_channel_ids(),
                  duration_s=n_samples / 20000.0)

    assert len(calls) == -(-n_channels // block), (
        f"{n_channels} channels at a block of {block} should take "
        f"{-(-n_channels // block)} estimates, not {len(calls)}"
    )
    assert max(calls) <= block, (
        f"an estimate covered {max(calls)} channels, over the block of {block}"
    )


def test_one_channel_still_returns_a_column():
    """A flat read is widened, not estimated as if it were many channels."""
    recording = _Recording(_signal(20_000, 1))
    result = welch_spectra(recording, [0], duration_s=1.0)
    assert result["power"].ndim == 2
    assert result["power"].shape[1] == 1


def test_no_resolved_channels_keeps_the_frequency_axis():
    """An empty pool is an empty answer, not a crash and not a fabricated axis."""
    recording = _Recording(_signal(20_000, 3))
    result = welch_spectra(recording, [], duration_s=1.0)
    assert result["power"].shape[1] == 0
    assert result["freqs"].ndim == 1 and result["freqs"].size > 0
    # the axis is the real one, not a stand-in
    expected, _ = scipy_welch(np.zeros((20_000, 1)), fs=20000.0, nperseg=4096, axis=0)
    assert np.array_equal(result["freqs"], expected)
