"""A chunked read of the chain gives the samples a continuous read gives.

SpikeInterface filters a lazy read chunk by chunk with `margin_ms` of signal on
either side of each chunk. Its default, 5 ms, is shorter than a 300 Hz filter
takes to settle, so the samples beside every chunk join came out different from
one continuous pass: the same segment gave different numbers depending on how it
was read (measured on a real segment, 2026-09-21). The chain's filters now read
a margin computed from the filter they design.
"""
import numpy as np
import pytest
from spikeinterface.core import NumpyRecording, generate_recording

from mea_modules.diagnostics.buffer import buffered_signal
from mea_modules.preprocessing import ensure_signed, preprocess_segment, to_float32
from mea_modules.preprocessing.filters import bandpass, highpass, settling_margin_ms
from mea_modules.quality.metrics import _prepare

FS = 20_000.0


def _raw(seconds=3.0, n_channels=8):
    """int16 counts with a large DC offset and slow drift, as a Maxwell trace
    has before the high-pass -- the edge a chunk restart has to settle from."""
    rng = np.random.default_rng(0)
    n = int(seconds * FS)
    t = np.arange(n) / FS
    traces = 1000.0 + 40.0 * np.sin(2 * np.pi * 3.0 * t)[:, None] + rng.normal(0, 10, (n, n_channels))
    recording = NumpyRecording([traces.astype(np.int16)], sampling_frequency=FS,
                               channel_ids=[str(i) for i in range(n_channels)])
    probe = generate_recording(num_channels=n_channels, sampling_frequency=FS, durations=[1.0])
    recording.set_probe(probe.get_probe(), in_place=True)
    return recording


def test_the_margin_follows_the_filter_it_protects():
    standard = settling_margin_ms(FS, 300.0, "highpass")
    assert 40.0 < standard < 60.0
    # A lower cut-off rings longer; so does a steeper filter.
    assert settling_margin_ms(FS, 100.0, "highpass") > 2 * standard
    assert settling_margin_ms(FS, 300.0, "highpass", filter_order=8) > standard
    # The band-pass is governed by its low edge.
    assert settling_margin_ms(FS, [300.0, 6000.0], "bandpass") == pytest.approx(standard, rel=0.25)


def test_a_margin_of_seconds_is_announced(caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="mea_modules.preprocessing.filters"):
        assert settling_margin_ms(FS, 300.0, "highpass") < 1000.0
        assert not caplog.records
        assert settling_margin_ms(FS, 1.0, "highpass") > 1000.0
    assert any("margin per chunk edge" in r.getMessage() for r in caplog.records)


def test_the_chain_filters_carry_the_margin_unless_the_caller_chose_one():
    raw = _raw(seconds=0.2)
    assert highpass(raw)._kwargs["margin_ms"] == settling_margin_ms(FS, 300.0, "highpass")
    assert bandpass(raw)._kwargs["margin_ms"] == settling_margin_ms(FS, [300.0, 6000.0], "bandpass")
    assert highpass(raw, margin_ms=5.0)._kwargs["margin_ms"] == 5.0


def test_a_chunked_buffer_of_the_chain_equals_one_continuous_read():
    """The real producer (the segment chain) into the real consumer (the
    diagnostics buffer, filled in 1 s chunks)."""
    qc = preprocess_segment(to_float32(ensure_signed(_raw())), rename_to_electrodes=False)
    continuous = qc.get_traces()
    with buffered_signal(qc, "memory") as (buffered, _):
        assert np.array_equal(buffered.get_traces(), continuous)


def test_the_noise_estimate_high_passes_with_the_same_margin():
    prepared, _, _, applied = _prepare(_raw(seconds=0.2), 300.0, True)
    assert applied == 300.0
    assert prepared._kwargs["margin_ms"] == settling_margin_ms(FS, 300.0, "highpass")
