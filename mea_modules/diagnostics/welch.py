"""The Welch power spectral density a segment's spectra are cached from.

Kept apart from :mod:`.spectra`, which draws them and imports this back, so the
compute side imports no drawing code.
"""

import numpy as np

from .selection import _effective_uv, _read_traces, _resolve_frame_window


# ~2.4 Hz resolution at 10 kHz, ~4.9 Hz at 20 kHz.
DEFAULT_NPERSEG = 4096


# A perfectly-zero bin (an all-flat channel) has no place on a log axis, and one
# of them takes the whole shared axis with it. Floored far below any physical
# density rather than dropped, so every channel still draws a full curve.
DEFAULT_POWER_FLOOR = 1e-12


# Matches the read window the trace figures default to: None would mean the
# whole recording, which on an AxonTracking segment is gigabytes.
_DEFAULT_DURATION_S = 60.0


# The unit :func:`welch_spectra` RECORDS. ASCII, because this string is data: it
# travels in the result dict and through log lines, neither of which renders
# markup, and a consumer would have to strip it back out.
_UV_PSD_UNIT = "uV^2/Hz"


_COUNTS_PSD_UNIT = "adc^2/Hz"


# How much transient memory one `welch` call may hold while it works.
#
# `scipy.signal.welch` is `csd(x, x)`, which ends in ShortTimeFFT.spectrogram at
# one line -- `return Sx.real**2 + Sx.imag**2`. `Sx` is a complex128 array of
# (channels, frequencies, segments) and is still referenced while both squares
# are built, and the addition allocates its own result before either temporary
# is freed: four full-size arrays alive at once, 40 bytes per cell.
#
# On a 30 s window of a 354-channel AxonTracking segment at 20 kHz that is
# 8.5 GB of transients to produce a 5.8 MB answer, and it was the peak of the
# whole segment capsule -- more than the filtered signal it was estimated from
# (measured 2026-09-22). Every one of those arrays has channels as its leading
# dimension, so estimating a block of channels at a time divides the transient
# without touching the arithmetic: the columns are independent under `axis=0`,
# and the result is identical byte for byte.
_BLOCK_TARGET_BYTES = 1_000_000_000
_BYTES_PER_CELL = 40


def _channel_block(n_samples, nperseg):
    """How many channels to estimate at once to stay near the target.

    Derived from the shape rather than fixed, so a longer window or a larger
    `nperseg` shrinks the block instead of quietly costing more memory.
    """
    noverlap = nperseg // 2
    n_freqs = nperseg // 2 + 1
    n_segments = max(1, (n_samples - noverlap) // max(1, nperseg - noverlap) + 1)
    per_channel = n_freqs * n_segments * _BYTES_PER_CELL
    return max(1, int(_BLOCK_TARGET_BYTES // max(1, per_channel)))


def welch_spectra(
    recording,
    channel_ids,
    start_time_s=0.0,
    duration_s=_DEFAULT_DURATION_S,
    nperseg=DEFAULT_NPERSEG,
    power_floor=DEFAULT_POWER_FLOOR,
    return_in_uV=True,
):
    """Per-channel Welch power spectral density over a bounded window.

    Parameters
    ----------
    recording
        Any recording exposing the SpikeInterface read interface.
    channel_ids : sequence
        The channels to estimate, in the order they should be drawn. For the
        cross-segment comparison this figure is for, that means one channel
        per electrode cluster, drawn from the SHARED electrode set — the
        electrodes every segment being compared kept routed — rather than an
        arbitrary top-N by activity: comparing a raw panel to a filtered one
        is only meaningful over electrodes both sides actually have. Build
        that set with :func:`.stitch_consistency.backbone_channel_ids` (the shared
        electrodes) and :func:`.channel_layout.cluster_center_channels` (one
        representative per cluster within them) — or, equivalently,
        :func:`.traces.select_representative_channels` with `restrict_to` set
        to the shared set and `n_channels=0` to keep every cluster. The count
        is DYNAMIC, whatever the routing produced (about thirty on the
        current MaxWell configuration); this function draws however many
        channels it is given rather than assuming a number. Channel
        SELECTION is the caller's job — this function only estimates spectra
        for the ids it is handed.
    start_time_s, duration_s : float
        The window to read, as an offset from the first sample. Pass the same
        window the trace figures use.
    nperseg : int
        Welch segment length, clamped to the window when the window is shorter.
    power_floor : float
        Densities below this are raised to it (see the module note).
    return_in_uV : bool
        Microvolts when the recording can scale; a recording carrying no
        gain/offset degrades to device counts with a warning, and ``unit``
        records which was used.

    Returns
    -------
    dict
        ``channel_ids``, ``freqs`` (n_freqs,), ``power`` (n_freqs, n_channels),
        ``unit``, ``fs_hz``, ``n_samples`` and the ``nperseg`` actually used.
        Arrays stay numpy — this result feeds a figure, not a JSON file.
    """
    from scipy.signal import welch

    channel_ids = list(channel_ids)
    start_frame, end_frame, fs = _resolve_frame_window(recording, start_time_s, duration_s)
    use_uV = _effective_uv(recording, return_in_uV)
    traces = np.asarray(_read_traces(recording, start_frame, end_frame, channel_ids, use_uV))

    # A single channel can arrive as a flat vector; the estimate below indexes
    # columns, and the result is documented as (n_freqs, n_channels).
    if traces.ndim == 1:
        traces = traces[:, None]

    nperseg = min(int(nperseg), traces.shape[0])
    # A block of channels at a time (see `_BLOCK_TARGET_BYTES`). The float64
    # cast moved in here with it: casting the whole read up front held a second
    # full copy of the signal for the length of the estimate, and only one
    # block of it is ever in use.
    block = _channel_block(traces.shape[0], nperseg)
    freqs, columns = None, []
    for start in range(0, traces.shape[1], block):
        chunk = np.asarray(traces[:, start:start + block], dtype=np.float64)
        freqs, part = welch(chunk, fs=fs, nperseg=nperseg, axis=0)
        columns.append(part)
        del chunk
    if columns:
        power = columns[0] if len(columns) == 1 else np.concatenate(columns, axis=1)
    else:
        # No channels resolved. `welch` of a zero-column array does not return a
        # usable frequency axis, so estimate one silent channel for the axis and
        # keep the (n_freqs, 0) shape the caller is promised -- rather than
        # inventing the axis or handing back an empty one.
        freqs, one = welch(np.zeros((traces.shape[0], 1)), fs=fs, nperseg=nperseg, axis=0)
        power = one[:, :0]
    del columns
    power = np.maximum(power, power_floor)

    return {
        "channel_ids": channel_ids,
        "freqs": freqs,
        "power": power,
        "unit": _UV_PSD_UNIT if use_uV else _COUNTS_PSD_UNIT,
        "fs_hz": float(fs),
        "n_samples": int(traces.shape[0]),
        "nperseg": int(nperseg),
    }
