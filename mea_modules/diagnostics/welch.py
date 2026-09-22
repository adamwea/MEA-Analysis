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
    traces = np.asarray(
        _read_traces(recording, start_frame, end_frame, channel_ids, use_uV),
        dtype=np.float64,
    )

    nperseg = min(int(nperseg), traces.shape[0])
    freqs, power = welch(traces, fs=fs, nperseg=nperseg, axis=0)
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
