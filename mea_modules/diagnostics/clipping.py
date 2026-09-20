"""Per-channel saturation census: how often each channel sat on a rail.

Saturation is the failure mode MAD noise cannot see, because a railed channel
is *quiet* — its baseline deviation collapses exactly where its signal is being
destroyed. The only way to catch it is to count samples sitting at an extreme,
which is what this does.

Counted against two references, both reported so a consumer can choose:

* the **dtype rails** — the bounds of the integer word the samples are stored
  in, which is the true ADC ceiling when acquisition uses the full word;
* the **observed extremes** over the sampled windows — the effective rails when
  the ADC range is narrower than the storage dtype. A MaxWell chip writes
  10-bit counts into a 16-bit word, so dtype rails may simply never occur and
  the observed pair is the only thing that fires.

**This must be handed the RAW, unfiltered recording, in its integer dtype.**
Two separate reasons, and both silently produce a useless report rather than an
error: a filter smears a flat rail into a curve, so no sample sits at the
extreme any more; and a float recording has no ``iinfo`` bounds, so every
dtype-rail number degrades to ``None``. This is the exact mirror of the noise
path, which needs a float cast before filtering — the two diagnostics want
opposite dtypes and must be given different views of the same recording.

Reads a bounded budget: `duration_s` seconds split into `num_chunks` windows
strided across the recording, the same sampling shape
:mod:`mea_modules.quality` uses, so runtime is a budget rather than a function
of the file size. Returns a JSON-serializable dict; writes nothing.
"""

import logging

import numpy as np

from mea_modules.quality import DEFAULT_DURATION_S, DEFAULT_NUM_CHUNKS

logger = logging.getLogger(__name__)


def clipping_census(recording, duration_s=DEFAULT_DURATION_S, num_chunks=DEFAULT_NUM_CHUNKS):
    """Count per-channel rail hits over a bounded sample of `recording`.

    Parameters
    ----------
    recording
        The RAW, unfiltered view, in its integer dtype (see the module note).
    duration_s : float
        Total seconds read, split evenly across the windows.
    num_chunks : int
        How many windows that budget is spread over, evenly strided from the
        first sample to the last full window.

    Returns
    -------
    dict
        ``n_channels``, ``channel_ids``, ``sampled_s``, the ``windows`` read,
        the ``dtype`` and its ``dtype_rails``, the ``observed_extremes``,
        per-channel ``dtype_rail_fraction`` and ``observed_extreme_fraction``,
        ``per_channel_observed_min`` / ``_max``, and a ``summary`` carrying the
        headline ``channels_at_dtype_rail`` count. On a non-integer recording
        every dtype-rail entry is ``None``.
    """
    fs = float(recording.get_sampling_frequency())
    n_samples = int(recording.get_num_samples())
    n_channels = int(recording.get_num_channels())
    # numpy scalars are not JSON-serializable; .item() is what turns a numpy
    # channel id back into the plain int or str it came from.
    channel_ids = [
        cid.item() if hasattr(cid, "item") else cid for cid in recording.get_channel_ids()
    ]

    num_chunks = max(1, int(num_chunks))
    window = max(1, int(round(float(duration_s) * fs / num_chunks)))
    window = min(window, n_samples)
    max_start = max(0, n_samples - window)
    starts = np.unique(np.linspace(0, max_start, num=num_chunks, dtype=np.int64))

    dtype = np.dtype(recording.get_dtype())
    if dtype.kind in "iu":
        info = np.iinfo(dtype)
        rail_low, rail_high = int(info.min), int(info.max)
    else:
        rail_low = rail_high = None
        logger.warning("raw dtype %s is not integer; dtype-rail counts unavailable", dtype)

    total = 0
    at_low = np.zeros(n_channels, dtype=np.int64)
    at_high = np.zeros(n_channels, dtype=np.int64)
    observed_min = np.full(n_channels, np.inf)
    observed_max = np.full(n_channels, -np.inf)
    windows_read = []
    for start in starts:
        end = min(n_samples, int(start) + window)
        traces = np.asarray(
            recording.get_traces(start_frame=int(start), end_frame=end, return_in_uV=False)
        )
        total += traces.shape[0]
        if rail_low is not None:
            at_low += (traces == rail_low).sum(axis=0)
            at_high += (traces == rail_high).sum(axis=0)
        observed_min = np.minimum(observed_min, traces.min(axis=0))
        observed_max = np.maximum(observed_max, traces.max(axis=0))
        windows_read.append([int(start), int(end)])

    # The observed extremes are only known once every window has been seen, so
    # counting hits against them needs a second pass — over the same windows,
    # never a wider read.
    at_obs_min = np.zeros(n_channels, dtype=np.int64)
    at_obs_max = np.zeros(n_channels, dtype=np.int64)
    global_min = float(observed_min.min()) if total else 0.0
    global_max = float(observed_max.max()) if total else 0.0
    for start, end in windows_read:
        traces = np.asarray(
            recording.get_traces(start_frame=start, end_frame=end, return_in_uV=False)
        )
        at_obs_min += (traces == global_min).sum(axis=0)
        at_obs_max += (traces == global_max).sum(axis=0)

    def fractions(counts):
        return [float(c) / total if total else 0.0 for c in counts]

    dtype_fraction = (
        fractions(at_low + at_high) if rail_low is not None else [None] * n_channels
    )
    census = {
        "n_channels": n_channels,
        "channel_ids": channel_ids,
        "sampled_s": total / fs if fs else 0.0,
        "windows": windows_read,
        "dtype": str(dtype),
        "dtype_rails": [rail_low, rail_high],
        "observed_extremes": {"min": global_min, "max": global_max},
        "dtype_rail_fraction": dtype_fraction,
        "observed_extreme_fraction": fractions(at_obs_min + at_obs_max),
        "per_channel_observed_min": [float(v) for v in observed_min],
        "per_channel_observed_max": [float(v) for v in observed_max],
        "summary": {
            "max_dtype_rail_fraction": (
                max(v for v in dtype_fraction) if rail_low is not None else None
            ),
            "channels_at_dtype_rail": (
                int(np.count_nonzero((at_low + at_high) > 0)) if rail_low is not None else None
            ),
        },
    }
    logger.info(
        "clipping census over %.2f s: %s channel(s) touched a dtype rail",
        census["sampled_s"], census["summary"]["channels_at_dtype_rail"],
    )
    return census
