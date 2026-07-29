"""Per-channel quality metrics for an MEA recording.

Every metric here answers the same question from a different angle: is this
channel — and by extension this well — carrying signal worth spending a sort on?

* :func:`mad_noise` — how loud is the baseline on each channel,
* :func:`activity_rate` — how often does each channel cross a spike threshold,
* :func:`detect_bad_channels` — SpikeInterface's own dead/noisy classifier,
* :func:`dead_well_flags` — reduces the above to one verdict per well.

These are the only functions in the package that read traces, and they do so on
a *bounded* sample: a handful of short windows drawn from the recording, never
the whole file. A Maxwell AxonTracking scan is tens of GB per segment, so the
sampling budget (`duration_s`) is the parameter that decides the runtime, not
the file size. Peak memory is one window, because windows are streamed rather
than concatenated.

Results are plain dicts of Python scalars and lists — JSON-serializable as-is,
no plots, no files. Thresholds are always arguments with documented defaults.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Sampling budget. Ten half-second windows spread over the recording: enough to
# average out a transient burst, small enough to stay in RAM at 1024 channels.
DEFAULT_DURATION_S = 5.0
DEFAULT_NUM_CHUNKS = 10

# Maxwell traces come off the chip essentially unfiltered, and the DC/drift
# component dominates MAD if you leave it in. 300 Hz is the usual spike-band
# corner. Applied only to recordings that do not already report themselves as
# filtered; pass None to skip it entirely.
DEFAULT_HIGHPASS_HZ = 300.0

# Fixed by default so a QC re-run on the same file reproduces the same numbers.
DEFAULT_SEED = 0

# MAD -> Gaussian sigma. Same constant SpikeInterface's get_noise_levels uses,
# so "5 sd" means the same thing here as it does everywhere else in the stack.
_MAD_TO_SIGMA = 1.0 / 0.6744897501960817


# --------------------------------------------------------------------------
# internals: sampling
# --------------------------------------------------------------------------


def _json_scalar(value):
    """Convert a numpy scalar to its Python equivalent; pass others through."""
    return value.item() if hasattr(value, "item") else value


def _channel_ids(recording):
    """Channel ids as a JSON-safe list (they arrive as numpy str_/int64)."""
    return [_json_scalar(cid) for cid in recording.get_channel_ids()]


def _to_signed(recording):
    """Convert unsigned traces to signed, which the preprocessing chain requires.

    Maxwell stores its ADC counts as uint16, and SpikeInterface's filters refuse
    unsigned dtypes outright. The conversion subtracts a constant (2**(nbits-1)),
    which every metric here is blind to: MAD is shift-invariant, and crossings
    are counted on median-centered traces.
    """
    if np.dtype(recording.get_dtype()).kind != "u":
        return recording

    from spikeinterface.preprocessing import unsigned_to_signed

    logger.debug("recording dtype is unsigned; converting to signed for preprocessing")
    return unsigned_to_signed(recording)


def _prepare(recording, highpass_hz, return_in_uV):
    """Build the lazy recording the metrics read from.

    Signed conversion, then an optional high-pass, then a decision about units.
    Nothing is read here — every step returns a lazy SpikeInterface object, so
    the filtering happens per window when traces are actually pulled.

    Returns ``(prepared_recording, return_in_uV, unit, applied_highpass_hz)``.
    """
    prepared = _to_signed(recording)

    applied_hp = None
    if highpass_hz is None:
        pass
    elif recording.is_filtered():
        logger.debug("recording reports is_filtered; skipping the %s Hz high-pass", highpass_hz)
    else:
        from spikeinterface.preprocessing import highpass_filter

        prepared = highpass_filter(prepared, freq_min=float(highpass_hz))
        applied_hp = float(highpass_hz)

    # A recording without gain/offset cannot be scaled; asking anyway raises deep
    # inside get_traces. Degrade to raw ADC counts and say so in the result,
    # since a noise number means nothing without its unit.
    if return_in_uV and not prepared.has_scaleable_traces():
        logger.warning("recording has no gain/offset; reporting metrics in raw ADC units")
        return_in_uV = False

    return prepared, return_in_uV, ("uV" if return_in_uV else "adc"), applied_hp


def _plan_windows(recording, duration_s, num_chunks, seed, placement):
    """Pick the ``(segment_index, start_frame, end_frame)`` windows to read.

    The total sampled duration is held at `duration_s` however many segments and
    windows are involved: the per-window length is the budget divided by the
    window count, not a fixed constant. `placement` is "random" (seeded, the
    SpikeInterface convention) or "strided" (evenly spaced, seed ignored).
    """
    if duration_s <= 0:
        raise ValueError(f"duration_s must be positive, got {duration_s}")

    fs = float(recording.get_sampling_frequency())
    n_segments = int(recording.get_num_segments())
    per_segment = max(1, int(num_chunks) // n_segments)
    total_windows = per_segment * n_segments
    window_frames = max(1, int(round(duration_s * fs / total_windows)))

    rng = np.random.default_rng(seed)
    windows = []
    for segment_index in range(n_segments):
        n_frames = int(recording.get_num_frames(segment_index))
        size = min(window_frames, n_frames)
        # Last legal start; 0 when the segment is shorter than one window.
        high = n_frames - size
        if high <= 0:
            starts = np.zeros(per_segment, dtype=np.int64)
        elif placement == "strided":
            starts = np.linspace(0, high, per_segment, dtype=np.int64)
        elif placement == "random":
            starts = np.sort(rng.integers(low=0, high=high, size=per_segment, endpoint=True))
        else:
            raise ValueError(f"placement must be 'random' or 'strided', got {placement!r}")

        # A segment shorter than the budget is read once, not `per_segment` times.
        if high <= 0:
            starts = starts[:1]
        for start in starts:
            windows.append((segment_index, int(start), int(start) + size))

    return windows


def _iter_traces(recording, windows, return_in_uV):
    """Yield one window of traces at a time, shaped (n_samples, n_channels)."""
    for segment_index, start, end in windows:
        yield recording.get_traces(
            start_frame=start,
            end_frame=end,
            segment_index=segment_index,
            return_in_uV=return_in_uV,
        )


def _chunk_mad(traces):
    """Per-channel MAD of one window, rescaled to a Gaussian-sigma equivalent."""
    traces = np.asarray(traces, dtype=np.float32)
    median = np.median(traces, axis=0, keepdims=True)
    return np.median(np.abs(traces - median), axis=0) * _MAD_TO_SIGMA


def _sampling_meta(recording, windows, unit, applied_hp, seed, placement):
    """The provenance block every metric dict carries."""
    fs = float(recording.get_sampling_frequency())
    sampled_frames = sum(end - start for _, start, end in windows)
    return {
        "unit": unit,
        "n_channels": int(recording.get_num_channels()),
        "n_chunks": len(windows),
        "sampled_s": sampled_frames / fs if fs else 0.0,
        "fs_hz": fs,
        "highpass_hz": applied_hp,
        "seed": seed,
        "placement": placement,
    }


# --------------------------------------------------------------------------
# public metrics
# --------------------------------------------------------------------------


def mad_noise(
    recording,
    duration_s=DEFAULT_DURATION_S,
    num_chunks=DEFAULT_NUM_CHUNKS,
    seed=DEFAULT_SEED,
    placement="random",
    highpass_hz=DEFAULT_HIGHPASS_HZ,
    return_in_uV=True,
):
    """Per-channel baseline noise, estimated by median absolute deviation.

    MAD rather than standard deviation because spikes are outliers: on an active
    channel the std tracks the firing rate, while the MAD keeps reporting the
    baseline. The value returned is the MAD divided by 0.6745, i.e. the sigma of
    the Gaussian that would produce it — the same convention as SpikeInterface's
    ``get_noise_levels(method="mad")``, so it can be compared directly against
    spike-detection thresholds expressed in sd.

    Reads `num_chunks` windows totalling `duration_s` seconds, one at a time.
    The per-window MADs are combined by median, so a single window landing on an
    artifact does not move the estimate.

    Returns a dict with ``channel_ids``, ``noise`` (one value per channel, in
    ``unit``), ``median_noise``, and the sampling parameters used.
    """
    prepared, return_in_uV, unit, applied_hp = _prepare(recording, highpass_hz, return_in_uV)
    windows = _plan_windows(recording, duration_s, num_chunks, seed, placement)

    per_window = [_chunk_mad(traces) for traces in _iter_traces(prepared, windows, return_in_uV)]
    noise = np.median(np.stack(per_window), axis=0)

    result = {
        "channel_ids": _channel_ids(recording),
        "noise": [float(v) for v in noise],
        "median_noise": float(np.median(noise)),
    }
    result.update(_sampling_meta(recording, windows, unit, applied_hp, seed, placement))
    return result


def activity_rate(
    recording,
    threshold_sd=5.0,
    polarity="negative",
    refractory_ms=1.0,
    noise=None,
    duration_s=DEFAULT_DURATION_S,
    num_chunks=DEFAULT_NUM_CHUNKS,
    seed=DEFAULT_SEED,
    placement="random",
    highpass_hz=DEFAULT_HIGHPASS_HZ,
    return_in_uV=True,
):
    """Per-channel threshold-crossing rate, in events per second.

    A cheap stand-in for a firing rate: no sorting, no templates, just how often
    the trace leaves the noise band. Useful for telling a silent channel from a
    live one before committing to a sort.

    `threshold_sd` is in units of that channel's own noise (see :func:`mad_noise`),
    and `polarity` selects which side to count — extracellular spikes are
    negative-going, which is the default. `refractory_ms` is the dead time after
    a crossing: without it the ringing of one spike is counted several times.

    By default the threshold is recomputed from each window's own MAD, which
    keeps it honest under slow drift. Pass `noise` — the dict from
    :func:`mad_noise`, or a per-channel sequence in the same units — to hold the
    threshold fixed across windows instead.

    Rates are counts over the *sampled* seconds, not the recording's full
    duration. Counting per window costs at most one spurious event per window,
    when a window happens to open mid-excursion; at realistic duty cycles that
    is far below the noise on the estimate.

    Returns a dict with ``channel_ids``, ``rate_hz``, ``n_events``, the threshold
    settings, and the sampling parameters used.
    """
    if polarity not in ("negative", "positive", "both"):
        raise ValueError(f"polarity must be 'negative', 'positive' or 'both', got {polarity!r}")

    prepared, return_in_uV, unit, applied_hp = _prepare(recording, highpass_hz, return_in_uV)
    windows = _plan_windows(recording, duration_s, num_chunks, seed, placement)

    fixed_noise = None
    if noise is not None:
        fixed_noise = np.asarray(_metric_values(noise, "noise"), dtype=np.float64)
        n_channels = recording.get_num_channels()
        if fixed_noise.size != n_channels:
            raise ValueError(f"noise has {fixed_noise.size} values but recording has {n_channels} channels")

    refractory_frames = int(round(float(refractory_ms) * 1e-3 * recording.get_sampling_frequency()))
    counts = np.zeros(recording.get_num_channels(), dtype=np.int64)
    for traces in _iter_traces(prepared, windows, return_in_uV):
        level = fixed_noise if fixed_noise is not None else _chunk_mad(traces)
        counts += _count_crossings(traces, level * float(threshold_sd), polarity, refractory_frames)

    meta = _sampling_meta(recording, windows, unit, applied_hp, seed, placement)
    sampled_s = meta["sampled_s"]
    rates = counts / sampled_s if sampled_s else np.zeros_like(counts, dtype=np.float64)

    result = {
        "channel_ids": _channel_ids(recording),
        "rate_hz": [float(v) for v in rates],
        "n_events": [int(v) for v in counts],
        "threshold_sd": float(threshold_sd),
        "polarity": polarity,
        "refractory_ms": float(refractory_ms),
        "fixed_threshold": fixed_noise is not None,
    }
    result.update(meta)
    return result


def _count_crossings(traces, threshold, polarity, refractory_frames):
    """Count threshold excursions per channel, one dead time apart at minimum.

    An excursion counts when the trace is over threshold *and* nothing was over
    threshold in the preceding `refractory_frames` samples. That collapses both
    a long excursion and a ringing multi-crossing into a single event, without
    a per-channel Python loop: the "was anything recently over" test is a
    difference of cumulative counts.
    """
    traces = np.asarray(traces, dtype=np.float32)
    centered = traces - np.median(traces, axis=0, keepdims=True)
    threshold = np.abs(np.asarray(threshold, dtype=np.float32))

    if polarity == "negative":
        over = centered < -threshold
    elif polarity == "positive":
        over = centered > threshold
    else:
        over = np.abs(centered) > threshold

    if refractory_frames <= 0:
        # Plain rising edges: sample over threshold, previous sample not.
        events = over.copy()
        events[1:] &= ~over[:-1]
        return events.sum(axis=0, dtype=np.int64)

    cumulative = np.cumsum(over, axis=0, dtype=np.int32)
    before = np.zeros_like(cumulative)
    before[1:] = cumulative[:-1]  # crossings up to and including i-1
    window_start = np.zeros_like(cumulative)
    window_start[refractory_frames + 1 :] = cumulative[: -(refractory_frames + 1)]
    recent = before - window_start  # crossings inside [i - refractory, i - 1]

    events = over & (recent == 0)
    return events.sum(axis=0, dtype=np.int64)


def detect_bad_channels(
    recording,
    method="mad",
    duration_s=DEFAULT_DURATION_S,
    num_chunks=DEFAULT_NUM_CHUNKS,
    seed=DEFAULT_SEED,
    **detect_kwargs,
):
    """Classify channels with SpikeInterface's bad-channel detector.

    A thin wrapper: it maps this module's sampling vocabulary onto SpikeInterface's
    (`duration_s`/`num_chunks` become ``chunk_duration_s``/``num_random_chunks``),
    keeps the default sample bounded, and returns JSON-serializable results
    instead of numpy arrays. Any other keyword — ``std_mad_threshold``,
    ``psd_hf_threshold``, ``n_neighbors``, … — is forwarded untouched.

    `method` defaults to "mad" rather than SpikeInterface's "coherence+psd"
    because the latter is written for a linear depth-ordered probe: it ranks
    channels by depth and labels a contiguous top block "out" (of brain), a
    notion with no meaning on a planar MEA grid. "mad" is geometry-free — it
    flags a channel whose MAD exceeds ``std_mad_threshold`` (default 5) times
    the array median — and labels channels good/noise. Pass
    ``method="coherence+psd"`` explicitly if you want dead/out labels too.

    Note that SpikeInterface holds the whole sample in memory at once here, so
    `duration_s` is the knob that bounds footprint.

    Returns a dict with ``bad_channel_ids``, per-channel ``labels`` aligned with
    ``channel_ids``, ``label_counts``, and ``bad_fraction``.
    """
    from spikeinterface.preprocessing import detect_bad_channels as si_detect_bad_channels

    if duration_s <= 0:
        raise ValueError(f"duration_s must be positive, got {duration_s}")
    num_chunks = max(1, int(num_chunks))

    # SpikeInterface high-pass filters internally when the recording is not
    # already filtered, and that filter rejects Maxwell's unsigned dtype.
    bad_ids, labels = si_detect_bad_channels(
        _to_signed(recording),
        method=method,
        chunk_duration_s=float(duration_s) / num_chunks,
        num_random_chunks=num_chunks,
        seed=seed,
        **detect_kwargs,
    )

    labels = [str(label) for label in labels]
    n_channels = int(recording.get_num_channels())
    unique, counts = np.unique(labels, return_counts=True)

    return {
        "method": method,
        "channel_ids": _channel_ids(recording),
        "labels": labels,
        "bad_channel_ids": [_json_scalar(cid) for cid in bad_ids],
        "label_counts": {str(k): int(v) for k, v in zip(unique, counts)},
        "bad_fraction": len(bad_ids) / n_channels if n_channels else 0.0,
        "n_channels": n_channels,
        "n_chunks": num_chunks,
        "sampled_s": float(duration_s),
        "seed": seed,
    }


def dead_well_flags(
    noise=None,
    activity=None,
    bad_channels=None,
    dead_noise_ratio=0.1,
    noisy_noise_ratio=5.0,
    min_activity_hz=0.01,
    max_unusable_fraction=0.5,
    min_active_fraction=0.02,
):
    """Reduce per-channel metrics to a single verdict: is this well dead?

    Takes the dicts returned by :func:`mad_noise`, :func:`activity_rate` and
    :func:`detect_bad_channels` — any subset, and each may equally be a plain
    per-channel sequence — and turns them into array-level fractions.

    The channel rules are *relative to the well itself*, never absolute
    voltages, so they survive a change of gain, of units, or of chip revision:

    * dead — noise at or below `dead_noise_ratio` (default 0.1) times the array
      median noise, i.e. a flat or disconnected electrode. A NaN counts as dead.
    * noisy — noise at or above `noisy_noise_ratio` (default 5.0) times the
      median, the same multiplier SpikeInterface's "mad" method uses.
    * bad — whatever :func:`detect_bad_channels` flagged.
    * active — threshold-crossing rate at or above `min_activity_hz` (default
      0.01 events/s, about one event per sampled 100 s).

    The well is called dead when the union of dead/noisy/bad channels exceeds
    `max_unusable_fraction` (default 0.5), or when the active fraction falls
    below `min_active_fraction` (default 0.02) — a well can be electrically
    perfect and still have nothing growing on it.

    Returns a dict of fractions plus ``is_dead`` and a ``reasons`` list naming
    every rule that fired, so a downstream report can say *why*.
    """
    noise_values = _metric_values(noise, "noise")
    rate_values = _metric_values(activity, "rate_hz")
    bad_ids = bad_channels.get("bad_channel_ids") if isinstance(bad_channels, dict) else bad_channels

    n_channels = _resolve_n_channels(noise_values, rate_values, bad_channels)

    dead_mask = np.zeros(n_channels, dtype=bool)
    noisy_mask = np.zeros(n_channels, dtype=bool)
    if noise_values is not None:
        values = np.asarray(noise_values, dtype=np.float64)
        median = float(np.nanmedian(values)) if np.any(np.isfinite(values)) else 0.0
        if median > 0:
            dead_mask = ~(values > dead_noise_ratio * median)  # NaN-safe: NaN is dead
            noisy_mask = values >= noisy_noise_ratio * median
        else:
            # Every channel flat (or unmeasurable): nothing to be relative to.
            logger.warning("median channel noise is not positive; treating all channels as dead")
            dead_mask = np.ones(n_channels, dtype=bool)

    bad_mask = np.zeros(n_channels, dtype=bool)
    if bad_ids is not None:
        channel_ids = bad_channels.get("channel_ids") if isinstance(bad_channels, dict) else None
        if channel_ids is not None:
            index = {cid: i for i, cid in enumerate(channel_ids)}
            bad_mask[[index[cid] for cid in bad_ids if cid in index]] = True
        else:
            # No id list to join on; all we can honour is the count.
            bad_mask[: len(bad_ids)] = True

    active_fraction = None
    if rate_values is not None:
        rates = np.asarray(rate_values, dtype=np.float64)
        active_fraction = float(np.count_nonzero(rates >= min_activity_hz) / n_channels)

    unusable = dead_mask | noisy_mask | bad_mask
    unusable_fraction = float(np.count_nonzero(unusable) / n_channels)

    reasons = []
    if unusable_fraction > max_unusable_fraction:
        reasons.append(f"unusable_fraction {unusable_fraction:.3f} > {max_unusable_fraction}")
    if active_fraction is not None and active_fraction < min_active_fraction:
        reasons.append(f"active_fraction {active_fraction:.3f} < {min_active_fraction}")

    return {
        "n_channels": int(n_channels),
        "dead_fraction": float(np.count_nonzero(dead_mask) / n_channels),
        "noisy_fraction": float(np.count_nonzero(noisy_mask) / n_channels),
        "bad_fraction": float(np.count_nonzero(bad_mask) / n_channels),
        "unusable_fraction": unusable_fraction,
        "active_fraction": active_fraction,
        "is_dead": bool(reasons),
        "reasons": reasons,
        "thresholds": {
            "dead_noise_ratio": float(dead_noise_ratio),
            "noisy_noise_ratio": float(noisy_noise_ratio),
            "min_activity_hz": float(min_activity_hz),
            "max_unusable_fraction": float(max_unusable_fraction),
            "min_active_fraction": float(min_active_fraction),
        },
    }


def _metric_values(metric, key):
    """Accept either a metric dict from this module or a bare per-channel list."""
    if metric is None:
        return None
    if isinstance(metric, dict):
        if key not in metric:
            raise ValueError(f"metric dict has no {key!r} entry; keys: {sorted(metric)}")
        return metric[key]
    return metric


def _resolve_n_channels(noise_values, rate_values, bad_channels):
    """Channel count agreed on by whichever metrics were supplied."""
    sizes = {len(v) for v in (noise_values, rate_values) if v is not None}
    if isinstance(bad_channels, dict) and bad_channels.get("n_channels"):
        sizes.add(int(bad_channels["n_channels"]))
    if not sizes:
        raise ValueError("dead_well_flags needs at least one of noise, activity or bad_channels")
    if len(sizes) > 1:
        raise ValueError(f"metrics disagree on channel count: {sorted(sizes)}")
    return sizes.pop()
