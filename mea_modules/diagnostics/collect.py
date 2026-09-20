"""Compute one segment's diagnostics once, where the recording is already open.

This is the producer half of the compute-in-capsule / plot-from-cache split.
The capsule that opens a segment calls :func:`collect_segment_diagnostics` while
it has the recording in hand, and hands the result straight to
:func:`mea_modules.diagnostics.cache.write_cache`. The plot tool then reads that
cache and draws. Neither side re-derives anything the other computed.

**Why it lives here and not in the capsule.** Put this computation in the
pipeline and there are immediately two definitions of every metric: the
capsule's, and whatever a figure tool or an analysis repo does when it wants the
same number. One mechanic, one definition -- so the mechanic is here, the capsule
is the glue that decides WHICH recording to point it at and WHERE to write the
result, and a second consumer calls exactly this function.

**The one thing a caller must get right: the QC chain is not the compute chain.**
The compute chain a segment is sorted from keeps the recording's integer dtype
through the filters. Filtering in integers quantizes traces to whole ADC counts,
and the MAD of a signal quantized to ~1 LSB collapses onto multiples of
``LSB x 1.4826`` -- so noise measured on that chain describes the converter, not
the tissue. QC therefore runs on a chain cast to float32 BEFORE the filters
(revision R10), which is a THIRD object, distinct from both the raw view and the
capsule's own processed chain::

    raw_view = rename_channels_to_electrodes(ensure_signed(recording))
    qc_rec   = preprocess_segment(to_float32(ensure_signed(recording)), **kwargs)

Passing the capsule's existing `processed` here instead would produce wrong
numbers that no byte-identity test could catch, because the live path a cached
figure is compared against would be equally wrong.

The clipping census needs the exact opposite -- the raw, integer, unfiltered
view, because a filter smears a flat rail into a curve -- which is why both
recordings are arguments rather than one.
"""

import logging

import numpy as np

from ..quality import (
    activity_rate,
    dead_well_flags,
    detect_bad_channels,
    mad_noise,
)
from .artifacts import artifact_census
from .channel_flags import flag_channels
from .clipping import clipping_census
from .raster import _select_channels, detect_threshold_crossings
from .spectra import welch_spectra
from .traces import select_representative_channels

logger = logging.getLogger(__name__)

# What the cached trace windows are stored in. The unit travels with the samples
# because the emitters label their axes from it: a window cached in microvolts
# that reported device counts would draw real microvolts under an "ADC counts"
# label, with nothing on the figure to catch it.
_TRACE_UNIT = "uV"


def _effective_trace_unit(recording, want_uv):
    """The unit a read of `recording` will ACTUALLY come back in.

    ``return_in_uV=True`` is a request, not a guarantee: a recording with no
    gain serves device counts regardless. Recording what came back, rather than
    what was asked for, is what keeps the cached figure honest.
    """
    if not want_uv:
        return "ADC"
    try:
        return _TRACE_UNIT if recording.has_scaleable_traces() else "ADC"
    except Exception:  # noqa: BLE001 - a view that cannot say is device counts
        return "ADC"


def _window_block(recording, channel_ids, window_s, start_s=0.0):
    """One bounded, frame-faithful sample window, in the unit it came back in.

    `frame_offset` is the block's start in the recording's own numbering, and it
    is load-bearing: decimation and real-elapsed gap shading are computed from
    frame numbers, so a renumbered window shades the wrong stretch.
    """
    fs = float(recording.get_sampling_frequency())
    start = int(round(start_s * fs))
    end = min(start + int(round(window_s * fs)), int(recording.get_num_samples()))
    unit = _effective_trace_unit(recording, True)
    traces = recording.get_traces(
        start_frame=start, end_frame=end, channel_ids=list(channel_ids),
        return_in_uV=(unit == _TRACE_UNIT),
    )
    return {
        "channel_ids": list(channel_ids),
        "traces": np.asarray(traces),
        "frame_offset": start,
        "sampling_frequency": fs,
        "unit": unit,
    }


def _tolerant(what, fn, default=None):
    """Run one diagnostic; a failure costs that diagnostic, never the segment.

    Returns ``(value, error_string_or_None)``. A capsule that refused to write a
    descriptor because a PSD failed would be trading a whole segment for a
    figure, which is never the right trade.
    """
    try:
        return fn(), None
    except Exception as exc:  # noqa: BLE001 - one diagnostic, not the segment
        logger.warning("%s failed (%s); continuing without it", what, exc)
        return default, str(exc)


def collect_segment_diagnostics(
    raw_view,
    qc_rec,
    *,
    source="preprocessed",
    duration_s,
    num_chunks,
    seed,
    mad_threshold,
    dead_noise_ratio,
    artifacts_duration_s,
    window_s,
    trace_channels,
    raster_max_channels,
    psd_channel_ids=None,
    meta=None,
):
    """Everything a segment's diagnostic figures need, computed once.

    Returns the keyword payload :func:`..cache.write_cache` takes --
    ``{"probe", "metrics", "traces", "events", "spectra", "meta"}`` -- so a
    capsule's whole obligation is::

        write_cache(out_dir, **collect_segment_diagnostics(raw, qc, ...))

    `raw_view` is the signed, electrode-keyed, UNFILTERED recording; `qc_rec` is
    the float32-early preprocessed chain (see the module docstring -- they are
    not interchangeable). `psd_channel_ids` is the pool the spectra are taken
    over, usually the electrodes every segment of the well shares, so the panels
    are comparable across segments; None falls back to the representative
    channels.

    Every individual diagnostic is tolerated failing. What comes back always has
    the geometry and whatever else succeeded, and `metrics["errors"]` names what
    did not, so the cache records the gap rather than hiding it.
    """
    from .cache import CachedProbe

    errors = {}
    metrics = {}

    # ---------------------------------------------------------------- QC
    # Three bounded passes over the SAME windows (same seed -> same placement),
    # so the numbers describe one slice of the segment rather than three.
    noise, err = _tolerant("noise", lambda: mad_noise(
        qc_rec, duration_s=duration_s, num_chunks=num_chunks, seed=seed,
    ))
    if err:
        errors["noise"] = err
    metrics["noise"] = noise

    activity = None
    if noise is not None:
        # Handing the noise dict back in fixes the threshold across windows, so
        # the rates reported here are the rates implied by the noise reported
        # here -- the two halves of the report cannot drift apart.
        activity, err = _tolerant("activity", lambda: activity_rate(
            qc_rec, threshold_sd=mad_threshold, noise=noise,
            duration_s=duration_s, num_chunks=num_chunks, seed=seed,
        ))
        if err:
            errors["activity"] = err
    metrics["activity"] = activity

    # The most fragile of the three: the only one that is not ours, and its
    # non-"mad" methods assume a depth-ordered linear probe that a planar MEA is
    # not. dead_well_flags accepts None and falls back to the other two rules.
    bad, err = _tolerant("bad-channel detection", lambda: detect_bad_channels(
        qc_rec, method="mad", duration_s=duration_s, num_chunks=num_chunks,
        seed=seed, std_mad_threshold=float(mad_threshold),
    ))
    metrics["bad_channels"] = bad
    metrics["bad_channels_error"] = err

    flags = flagged = None
    if noise is not None:
        flags, err = _tolerant("dead-well verdict", lambda: dead_well_flags(
            noise=noise, activity=activity, bad_channels=bad,
            dead_noise_ratio=dead_noise_ratio, noisy_noise_ratio=mad_threshold,
        ))
        if err:
            errors["flags"] = err
        flagged, err = _tolerant("channel flags", lambda: flag_channels(
            noise, bad_channels=bad, dead_noise_ratio=dead_noise_ratio,
            noisy_noise_ratio=mad_threshold,
        ))
        if err:
            errors["flagged"] = err
    metrics["flags"] = flags
    metrics["flagged"] = flagged

    # ------------------------------------------------- clipping + artifacts
    # Clipping on the RAW integer view: a filter smears a flat rail into a
    # curve, and on a float dtype the rail bounds do not exist at all.
    clipping, err = _tolerant("clipping census", lambda: clipping_census(
        raw_view, duration_s=duration_s, num_chunks=num_chunks,
    ))
    if err:
        errors["clipping"] = err
    metrics["clipping"] = clipping

    # Artifacts on the chain the noise was measured on: the per-channel
    # thresholds derive from it, so scanning a raw view and reporting against a
    # filtered one would compare different noise floors.
    #
    # A non-positive budget means the caller asked for this scan to be skipped.
    # It records as None with a reason rather than as a failure, because "not
    # asked for" and "could not be computed" are different things to a reader.
    artifacts = None
    if artifacts_duration_s > 0:
        artifacts, err = _tolerant("artifact census", lambda: artifact_census(
            qc_rec, duration_s=artifacts_duration_s,
        ))
        if err:
            errors["artifacts"] = err
    else:
        metrics["artifacts_skipped"] = "not requested (artifacts_duration_s <= 0)"
    metrics["artifacts"] = artifacts

    # ----------------------------------------------- representative channels
    def _representatives():
        return select_representative_channels(
            qc_rec, n_channels=trace_channels, seed=int(seed),
            start_time_s=0.0, duration_s=window_s, return_in_uV=True,
        )

    rep, err = _tolerant("representative-channel selection", _representatives)
    if rep is None:
        # A poorer figure beats no figure, and the fallback is recorded so a
        # reader knows the channels were not chosen by activity.
        rep = list(qc_rec.get_channel_ids())[:trace_channels]
        errors["representative_channels"] = err
    # NOT stringified: these ids go straight back to a recording, and an id
    # rewritten as "5" is not the id 5 as far as SpikeInterface is concerned.
    # The cache round-trips them through JSON, which keeps an int an int.
    metrics["representative_channels"] = list(rep)

    # ------------------------------------------------------------- traces
    traces = {}
    for name, recording in (("preprocessed", qc_rec), ("raw", raw_view)):
        block, err = _tolerant(f"{name} trace window", lambda r=recording: _window_block(
            r, rep, window_s,
        ))
        if block is not None:
            traces[name] = block
        else:
            errors[f"traces_{name}"] = err

    # ------------------------------------------------------------- events
    # The expensive half of the raster, and it is run ONCE here for both the
    # file-time figure and its real-elapsed twin -- which today re-detects the
    # identical events. The channel selection is the emitter's own, so the tool
    # can draw with channel_ids=None and land on exactly this list.
    events = {}
    if raster_max_channels > 0:
        raster_channels = _select_channels(qc_rec, None, raster_max_channels)
        detected, err = _tolerant("threshold crossings", lambda: detect_threshold_crossings(
            qc_rec, channel_ids=raster_channels, threshold_factor=mad_threshold,
            start_time_s=0.0, duration_s=window_s,
        ))
        if detected is not None:
            times_s, labels = detected
            events["raster"] = {"times_s": times_s, "labels": labels}
            metrics["raster"] = {
                "channel_ids": list(raster_channels),
                "threshold_factor": float(mad_threshold),
                "window_s": float(window_s),
                "n_events": int(np.asarray(times_s).size),
            }
        else:
            errors["events"] = err
    else:
        metrics["raster_skipped"] = "not requested (raster_max_channels <= 0)"

    # ------------------------------------------------------------ spectra
    psd_pool = list(psd_channel_ids) if psd_channel_ids else list(rep)
    spectra = {}
    for name, recording in (("raw", raw_view), ("preprocessed", qc_rec)):
        if name == "preprocessed" and source != "preprocessed":
            # No descriptor: there is no preprocessed panel to compare against,
            # and the filename says so rather than the figure implying a pair.
            continue
        block, err = _tolerant(f"{name} spectra", lambda r=recording: welch_spectra(
            r, psd_pool, start_time_s=0.0, duration_s=window_s,
        ))
        if block is not None:
            spectra[name] = block
        else:
            errors[f"spectra_{name}"] = err
    metrics["psd_channels"] = list(psd_pool)

    metrics["errors"] = errors

    payload_meta = {
        "source": source,
        # R10, recorded so nobody diffs the QC numbers against the compute
        # chain and calls the sub-LSB disagreement a bug.
        "qc_chain": "float32-early (R10: filters run in float; compute chain unchanged)",
        "sampling": {
            "duration_s": float(duration_s),
            "num_chunks": int(num_chunks),
            "seed": int(seed),
        },
        "mad_threshold": float(mad_threshold),
        "dead_noise_ratio": dead_noise_ratio,
        "artifacts_duration_s": float(artifacts_duration_s),
        "window_s": float(window_s),
        "trace_channels": int(trace_channels),
        "raster_max_channels": int(raster_max_channels),
        # Both counts, because a figure name carries them separately and the
        # renderer has no recording left to ask.
        "n_channels": int(qc_rec.get_num_channels()),
        "n_layout_channels": int(raw_view.get_num_channels()),
        **(meta or {}),
    }

    return {
        "probe": CachedProbe.from_recording(qc_rec),
        "metrics": metrics,
        "traces": traces,
        "events": events,
        "spectra": spectra,
        "meta": payload_meta,
    }
