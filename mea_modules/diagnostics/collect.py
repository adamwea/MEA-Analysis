"""Compute one segment's diagnostics once, where the recording is already open.

This is the producer half of the compute-in-capsule / plot-from-cache split.
The capsule that opens a segment calls :func:`collect_segment_diagnostics` while
it has the recording in hand, and hands the result straight to
:func:`mea_modules.diagnostics.cache.write_cache`. The plot suite then reads that
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

**The QC chain is read once.** A lazy chain re-filters on every read, so six
diagnostics used to pay for six filter passes (126 s of a 127 s detection was
the chain). `signal_buffer` holds the filtered segment for the duration of this
call -- in memory, in a scratch file, or not at all -- and every diagnostic
reads that one copy. Which of the three is the caller's decision: it depends
on how many siblings the process runs beside, which only the caller knows.

**One detection, two consumers.** The activity rate and the raster are both
threshold-crossing events, and they come from ONE call to the same detector
(:func:`mea_modules.quality.detection.detect_events`) over the whole segment:
the rate divides its counts by the segment's duration, the raster keeps the
events inside its window. They can no longer disagree about what an event is.
"""

import logging
import time
from dataclasses import dataclass

import numpy as np

from ..quality import (
    dead_well_flags,
    detect_bad_channels,
    mad_noise,
    rms_noise,
    rms_over_mad,
)
from ..quality.detection import (
    DEFAULT_EXCLUDE_SWEEP_MS,
    detect_events,
    detector_description,
    event_rates,
    resampled_for_detection,
)
from .artifacts import artifact_census
from .buffer import buffered_signal
from .channel_flags import flag_channels
from .clipping import clipping_census
from .selection import _select_channels
from .welch import welch_spectra
from .selection import _frames_to_seconds, _has_time_vector, select_representative_channels

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
        "time_gaps": "native",
    }


def _resolve_channel_pool(recording, wanted):
    """`wanted` expressed in the recording's OWN channel ids, or None.

    A shared-electrode list read out of JSON arrives as strings; a recording
    keyed on electrode ids answers to ints. Handing the strings straight to a
    reader raises "IDs ['1847'] are not ids of the extractor" and the whole
    spectra pass is lost -- which is exactly what happened the first time this
    ran on real data.

    Matching on the string form and returning the recording's own objects is
    what makes the pool survive the round trip in either direction. Ids the
    recording does not have are dropped rather than raising: a shared set is
    computed across the well, and a segment that did not route one of them is
    the normal case, not an error.
    """
    if not wanted:
        return None
    own = list(recording.get_channel_ids())
    by_text = {str(cid): cid for cid in own}
    resolved = [by_text[str(cid)] for cid in wanted if str(cid) in by_text]
    if not resolved:
        logger.warning(
            "none of the %d requested channels are in this recording's %d; "
            "falling back", len(list(wanted)), len(own),
        )
        return None
    missing = len(list(wanted)) - len(resolved)
    if missing:
        logger.info("%d of the requested channels are not routed here", missing)
    return resolved


@dataclass(frozen=True)
class DiagnosticSpec:
    """One discrete diagnostic a capsule can be asked to compute.

    Declared HERE, beside the code that computes it, so a consumer offering one
    switch per diagnostic can generate the switches from this registry instead
    of restating the list -- and a diagnostic then cannot exist in one place and
    be missing from the other.

    `summary` is written to serve as that switch's one-line description. What a
    diagnostic COSTS is deliberately not recorded here: expense is a fact about
    running it many times in a pipeline, and a consumer running it once would be
    misled by a warning written for a thousand repetitions. A consumer that runs
    it many times states the cost in its own configuration.
    """

    name: str
    summary: str
    default: bool = True
    requires: tuple = ()


# The whole set, in the order they run. `requires` is a real dependency, not a
# preference: activity and the raster threshold against the noise estimate, so
# with noise off there is nothing for them to threshold against.
SEGMENT_DIAGNOSTICS = (
    DiagnosticSpec("noise", "per-electrode MAD noise over the sampled windows"),
    DiagnosticSpec(
        "rms", "per-electrode RMS over the same windows, and its ratio to the MAD",
    ),
    DiagnosticSpec(
        "activity", "event rate per electrode over the whole segment",
        requires=("noise",),
    ),
    DiagnosticSpec("bad_channels", "SpikeInterface's own bad-channel verdict"),
    DiagnosticSpec(
        "flags", "dead-well verdict and per-channel flags from the above",
        requires=("noise",),
    ),
    DiagnosticSpec("clipping", "census of samples sitting on the converter rails"),
    DiagnosticSpec("artifacts", "census of array-wide excursions"),
    DiagnosticSpec("traces", "the raw and preprocessed trace windows the figures draw"),
    DiagnosticSpec(
        "raster", "the detected events the raster and its real-elapsed twin draw",
        requires=("noise",),
    ),
    DiagnosticSpec("spectra", "Welch power spectra, raw against preprocessed"),
)

SEGMENT_DIAGNOSTIC_NAMES = tuple(spec.name for spec in SEGMENT_DIAGNOSTICS)


# Which diagnostics read the FILTERED signal -- the one `buffered_signal` would
# materialise. Declared beside the registry rather than on DiagnosticSpec: a
# spec says what a diagnostic IS, never what it costs here. `clipping` reads the
# raw view, and `flags` only re-reads values the others already produced.
FILTERED_READERS = frozenset({
    "noise", "rms", "activity", "bad_channels", "artifacts", "traces", "raster", "spectra",
})
RAW_ONLY = frozenset({"clipping", "flags"})


def reads_filtered_signal(enabled=None, *, artifacts_duration_s=None):
    """True when a diagnostic that will actually RUN reads the filtered signal.

    A diagnostic that is skipped reads nothing, so the runner's dependency gate
    is mirrored here: `raster` asked for without `noise` never runs. A zero
    `artifacts_duration_s` switches its census off the same way. The raster's
    own budget cannot decide this one: it requires `noise`, which reads the
    filtered signal itself.
    """
    wanted = set(_wanted(enabled))
    specs = {spec.name: spec for spec in SEGMENT_DIAGNOSTICS}
    while True:
        runnable = {name for name in wanted
                    if all(dep in wanted for dep in specs[name].requires)}
        if runnable == wanted:
            break
        wanted = runnable
    if artifacts_duration_s is not None and float(artifacts_duration_s) <= 0:
        wanted.discard("artifacts")
    return bool(wanted & FILTERED_READERS)


def _wanted(enabled, registry=SEGMENT_DIAGNOSTICS):
    """The set of diagnostics to compute; `None` means every default.

    A name not in the registry is a typo, and a typo that silently switches
    nothing off is worse than a crash -- the run looks like it honoured a
    setting it never saw.
    """
    names = tuple(spec.name for spec in registry)
    if enabled is None:
        return {spec.name for spec in registry if spec.default}
    if isinstance(enabled, dict):
        chosen = {name for name, on in enabled.items() if on}
        named = set(enabled)
    else:
        chosen = set(enabled)
        named = chosen
    unknown = named - set(names)
    if unknown:
        raise ValueError(
            f"unknown diagnostics {sorted(unknown)}; this capsule computes {list(names)}"
        )
    return chosen


def _tolerant(what, fn, default=None):
    """Run one diagnostic; a failure costs that diagnostic, never the entity.

    Returns ``(value, error_string_or_None, timing)``, where `timing` is
    ``{"seconds", "started", "ended"}`` -- the duration from a monotonic clock,
    the two instants as epoch seconds, so a profile can lay every diagnostic on
    one timeline and see what ran when, not only how long each took.

    The timing rides along here because this is the one place every discrete
    computation passes through. A failed diagnostic is timed too: how long
    something took to not work is part of what a profile is for, and a missing
    entry would silently read as free.
    """
    started_at = time.time()
    started = time.perf_counter()
    try:
        value, error = fn(), None
    except Exception as exc:  # noqa: BLE001 - one diagnostic, not the entity
        logger.warning("%s failed (%s); continuing without it", what, exc)
        value, error = default, str(exc)
    timing = {
        "seconds": round(time.perf_counter() - started, 3),
        "started": round(started_at, 3),
        "ended": round(time.time(), 3),
    }
    return value, error, timing


class _Runner:
    """The registry's bookkeeping, shared by every collector.

    Three facts are kept apart for every diagnostic, because a reader downstream
    has to be able to tell them apart: not asked for (with the reason,
    including a dependency that was the one switched off), asked for and
    failed, and ran and took N seconds.
    """

    def __init__(self, registry, wanted):
        self.specs = {spec.name: spec for spec in registry}
        self.wanted = wanted
        self.errors = {}
        self.timings = {}
        self.skipped = {}
        self.produced = {}

    def skip(self, name, reason):
        self.skipped[name] = reason

    def record(self, key, value, error, timing):
        self.timings[key] = timing
        if error:
            self.errors[key] = error
        return value

    def run(self, name, what, fn, default=None, key=None):
        """Compute one diagnostic if it was asked for and its inputs exist.

        `key` separates the TIMING key from the gate when one setting covers
        two computations -- `flags` produces both the well verdict and the
        per-channel flags, and one setting switching both is the honest shape,
        but a profile that reported only the second would hide the first.
        """
        spec = self.specs[name]
        key = key or name
        if name not in self.wanted:
            self.skipped[name] = "not requested"
            return default
        missing = [
            dep for dep in spec.requires
            if dep in self.skipped or self.produced.get(dep) is None
        ]
        if missing:
            self.skipped[name] = f"requires {', '.join(missing)}"
            return default
        value, err, timing = _tolerant(what, fn, default=default)
        self.record(key, value, err, timing)
        if value is not None:
            self.produced[name] = value
        return value

    def close(self, metrics):
        metrics["errors"] = self.errors
        metrics["timings"] = self.timings
        metrics["skipped"] = self.skipped
        return metrics


def _raster_events(qc, noise, events_all, *, raster_channels, window_s,
                   detect_threshold, exclude_sweep_ms, downsample_hz, duration_s,
                   num_chunks, seed):
    """The raster's events: the shared detection's, inside the window.

    At the native rate the raster is a VIEW of the one detection the activity
    rate also counts -- the events on the raster's channels inside its window.
    With `downsample_hz` the raster detects on an anti-aliased decimation
    instead, with the noise re-estimated on that band: the anti-alias filter
    removes real signal power, so a threshold calibrated on the full band would
    be applied to a band the detector never sees.
    """
    fs = float(qc.get_sampling_frequency())
    window_end = min(int(qc.get_num_samples()), int(round(float(window_s) * fs)))
    if downsample_hz:
        decimated, factor, effective_hz = resampled_for_detection(qc, downsample_hz)
        band_noise = mad_noise(
            decimated, duration_s=duration_s, num_chunks=num_chunks, seed=seed,
        )
        detected = detect_events(
            decimated, band_noise, detect_threshold=detect_threshold,
            exclude_sweep_ms=exclude_sweep_ms, channel_ids=raster_channels,
            end_frame=window_end // factor,
        )
        frames = detected["frames"] * factor
        labels = detected["labels"]
        noise_source = "re-estimated on the decimated band"
    else:
        if events_all is None:
            raise RuntimeError("the shared detection did not run, so there are no events to draw")
        factor, effective_hz = 1, fs
        from ..quality.detection import channel_labels

        keep = (events_all["frames"] < window_end) & np.isin(
            events_all["labels"], channel_labels(raster_channels)
        )
        frames = events_all["frames"][keep]
        labels = events_all["labels"][keep]
        noise_source = "the segment's MAD noise"
    times_s = _frames_to_seconds(qc, frames, fs, has_times=_has_time_vector(qc))
    summary = {
        "channel_ids": list(raster_channels),
        "threshold_factor": float(detect_threshold),
        "exclude_sweep_ms": float(exclude_sweep_ms),
        "window_s": float(window_s),
        "n_events": int(frames.size),
        # The rate DETECTED AT, not the rate asked for.
        "detection_hz": float(effective_hz),
        "decimation_factor": int(factor),
        "noise_source": noise_source,
    }
    return {"times_s": np.asarray(times_s, dtype=float), "labels": labels}, summary


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
    raster_downsample_hz=None,
    exclude_sweep_ms=DEFAULT_EXCLUDE_SWEEP_MS,
    signal_buffer="lazy",
    scratch_dir=None,
    time_gaps=None,
    enabled=None,
    meta=None,
):
    """Everything a segment's diagnostic figures need, computed once.

    Returns the keyword payload :func:`..cache.write_cache` takes --
    ``{"probe", "metrics", "traces", "events", "spectra", "time_gaps", "meta"}``
    -- so a capsule's whole obligation is::

        write_cache(out_dir, **collect_segment_diagnostics(raw, qc, ...))

    `raw_view` is the signed, electrode-keyed, UNFILTERED recording; `qc_rec` is
    the float32-early preprocessed chain (see the module docstring -- they are
    not interchangeable). `psd_channel_ids` is the pool the spectra are taken
    over, usually the electrodes every segment of the well shares, so the panels
    are comparable across segments; None falls back to the representative
    channels. `time_gaps` is the segment's frame-counter breaks, read off the
    source by the caller and cached so a redraw never needs the file.

    `mad_threshold` is the detection threshold in MAD-sigma, and the noisy-
    channel and bad-channel ratio; `exclude_sweep_ms` is the detector's
    isolation window. `signal_buffer` is ``"memory"``, ``"disk"`` (under
    `scratch_dir`) or ``"lazy"`` -- the RESOLVED mode; see
    :mod:`.buffer`.

    `enabled` selects which of :data:`SEGMENT_DIAGNOSTICS` to compute -- a set
    of names, or a name->bool mapping, or None for every default. A diagnostic
    that was not asked for is recorded as skipped WITH ITS REASON.

    Every individual diagnostic is tolerated failing. What comes back always has
    the geometry and whatever else succeeded, `metrics["errors"]` names what did
    not, and `metrics["timings"]` says what each one cost and when it ran --
    including the buffer read, the one-off price every diagnostic after it no
    longer pays.
    """
    from .cache import CachedProbe

    runner = _Runner(SEGMENT_DIAGNOSTICS, _wanted(enabled))
    wanted = runner.wanted
    metrics = {}
    traces = {}
    events = {}
    spectra = {}

    # Nothing enabled reads the filtered signal: materialising the segment
    # would be paid for and never read. What was ASKED for is kept, in the cache
    # meta and in `requested`, so a reader can tell this from a lazy request.
    effective_buffer = signal_buffer
    if signal_buffer != "lazy" and not reads_filtered_signal(
        enabled, artifacts_duration_s=artifacts_duration_s,
    ):
        logger.info("no enabled diagnostic reads the filtered signal; not buffering it "
                    "(%s was asked for)", signal_buffer)
        effective_buffer = "lazy"

    with buffered_signal(qc_rec, effective_buffer, scratch_dir=scratch_dir) as (qc, buffer_info):
        metrics["buffer"] = buffer_info if effective_buffer == signal_buffer else dict(
            buffer_info, requested=signal_buffer,
            skipped="no enabled diagnostic reads the filtered signal")
        if effective_buffer != "lazy":
            runner.timings["buffer"] = {
                "seconds": buffer_info["seconds"],
                "started": buffer_info.get("started"),
                "ended": buffer_info.get("ended"),
            }

        # ------------------------------------------------------------ QC
        # Bounded passes over the SAME windows (same seed -> same placement),
        # so the numbers describe one slice of the segment rather than several.
        noise = runner.run("noise", "noise", lambda: mad_noise(
            qc, duration_s=duration_s, num_chunks=num_chunks, seed=seed,
        ))
        metrics["noise"] = noise

        def _rms():
            result = rms_noise(qc, duration_s=duration_s, num_chunks=num_chunks, seed=seed)
            # The ratio only exists beside a MAD; without one the RMS stands
            # alone and says why it has no ratio.
            if noise is not None:
                # The ratio's own unit is "ratio"; merged whole it would
                # relabel the RMS it sits beside.
                ratio = rms_over_mad(result, noise)
                result.update({k: v for k, v in ratio.items() if k not in ("unit", "channel_ids")})
            else:
                result["rms_over_mad"] = None
                result["rms_over_mad_skipped"] = "requires noise"
            return result

        metrics["rms"] = runner.run("rms", "RMS", _rms)

        # ONE detection over the whole segment, serving the activity rate and,
        # at the native rate, the raster. Timed as its own entry: it is the
        # expensive step, and splitting it between its two consumers would
        # make neither number true.
        events_all = None
        needs_detection = noise is not None and (
            "activity" in wanted
            or ("raster" in wanted and raster_max_channels > 0 and not raster_downsample_hz)
        )
        if needs_detection:
            events_all = runner.record("detection", *_tolerant(
                "peak detection",
                lambda: detect_events(
                    qc, noise, detect_threshold=mad_threshold,
                    exclude_sweep_ms=exclude_sweep_ms,
                ),
            ))

        def _activity():
            if events_all is None:
                raise RuntimeError(runner.errors.get("detection") or "detection did not run")
            return event_rates(events_all)

        activity = runner.run("activity", "activity", _activity)
        metrics["activity"] = activity

        # The most fragile of the set: the only one that is not ours, and its
        # non-"mad" methods assume a depth-ordered linear probe that a planar
        # MEA is not. dead_well_flags accepts None and falls back.
        bad = runner.run("bad_channels", "bad-channel detection", lambda: detect_bad_channels(
            qc, method="mad", duration_s=duration_s, num_chunks=num_chunks,
            seed=seed, std_mad_threshold=float(mad_threshold),
        ))
        metrics["bad_channels"] = bad
        metrics["bad_channels_error"] = runner.errors.get("bad_channels")

        metrics["flags"] = runner.run("flags", "dead-well verdict", lambda: dead_well_flags(
            noise=noise, activity=activity, bad_channels=bad,
            dead_noise_ratio=dead_noise_ratio, noisy_noise_ratio=mad_threshold,
        ), key="flags")
        metrics["flagged"] = runner.run("flags", "channel flags", lambda: flag_channels(
            noise, bad_channels=bad, dead_noise_ratio=dead_noise_ratio,
            noisy_noise_ratio=mad_threshold,
        ), key="flagged")

        # --------------------------------------------- clipping + artifacts
        # Clipping on the RAW integer view: a filter smears a flat rail into a
        # curve, and on a float dtype the rail bounds do not exist at all.
        metrics["clipping"] = runner.run("clipping", "clipping census", lambda: clipping_census(
            raw_view, duration_s=duration_s, num_chunks=num_chunks,
        ))

        # Artifacts on the chain the noise was measured on: the per-channel
        # thresholds derive from it, so scanning a raw view and reporting
        # against a filtered one would compare different noise floors.
        if artifacts_duration_s <= 0:
            runner.skip("artifacts", "not requested (artifacts_duration_s <= 0)")
            metrics["artifacts"] = None
        else:
            metrics["artifacts"] = runner.run("artifacts", "artifact census", lambda: artifact_census(
                qc, duration_s=artifacts_duration_s,
            ))

        # ------------------------------------------- representative channels
        # Not a diagnostic in its own right: it is the channel list the trace
        # and spectra figures are drawn on, so it runs whenever either does.
        rep = None
        if {"traces", "spectra"} & wanted:
            rep = runner.record("representative_channels", *_tolerant(
                "representative-channel selection",
                lambda: select_representative_channels(
                    qc, n_channels=trace_channels, seed=int(seed),
                    start_time_s=0.0, duration_s=window_s, return_in_uV=True,
                ),
            ))
        if rep is None:
            # A poorer figure beats no figure, and the fallback is recorded so
            # a reader knows the channels were not chosen by activity.
            rep = list(qc.get_channel_ids())[:trace_channels]
        # NOT stringified: these ids go straight back to a recording, and an id
        # rewritten as "5" is not the id 5 as far as SpikeInterface is concerned.
        metrics["representative_channels"] = list(rep)

        # --------------------------------------------------------- traces
        if "traces" in wanted:
            for name, recording in (("preprocessed", qc), ("raw", raw_view)):
                block = runner.record(f"traces_{name}", *_tolerant(
                    f"{name} trace window",
                    lambda r=recording: _window_block(r, rep, window_s),
                ))
                if block is not None:
                    traces[name] = block
        else:
            runner.skip("traces", "not requested")

        # --------------------------------------------------------- events
        if raster_max_channels <= 0:
            runner.skip("raster", "not requested (raster_max_channels <= 0)")
        else:
            detected = runner.run("raster", "raster events", lambda: _raster_events(
                qc, noise, events_all,
                raster_channels=_select_channels(qc, None, raster_max_channels),
                window_s=window_s, detect_threshold=mad_threshold,
                exclude_sweep_ms=exclude_sweep_ms, downsample_hz=raster_downsample_hz,
                duration_s=duration_s, num_chunks=num_chunks, seed=seed,
            ))
            if detected is not None:
                events["raster"], metrics["raster"] = detected

        # -------------------------------------------------------- spectra
        psd_pool = _resolve_channel_pool(qc, psd_channel_ids) or list(rep)
        if "spectra" in wanted:
            for name, recording in (("raw", raw_view), ("preprocessed", qc)):
                if name == "preprocessed" and source != "preprocessed":
                    # No descriptor: there is no preprocessed panel to compare
                    # against, and the filename says so rather than the figure
                    # implying a pair.
                    continue
                block = runner.record(f"spectra_{name}", *_tolerant(
                    f"{name} spectra",
                    lambda r=recording: welch_spectra(
                        r, psd_pool, start_time_s=0.0, duration_s=window_s,
                    ),
                ))
                if block is not None:
                    spectra[name] = block
        else:
            runner.skip("spectra", "not requested")
        metrics["psd_channels"] = list(psd_pool)

    runner.close(metrics)

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
        "signal_buffer": signal_buffer,
        "detection": detector_description(mad_threshold, "neg", exclude_sweep_ms),
        # What was asked for, so a later reader can tell a cache that is thin by
        # choice from one that is thin because something broke.
        "diagnostics": {name: (name in wanted) for name in SEGMENT_DIAGNOSTIC_NAMES},
        "raster_downsample_hz": (
            None if not raster_downsample_hz else float(raster_downsample_hz)
        ),
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
        "time_gaps": {"native": time_gaps} if time_gaps is not None else {},
        "meta": payload_meta,
    }
