"""Compute one concatenated well's diagnostics once, where the recording is open.

The well-level counterpart of :func:`.collect.collect_segment_diagnostics`, and
the same split: the capsule that writes the concatenated binary calls
:func:`collect_concat_diagnostics` straight after, hands the result to
:func:`mea_modules.diagnostics.cache.write_cache`, and the plot suite draws from
that cache without opening the recording, the source file or anything else.

It reads the SAVED binary, not the lazy chain that produced it. The binary is
filtered signal on disk, so a whole-timeline read costs seconds; reading the
lazy concatenation instead would re-run every segment's filter chain for each
diagnostic.

What it computes, and the one rule each follows:

* **one detection** over the whole timeline, on every channel, thresholds at
  k x the well's own MAD-sigma. The rasters and the per-segment activity summary
  are both views of it, so they cannot disagree about what an event is.
* **decimated traces** of the representative channels over the whole timeline,
  kept at a display rate (~500 Hz by default). The figure spans ~280 s across a
  few thousand pixels -- about a hundred native samples per pixel -- so the full
  rate buys nothing it can show, and caching it would be the size of the
  recording. The frames, the joins and the gap table are all re-addressed onto
  the decimated numbering, so the cached figure shades and marks exactly where
  the full-rate one did.
* **the gap table of the segments actually joined**, in the order they were
  joined, read off the source file once.
* **the per-segment table** with each segment's own routed electrode count
  beside the shared set it was cut down to.

Nothing is computed across a join. Every per-segment number comes from that
segment's own samples and duration; see
:data:`mea_modules.diagnostics.figure_text.PER_SEGMENT_ONLY`.
"""

import logging

import numpy as np

from ..quality import mad_noise
from ..quality.detection import (
    DEFAULT_EXCLUDE_SWEEP_MS,
    detect_events,
    detector_description,
)
from .collect import DiagnosticSpec, _effective_trace_unit, _Runner, _wanted
from .segment_event_rates import segment_event_rate_summary
from .timebase import rescale_time_gaps
from .traces import select_representative_channels

logger = logging.getLogger(__name__)

# The cached trace series' rate. ~500 Hz matches what the whole-timeline trace
# figure drew at when it read the recording itself (150 k points over ~280 s).
DEFAULT_TRACE_HZ = 500.0

# Frames read per block when building the decimated series: bounded memory on a
# timeline of any length.
_TRACE_BLOCK_FRAMES = 200_000

CONCAT_DIAGNOSTICS = (
    DiagnosticSpec(
        "noise", "per-electrode MAD noise over seeded windows of the concatenated recording",
    ),
    DiagnosticSpec(
        "gaps", "the gap table of the joined segments, for the real-elapsed figures",
    ),
    DiagnosticSpec(
        "traces", "whole-timeline traces of the representative channels, at a display rate",
    ),
    DiagnosticSpec(
        "events", "one peak detection over the whole timeline, for the rasters and the activity summary",
        requires=("noise",),
    ),
    DiagnosticSpec(
        "activity_summary", "per-segment, per-electrode event rates and their stability statistics",
        requires=("events",),
    ),
    DiagnosticSpec(
        "motion", "SpikeInterface drift estimate over the concatenated recording (unvalidated here)",
        default=False,
    ),
)

CONCAT_DIAGNOSTIC_NAMES = tuple(spec.name for spec in CONCAT_DIAGNOSTICS)


def segment_table(segments, gap_summary, fs_hz):
    """The per-segment table: the capsule's own rows joined with the source's.

    `segments` are the concatenation's rows in joined order (``rec``,
    ``n_samples``, ``start_frame``, ``end_frame`` and, when known,
    ``n_electrodes`` -- the segment's own routed count before the shared-set
    cut). `gap_summary` is :func:`mea_modules.io.gaps.concatenated_gaps`'s
    ``summary``, joined by rec name, so a mismatch shows up as missing
    wall-clock fields rather than silently misaligned rows. ``source_n_samples``
    is the source file's own frame count for the segment -- the independent
    number a continuity check compares the binary against.
    """
    by_rec = {
        str(entry.get("rec")): entry
        for entry in ((gap_summary or {}).get("segments") or [])
    }
    rows = []
    for index, seg in enumerate(segments):
        gap_entry = by_rec.get(str(seg.get("rec")), {})
        n_samples = int(seg.get("n_samples", 0))
        rows.append({
            "segment_index": index,
            "rec": seg.get("rec"),
            "n_samples": n_samples,
            "start_frame": int(seg.get("start_frame", 0)),
            "end_frame": int(seg.get("end_frame", 0)),
            "recorded_s": (n_samples / fs_hz) if fs_hz else None,
            "n_electrodes": seg.get("n_electrodes"),
            "start_time": gap_entry.get("start_time"),
            "stop_time": gap_entry.get("stop_time"),
            "gap_before_s": gap_entry.get("gap_before_s"),
            "n_breaks": gap_entry.get("n_breaks"),
            "missing_frames": gap_entry.get("missing_frames"),
            "missing_within_s": gap_entry.get("missing_within_s"),
            "source_n_samples": gap_entry.get("n_samples"),
        })
    return rows


def _decimated_traces(recording, channel_ids, step):
    """The whole timeline of `channel_ids`, every `step`-th frame, in its unit.

    Read in bounded blocks and strided with the global phase kept across block
    boundaries, exactly as the trace figure decimates when it reads a recording
    itself -- so the cached series is the samples that figure would have drawn.
    """
    fs = float(recording.get_sampling_frequency())
    total = int(recording.get_num_samples())
    unit = _effective_trace_unit(recording, True)
    parts = []
    for start in range(0, total, _TRACE_BLOCK_FRAMES):
        end = min(total, start + _TRACE_BLOCK_FRAMES)
        block = recording.get_traces(
            start_frame=start, end_frame=end, channel_ids=list(channel_ids),
            return_in_uV=(unit == "uV"),
        )
        offset = (-start) % step
        parts.append(np.asarray(block[offset::step, :], dtype=np.float32))
    traces = np.concatenate(parts, axis=0) if parts else np.zeros((0, len(channel_ids)), np.float32)
    return {
        "channel_ids": list(channel_ids),
        "traces": traces,
        "frame_offset": 0,
        "sampling_frequency": fs / step,
        "unit": unit,
        "frame_step": int(step),
        "time_gaps": "traces",
    }


def collect_concat_diagnostics(
    recording,
    *,
    segments,
    stitch_frames,
    seed,
    detect_threshold,
    duration_s,
    num_chunks,
    trace_channels,
    trace_hz=DEFAULT_TRACE_HZ,
    exclude_sweep_ms=DEFAULT_EXCLUDE_SWEEP_MS,
    data_h5=None,
    well=None,
    enabled=None,
    meta=None,
):
    """Everything a concatenated well's figures need, computed once.

    Returns the keyword payload :func:`..cache.write_cache` takes. `recording`
    is the SAVED concatenated binary; `segments` its rows in joined order (see
    :func:`segment_table`); `stitch_frames` the joins in native frames. `data_h5`
    and `well` locate the source for the gap table; without them the gap table
    is skipped with that reason and the figures draw on file time only.

    `enabled` selects from :data:`CONCAT_DIAGNOSTICS` exactly as the segment
    collector's does, and every diagnostic is tolerated failing.
    """
    from .cache import CachedProbe

    runner = _Runner(CONCAT_DIAGNOSTICS, _wanted(enabled, CONCAT_DIAGNOSTICS))
    fs = float(recording.get_sampling_frequency())
    stitch_frames = [int(frame) for frame in stitch_frames]
    metrics = {"stitch_frames": stitch_frames}
    traces, events, time_gaps = {}, {}, {}

    noise = runner.run("noise", "noise", lambda: mad_noise(
        recording, duration_s=duration_s, num_chunks=num_chunks, seed=seed,
    ))
    metrics["noise"] = noise

    def _gaps():
        from ..io.gaps import concatenated_gaps

        return concatenated_gaps(
            data_h5, well, fs_hz=fs, recs=[seg.get("rec") for seg in segments],
        )

    if "gaps" in runner.wanted and not (data_h5 and well):
        # Not a failure: nothing was named to read from. The figures draw on
        # file time only, and the record says why.
        runner.skip("gaps", "no source file named, so no gap table can be read")
        gaps = None
    else:
        gaps = runner.run("gaps", "gap table", _gaps)
    if gaps is not None:
        time_gaps["native"] = gaps
    table = segment_table(segments, (gaps or {}).get("summary"), fs)
    metrics["segment_table"] = table

    if "traces" in runner.wanted:
        step = max(1, int(round(fs / float(trace_hz)))) if trace_hz else 1

        def _traces():
            rep = select_representative_channels(
                recording, n_channels=trace_channels, seed=int(seed), return_in_uV=True,
            )
            return rep, _decimated_traces(recording, rep, step)

        picked = runner.run("traces", "decimated traces", _traces)
        if picked is not None:
            rep, block = picked
            traces["preprocessed"] = block
            metrics["representative_channels"] = list(rep)
            metrics["traces"] = {
                "channel_ids": list(rep),
                "frame_step": step,
                "sampling_frequency": fs / step,
                # The joins on the decimated numbering: the first kept sample
                # of each new segment, the same rule the gap table follows.
                "stitch_frames": [int(-(-frame // step)) for frame in stitch_frames],
            }
            if gaps is not None:
                time_gaps["traces"] = rescale_time_gaps(gaps, step)
    else:
        runner.skip("traces", "not requested")

    detected = runner.run("events", "peak detection", lambda: detect_events(
        recording, noise, detect_threshold=detect_threshold, exclude_sweep_ms=exclude_sweep_ms,
    ))
    if detected is not None:
        times_s = detected["frames"] / fs
        events["raster"] = {"times_s": times_s, "labels": detected["labels"]}
        metrics["events"] = {
            "channel_ids": [cid.item() if hasattr(cid, "item") else cid for cid in detected["channel_ids"]],
            "n_events": int(detected["frames"].size),
            "threshold_factor": float(detect_threshold),
            "exclude_sweep_ms": float(exclude_sweep_ms),
            "window": "whole timeline",
        }

    def _activity():
        if "raster" not in events:
            raise RuntimeError(runner.errors.get("events") or "the detection did not run")
        return segment_event_rate_summary(
            table,
            events["raster"]["times_s"],
            fs,
            stitch_frames=stitch_frames,
            event_channel_labels=events["raster"]["labels"],
            channel_ids=detected["channel_ids"],
            threshold_factor=detect_threshold,
        )

    metrics["segment_activity"] = runner.run("activity_summary", "activity summary", _activity)

    def _motion():
        from .motion import estimate_motion_over_recording

        return estimate_motion_over_recording(recording, well=well, seed=int(seed))

    metrics["motion"] = runner.run("motion", "motion estimate", _motion)
    runner.close(metrics)

    payload_meta = {
        "sampling": {
            "duration_s": float(duration_s),
            "num_chunks": int(num_chunks),
            "seed": int(seed),
        },
        "detect_threshold": float(detect_threshold),
        "detection": detector_description(detect_threshold, "neg", exclude_sweep_ms),
        "trace_hz": None if not trace_hz else float(trace_hz),
        "trace_channels": int(trace_channels),
        "diagnostics": {name: (name in runner.wanted) for name in CONCAT_DIAGNOSTIC_NAMES},
        "n_channels": int(recording.get_num_channels()),
        "n_segments": len(segments),
        "fs_hz": fs,
        "well": well,
        **(meta or {}),
    }
    return {
        "probe": CachedProbe.from_recording(recording),
        "metrics": metrics,
        "traces": traces,
        "events": events,
        "spectra": {},
        "time_gaps": time_gaps,
        "meta": payload_meta,
    }


__all__ = [
    "CONCAT_DIAGNOSTICS",
    "CONCAT_DIAGNOSTIC_NAMES",
    "DEFAULT_TRACE_HZ",
    "collect_concat_diagnostics",
    "segment_table",
]
