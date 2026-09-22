"""Readable diagnostics: JSON a person or a review script reads.

The cache beside them (:mod:`.cache`) is what figures are drawn from. These
files re-serve the numbers a reader looks up rather than plots -- the QC report,
the flagged channels, the clipping and artifact censuses -- copied out of the
cache, so the numbers in them and the numbers behind the figures are one set.
A file whose diagnostic was not computed is removed: a refresh that switched a
diagnostic off never leaves the previous numbers behind to be mistaken for new.

What these files cannot say is whether they are current. Nothing is rewritten
when the diagnostics are switched off altogether, or when collecting or writing
them fails: the previous files stay. The capsule that writes them records a
completion marker beside the cache only after a successful write, so a reader
trusts the files (and the cache) while that marker stands, never the files alone.
"""

import json

from .atomic import write_json
from .cache import RECORD_NAME, jsonable, read_cache

SEGMENT_REPORTS = ("qc_report.json", "bad_channels.json", "clipping.json", "artifacts.json")
CONCAT_REPORTS = ("segment_activity.json", "gap_table.json", "motion.json")


def write_segment_reports(capsule_out_dir, *, capsule, well, rec=None, label=None):
    """Write one segment's readable JSON beside its cache.

    `capsule` is the capsule that computed the cache (recorded as
    ``computed_by``, and named if the cache is missing). Returns
    ``{file name: path, or None when it was not written}``.
    """
    cache = read_cache(capsule_out_dir, capsule=capsule)
    common = {"computed_by": capsule, "well": well, "rec": rec,
              "source": cache.meta.get("source")}
    reports = {}
    noise = cache.metric("noise")
    if noise is not None:
        reports["qc_report.json"] = _qc_report(cache, common, label, noise)
        reports["bad_channels.json"] = _bad_channels(cache, common)
    for name in ("clipping", "artifacts"):
        census = cache.metric(name)
        if census is not None:
            reports[f"{name}.json"] = {**common, **census}

    return _write(cache, reports, SEGMENT_REPORTS)


def write_concat_reports(capsule_out_dir, *, capsule, well):
    """Write one concatenated well's readable JSON beside its cache.

    ``segment_activity.json`` (the per-segment table and the activity summary
    with its stability statistics), ``gap_table.json`` (the joined segments'
    wall-clock table, when the gap table was read) and ``motion.json`` (when
    motion was estimated). Returns ``{file name: path or None}``.
    """
    cache = read_cache(capsule_out_dir, capsule=capsule)
    common = {"computed_by": capsule, "well": well}
    table = cache.metric("segment_table") or []
    reports = {}
    activity = cache.metric("segment_activity")
    if activity is not None:
        reports["segment_activity.json"] = {
            **common, "segments": table, "activity": activity,
            "detection": cache.meta.get("detection"),
            "threshold_factor": cache.meta.get("detect_threshold"),
        }
    if "native" in cache.time_gap_names():
        reports["gap_table.json"] = {**common, "segments": table}
    motion = cache.metric("motion")
    if motion is not None:
        reports["motion.json"] = {**common, **motion}
    return _write(cache, reports, CONCAT_REPORTS)


def _write(cache, reports, names):
    """Write `reports` beside the cache; remove any of `names` not among them."""
    written = {}
    for name in names:
        path = cache.path / name
        if name in reports:
            write_json(path, jsonable(reports[name]))
            written[name] = path
        else:
            path.unlink(missing_ok=True)
            written[name] = None
    return written


def _qc_report(cache, common, label, noise):
    meta = cache.meta
    activity = cache.metric("activity")
    flags = cache.metric("flags")
    flagged = cache.metric("flagged")
    n_channels = int(cache.probe.get_num_channels())
    return {
        **common,
        "cache": str(cache.path / RECORD_NAME),
        "label": label,
        "data_h5": meta.get("data_h5"),
        # R10: the QC chain runs the filters in float32; the compute chain keeps
        # the integer dtype. Recorded so the sub-LSB difference is not a surprise.
        "qc_chain": meta.get("qc_chain"),
        "n_channels": n_channels,
        "sampling": meta.get("sampling"),
        "mad_threshold": meta.get("mad_threshold"),
        "noise": noise,
        "activity": activity,
        "bad_channels": cache.metric("bad_channels"),
        "bad_channels_error": cache.metric("bad_channels_error"),
        "flags": flags,
        "incomplete": cache.metric("errors") or {},
        "summary": {
            "source": meta.get("source"),
            "median_noise": noise["median_noise"],
            "noise_unit": noise["unit"],
            # the detector's own number, copied, never recomputed here
            "median_rate_hz": (activity or {}).get("median_rate_hz"),
            "active_fraction": (flags or {}).get("active_fraction"),
            "flagged_fraction": (flagged or {}).get("flagged_fraction"),
            "is_dead": (flags or {}).get("is_dead"),
            "reasons": (flags or {}).get("reasons"),
        },
    }


def _bad_channels(cache, common):
    flagged = cache.metric("flagged") or {}
    return {
        **common,
        "n_channels": int(cache.probe.get_num_channels()),
        **flagged,
        # the flag rules' own ratios, with the detection threshold and the
        # bad-channel detector beside them (one block, so neither hides the other)
        "thresholds": {
            **(flagged.get("thresholds") or {}),
            "mad_threshold": cache.meta.get("mad_threshold"),
            "detector_method": (cache.metric("bad_channels") or {}).get("method"),
        },
    }
