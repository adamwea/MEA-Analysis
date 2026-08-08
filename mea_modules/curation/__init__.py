"""Curation-stage arithmetic: applying merge maps to dense stitched templates.

Public API::

    from mea_modules.curation import (
        merge_group_dense,
        plan_merge_output,
        dedup_coincident_spikes,
        DEFAULT_CENSOR_SAMPLES,
    )

SLAy (capsule `17 slay_propose_merges`) proposes merge GROUPS and stops --
the SpikeInterface re-implementation has no apply path. SpikeInterface's own
`merge_units` is the wrong applier for this pipeline: it would re-derive
merged templates from the analyzer it is given -- the ~350-channel backbone
recording -- discarding the full-array dense stitch that is the pipeline's
whole point. This package is the correct applier: a closed-form per-channel
coverage-weighted merge over `10 stitch_templates`' outputs, exact under
`spike_count` averaging (see :func:`merge_group_dense` for the theorem), plus
the coincident-spike dedup SLAy's old version applied and the new one
dropped.

Pure by construction: nothing here reads a file, opens a SortingAnalyzer, or
imports SpikeInterface. Capsule `18 dense_merge_apply` owns the disk/stream
orchestration (capsules wire modules to disk; the science lives here).
"""

from .dense_merge import (
    DEFAULT_CENSOR_SAMPLES,
    dedup_coincident_spikes,
    merge_group_dense,
    plan_merge_output,
)

__all__ = [
    "DEFAULT_CENSOR_SAMPLES",
    "dedup_coincident_spikes",
    "merge_group_dense",
    "plan_merge_output",
]
