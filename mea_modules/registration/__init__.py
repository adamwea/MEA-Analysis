"""Putting a backbone sort back onto the segments it came from.

Public API::

    from mea_modules.registration import (
        segment_bounds_from_manifest,
        register_sorting_to_segment,
        registration_summary,
    )

Concatenation traded the array for a timeline: the sort ran on the electrodes
routed in every segment (353 of 13,384 on the reference scan) because only those
sit on one continuous recording. This package does the inverse — re-indexes the
sorted spike times onto each segment's own clock so each segment's full
electrode set can be read against them.

Time only. Unit identity comes from the backbone sort and is never revisited
here; :mod:`mea_modules.concatenation.union` handles the channel side.
"""

from .coverage import (
    coverage_counts,
    rescale_union_templates,
    routing_table,
    selected_spikes_per_segment,
)
from .segments import (
    register_sorting_to_segment,
    registration_summary,
    segment_bounds_from_manifest,
    split_spike_train,
)

__all__ = [
    "segment_bounds_from_manifest",
    "register_sorting_to_segment",
    "registration_summary",
    "split_spike_train",
    # correcting templates taken straight off a zero-padded union recording,
    # which is the cheap alternative to one analyzer per segment
    "routing_table",
    "selected_spikes_per_segment",
    "coverage_counts",
    "rescale_union_templates",
]
