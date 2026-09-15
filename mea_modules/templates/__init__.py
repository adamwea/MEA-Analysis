"""Recombining per-segment templates into one full-array template, per unit.

Public API::

    from mea_modules.templates import merge_segment_templates

Capsule `register_segment` fans the backbone sort out onto N per-segment
`SortingAnalyzer`s, one per AxonTracking configuration, each on that segment's
own FULL native electrode set (hundreds to low-thousands of channels) with no
extensions computed. This package is the other half of array recovery: it
opens those analyzers ONE AT A TIME, computes each one's `templates` extension
on demand, and folds every unit's per-segment channel template into a running
weighted-average accumulator — so the union across segments recovers the
spatial coverage the 353-electrode backbone sort could never see on its own.
(`mea_modules.registration` is the time-domain half of the same recovery —
re-indexing spikes onto each segment's own clock; this package is the
channel-domain half, done after that.)

Streaming by construction: :func:`merge_segment_templates` never holds more
than one segment's analyzer in memory at once, which is what makes a real
array (low thousands of channels x hundreds of units x tens of segments)
affordable, where materializing every analyzer at once was not — see
`merge.py`'s module docstring for the failure this replaces.

The pure per-channel merge arithmetic — :func:`new_unit_accumulator`,
:func:`accumulate_channel_contributions`, :func:`finalize_unit_accumulator` —
is exposed separately because it has no SpikeInterface dependency and is
unit-tested on its own, independent of the streaming orchestrator around it.
"""

from .load import (
    FS_MANIFEST_KEY,
    LOCATIONS_FILENAME,
    MANIFEST_FILENAME,
    TEMPLATES_FILENAME,
    UNIT_IDS_FILENAME,
    WEIGHT_FILENAME,
    discover_unit_ids,
    load_unit_inputs,
    load_well_inputs,
)
from .merge import (
    AVERAGING_METHODS,
    DEFAULT_AVERAGING_METHOD,
    DEFAULT_MAX_SPIKES_PER_UNIT,
    DEFAULT_MS_AFTER,
    DEFAULT_MS_BEFORE,
    DEFAULT_SEED,
    WEIGHTING_MODES,
    accumulate_channel_contributions,
    finalize_unit_accumulator,
    merge_segment_templates,
    new_unit_accumulator,
)

__all__ = [
    "merge_segment_templates",
    "new_unit_accumulator",
    "accumulate_channel_contributions",
    "finalize_unit_accumulator",
    "AVERAGING_METHODS",
    "DEFAULT_AVERAGING_METHOD",
    "WEIGHTING_MODES",
    "DEFAULT_MS_BEFORE",
    "DEFAULT_MS_AFTER",
    "DEFAULT_MAX_SPIKES_PER_UNIT",
    "DEFAULT_SEED",
    "discover_unit_ids",
    "load_well_inputs",
    "load_unit_inputs",
    "TEMPLATES_FILENAME",
    "WEIGHT_FILENAME",
    "LOCATIONS_FILENAME",
    "UNIT_IDS_FILENAME",
    "MANIFEST_FILENAME",
    "FS_MANIFEST_KEY",
]
