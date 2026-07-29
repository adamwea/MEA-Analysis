"""Quality control metrics for MEA recordings.

Public API::

    from mea_modules.quality import mad_noise, activity_rate, detect_bad_channels, dead_well_flags

The first three measure one recording (a well/segment) and return per-channel
numbers; `dead_well_flags` reduces those numbers to one verdict per well. All
four return JSON-serializable dicts and read only a bounded sample of traces.
"""

from .metrics import (
    DEFAULT_DURATION_S,
    DEFAULT_HIGHPASS_HZ,
    DEFAULT_NUM_CHUNKS,
    DEFAULT_SEED,
    activity_rate,
    dead_well_flags,
    detect_bad_channels,
    mad_noise,
)

__all__ = [
    # metrics
    "mad_noise",
    "activity_rate",
    "detect_bad_channels",
    "dead_well_flags",
    # sampling defaults, so callers can report what they used
    "DEFAULT_DURATION_S",
    "DEFAULT_NUM_CHUNKS",
    "DEFAULT_HIGHPASS_HZ",
    "DEFAULT_SEED",
]
