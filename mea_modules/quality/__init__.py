"""Quality control metrics for MEA recordings.

Public API::

    from mea_modules.quality import mad_noise, rms_noise, detect_events, event_rates
    from mea_modules.quality import detect_bad_channels, dead_well_flags

`mad_noise`, `rms_noise` and `detect_bad_channels` measure one recording (a
well/segment) over a bounded sample of traces; `detect_events` is the one
threshold-crossing detector (SpikeInterface's, called one way) and
`event_rates` reduces its events to a rate per channel; `dead_well_flags`
reduces all of it to one verdict per well.
"""

from .robust import MAD_TO_SIGMA, mad_sigma
from .metrics import (
    DEFAULT_DURATION_S,
    DEFAULT_HIGHPASS_HZ,
    DEFAULT_NUM_CHUNKS,
    DEFAULT_SEED,
    dead_well_flags,
    detect_bad_channels,
    mad_noise,
    rms_noise,
    rms_over_mad,
)
from .detection import (
    DEFAULT_DETECT_THRESHOLD,
    DEFAULT_EXCLUDE_SWEEP_MS,
    DEFAULT_PEAK_SIGN,
    detect_events,
    event_rates,
)

__all__ = [
    # metrics
    "mad_noise",
    "mad_sigma",
    "MAD_TO_SIGMA",
    "rms_noise",
    "rms_over_mad",
    "detect_events",
    "event_rates",
    "DEFAULT_DETECT_THRESHOLD",
    "DEFAULT_EXCLUDE_SWEEP_MS",
    "DEFAULT_PEAK_SIGN",
    "detect_bad_channels",
    "dead_well_flags",
    # sampling defaults, so callers can report what they used
    "DEFAULT_DURATION_S",
    "DEFAULT_NUM_CHUNKS",
    "DEFAULT_HIGHPASS_HZ",
    "DEFAULT_SEED",
]
