"""Lazy preprocessing chains for MEA segment recordings.

Public API::

    from mea_modules.preprocessing import preprocess_segment, detect_artifacts

Everything here except :func:`detect_artifacts` and :func:`estimate_noise_levels`
returns a lazy SpikeInterface recording — no traces read, no files written. The
two exceptions read traces because measuring is their entire job.

:func:`preprocess_segment` is the assembled standard recipe (signed cast ->
high-pass -> local median reference -> float32); the individual steps are
exported alongside it so a caller can build a different chain without forking.
"""

from .artifacts import (
    blank_artifacts,
    blank_detected_artifacts,
    detect_artifacts,
    estimate_noise_levels,
)
from .filters import (
    rename_channels_to_electrodes,
    bandpass,
    center,
    common_median_reference,
    ensure_signed,
    highpass,
    preprocess_segment,
    to_float32,
)

__all__ = [
    # assembled chain
    "preprocess_segment",
    "rename_channels_to_electrodes",
    # individual filter steps
    "ensure_signed",
    "highpass",
    "bandpass",
    "common_median_reference",
    "center",
    "to_float32",
    # artifacts
    "detect_artifacts",
    "blank_artifacts",
    "blank_detected_artifacts",
    "estimate_noise_levels",
]
