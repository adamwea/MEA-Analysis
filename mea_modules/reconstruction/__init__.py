"""Axon tracking on a recovered full-array template — the final recovery step.

Public API::

    from mea_modules.reconstruction import (
        default_params,
        track_unit_axon,
        save_reconstruction,
        plot_unit_reconstruction,
    )

Where this sits in the recovery path: `mea_modules.registration` puts a
segment-bound sort back onto the full electrode set it was sorted from, one
segment at a time; a sibling stage merges those per-segment templates into one
`(n_channels, n_samples)` template per unit covering the whole array. This
package starts from THAT — a merged `(template, locations, fs)` for one unit —
and runs `axon_velocity`'s graph-based tracker on it, persists what it finds,
and (via `plots`) renders it as a PNG. Nothing here reads a segment, a sort, or
a manifest; the merge stage's output is this package's entire input contract.
"""

from .axon_velocity_track import (
    GTR_FILENAME,
    SUMMARY_FILENAME,
    default_params,
    save_reconstruction,
    track_unit_axon,
)
from .plots import (
    DEFAULT_DPI,
    DEFAULT_FIGSIZE,
    DEFAULT_FOOTPRINT_DPI,
    DEFAULT_FOOTPRINT_FIGSIZE,
    DEFAULT_MARKER_MAX_DIAMETER_PT,
    DEFAULT_MARKER_MIN_DIAMETER_PT,
    DEFAULT_MAX_RADIUS_PITCH_FRACTION,
    DEFAULT_MIN_RADIUS_MAX_FRACTION,
    FOOTPRINT_RECONSTRUCTION_PLOT_FILENAME,
    RECONSTRUCTION_PLOT_FILENAME,
    plot_unit_footprint_reconstruction,
    plot_unit_reconstruction,
)

__all__ = [
    "default_params",
    "track_unit_axon",
    "save_reconstruction",
    "GTR_FILENAME",
    "SUMMARY_FILENAME",
    "plot_unit_reconstruction",
    "RECONSTRUCTION_PLOT_FILENAME",
    "DEFAULT_FIGSIZE",
    "DEFAULT_DPI",
    "plot_unit_footprint_reconstruction",
    "FOOTPRINT_RECONSTRUCTION_PLOT_FILENAME",
    "DEFAULT_FOOTPRINT_FIGSIZE",
    "DEFAULT_FOOTPRINT_DPI",
    "DEFAULT_MARKER_MIN_DIAMETER_PT",
    "DEFAULT_MARKER_MAX_DIAMETER_PT",
    "DEFAULT_MAX_RADIUS_PITCH_FRACTION",
    "DEFAULT_MIN_RADIUS_MAX_FRACTION",
]
