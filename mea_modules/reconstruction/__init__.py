"""Axon tracking on a recovered full-array template — the final recovery step.

Public API::

    from mea_modules.reconstruction import (
        default_params,
        track_unit_axon,
        save_reconstruction,
    )

Where this sits in the recovery path: `mea_modules.registration` puts a
segment-bound sort back onto the full electrode set it was sorted from, one
segment at a time; a sibling stage merges those per-segment templates into one
`(n_channels, n_samples)` template per unit covering the whole array. This
package starts from THAT — a merged `(template, locations, fs)` for one unit —
and runs `axon_velocity`'s graph-based tracker on it, then persists what it
finds. Nothing here reads a segment, a sort, or a manifest; the merge stage's
output is this package's entire input contract.
"""

from .axon_velocity_track import (
    GTR_FILENAME,
    SUMMARY_FILENAME,
    default_params,
    save_reconstruction,
    track_unit_axon,
)

__all__ = [
    "default_params",
    "track_unit_axon",
    "save_reconstruction",
    "GTR_FILENAME",
    "SUMMARY_FILENAME",
]
