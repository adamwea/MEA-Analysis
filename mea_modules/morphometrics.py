"""Canonical per-unit axon morphometrics over the reconstruction-geometry schema.

Single source of truth (one canonical definition per mechanic, computed ONCE and
EXTRACTED, never re-derived downstream). Analysis repos and figures call these;
they must not reimplement branch length locally -- that is the drift this module
prevents.

Input is the portable ``reconstruction_geometry`` artifact
(``mea_recon.reconstruction_geometry`` v1, see
``mea_modules.reconstruction.geometry``): each reconstructed unit's branches are
independent ordered polylines, concatenated into ``branch_points_xy`` (micrometre
(x, y) vertices) and split by ``branch_node_counts``. Positions are micrometres,
so every length returned here is in micrometres -- no unit conversion happens and
none should be introduced (guards the um-vs-mm drift class).

Metrics (per unit):
  * ``per_branch_length_um`` -- Euclidean length of each branch polyline (sum of
    consecutive-vertex distances). A branch with fewer than two vertices is 0.0.
  * ``mean_branch_length_um`` -- mean of the per-branch lengths (NaN if the unit
    has no branches).
  * ``longest_branch_um`` -- the maximum per-branch length (NaN if none).
  * ``total_axon_length_um`` -- the SUM of the per-branch lengths. **PROVISIONAL
    DEFINITION, pending Roy/Adam.** Branches are independent polylines that share
    an initiation site and OVERLAP over their proximal segments (axon_velocity
    exposes no parent/child tree, so shared segments are not deduplicated). This
    sum therefore DOUBLE-COUNTS shared proximal path. Whether "total axon length"
    should dedupe shared segments is a biological definitional choice; treat this
    field as provisional until a domain expert confirms the intended semantics.
    The returned dict carries ``total_axon_length_um_provisional: True`` as a
    machine-readable marker, so a consumer that reads the value but not this
    docstring still cannot mistake it for a validated metric.

``mean_branch_length_um`` and ``longest_branch_um`` are the uncontroversial
Euclidean-polyline metrics; only ``total_axon_length_um`` carries the caveat above.

Pure numpy + stdlib; no ``axon_velocity`` import.
"""

import numpy as np

from .reconstruction.geometry import (
    RECON_GEOMETRY_NPZ_FILENAME,
    split_branch_polylines,
)

__all__ = [
    "segment_lengths_um",
    "polyline_length_um",
    "unit_morphometrics",
    "unit_morphometrics_from_npz",
]


def segment_lengths_um(points):
    """Euclidean length of each consecutive-vertex segment of one polyline.

    ``points``: an ``(n, 2)`` array of ordered (x, y) vertices in micrometres.
    Returns an ``(n-1,)`` float array of per-segment distances (empty for fewer
    than two vertices). This is the canonical per-segment primitive: the branch
    LENGTH (``polyline_length_um``) is its sum, and a per-segment DISTRIBUTION
    metric (e.g. inter-node gap statistics) is built from the array itself --
    both route through here so the segment-distance computation lives in one place.
    """
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[0] < 2:
        return np.empty((0,), dtype=float)
    return np.linalg.norm(np.diff(p, axis=0), axis=1)


def polyline_length_um(points):
    """Euclidean length of one branch polyline.

    ``points``: an ``(n, 2)`` array of ordered (x, y) vertices in micrometres.
    Returns the sum of consecutive-vertex distances; 0.0 for fewer than two
    vertices (nothing to measure). This is THE canonical branch-length
    definition -- every branch/axon length in the codebase routes through here.
    """
    return float(segment_lengths_um(points).sum())


def unit_morphometrics(branch_points_xy, branch_node_counts):
    """Per-unit morphometrics from the concatenated branch arrays.

    ``branch_points_xy``: ``(n_points_total, 2)`` float, micrometres -- all branch
    polyline vertices concatenated (the ``reconstruction_geometry`` npz array).
    ``branch_node_counts``: ``(n_branches,)`` int -- vertices per branch (the
    split key).

    Returns a dict: ``n_branches``, ``per_branch_length_um`` (list),
    ``mean_branch_length_um``, ``longest_branch_um``, ``total_axon_length_um``
    (see the module docstring; ``total_axon_length_um`` is PROVISIONAL). The three
    scalar length fields are ``float('nan')`` when the unit has no branches.
    """
    polylines = split_branch_polylines(branch_points_xy, branch_node_counts)
    per_branch = [polyline_length_um(pl) for pl in polylines]
    n = len(per_branch)
    if n == 0:
        return {
            "n_branches": 0,
            "per_branch_length_um": [],
            "mean_branch_length_um": float("nan"),
            "longest_branch_um": float("nan"),
            "total_axon_length_um": 0.0,
            # Machine-readable caveat: total_axon_length_um is NOT a validated
            # metric (see docstring). A consumer that reads the total but not the
            # docstring still sees this flag, so it cannot silently treat the
            # total as final.
            "total_axon_length_um_provisional": True,
        }
    return {
        "n_branches": n,
        "per_branch_length_um": per_branch,
        "mean_branch_length_um": float(np.mean(per_branch)),
        "longest_branch_um": float(np.max(per_branch)),
        # PROVISIONAL: sum double-counts shared proximal segments (see docstring).
        "total_axon_length_um": float(np.sum(per_branch)),
        "total_axon_length_um_provisional": True,
    }


def unit_morphometrics_from_npz(npz_path):
    """Load a unit's ``reconstruction_geometry.npz`` and return its morphometrics.

    ``npz_path`` may be the npz file itself or the directory that contains it.
    """
    import os

    if os.path.isdir(npz_path):
        npz_path = os.path.join(npz_path, RECON_GEOMETRY_NPZ_FILENAME)
    with np.load(npz_path, allow_pickle=False) as z:
        return unit_morphometrics(z["branch_points_xy"], z["branch_node_counts"])
