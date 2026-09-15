"""Export one reconstructed unit's AXON BRANCH GEOMETRY — the tracked
reconstruction itself, as drawable polylines — to a PORTABLE companion artifact
(`reconstruction_geometry.npz` + `.json`), loadable with `numpy` + stdlib `json`
alone.

**Why this exists (companion to `export_raw_traces`).** The raw-traces artifact
carries the per-channel signal (the footprint backdrop) but only a COUNT of
branches — not the branches themselves. A "reconstruction traces" figure needs
the reconstruction: for each tracked branch, the ORDERED sequence of electrode
(x, y) positions (µm) that define the traced axon polyline, so it can be drawn
as connected lines over the electrode array. This module reads that straight off
an already-unpickled `gtr` and writes it next to `raw_traces.*`. It is strictly
additive: it does NOT touch `gtr.pkl`, `reconstruction_summary.json`, or the
`raw_traces.*` files.

**Where the geometry lives on `gtr`.** Confirmed against real P005843 `gtr.pkl`
and `mea_modules.reconstruction.overlay.unit_arbor_record` (the reviewed
extractor the existing per-well overlay figure already draws from): each entry
of `gtr.branches` is a dict whose `'channels'` is the branch's electrode path IN
ORDER (int indices into `gtr.locations` / the raw-traces channel axis), so the
branch polyline is simply `locations[channels]` — the SAME (x, y) µm frame as
`channel_locations` in `raw_traces.npz`. Each branch dict also carries the
velocity-fit summary (`velocity`, `r2`, `pval`, `offset`, `raw_path_idx`) and
per-node fit arrays (`distances`, `peak_times`, length `n_nodes - 1`).
`gtr.init_channel` is the largest-amplitude electrode — the tracker's initiation
site and the closest thing to a soma position (a shared marker the branches fan
out from).

**Connectivity.** Each branch is exported as an independent ordered polyline.
`axon_velocity`'s branch list does NOT expose a branch-to-branch parent/child
tree (the `networkx` DiGraph on `gtr` is the raw search graph over every
selected channel, not a cleaned branch tree, and branches do not all start at
`init_channel`), so the drawable structure is "polylines + a shared initiation
site". The raw DiGraph stays available in `gtr.pkl` for anyone who wants it; it
is deliberately not serialized here (heavy, and not the reconstruction overlay).

Public API::

    from mea_modules.reconstruction import (
        export_reconstruction_geometry,
        RECON_GEOMETRY_NPZ_FILENAME,
        RECON_GEOMETRY_JSON_FILENAME,
    )

Zero argparse and no `axon_velocity` import — pure serialization over plain
attributes (numpy + stdlib only).
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

RECON_GEOMETRY_NPZ_FILENAME = "reconstruction_geometry.npz"
RECON_GEOMETRY_JSON_FILENAME = "reconstruction_geometry.json"

SCHEMA = "mea_recon.reconstruction_geometry"
SCHEMA_VERSION = 1

_COORDINATE_FRAME_NOTE = (
    "Positions are micrometers (um) in the SAME (x, y) frame as "
    "`channel_locations` in raw_traces.npz; branch_channels index the same "
    "channel axis as raw_traces' template rows (0..n_channels-1)."
)
_HOW_TO_SPLIT_NOTE = (
    "Per-branch polylines are concatenated. To recover them: "
    "offsets = numpy.cumsum(branch_node_counts)[:-1]; "
    "polylines = numpy.split(branch_points_xy, offsets); "
    "channels_per_branch = numpy.split(branch_channels, offsets). "
    "Draw each polyline as a connected line; all branches share the "
    "initiation site init_xy (soma stand-in). The JSON `branches` list carries "
    "the same points_xy inline for stdlib-only consumers."
)


def _float(v, default=float("nan")):
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _int(v, default=-1):
    if v is None:
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def export_reconstruction_geometry(gtr, out_dir, unit_id=None):
    """Write one unit's reconstructed branch polylines to `out_dir` as
    `reconstruction_geometry.npz` + `.json`.

    `gtr` carries `locations` `(n_channels, 2)` µm, `init_channel`, and
    `branches` (a list of dicts, each with an ordered `channels` path). A unit
    with zero branches still writes both files, with empty arrays / an empty
    `branches` list — a legitimate outcome (most units are not clean axons).

    Returns the metadata dict written to the JSON file. Raises `ValueError` if
    `gtr.locations` is not `(n_channels, 2+)`.
    """
    out_dir = Path(out_dir)
    locations = np.asarray(getattr(gtr, "locations"), dtype=float)
    if locations.ndim != 2 or locations.shape[1] < 2:
        raise ValueError(
            f"gtr.locations must be (n_channels, 2+); got shape {locations.shape}"
        )
    n_channels = locations.shape[0]

    branches = list(getattr(gtr, "branches", None) or [])

    # Single pass: build both the concatenated arrays and the inline JSON list.
    points_list, node_counts, channels_flat = [], [], []
    ids, velocity, r2, pval, offset, raw_path_idx = [], [], [], [], [], []
    branches_json = []
    for i, br in enumerate(branches):
        raw_ch = br.get("channels") if isinstance(br, dict) else None
        if raw_ch is None:
            continue
        ch = np.asarray(raw_ch).ravel().astype(np.int64)
        # Keep only in-range channel indices (defensive; real branches are all
        # in range). An empty result means nothing drawable — skip the branch.
        ch = ch[(ch >= 0) & (ch < n_channels)]
        if ch.size == 0:
            continue
        pts = locations[ch, :2]

        points_list.append(pts)
        node_counts.append(int(ch.size))
        channels_flat.append(ch)
        ids.append(i)
        velocity.append(_float(br.get("velocity")))
        r2.append(_float(br.get("r2")))
        pval.append(_float(br.get("pval")))
        offset.append(_float(br.get("offset")))
        raw_path_idx.append(_int(br.get("raw_path_idx")))

        branches_json.append({
            "branch_id": i,
            "n_nodes": int(ch.size),
            "channels": [int(c) for c in ch.tolist()],
            "points_xy": [[float(x), float(y)] for x, y in pts.tolist()],
            "velocity": _float(br.get("velocity")),
            "r2": _float(br.get("r2")),
            "pval": _float(br.get("pval")),
            "offset": _float(br.get("offset")),
            "raw_path_idx": _int(br.get("raw_path_idx")),
            # per-node velocity-fit arrays; length n_nodes-1 in this data
            # (axon_velocity excludes the reference node from the fit).
            "distances": [float(d) for d in np.asarray(br.get("distances", [])).ravel().tolist()],
            "peak_times": [float(t) for t in np.asarray(br.get("peak_times", [])).ravel().tolist()],
        })

    n_branches = len(branches_json)
    if points_list:
        branch_points_xy = np.vstack(points_list).astype(np.float64)
        branch_channels = np.concatenate(channels_flat).astype(np.int64)
    else:
        branch_points_xy = np.empty((0, 2), dtype=np.float64)
        branch_channels = np.empty((0,), dtype=np.int64)
    branch_node_counts = np.asarray(node_counts, dtype=np.int64)
    branch_ids = np.asarray(ids, dtype=np.int64)
    branch_velocity = np.asarray(velocity, dtype=np.float64)
    branch_r2 = np.asarray(r2, dtype=np.float64)
    branch_pval = np.asarray(pval, dtype=np.float64)
    branch_offset = np.asarray(offset, dtype=np.float64)
    branch_raw_path_idx = np.asarray(raw_path_idx, dtype=np.int64)

    init_raw = getattr(gtr, "init_channel", None)
    init_channel = _int(init_raw, default=-1)
    if 0 <= init_channel < n_channels:
        init_xy = [float(locations[init_channel, 0]), float(locations[init_channel, 1])]
    else:
        init_xy = None

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / RECON_GEOMETRY_NPZ_FILENAME,
        branch_points_xy=branch_points_xy,
        branch_node_counts=branch_node_counts,
        branch_channels=branch_channels,
        branch_ids=branch_ids,
        branch_velocity=branch_velocity,
        branch_r2=branch_r2,
        branch_pval=branch_pval,
        branch_offset=branch_offset,
        branch_raw_path_idx=branch_raw_path_idx,
        init_channel=np.asarray(init_channel, dtype=np.int64),
        init_xy=np.asarray(init_xy if init_xy is not None else [np.nan, np.nan], dtype=np.float64),
    )

    metadata = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "unit_id": None if unit_id is None else str(unit_id),
        "n_branches": int(n_branches),
        "n_points_total": int(branch_points_xy.shape[0]),
        "position_units": "um",
        "velocity_units": "um/ms",
        "velocity_units_note": "axon_velocity's reported per-branch velocity (um/ms); can be negative for a retrograde/ill-fit branch.",
        "coordinate_frame": _COORDINATE_FRAME_NOTE,
        "channel_index_space": "branch_channels index the same axis as raw_traces' template / channel_locations rows (0..n_channels-1).",
        "how_to_split": _HOW_TO_SPLIT_NOTE,
        "connectivity": (
            "Each branch is an independent ordered polyline (the tracker's "
            "cleaned path). No branch-to-branch parent/child tree is exposed by "
            "axon_velocity; branches share the initiation site init_xy (soma "
            "stand-in) as the visual root."
        ),
        "init_channel": init_channel,
        "init_xy": init_xy,
        "npz_file": RECON_GEOMETRY_NPZ_FILENAME,
        "npz_arrays": {
            "branch_points_xy": "(n_points_total, 2) float64, um — all branch polyline vertices concatenated",
            "branch_node_counts": "(n_branches,) int64 — vertices per branch (split key)",
            "branch_channels": "(n_points_total,) int64 — channel index per vertex",
            "branch_ids": "(n_branches,) int64 — branch id (index into gtr.branches)",
            "branch_velocity": "(n_branches,) float64 — um/ms",
            "branch_r2": "(n_branches,) float64 — velocity-fit R^2",
            "branch_pval": "(n_branches,) float64 — velocity-fit p-value",
            "branch_offset": "(n_branches,) float64 — velocity-fit intercept",
            "branch_raw_path_idx": "(n_branches,) int64 — axon_velocity raw-path index",
            "init_channel": "() int64 — initiation-site channel index (-1 if none)",
            "init_xy": "(2,) float64 — initiation-site xy, um (NaN if none)",
        },
        "provenance": {
            "source": "gtr.pkl",
            "source_capsule": "reconstruct_axons",
            "generator": "export_recon_traces",
            "gtr_class": f"{type(gtr).__module__}.{type(gtr).__name__}",
        },
        "branches": branches_json,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }

    (out_dir / RECON_GEOMETRY_JSON_FILENAME).write_text(json.dumps(metadata, indent=2))

    logger.info(
        "exported reconstruction geometry%s: %d branch(es), %d polyline vertex(es) -> %s",
        f" for unit {unit_id}" if unit_id is not None else "",
        n_branches, int(branch_points_xy.shape[0]), out_dir,
    )
    return metadata


__all__ = [
    "export_reconstruction_geometry",
    "RECON_GEOMETRY_NPZ_FILENAME",
    "RECON_GEOMETRY_JSON_FILENAME",
    "SCHEMA",
    "SCHEMA_VERSION",
]


def rg_circle(points):
    """Radius of gyration of a 2-D point cloud, returned as a drawable circle.

    ``points``: (N, 2) array of (x, y). Returns ``(center_xy (2,), rg)`` with
    ``rg = sqrt(mean(sum((p - centroid)**2)))`` -- the RMS distance of the points
    from their centroid. General geometry helper; the recon-vs-maxlive figure
    uses it to draw each arbor's spatial-spread circle at a common scale.
    (Relocated from the figure repo per the mechanics-in-mea_modules policy,
    2026-09-11.)
    """
    P = np.asarray(points, float)
    cen = P.mean(axis=0)
    rg = float(np.sqrt(np.mean(np.sum((P - cen) ** 2, axis=1))))
    return cen, rg
