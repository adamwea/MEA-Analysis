"""Read OUR axon-reconstruction pipeline's per-unit output for one well, in the
portable form the comparison needs — arbors in MaxWell µm, without axon_velocity.

Per well the pipeline writes (stage numbers vary by run, so we discover by keyword)::

    <well_dir>/<NN>_stitch_templates/channel_locations_xy.npy    # (n_ch, 2) µm — the
                                                                 # index space branch
                                                                 # `channels` refer to
    <well_dir>/<NN>_reconstruct_axons/units/unit<NNNN>/reconstruction_summary.json

`reconstruction_summary.json` is fully portable (plain JSON): per unit an
`init_channel` and a list of `branches`, each carrying `channels` (indices into
channel_locations_xy), `velocity` (µm/ms), `distances` (µm), `peak_times` (ms),
`r2`, `pval`. Mapping `channels -> channel_locations_xy` gives each branch as an
(n,2) xy polyline in the same MaxWell µm frame MaxLive reports, so our arbors and
MaxLive's are directly overlayable (subject to the orientation check in `match`).

This reader never touches `gtr.pkl` (that needs `axon_velocity`); everything comes
from the JSON + the xy npy. numpy only.
"""
from __future__ import annotations

import os
import glob
import json
from dataclasses import dataclass, field

import numpy as np

STITCH_KEYWORD = "stitch_templates"
RECON_KEYWORD = "reconstruct_axons"
CHANNEL_XY_NAME = "channel_locations_xy.npy"


def discover_stage(well_dir: str, keyword: str) -> str | None:
    """The `<NN>_<keyword>` stage dir under a well dir (highest stage number wins)."""
    hits = sorted(glob.glob(os.path.join(well_dir, "*" + keyword)))
    return hits[-1] if hits else None


def load_channel_xy(well_dir: str, stitch_dir: str | None = None) -> np.ndarray:
    """(n_ch, 2) µm channel locations — the index space branch `channels` use."""
    sd = stitch_dir or discover_stage(well_dir, STITCH_KEYWORD)
    if sd is None:
        raise FileNotFoundError("no *%s stage under %s" % (STITCH_KEYWORD, well_dir))
    return np.load(os.path.join(sd, CHANNEL_XY_NAME))


@dataclass
class ReconUnit:
    """One reconstructed unit, arbor already mapped to xy (MaxWell µm)."""
    unit_id: int
    init_channel: int
    init_xy: np.ndarray                       # (2,)
    branches: list = field(default_factory=list)   # list of dicts (see below)

    @property
    def n_branches(self) -> int:
        return len(self.branches)

    def arbor(self) -> list:
        """List of (n,2) xy polylines, one per branch."""
        return [b["xy"] for b in self.branches]

    def total_length_um(self) -> float:
        tot = 0.0
        for b in self.branches:
            p = b["xy"]
            if len(p) > 1:
                tot += float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
        return tot

    def median_velocity(self) -> float:
        v = np.array([b["velocity"] for b in self.branches
                      if b.get("velocity") is not None], float)
        v = v[np.isfinite(v) & (v > 0)]
        return float(np.median(v)) if v.size else float("nan")


@dataclass
class PipelineWell:
    """One well of our pipeline's reconstruction output."""
    well_dir: str
    channel_xy: np.ndarray
    units: dict                                # {unit_id: ReconUnit}

    def unit_ids(self) -> list:
        return sorted(self.units)

    def somata(self) -> dict:
        return {u: self.units[u].init_xy for u in self.units}

    def arbors(self) -> dict:
        return {u: self.units[u].arbor() for u in self.units}


def _branch_xy(channels, channel_xy: np.ndarray) -> np.ndarray:
    ch = np.asarray(channels, int)
    ch = ch[(ch >= 0) & (ch < len(channel_xy))]
    return channel_xy[ch].astype(float) if ch.size else np.empty((0, 2))


def load_recon_units(well_dir: str, recon_dir: str | None = None,
                     channel_xy: np.ndarray | None = None) -> PipelineWell:
    """Load every reconstructed unit for a well into a :class:`PipelineWell`.

    Reads `reconstruction_summary.json` per unit and maps branch `channels` to xy
    via `channel_locations_xy.npy`. Units with no branches are skipped.
    """
    rd = recon_dir or discover_stage(well_dir, RECON_KEYWORD)
    if rd is None:
        raise FileNotFoundError("no *%s stage under %s" % (RECON_KEYWORD, well_dir))
    cxy = channel_xy if channel_xy is not None else load_channel_xy(well_dir)
    units = {}
    for udir in sorted(glob.glob(os.path.join(rd, "units", "unit*"))):
        summ = os.path.join(udir, "reconstruction_summary.json")
        if not os.path.isfile(summ):
            continue
        try:
            rs = json.load(open(summ))
        except Exception:
            continue
        brs_raw = rs.get("branches") or []
        if not brs_raw:
            continue
        uid = int(rs["unit_id"])
        init_ch = int(rs.get("init_channel", -1))
        branches = []
        for b in brs_raw:
            xy = _branch_xy(b.get("channels", []), cxy)
            if len(xy) < 1:
                continue
            branches.append(dict(
                channels=np.asarray(b.get("channels", []), int),
                xy=xy,
                velocity=b.get("velocity"),
                distances=np.asarray(b.get("distances", []), float),
                peak_times=np.asarray(b.get("peak_times", []), float),
                r2=b.get("r2"),
                pval=b.get("pval"),
            ))
        if not branches:
            continue
        init_xy = (cxy[init_ch].astype(float)
                   if 0 <= init_ch < len(cxy) else branches[0]["xy"][0])
        units[uid] = ReconUnit(uid, init_ch, init_xy, branches)
    return PipelineWell(well_dir, cxy, units)
