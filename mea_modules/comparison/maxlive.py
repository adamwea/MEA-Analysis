"""Read MaxLab Live's AxonTracking analysis output — the "SOTA" side of the
axon-reconstruction comparison.

MaxLab Live ("MaxLive") is MaxWell Biosystems' proprietary acquisition/analysis
software. When its offline *Axon Tracking analysis* is run on an AxonTracking
assay recording, it writes a per-recording results tree::

    <run_dir>/analysis/AxonAnalysis_v1/<round>/
        .swlib/internal_results.h5                     # the tables (read here)
        Well<W>/AxonAnalysis_Well<W>.png               # per-well summary image
        Well<W>/Neuron#<n>/Neuron#<n>Branch#<b>.png    # per-neuron/branch recon image

`internal_results.h5` stores pandas *fixed*-format tables under
`dataframes/<well>/{neuron_metrics, tracking_info, ...}`. We read them with plain
h5py + numpy (NO pandas, NO pytables): a fixed table is a set of `block<b>_items`
(column-name arrays) plus `block<b>_values` (value matrices).

Two coordinate conventions to keep straight:
  * table well index is **0-based** (`dataframes/0` … `dataframes/5`);
  * on-disk image dirs are **1-based** (`Well1` … `Well6`).

`neuron_metrics` carries one row per tracked neuron; the key columns for the
comparison are `neuron` (id), `electrodeNo` + `x,y` (the fixed init-site electrode
and its µm position — the hook for electrode-cluster matching), plus
`neuronConductionVel`, `totalAxonLen`, `longestBranchLen`, `amplitudeInitSite`.
`tracking_info` carries one row per branch vertex (`x,y,neuron,branch,cumDistance,
latFromInitSite`) — the reconstructed arbor.

Public API::

    from mea_modules.comparison.maxlive import (
        AXON_ANALYSIS_RELDIR, MaxLiveWell,
        results_h5_path, list_wells, read_internal_results, read_fixed_table,
        load_well, neuron_somata, neuron_arbors, recon_image_paths,
    )

Read-only: this module never runs MaxLive (that needs the GUI). It depends only
on numpy + h5py — nothing else in `mea_modules`.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np
import h5py

# internal_results.h5 location relative to a recording's AxonTracking/<run> dir.
AXON_ANALYSIS_RELDIR = os.path.join("analysis", "AxonAnalysis_v1")
DEFAULT_ROUND = "0001"
_RESULTS_REL = os.path.join(".swlib", "internal_results.h5")
_WELL_DIRNAME = "Well%d"          # on-disk image dirs are 1-based


def _dec(x):
    """bytes -> str, pass everything else through (MaxLive strings are bytes)."""
    return x.decode("utf-8", "replace") if isinstance(x, (bytes, bytearray)) else x


def read_fixed_table(group) -> dict:
    """A pandas *fixed*-format HDF5 table group -> {column_name: 1-D np.ndarray}.

    Pure h5py/numpy: reassembles the block layout `DataFrame.to_hdf(format="fixed")`
    writes, so reading MaxLive's tables needs neither pandas nor pytables.
    """
    cols: dict = {}
    for b in range(int(group.attrs.get("nblocks", 0))):
        items = [_dec(v) for v in group["block%d_items" % b][:]]
        vals = group["block%d_values" % b][:]
        if vals.ndim == 1:
            vals = vals.reshape(1, -1)
        if vals.shape[0] == len(items):        # axis 0 already indexes columns
            arr = vals
        elif vals.shape[1] == len(items):      # transposed
            arr = vals.T
        else:                                  # single-column / degenerate block
            arr = vals
        for i, name in enumerate(items):
            col = arr[i]
            if col.dtype.kind in ("S", "O"):
                col = np.array([_dec(c) for c in col])
            cols[_dec(name)] = col
    return cols


def results_h5_path(run_dir: str, round_name: str = DEFAULT_ROUND) -> str:
    """Path to internal_results.h5 for a recording's AxonTracking/<run> dir."""
    return os.path.join(run_dir, AXON_ANALYSIS_RELDIR, round_name, _RESULTS_REL)


def list_wells(h5_path: str) -> list:
    """Sorted well keys present in an internal_results.h5 (as strings)."""
    with h5py.File(h5_path, "r") as f:
        if "dataframes" not in f:
            return []
        keys = list(f["dataframes"].keys())
    return sorted(keys, key=lambda s: int(s) if s.isdigit() else s)


def read_internal_results(h5_path: str, well) -> dict:
    """{table_name: {col: array}} for one well of an internal_results.h5.

    Reads every fixed-format table under `dataframes/<well>/` (neuron_metrics,
    tracking_info, summary_metrics, well_result, neurons_for_well, ...).
    """
    out: dict = {}
    with h5py.File(h5_path, "r") as f:
        g = f["dataframes/%s" % str(well)]
        for tbl in g.keys():
            sub = g[tbl]
            if isinstance(sub, h5py.Group) and any(k.startswith("block") for k in sub.keys()):
                out[tbl] = read_fixed_table(sub)
    return out


@dataclass
class MaxLiveWell:
    """One well of MaxLive AxonTracking analysis (tables only; images via
    :func:`recon_image_paths`)."""
    well: int
    run_dir: str
    round_name: str
    neuron_metrics: dict     # {col: array}, one row per tracked neuron
    tracking_info: dict      # {col: array}, one row per branch vertex

    @property
    def neuron_ids(self) -> np.ndarray:
        return np.asarray(self.neuron_metrics["neuron"], float).astype(int)

    def somata(self) -> dict:
        return neuron_somata(self.neuron_metrics)

    def arbors(self) -> dict:
        return neuron_arbors(self.tracking_info)

    def images(self) -> dict:
        return recon_image_paths(self.run_dir, self.well, self.round_name)


def load_well(run_dir: str, well, round_name: str = DEFAULT_ROUND) -> MaxLiveWell:
    """Load one well's neuron_metrics + tracking_info from run_dir's AxonAnalysis."""
    tabs = read_internal_results(results_h5_path(run_dir, round_name), well)
    return MaxLiveWell(int(well), run_dir, round_name,
                       tabs.get("neuron_metrics", {}), tabs.get("tracking_info", {}))


def neuron_somata(neuron_metrics: dict) -> dict:
    """{neuron_id: {"electrodeNo": int|None, "x": µm, "y": µm}} init-site per neuron."""
    nm = neuron_metrics
    ids = np.asarray(nm["neuron"], float).astype(int)
    x = np.asarray(nm["x"], float)
    y = np.asarray(nm["y"], float)
    eno = np.asarray(nm["electrodeNo"], float) if "electrodeNo" in nm else None
    out = {}
    for i, nid in enumerate(ids):
        out[int(nid)] = {
            "electrodeNo": (int(eno[i]) if eno is not None and np.isfinite(eno[i]) else None),
            "x": float(x[i]),
            "y": float(y[i]),
        }
    return out


def neuron_arbors(tracking_info: dict) -> dict:
    """{neuron_id: {branch_id: (n,2) xy array}} — each branch an ordered polyline.

    tracking_info is one row per branch vertex; we group by (neuron, branch) and
    order each branch by its monotone `cumDistance`.
    """
    ti = tracking_info
    if not ti or "neuron" not in ti:
        return {}
    neuron = np.asarray(ti["neuron"], float).astype(int)
    branch = np.asarray(ti["branch"], float).astype(int)
    x = np.asarray(ti["x"], float)
    y = np.asarray(ti["y"], float)
    cd = np.asarray(ti.get("cumDistance", np.arange(len(x))), float)
    out: dict = {}
    for nid in np.unique(neuron):
        mn = neuron == nid
        branches = {}
        for bid in np.unique(branch[mn]):
            m = mn & (branch == bid)
            order = np.argsort(cd[m], kind="stable")
            branches[int(bid)] = np.column_stack([x[m][order], y[m][order]])
        out[int(nid)] = branches
    return out


def _well_image_dir(run_dir: str, well: int, round_name: str = DEFAULT_ROUND) -> str:
    # table well is 0-based; on-disk Well dirs are 1-based
    return os.path.join(run_dir, AXON_ANALYSIS_RELDIR, round_name, _WELL_DIRNAME % (well + 1))


def recon_image_paths(run_dir: str, well: int, round_name: str = DEFAULT_ROUND) -> dict:
    """Locate MaxLive's own rendered images for a well (F2P2 source — no regen).

    Returns::

        {"summary": <AxonAnalysis_Well<W>.png or None>,
         "neurons": {neuron_id: {branch_id: png_path}}}
    """
    wdir = _well_image_dir(run_dir, well, round_name)
    res = {"summary": None, "neurons": {}}
    if not os.path.isdir(wdir):
        return res
    summ = os.path.join(wdir, "AxonAnalysis_Well%d.png" % (well + 1))
    res["summary"] = summ if os.path.isfile(summ) else None
    n_re = re.compile(r"^Neuron#(\d+)$")
    b_re = re.compile(r"^Neuron#(\d+)Branch#(\d+)\.png$")
    for entry in sorted(os.listdir(wdir)):
        m = n_re.match(entry)
        if not m:
            continue
        nid = int(m.group(1))
        ndir = os.path.join(wdir, entry)
        if not os.path.isdir(ndir):
            continue
        branches = {}
        for fn in os.listdir(ndir):
            bm = b_re.match(fn)
            if bm and int(bm.group(1)) == nid:
                branches[int(bm.group(2))] = os.path.join(ndir, fn)
        res["neurons"][nid] = branches
    return res
