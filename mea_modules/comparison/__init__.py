"""Compare our axon-tracking pipeline against MaxLab Live's AxonTracking analysis.

The reusable *machinery* (readers + matcher) for the "vs state-of-the-art"
comparison. MaxWell's MaxLab Live ("MaxLive") is the SOTA reference; this
subpackage reads its output and pairs it with our pipeline's reconstruction so
they can be compared and rendered side by side.

Two scopes:
  1. **individual reconstruction comparison** — MaxLive assumes one tracked
     neuron per fixed-electrode cluster, whereas our sorter can resolve several
     units at the same cluster. So we match *all our units whose extremum
     electrode falls in a cluster* against *MaxLive's single neuron there* and
     show them together;
  2. **population comparison** — per-unit metric distributions, ours vs MaxLive.

Figure-specific composition (the F2 panels) lives in the figures repo that
consumes this — this subpackage stays general pipeline machinery.

Modules:
  * ``maxlive``  — read MaxLive's ``internal_results.h5`` tables + rendered images
    (numpy + h5py only).
  * ``pipeline`` — read our pipeline's per-unit reconstruction for the same recording.
  * ``match``    — electrode-cluster matching (+ coordinate/orientation alignment).
"""
from .maxlive import (
    AXON_ANALYSIS_RELDIR,
    DEFAULT_ROUND,
    MaxLiveWell,
    load_well,
    list_wells,
    read_internal_results,
    read_fixed_table,
    results_h5_path,
    neuron_somata,
    neuron_arbors,
    recon_image_paths,
)

__all__ = [
    "AXON_ANALYSIS_RELDIR",
    "DEFAULT_ROUND",
    "MaxLiveWell",
    "load_well",
    "list_wells",
    "read_internal_results",
    "read_fixed_table",
    "results_h5_path",
    "neuron_somata",
    "neuron_arbors",
    "recon_image_paths",
]
