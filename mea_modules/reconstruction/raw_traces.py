"""Export one reconstructed unit's RAW per-channel template traces to a
PORTABLE, self-describing artifact — a `.npz` of arrays plus a sibling `.json`
of metadata, loadable with `numpy` + stdlib `json` alone.

**Why this exists (and why it is separate from `save_reconstruction`).**
`reconstruct_axons` already persists the whole tracked `GraphAxonTracking`
instance as `gtr.pkl` (see :func:`axon_velocity_track.save_reconstruction`).
That pickle carries everything — but it is a pickle: it can only be reopened by
a Python process with a compatible `axon_velocity` installed, which makes it a
poor hand-off to a downstream, standalone figures repo that just wants the raw
traces. This module reads the SAME data the pickle already holds — the raw
per-channel template, the electrode geometry, the sampling rate — and re-emits
it in a durable, tool-independent form: `numpy.load(...)` + `json.load(...)`
and nothing else. It reads a `gtr` object (already unpickled by the caller) and
writes next to it; it does NOT re-run reconstruction, and it does NOT touch
`gtr.pkl` or `reconstruction_summary.json` — a strictly additive third artifact.

**RAW-FIRST (binding).** The whole point is the *raw* picture, so the template
is written VERBATIM: every channel the reconstruction covered, in the unit's
own channel order, with no filtering, no channel elimination, and no
noise-gating. The low-amplitude "noise blanket" channels are information and are
kept. In particular this does NOT restrict the export to `gtr.selected_channels`
(the subset the tracker's graph search chose) — that subset is recorded as
provenance only. `selected_channels`, `init_channel`, and each branch's channel
indices all index into the SAME channel axis as the exported `template` /
`channel_locations` rows (`0 .. n_channels - 1`), so a downstream consumer can
map them back onto the raw traces.

**Where the data lives on `gtr`.** Confirmed against
`axon_velocity.tracking_classes.AxonTracking.__init__` (the base class
`GraphAxonTracking` inherits from): `self.template` `(n_channels, n_samples)`
µV and `self.locations` `(n_channels, 2)` µm are stored verbatim (only
reassigned when `upsample > 1`; this lab pins `upsample=1`), alongside
`self.fs`. This is the same contract `plot_unit_footprint_reconstruction` and
the `overlay` module already read from an unpickled `gtr`.

Public API::

    from mea_modules.reconstruction import (
        export_raw_traces,
        RAW_TRACES_NPZ_FILENAME,
        RAW_TRACES_JSON_FILENAME,
    )

Zero argparse and no `axon_velocity` import here — this is pure serialization
logic over plain attributes (numpy + stdlib only), so the capsule that drives it
stays a thin CLI adapter and a pure test can exercise it with a duck-typed
stand-in.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

RAW_TRACES_NPZ_FILENAME = "raw_traces.npz"
RAW_TRACES_JSON_FILENAME = "raw_traces.json"

SCHEMA = "mea_recon.raw_traces"
SCHEMA_VERSION = 1

# One place, so the .json note and this module's docstring cannot drift.
_CHANNEL_INDEX_SPACE_NOTE = (
    "Channels are indexed 0..n_channels-1, matching the rows of `template` and "
    "`channel_locations` (array `channel_indices`). `selected_channels`, "
    "`init_channel`, and every branch's channel indices in the reconstruction "
    "index into this same axis."
)
_RAW_FIRST_NOTE = (
    "RAW per-channel traces, written verbatim: every channel the reconstruction "
    "covered, in the unit's own channel order, with no filtering, no channel "
    "elimination, and no noise-gating. The low-amplitude noise blanket is kept "
    "by design. This is NOT restricted to selected_channels (recorded as "
    "provenance only)."
)


def _as_int_list(values):
    """A 1-D int array (or anything array-like) as a plain list of Python ints.

    JSON-safe: `np.int64`/`np.ndarray` do not survive `json.dumps`, so the
    provenance channel lists are converted here rather than at write time.
    """
    arr = np.asarray(values).ravel()
    return [int(v) for v in arr.tolist()]


def export_raw_traces(gtr, out_dir, unit_id=None):
    """Write one unit's raw traces + geometry to `out_dir` as `.npz` + `.json`.

    `gtr` is any object carrying `template` `(n_channels, n_samples)` µV,
    `locations` `(n_channels, 2)` µm, and `fs` Hz (an unpickled
    `GraphAxonTracking`, or a duck-typed stand-in in tests). `selected_channels`,
    `init_channel`, and `branches` are read for provenance when present and are
    optional — a `gtr` missing them still exports its raw traces, with the
    corresponding provenance fields left null / zero.

    Returns the metadata dict written to `raw_traces.json` (so a caller — e.g.
    the capsule building a per-well summary — need not re-read the file). Raises
    `ValueError` if `template`/`locations` are not 2-D or their channel axes
    disagree; that is a corrupt-input problem, not a per-unit skip.
    """
    out_dir = Path(out_dir)

    template = np.asarray(gtr.template)
    locations = np.asarray(gtr.locations)

    if template.ndim != 2:
        raise ValueError(
            f"gtr.template must be 2-D (n_channels, n_samples); got shape {template.shape}"
        )
    if locations.ndim != 2 or locations.shape[1] < 2:
        raise ValueError(
            f"gtr.locations must be (n_channels, 2+); got shape {locations.shape}"
        )
    if template.shape[0] != locations.shape[0]:
        raise ValueError(
            f"gtr.template has {template.shape[0]} channel(s) but gtr.locations has "
            f"{locations.shape[0]} -- must match 1:1 (the channel index space "
            "selected_channels / init_channel / branch channels all assume)"
        )

    n_channels, n_samples = int(template.shape[0]), int(template.shape[1])
    channel_indices = np.arange(n_channels, dtype=np.int64)

    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / RAW_TRACES_NPZ_FILENAME
    # savez_compressed writes template/locations VERBATIM (no dtype coercion) —
    # raw-first. channel_indices is the explicit, self-describing channel axis.
    np.savez_compressed(
        npz_path,
        template=template,
        channel_locations=locations,
        channel_indices=channel_indices,
    )

    selected = getattr(gtr, "selected_channels", None)
    init_channel = getattr(gtr, "init_channel", None)
    branches = getattr(gtr, "branches", None)

    metadata = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "unit_id": None if unit_id is None else str(unit_id),
        "n_channels": n_channels,
        "n_samples": n_samples,
        "sampling_rate_hz": None if getattr(gtr, "fs", None) is None else float(gtr.fs),
        "trace_units": "uV",
        "location_units": "um",
        "location_axes": ["x", "y"],
        "channel_index_space": _CHANNEL_INDEX_SPACE_NOTE,
        "raw_first": True,
        "raw_first_note": _RAW_FIRST_NOTE,
        "provenance": {
            "source": "gtr.pkl",
            "source_capsule": "reconstruct_axons",
            "generator": "export_recon_traces",
            "gtr_class": f"{type(gtr).__module__}.{type(gtr).__name__}",
            "selected_channels": [] if selected is None else _as_int_list(selected),
            "n_selected_channels": 0 if selected is None else int(np.asarray(selected).size),
            "init_channel": None if init_channel is None else int(init_channel),
            "n_branches": 0 if branches is None else int(len(branches)),
        },
        "npz_file": RAW_TRACES_NPZ_FILENAME,
        "npz_arrays": {
            "template": "(n_channels, n_samples) float, uV — raw per-channel template traces",
            "channel_locations": "(n_channels, 2) float, um — electrode xy per channel",
            "channel_indices": "(n_channels,) int64 — the channel index axis (0..n_channels-1)",
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }

    json_path = out_dir / RAW_TRACES_JSON_FILENAME
    json_path.write_text(json.dumps(metadata, indent=2))

    logger.info(
        "exported raw traces%s: %d channel(s) x %d sample(s) -> %s",
        f" for unit {unit_id}" if unit_id is not None else "",
        n_channels, n_samples, out_dir,
    )
    return metadata


__all__ = [
    "export_raw_traces",
    "RAW_TRACES_NPZ_FILENAME",
    "RAW_TRACES_JSON_FILENAME",
    "SCHEMA",
    "SCHEMA_VERSION",
]
