"""Reading `merge_templates`' own output back — the read-side counterpart to
`merge.py`'s write-side.

Public API::

    from mea_modules.templates import discover_unit_ids, load_well_inputs, load_unit_inputs

Extracted 2026-08-04: `capsules/reconstruct_axons/run_capsule.py` already had
this exact loading logic (private `_discover_units`/`_load_well_inputs`/
`_load_unit_inputs`), written and verified against the real per-well contract
during chain-integration and since run clean over all 780 real units in a
well. When `footprint_diagnostics`/`footprint_plots` needed the IDENTICAL
loading (same `merged_templates.npy`/`contributing_weight.npy`/
`channel_locations_xy.npy`/`unit_ids.json`/`merge_manifest.json` contract,
same NaN-drop policy per unit), the choice was: (a) refactor
`reconstruct_axons` to import from here too, or (b) leave that
already-verified, already-run-on-real-data capsule exactly as it is and give
the two NEW capsules their own copy of the same logic here. Chose (b),
deliberately — touching a capsule that already has a full 780-unit real run
behind it, purely for a DRY cleanup, is a real-code-churn risk this session
has no reason to take on. `reconstruct_axons` keeps its own private copy;
this module exists for the two new consumers so THEY are not duplicating each
other. A future pass can fold `reconstruct_axons` onto this module once
there is no run in flight to risk.

Library functions, not CLI code: raise ordinary exceptions
(`FileNotFoundError`/`KeyError`/`ValueError`) rather than `SystemExit` —
unlike `reconstruct_axons`'s own capsule-local copy, callers here are
expected to be other library/capsule code that wants a normal Python
exception to catch or let propagate, not a CLI process exiting directly.
"""

import json
from pathlib import Path

TEMPLATES_FILENAME = "merged_templates.npy"
WEIGHT_FILENAME = "contributing_weight.npy"
LOCATIONS_FILENAME = "channel_locations_xy.npy"
UNIT_IDS_FILENAME = "unit_ids.json"
MANIFEST_FILENAME = "merge_manifest.json"
FS_MANIFEST_KEY = "sampling_frequency_hz"


def discover_unit_ids(merge_well_dir, only_unit_id=None):
    """Unit ids for `merge_well_dir`, read straight from `unit_ids.json`.

    `merge_templates` keeps every unit in ONE shared `unit_ids.json` (aligned
    to the unit axis of `merged_templates.npy`/`contributing_weight.npy`),
    not one directory per unit. Reading it is cheap, so this can run before
    the (potentially large) shared arrays are opened by `load_well_inputs`.
    """
    merge_well_dir = Path(merge_well_dir)
    unit_ids_path = merge_well_dir / UNIT_IDS_FILENAME
    if not unit_ids_path.is_file():
        raise FileNotFoundError(
            f"no {UNIT_IDS_FILENAME} at {merge_well_dir}; merge_templates must "
            "run for this well before its output can be read"
        )

    unit_ids = [str(unit_id) for unit_id in json.loads(unit_ids_path.read_text())]
    if not unit_ids:
        raise ValueError(f"{unit_ids_path} lists zero unit(s); nothing to read")

    if only_unit_id is not None:
        if str(only_unit_id) not in unit_ids:
            raise ValueError(
                f"unit {only_unit_id!r} not among the {len(unit_ids)} unit(s) "
                f"listed in {unit_ids_path}"
            )
        return [str(only_unit_id)]

    return unit_ids


def load_well_inputs(merge_well_dir):
    """Load one well's combined `merge_templates` arrays ONCE.

    Every unit's channel data lives in one shared `(n_units,
    n_channels_union, n_samples)` template array plus a matching
    `(n_units, n_channels_union)` weight array and one
    `(n_channels_union, 2)` locations array. Opening those three files once
    per well and slicing per unit (`load_unit_inputs` below) avoids
    reopening the same, potentially large, `.npy` files once per unit.

    Returns a dict: `{"unit_ids": [...], "templates": ndarray,
    "contributing_weight": ndarray, "channel_locations_xy": ndarray, "fs": float}`.

    Raises `FileNotFoundError`/`KeyError`/`ValueError` for any whole-well
    problem (missing file, missing sampling rate, internally inconsistent
    array shapes) — a caller processing many units should let this propagate
    rather than catch it per unit, since it means the well's data itself is
    unusable, not that one unit failed.
    """
    import numpy as np

    merge_well_dir = Path(merge_well_dir)

    manifest_path = merge_well_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{manifest_path} not found; the merge_templates contract carries "
            f"the sampling rate here under {FS_MANIFEST_KEY!r}"
        )
    manifest = json.loads(manifest_path.read_text())
    if FS_MANIFEST_KEY not in manifest:
        raise KeyError(
            f"{manifest_path} has no {FS_MANIFEST_KEY!r} key; cannot determine "
            "the sampling rate for this well"
        )
    fs = float(manifest[FS_MANIFEST_KEY])

    unit_ids = [str(unit_id) for unit_id in json.loads((merge_well_dir / UNIT_IDS_FILENAME).read_text())]
    templates = np.load(merge_well_dir / TEMPLATES_FILENAME)
    contributing_weight = np.load(merge_well_dir / WEIGHT_FILENAME)
    channel_locations_xy = np.load(merge_well_dir / LOCATIONS_FILENAME)

    if templates.shape[0] != len(unit_ids) or contributing_weight.shape[0] != len(unit_ids):
        raise ValueError(
            f"{merge_well_dir}: {TEMPLATES_FILENAME} ({templates.shape[0]} unit "
            f"row(s)) / {WEIGHT_FILENAME} ({contributing_weight.shape[0]} unit "
            f"row(s)) does not match {UNIT_IDS_FILENAME} ({len(unit_ids)} "
            "unit(s)) -- merge_templates' output is internally inconsistent"
        )
    if templates.shape[1] != channel_locations_xy.shape[0]:
        raise ValueError(
            f"{merge_well_dir}: {TEMPLATES_FILENAME} has {templates.shape[1]} "
            f"channel column(s) but {LOCATIONS_FILENAME} has "
            f"{channel_locations_xy.shape[0]} location(s) -- merge_templates' "
            "output is internally inconsistent"
        )

    return {
        "unit_ids": unit_ids,
        "templates": templates,
        "contributing_weight": contributing_weight,
        "channel_locations_xy": channel_locations_xy,
        "fs": fs,
    }


def load_unit_inputs(well_inputs, unit_id):
    """`(template, locations, fs)` for one unit, sliced from `well_inputs`.

    **NaN-handling policy.** A `(unit, channel)` entry no segment ever
    measured for that unit is `NaN` in `merged_templates.npy`, with
    `contributing_weight` at exactly `0.0` there. This restricts each unit's
    slice to channels where `contributing_weight[unit] > 0` -- i.e. DROPS
    unmeasured channels rather than zero-filling (which would assert a unit
    was measured at zero amplitude on an electrode it never routed through)
    or leaving NaN in place (which downstream numeric code has no
    NaN-handling for). Different units therefore end up on different,
    non-uniform channel subsets — the scientifically honest state of the
    data, not an artifact of this slicing.

    Raises `ValueError` if the unit has zero covered channels (never
    measured on any channel) -- nothing to plot or track for it.
    """
    unit_ids = well_inputs["unit_ids"]
    u_idx = unit_ids.index(str(unit_id))

    covered = well_inputs["contributing_weight"][u_idx] > 0.0
    template = well_inputs["templates"][u_idx, covered, :]
    locations = well_inputs["channel_locations_xy"][covered, :]
    if template.shape[0] == 0:
        raise ValueError(
            f"unit {unit_id!r}: zero covered channel(s) in the merged template "
            "(no segment ever measured this unit on any channel) -- nothing to "
            "read"
        )
    return template, locations, well_inputs["fs"]


__all__ = [
    "discover_unit_ids",
    "load_well_inputs",
    "load_unit_inputs",
    "TEMPLATES_FILENAME",
    "WEIGHT_FILENAME",
    "LOCATIONS_FILENAME",
    "UNIT_IDS_FILENAME",
    "MANIFEST_FILENAME",
    "FS_MANIFEST_KEY",
]
