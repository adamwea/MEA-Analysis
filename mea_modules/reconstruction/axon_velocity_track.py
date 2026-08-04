"""Run axon_velocity's graph-based tracker on one unit's full-array template.

This is the terminal step of the recovery path: earlier stages buy back the
electrodes a segment-bound sort never saw (`mea_modules.registration`) and
merge per-segment templates into one array-wide template per unit (a sibling
stage, not this module). Handed that — `(template, locations, fs)` for a
single unit — this module's only job is to call `axon_velocity`'s own
graph tracker correctly and persist what it finds.

**Why a wrapper at all**, when `axon_velocity.compute_graph_propagation_velocity`
is already one function call: two reasons, both about staying honest across an
upstream dependency this lab does not control.

1. **The library's own defaults are not this lab's defaults.** The values in
   :data:`_PRODUCTION_DEFAULTS` come from the pre-rebuild pipeline's tuned
   config (`axon_reconstructor__loop-cleanup`), validated against this lab's
   real recordings — e.g. `min_path_length=17.5` um vs the library's own 100,
   `detect_threshold=0.01` vs 0.1. Calling the bare library function would
   silently swap back to untuned values.
2. **A future axon_velocity upgrade should fail loudly, not quietly.** Kwargs
   are filtered against `inspect.signature` before the call: an override the
   caller passed on purpose that the installed version no longer accepts
   raises immediately (:class:`TypeError`), but a *default* this module carries
   that the installed version happens not to accept is dropped without
   complaint. The two cases look identical in the merged dict and must not be
   treated the same — one is the caller's mistake to see, the other is this
   module's own version-skew to absorb.

Persistence (`save_reconstruction`) is deliberately simple relative to the old
build, which wrote one JSON file per diagnostic stage (`branches_raw.json`,
`kurtosis_filter.json`, `channel_selection.json`, ...). That per-stage detail
lived on `GraphAxonTracking`'s private `_paths_raw` / `_selected_channels_sorted`
attributes, which are not part of any documented contract. This module instead
pickles the whole `gtr` object — every one of those attributes survives, so
nothing is actually lost — and additionally writes ONE
`reconstruction_summary.json` covering the public, documented surface
(`.branches`, `.selected_channels`, `.init_channel`) for callers that want a
quick look without unpickling. A single combined summary file, not a
directory of per-diagnostic files, is the simplification: it is a reasonable
default until a real consumer asks for one of those private stages back.
"""

import inspect
import json
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

# Recovered from the pre-rebuild build's tuned config
# (`axon_reconstructor__loop-cleanup/.../integrations/axon_velocity.py`, the
# params dict its capsule actually called with) rather than
# `axon_velocity.get_default_graph_velocity_params()`'s own defaults, which are
# untuned library values. These are what was validated on this lab's real MEA
# recordings; treat this dict, not the library's, as the starting point for any
# override.
_PRODUCTION_DEFAULTS = {
    "upsample": 1,
    "init_delay": 0.1,
    "min_points_after_branching": 2,
    "min_path_length": 17.5,
    "min_path_points": 4,
    "max_peak_latency_for_splitting": 0.25,
    "kurt_threshold": 0.3,
    "peak_std_threshold": 1.0,
    "peak_std_distance": 30,
    "n_neighbors": 8,
    "distance_exp": 2,
    "remove_isolated": False,
    "detection_type": "relative",
    "detect_threshold": 0.01,
    "min_selected_points": 30,
    "r2_threshold": 0.85,
    "max_distance_for_edge": 100,
    "max_distance_to_init": 200,
    "mad_threshold": 10,
    "init_amp_peak_ratio": 0.2,
    "edge_dist_amp_ratio": 0.3,
    "r2_threshold_for_outliers": 0.98,
    "min_outlier_tracking_error": 50,
    "neighbor_radius": 100,
    "split_paths": True,
    "theilsen_maxiter": 8000,
}

# Positional in axon_velocity's own signature; never accepted as an override
# kwarg here, since `track_unit_axon` already takes them as its own positional
# arguments. Excluded from kwarg filtering so an override accidentally named
# `fs` (say) cannot collide with the positional `fs` this module passes and
# raise "got multiple values for argument".
_POSITIONAL_ARGS = frozenset({"template", "locations", "fs"})

GTR_FILENAME = "gtr.pkl"
SUMMARY_FILENAME = "reconstruction_summary.json"


def default_params():
    """A fresh copy of this lab's production `axon_velocity` params.

    A plain `dict`, copied on every call, so a caller mutating the return
    value (e.g. ``p = default_params(); p["r2_threshold"] = 0.9``) never
    touches the module's own defaults.
    """
    return dict(_PRODUCTION_DEFAULTS)


def _accepted_kwargs(fn):
    """Keyword names `fn` will accept, minus the ones passed positionally."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        # Signature introspection can fail on some builtins/C extensions;
        # nothing here should hard-fail because of it. Fall back to "nothing
        # filtered" would risk quietly passing garbage through, so instead
        # fall back to the default keys only.
        return set(_PRODUCTION_DEFAULTS)
    return set(params) - _POSITIONAL_ARGS


def track_unit_axon(template, locations, fs, **overrides):
    """Track one unit's axon: thin, honest wrapper over `axon_velocity`.

    `template` is `(n_channels, n_samples)` in uV, `locations` is
    `(n_channels, 2)` in um, `fs` is the sampling rate in Hz — the exact
    contract `axon_velocity.compute_graph_propagation_velocity` documents.

    `overrides` are merged onto :func:`default_params`, then filtered to the
    keyword arguments the installed `axon_velocity` actually accepts. An
    override key that is NOT accepted raises `TypeError` immediately — the
    caller asked for something specific and silently dropping it would hide a
    real mistake (a typo, or a param renamed upstream). A *default* key that
    is not accepted (the installed `axon_velocity` version dropped or renamed
    a param this module still carries) is dropped without complaint — that is
    this module's own version-skew to absorb, not the caller's problem.

    Returns the `GraphAxonTracking` instance `compute_graph_propagation_velocity`
    itself returns (not a separate result wrapper) — already run through
    `select_channels()` -> `build_graph()` -> `find_paths()` -> `clean_paths()`.
    """
    import axon_velocity as av

    merged = default_params()
    merged.update(overrides)

    accepted = _accepted_kwargs(av.compute_graph_propagation_velocity)
    unknown_overrides = sorted(set(overrides) - accepted)
    if unknown_overrides:
        raise TypeError(
            f"track_unit_axon() got override(s) axon_velocity's "
            f"compute_graph_propagation_velocity does not accept: "
            f"{unknown_overrides} (installed axon_velocity version may have "
            "renamed or dropped these params)"
        )

    filtered = {k: v for k, v in merged.items() if k in accepted}
    dropped_defaults = sorted(set(merged) - set(filtered))
    if dropped_defaults:
        logger.warning(
            "%d production default(s) not accepted by the installed "
            "axon_velocity and dropped: %s",
            len(dropped_defaults), dropped_defaults,
        )

    logger.info(
        "tracking axon: %d channel(s), %d sample(s), fs=%.1f Hz",
        len(template), template.shape[1] if hasattr(template, "shape") else -1, float(fs),
    )
    gtr = av.compute_graph_propagation_velocity(template, locations, float(fs), **filtered)
    logger.info(
        "tracked %d branch(es) from %d selected channel(s) (init channel %s)",
        len(gtr.branches), len(gtr.selected_channels), gtr.init_channel,
    )
    return gtr


def _to_jsonable(value):
    """Recursively convert numpy arrays/scalars to plain JSON-safe Python.

    `axon_velocity`'s branch dicts hold a mix of numpy arrays (`channels`,
    `distances`, `peak_times`) and numpy scalar types (`velocity`, `r2`,
    `pval`, `raw_path_idx` are frequently `np.float64`/`np.int64`, depending
    on which internal estimator produced them) — neither survives
    `json.dumps` unconverted. `.tolist()` / `.item()` on ndarray/numpy-scalar
    duck-types rather than importing numpy here, since this helper is called
    from `save_reconstruction` which already needs numpy for nothing else.
    """
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "tolist"):  # numpy ndarray
        return value.tolist()
    if hasattr(value, "item"):  # numpy scalar (float64, int64, bool_, ...)
        return value.item()
    return value


def save_reconstruction(gtr, out_dir, unit_id=None):
    """Persist a tracked `gtr`: the full object pickled, plus a JSON summary.

    Writes `out_dir/gtr.pkl` (the whole `GraphAxonTracking` instance — every
    branch, selected channel, and internal diagnostic attribute, unpicklable
    back into a live object for further use with `axon_velocity`'s own
    plotting) and `out_dir/reconstruction_summary.json` (the documented public
    surface only — `.branches`, `.selected_channels`, `.init_channel` — for a
    caller that wants unit counts or branch geometry without unpickling
    anything, and for the capsule's own logging).

    Returns the summary dict (the same content written to the JSON file), so
    a caller doesn't have to re-read the file it was just given.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gtr_path = out_dir / GTR_FILENAME
    with open(gtr_path, "wb") as fh:
        pickle.dump(gtr, fh)

    summary = {
        "unit_id": unit_id,
        "n_branches": len(gtr.branches),
        "n_selected_channels": int(len(gtr.selected_channels)),
        "init_channel": _to_jsonable(gtr.init_channel),
        "selected_channels": _to_jsonable(gtr.selected_channels),
        "branches": [_to_jsonable(branch) for branch in gtr.branches],
    }

    summary_path = out_dir / SUMMARY_FILENAME
    summary_path.write_text(json.dumps(summary, indent=2))

    logger.info(
        "saved reconstruction%s: %d branch(es), %d selected channel(s) -> %s",
        f" for unit {unit_id}" if unit_id is not None else "",
        summary["n_branches"], summary["n_selected_channels"], out_dir,
    )
    return summary


__all__ = [
    "default_params",
    "track_unit_axon",
    "save_reconstruction",
    "GTR_FILENAME",
    "SUMMARY_FILENAME",
]
