"""Closed-form per-channel dense merge of stitched templates, plus spike dedup.

The math (the round-2 planning note's TOPOLOGY-RESOLVED block). Capsule
`10 stitch_templates` computes, per (unit, channel),

    T[u, c] = sum_s(w_s[u] * template_s[u, c]) / sum_s(w_s[u])

over the segments that routed channel `c` and saw unit `u` fire, and emits
`contributing_weight[u, c] = sum_s(w_s[u])` alongside. Merging units
`G = {u1, u2, ...}` (SLAy's verdict that they are one neuron) is then

    T_merged[c] = sum_{u in G}(T[u, c] * W[u, c]) / sum_{u in G}(W[u, c])
    W_merged[c] = sum_{u in G}(W[u, c])

**Exact under `spike_count` averaging** -- substituting the stitch formula
shows T_merged[c] equals the spike-count-weighted average over ALL (unit,
segment) contributions that measured channel c, i.e. exactly what
re-stitching the combined unit from the per-segment analyzers would produce
(and, under spike_count, what re-sorting the pooled selected spikes would
average). The weight algebra needs `W` to be the summed per-(unit, channel)
spike counts; under `uniform` averaging `contributing_weight` is a segment
COUNT instead, the substitution no longer telescopes, and this merge is an
approximation -- which is why capsule 18 asserts the stitch manifest records
`averaging_method == "spike_count"` before applying (round-2 revision R14:
the stitch default was flipped to `spike_count` for exactly this reason).

A channel only some members cover comes out as those members' weighted value
undiluted -- the divergent-coverage soma-axon case SLAy's own (now deleted)
global weighted average corrupted, and the reason this module exists. A
channel no member covers stays NaN with weight 0.0, preserving the stitch's
"no data is not zero" convention.

Pure library: numpy only, no SpikeInterface, no disk, no argparse. Capsule
`18 dense_merge_apply` streams these functions over memory-mapped arrays.
"""

import numpy as np

# Upstream SLAy's suggested censor period was censor_ms = 5/30000 s -- five
# SAMPLES at its native 30 kHz. The sample count, not the millisecond value,
# is the invariant to carry across platforms (20 kHz KCNT1 reference, 10 kHz
# FA MaxTwo plate): 5 samples everywhere, a different absolute window.
DEFAULT_CENSOR_SAMPLES = 5


def merge_group_dense(templates, contributing_weight, member_indices):
    """Merge one group of units' dense templates, closed-form per channel.

    Parameters
    ----------
    templates : array-like (n_units, n_channels, n_samples)
        Stitched dense templates, NaN where a (unit, channel) has no
        coverage. May be a numpy memmap; only `member_indices` rows are
        materialized.
    contributing_weight : array-like (n_units, n_channels)
        The stitch's summed per-(unit, channel) weight, 0.0 exactly where
        the template is NaN.
    member_indices : sequence of int
        Row indices of the group's member units. At least one.

    Returns
    -------
    merged_template : (n_channels, n_samples) float64
        NaN on channels no member covers.
    merged_weight : (n_channels,) float64
        Summed member weight per channel (0.0 where merged_template is NaN).

    Raises
    ------
    ValueError
        On empty groups, shape disagreement, or a member whose NaN pattern
        contradicts its weights (NaN template where weight > 0, or finite
        template where weight == 0) -- that inconsistency means the inputs
        are not a matched (templates, contributing_weight) pair from one
        stitch, and merging them would silently corrupt.
    """
    member_indices = [int(index) for index in member_indices]
    if not member_indices:
        raise ValueError("merge_group_dense: empty member_indices")

    group_templates = np.asarray(templates[member_indices], dtype=np.float64)
    group_weights = np.asarray(contributing_weight[member_indices], dtype=np.float64)

    if group_templates.ndim != 3:
        raise ValueError(
            f"templates rows must be (n_channels, n_samples); got a stacked "
            f"shape of {group_templates.shape}"
        )
    if group_weights.shape != group_templates.shape[:2]:
        raise ValueError(
            f"contributing_weight rows {group_weights.shape} do not match "
            f"templates rows {group_templates.shape[:2]}"
        )
    if np.any(group_weights < 0):
        raise ValueError("contributing_weight has negative entries")

    nan_mask = np.isnan(group_templates).any(axis=2)  # (k, n_channels)
    covered_mask = group_weights > 0
    # NaN <-> zero-weight must agree per member channel: the stitch writes
    # them together, so disagreement means mismatched inputs, not data.
    mismatch = nan_mask == covered_mask
    if np.any(mismatch):
        member, channel = np.argwhere(mismatch)[0]
        raise ValueError(
            f"member row {member_indices[member]} channel {channel}: template "
            f"NaN={bool(nan_mask[member, channel])} but weight="
            f"{group_weights[member, channel]} -- templates and "
            "contributing_weight are not a matched pair from one stitch"
        )

    weight_total = group_weights.sum(axis=0)  # (n_channels,)
    # Zero out uncovered members' NaN rows so they add nothing; covered
    # members contribute template * weight.
    contributions = np.where(
        covered_mask[:, :, None], np.nan_to_num(group_templates, nan=0.0), 0.0
    ) * group_weights[:, :, None]
    merged = contributions.sum(axis=0)

    covered = weight_total > 0
    merged[covered] /= weight_total[covered, None]
    merged[~covered] = np.nan
    return merged, weight_total


def plan_merge_output(unit_ids, groups):
    """Validate a merge map against a unit-id list and plan the output set.

    Parameters
    ----------
    unit_ids : sequence
        Unit ids in stitch row order (the unit axis of the template arrays).
    groups : sequence of sequences
        SLAy merge groups, unit IDS (not indices), disjoint.

    Returns
    -------
    plan : list of dict, one per OUTPUT unit, in a deterministic order:
        input order, with a merged group appearing at the position of its
        first-encountered member (later members are consumed silently).
        Each entry: {"unit_id": survivor id (the group's LOWEST member id;
        input id for passthrough units), "member_ids": [...],
        "member_indices": [...], "merged": bool}.

    Raises
    ------
    ValueError
        On a group member missing from `unit_ids`, appearing in two groups,
        appearing twice in one group, or a group of fewer than two units.
    """
    index_of = {}
    for index, unit_id in enumerate(unit_ids):
        index_of[_id_key(unit_id)] = index

    member_to_group = {}
    normalized_groups = []
    for group_number, group in enumerate(groups):
        members = [_id_key(unit) for unit in group]
        if len(members) < 2:
            raise ValueError(
                f"merge group {group_number} has {len(members)} member(s); "
                "a merge needs at least two"
            )
        if len(set(members)) != len(members):
            raise ValueError(f"merge group {group_number} repeats a unit id: {group}")
        for member in members:
            if member not in index_of:
                raise ValueError(
                    f"merge group {group_number} names unit {member!r} which is "
                    "not in the unit-id list -- was the merge map produced from "
                    "a different unit set?"
                )
            if member in member_to_group:
                raise ValueError(
                    f"unit {member!r} appears in merge groups "
                    f"{member_to_group[member]} and {group_number}; groups must "
                    "be disjoint"
                )
            member_to_group[member] = group_number
        normalized_groups.append(members)

    plan = []
    emitted_groups = set()
    for unit_id in unit_ids:
        key = _id_key(unit_id)
        group_number = member_to_group.get(key)
        if group_number is None:
            plan.append({
                "unit_id": unit_id,
                "member_ids": [unit_id],
                "member_indices": [index_of[key]],
                "merged": False,
            })
            continue
        if group_number in emitted_groups:
            continue
        emitted_groups.add(group_number)
        members = normalized_groups[group_number]
        plan.append({
            "unit_id": min(members),
            "member_ids": list(members),
            "member_indices": [index_of[member] for member in members],
            "merged": True,
        })
    return plan


def _id_key(unit_id):
    """Unit ids as comparable keys: ints where possible, else strings.

    Stitch `unit_ids.json`, SLAy merge maps and SpikeInterface sortings all
    carry KS4's integer unit ids, but json round-trips and numpy scalar
    types can shuffle int/str/np.int64 representations; normalizing here
    keeps `17`'s map joinable against `10`'s row order without caring which
    serializer touched which file.
    """
    if hasattr(unit_id, "item"):
        unit_id = unit_id.item()
    try:
        return int(unit_id)
    except (TypeError, ValueError):
        return str(unit_id)


def dedup_coincident_spikes(spike_times, spike_sources, censor_samples=DEFAULT_CENSOR_SAMPLES):
    """Drop cross-unit duplicate spikes after a merge, keep-first.

    When oversplit fragments of one neuron are recombined, the same physical
    spike can appear once per fragment a few samples apart. Old SLAy removed
    these with `remove_duplicate_spikes` (censor period 5 samples at its
    native rate); the re-implementation dropped that step, so it is
    replicated here.

    A spike is dropped iff it falls within `censor_samples` samples
    (inclusive) of the most recent KEPT spike AND comes from a DIFFERENT
    source unit. Same-source spikes are never dropped -- refractory
    violations inside one sorter-defined unit are the sorter's output, not
    this function's to edit.

    Parameters
    ----------
    spike_times : array-like of int
        Spike times in SAMPLES, any order (sorted internally).
    spike_sources : array-like
        Source unit id per spike (which pre-merge fragment it came from).
    censor_samples : int, default DEFAULT_CENSOR_SAMPLES
        Window in samples; 5 matches upstream's censor_ms=5/30000 at 30 kHz.

    Returns
    -------
    keep : (n_spikes,) bool, aligned to the INPUT order.
    """
    spike_times = np.asarray(spike_times)
    spike_sources = np.asarray(spike_sources)
    if spike_times.shape != spike_sources.shape:
        raise ValueError(
            f"{spike_times.shape[0]} spike time(s) but "
            f"{spike_sources.shape[0]} source id(s)"
        )
    censor_samples = int(censor_samples)
    if censor_samples < 0:
        raise ValueError("censor_samples must be >= 0")

    order = np.argsort(spike_times, kind="stable")
    keep_sorted = np.ones(len(order), dtype=bool)
    last_kept_time = None
    last_kept_source = None
    for position in range(len(order)):
        index = order[position]
        time = spike_times[index]
        source = spike_sources[index]
        if (
            last_kept_time is not None
            and time - last_kept_time <= censor_samples
            and source != last_kept_source
        ):
            keep_sorted[position] = False
            continue
        last_kept_time = time
        last_kept_source = source

    keep = np.empty(len(order), dtype=bool)
    keep[order] = keep_sorted
    return keep


__all__ = [
    "DEFAULT_CENSOR_SAMPLES",
    "dedup_coincident_spikes",
    "merge_group_dense",
    "plan_merge_output",
]
