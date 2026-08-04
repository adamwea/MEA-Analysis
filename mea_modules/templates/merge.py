"""Fold N per-segment templates into one full-array template, one unit at a time.

Capsule `register_segment` re-indexes the backbone sort onto each segment's own
clock and builds a bare `SortingAnalyzer` over that segment's FULL native
electrode set (hundreds to low-thousands of channels), with no extensions
computed. That analyzer only sees ITS segment's channels; the array only shows
up once the segments are put back together. This module is that "back
together" step, done per unit and per channel:

    T_true[u, c] = sum(weight_s * template_s[u, c] for s in segments that routed c)
                   / sum(weight_s for s in segments that routed c)

The divisor is never the segment count and never a global constant — it only
grows for a (unit, channel) pair when a segment actually measured that unit on
that channel. That is what makes a channel routed by one segment come out
exactly equal to that segment's own value (weight cancels top and bottom), and
a channel routed by three come out as their weighted average, with no dilution
from segments that never touched it. A hand-written union recording cannot get
this for free — SpikeInterface's `SortingAnalyzer` recomputes `templates` from
whatever recording it is given, at one divisor per UNIT (not per channel), so
feeding it a zero-padded union and asking it to average dilutes every padded
channel by the fraction of segments that did not route it (the failure that
retired the `synthesize_sorting` / union+rescale route — see this repo's
`Plans.md`, "Phase 2b"). Reading the per-segment analyzers directly and doing
the divisor by hand, per channel, is what this module buys instead.

Two layers, split on purpose:

* :func:`new_unit_accumulator`, :func:`accumulate_channel_contributions`,
  :func:`finalize_unit_accumulator` — plain-numpy bucket arithmetic with no
  SpikeInterface dependency. Given a segment's channel ids, locations, a
  unit's per-channel template on those channels, and a scalar weight, they
  fold one contribution into a running per-unit accumulator and, at the end,
  turn it into arrays. Nothing here reads a file or knows what a
  SortingAnalyzer is, so this arithmetic is tested in isolation.
* :func:`merge_segment_templates` — the streaming orchestrator. It is the only
  function in this module that touches SpikeInterface or disk.

**Streaming, not batch — the point of this rebuild.** The old (pre-rebuild)
merge held every segment's analyzer alive at once in a
`list[tuple[str, Any]]` before merging a single unit
(`materialize_unit_templates_by_unit_with_meta`, ~600 lines in, in the retired
`axon_recon.pipeline.stages.reconstruct.templates.core.merge`). On a real well
— tens of segments, each a dense analyzer over low-thousands of channels — that
is tens of GB resident at once for no reason: every segment's contribution to
the merge is fully consumed the moment it is folded into the accumulators.
`merge_segment_templates` below opens exactly one segment's analyzer at a time
(`mea_modules.postprocess.load_analyzer`), computes that analyzer's `templates`
extension on demand (`register_segment` deliberately leaves it uncomputed —
see its module docstring), folds every unit's contribution from it into the
(small) running per-unit accumulators, and then drops every reference to that
analyzer — see the `finally: del analyzer` in the loop — before opening the
next one. Peak memory is ONE segment's dense templates buffer plus the
accumulators (which grow by a few `(n_samples,)` float rows per unit per
segment, not by a whole analyzer), regardless of how many segments a well has.

**Simplification versus the old build, made deliberately.** The old build
matched channels across sources by a three-tier priority — electrode_id, then
channel_id, then a quantized xy location — because its sources could disagree
on identity. Here they cannot: every per-segment analyzer is built straight off
a Maxwell recording (`mea_modules.io.load_segment`), and on Maxwell the routed
channel id already *is* the electrode id (see
`mea_modules.registration.coverage`'s own note to the same effect). So this
module matches channels by `analyzer.channel_ids` alone, with no separate
electrode-id field and no location-tolerance fallback. If a future segment
source ever carries channel ids that are not electrode-stable, that assumption
breaks here first and loudly (a channel_id collision that should not exist), not
silently.

Pure library: no argparse, no printing, no ``__main__``.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Waveform window used when a segment analyzer's own `templates` extension has
# to be computed here (register_segment leaves it uncomputed by design). Match
# mea_modules.postprocess.analyzer's defaults so a merge is on the same window
# build_analyzer would have used, had the capsule computed it itself.
DEFAULT_MS_BEFORE = 1.0
DEFAULT_MS_AFTER = 2.0

# Spike cap for the `random_spikes` selection each segment's `templates` pass
# needs. Matches mea_modules.postprocess.analyzer.DEFAULT_MAX_SPIKES_PER_UNIT;
# kept as its own constant rather than imported so this module's pure layer
# never has to import that one (see the deferred import in
# merge_segment_templates for why).
DEFAULT_MAX_SPIKES_PER_UNIT = 500
DEFAULT_SEED = 0

WEIGHTING_MODES = ("uniform", "spike_count")


def _channel_sort_key(channel_id):
    """Sort key that orders numeric-looking ids numerically, others lexically.

    Maxwell channel ids are unpadded digit strings (``'0'..'13383'``), where a
    plain string sort puts ``'10'`` before ``'2'``. Falling back to the raw
    string for anything that does not parse as an int keeps a mixed or
    symbolic id scheme sorting deterministically instead of raising.
    """
    try:
        return (0, int(channel_id))
    except (TypeError, ValueError):
        return (1, str(channel_id))


def new_unit_accumulator():
    """A fresh streaming accumulator for one unit's cross-segment channel merge.

    A plain ``{channel_id: [weighted_wave_sum, weight_sum, xy]}`` dict —
    deliberately not a class. The orchestrator below holds one of these PER
    UNIT for the entire run (never one per segment), so it has to stay small
    and easy to reason about; a dict of numpy rows is both.
    """
    return {}


def accumulate_channel_contributions(
    accumulator, *, channel_ids, locations_xy, template_ch_by_t, weight,
):
    """Fold ONE segment's per-unit contribution into `accumulator`, in place.

    `template_ch_by_t` is that segment's `(n_channels, n_samples)` average
    waveform for this unit, on `channel_ids` (in the same row order), at their
    `locations_xy`. `weight` is a single scalar applied to every one of those
    channels — see :func:`merge_segment_templates` for how it is chosen
    (uniform 1.0, or that segment's spike count for this unit).

    A `weight` of zero or less is a no-op: the caller is expected to skip the
    call entirely when a unit had no spikes in a segment (its stored template
    row is then undefined, not a real zero — see the orchestrator's own guard),
    but treating a non-positive weight as a no-op here too makes this function
    safe to call unconditionally if a caller does not want to special-case it.

    A channel this segment did not route never enters the dict. That is the
    whole mechanism behind "the divisor only grows for segments that actually
    measured this channel" — there is no zero-fill anywhere in this function.
    """
    if weight is None or weight <= 0:
        return

    channel_ids = list(channel_ids)
    template_ch_by_t = np.asarray(template_ch_by_t, dtype=np.float64)
    locations_xy = np.asarray(locations_xy, dtype=np.float64)
    if template_ch_by_t.shape[0] != len(channel_ids):
        raise ValueError(
            f"template has {template_ch_by_t.shape[0]} channel row(s) but "
            f"{len(channel_ids)} channel_id(s) were given"
        )
    if locations_xy.shape[0] != len(channel_ids):
        raise ValueError(
            f"{locations_xy.shape[0]} location(s) given for {len(channel_ids)} "
            "channel_id(s)"
        )

    weight = float(weight)
    for row, channel_id in enumerate(channel_ids):
        wave = template_ch_by_t[row, :] * weight
        xy = locations_xy[row, :2]
        existing = accumulator.get(channel_id)
        if existing is None:
            accumulator[channel_id] = [wave, weight, xy]
            continue
        existing_wave, existing_weight, existing_xy = existing
        if existing_wave.shape != wave.shape:
            # Two segments disagreeing on n_samples means they disagree on the
            # ms_before/ms_after window their `templates` extension was
            # computed with. Silently truncating one to fit the other would
            # blur the response it actually captured, so this is left to
            # surface as a hard shape error rather than papered over.
            raise ValueError(
                f"channel {channel_id!r}: incoming template has "
                f"{wave.shape[0]} sample(s) but the accumulator already holds "
                f"{existing_wave.shape[0]}; segments disagree on the waveform "
                "window and cannot be merged"
            )
        existing_wave += wave
        existing[1] = existing_weight + weight


def finalize_unit_accumulator(accumulator):
    """Turn one unit's finished accumulator into arrays.

    Returns `(channel_ids, template_ch_by_t, locations_xy, weights)`:

    * `channel_ids` — sorted (see :func:`_channel_sort_key`) list of every
      channel that had at least one contributing segment for this unit,
    * `template_ch_by_t` — `(n_channels, n_samples)`, each row divided by its
      own accumulated weight — the weighted average, computed "by
      construction" rather than by dividing by a fixed N,
    * `locations_xy` — `(n_channels, 2)`, aligned to `channel_ids`,
    * `weights` — `(n_channels,)`, the summed weight behind each row (useful
      as a per-channel confidence / coverage diagnostic).

    An empty accumulator (a unit with zero spikes on every segment it was
    given) returns four length-0 arrays rather than raising, so a caller can
    decide per-unit whether that is fatal.
    """
    if not accumulator:
        return [], np.zeros((0, 0), dtype=np.float64), np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    channel_ids = sorted(accumulator.keys(), key=_channel_sort_key)
    n_samples = next(iter(accumulator.values()))[0].shape[0]
    template_ch_by_t = np.zeros((len(channel_ids), n_samples), dtype=np.float64)
    locations_xy = np.zeros((len(channel_ids), 2), dtype=np.float64)
    weights = np.zeros((len(channel_ids),), dtype=np.float64)
    for i, channel_id in enumerate(channel_ids):
        wave_sum, weight_sum, xy = accumulator[channel_id]
        # max(1e-12, ...) only guards a divide-by-zero that should be
        # unreachable (every entry got here via a positive weight); it is not
        # a smoothing term, just a refusal to ever raise ZeroDivisionError on
        # what should be dead code.
        template_ch_by_t[i, :] = wave_sum / max(1e-12, weight_sum)
        locations_xy[i, :] = xy
        weights[i] = weight_sum
    return channel_ids, template_ch_by_t, locations_xy, weights


def merge_segment_templates(
    segment_analyzer_dirs,
    *,
    well=None,
    weighting="uniform",
    unit_ids=None,
    ms_before=DEFAULT_MS_BEFORE,
    ms_after=DEFAULT_MS_AFTER,
    max_spikes_per_unit=DEFAULT_MAX_SPIKES_PER_UNIT,
    random_spikes_seed=DEFAULT_SEED,
):
    """Stream-merge N per-segment `SortingAnalyzer`s into one full-array template set.

    Opens ONE analyzer at a time (`mea_modules.postprocess.load_analyzer`),
    computes its `templates` extension if it is not already there (it is not,
    on a fresh `register_segment` output — see that capsule's module
    docstring), extracts every unit's per-channel contribution, folds it into
    small running per-unit accumulators, then drops the analyzer before
    opening the next `segment_analyzer_dirs` entry. See the module docstring
    for why this matters: the old build held every segment's analyzer alive
    at once, which does not fit in memory at real scale. The `finally: del
    analyzer` at the bottom of the loop is that invariant made explicit — a
    regression back to collecting analyzers into a list before merging would
    have to remove it.

    `weighting`:

    * `"uniform"` — every contributing segment counts equally (weight 1.0),
      i.e. a plain mean across segments that saw the unit on that channel.
      This is also what the published per-configuration protocol this pipeline
      otherwise follows uses (Buccino et al. 2022: per-config estimate, then
      an UNWEIGHTED mean across configs) — the default here for that reason.
    * `"spike_count"` — a segment's contribution is weighted by how many of
      that unit's spikes actually went into ITS template average there (the
      `random_spikes`-selected count, read via
      `mea_modules.postprocess.unit_random_spike_count`), so a segment that
      saw the unit fire 400 times counts for more than one where it fired
      twice.

    A unit with zero spikes in a given segment contributes NOTHING from that
    segment, for ANY channel, regardless of `weighting` — the template row
    SpikeInterface would report there is not a real zero, it is "no
    measurement", and folding it in (even at weight 1.0) would pull every
    channel that segment routed toward zero for no reason. This is the same
    "only segments that actually routed a channel contribute to its average"
    principle applied to units, not just channels.

    The returned `templates` array is `(n_units, n_channels_union, n_samples)`,
    where `n_channels_union` is every channel id ANY segment routed — the
    full-array claim in the module docstring. A `(unit, channel)` entry that no
    segment ever measured for that unit (channel routed by some OTHER segment
    only, or by this one for a DIFFERENT unit) is `NaN`, with the matching
    `contributing_weight` entry at `0.0` — so "no data" is distinguishable from
    "measured and happened to average near zero" by construction, never guessed
    at downstream.

    Returns a dict:

        {
            "well": well,
            "unit_ids": [...],                         # length n_units
            "channel_ids": [...],                       # length n_channels_union, sorted
            "channel_locations_xy": (n_channels_union, 2) float64,
            "templates": (n_units, n_channels_union, n_samples) float64, NaN where uncovered
            "contributing_weight": (n_units, n_channels_union) float64, 0.0 where uncovered
            "n_segments": int,
            "weighting": weighting,
            "ms_before": float,
            "ms_after": float,
        }

    Raises `ValueError` if `segment_analyzer_dirs` is empty, if any analyzer is
    sparse (`register_segment`'s contract is `sparse=False` — see its module
    docstring; a sparsity mask would silently hide channels this merge needs to
    see), if two segments disagree on the unit id set (a sign they were not
    registered from the same backbone sort — see
    `mea_modules.registration.segments.register_sorting_to_segment`), or if not
    one unit had a single spike across every segment given.
    """
    if weighting not in WEIGHTING_MODES:
        raise ValueError(f"weighting must be one of {WEIGHTING_MODES}, got {weighting!r}")

    segment_analyzer_dirs = [Path(p) for p in segment_analyzer_dirs]
    if not segment_analyzer_dirs:
        raise ValueError("merge_segment_templates: no segment analyzer directories given")

    # Deferred on purpose: this is the only function in the module that needs
    # SpikeInterface / mea_modules.postprocess. The pure accumulator functions
    # above stay importable -- and unit-testable -- without either.
    from mea_modules.postprocess import load_analyzer, unit_random_spike_count, unit_template

    well_suffix = f" well={well}" if well else ""
    seen_unit_ids = list(unit_ids) if unit_ids is not None else None
    unit_accumulators = {}
    channel_locations = {}  # channel_id -> xy, EVERY channel any segment routed
    n_segments_used = 0

    for index, seg_dir in enumerate(segment_analyzer_dirs, start=1):
        logger.info(
            "merge_segment_templates%s: opening segment analyzer %d/%d at %s",
            well_suffix, index, len(segment_analyzer_dirs), seg_dir,
        )
        analyzer = load_analyzer(seg_dir)
        try:
            if analyzer.is_sparse():
                raise ValueError(
                    f"{seg_dir}: analyzer is sparse; register_segment's contract is "
                    "sparse=False (see capsules/register_segment/run_capsule.py) -- "
                    "a sparsity mask would silently hide channels this merge needs "
                    "to see"
                )
            if not analyzer.has_extension("random_spikes"):
                analyzer.compute(
                    "random_spikes",
                    method="uniform",
                    max_spikes_per_unit=int(max_spikes_per_unit),
                    seed=int(random_spikes_seed),
                )
            if not analyzer.has_extension("templates"):
                # No `waveforms` extension requested: SpikeInterface then
                # accumulates the average template in place instead of keeping
                # every snippet (see mea_modules.postprocess.analyzer's own
                # note on this), which is what keeps opening one dense,
                # full-array analyzer at a time affordable here.
                analyzer.compute(
                    "templates",
                    ms_before=float(ms_before),
                    ms_after=float(ms_after),
                    operators=["average"],
                )

            these_unit_ids = list(analyzer.unit_ids)
            if seen_unit_ids is None:
                seen_unit_ids = these_unit_ids
                for uid in seen_unit_ids:
                    unit_accumulators[uid] = new_unit_accumulator()
            elif these_unit_ids != seen_unit_ids:
                raise ValueError(
                    f"{seg_dir}: carries {len(these_unit_ids)} unit id(s) that do "
                    f"not match the {len(seen_unit_ids)} seen in earlier segments -- "
                    "register_sorting_to_segment guarantees every segment keeps the "
                    "full backbone unit set (mea_modules.registration.segments), so "
                    "a mismatch means these analyzers were not registered from the "
                    "same backbone sort"
                )

            channel_ids = list(analyzer.channel_ids)
            locations_xy = np.asarray(analyzer.get_channel_locations(), dtype=np.float64)[:, :2]
            for channel_id, xy in zip(channel_ids, locations_xy):
                channel_locations.setdefault(channel_id, xy)

            skipped = 0
            for uid in these_unit_ids:
                spike_count = unit_random_spike_count(analyzer, uid)
                if spike_count <= 0:
                    # No spikes here -> the stored template row is undefined,
                    # not a real zero. Skipping keeps the eventual per-channel
                    # divisor exactly the weight of segments that actually
                    # measured this unit here (see the module docstring).
                    skipped += 1
                    continue
                weight = 1.0 if weighting == "uniform" else float(spike_count)
                template_ch_by_t = np.asarray(unit_template(analyzer, uid), dtype=np.float64).T
                accumulate_channel_contributions(
                    unit_accumulators[uid],
                    channel_ids=channel_ids,
                    locations_xy=locations_xy,
                    template_ch_by_t=template_ch_by_t,
                    weight=weight,
                )
            logger.info(
                "merge_segment_templates%s: %s contributed %d/%d unit(s) "
                "(%d had zero spikes in this segment)",
                well_suffix, seg_dir, len(these_unit_ids) - skipped,
                len(these_unit_ids), skipped,
            )
            n_segments_used += 1
        finally:
            # THE STREAMING INVARIANT (see the module docstring). Every
            # reference to this segment's analyzer -- and the templates
            # buffer `.compute("templates")` just built for it -- is dropped
            # here before the loop opens the next segment. `unit_accumulators`
            # only ever grows by a handful of (n_samples,) float rows per unit
            # per segment, never by a whole analyzer, so peak memory is ONE
            # analyzer's templates buffer plus the running accumulators,
            # regardless of segment count.
            del analyzer

    if seen_unit_ids is None:
        raise ValueError("merge_segment_templates: no segment analyzer directories given")

    union_channel_ids = sorted(channel_locations.keys(), key=_channel_sort_key)
    channel_index = {channel_id: i for i, channel_id in enumerate(union_channel_ids)}
    n_channels_union = len(union_channel_ids)

    n_samples = None
    for uid in seen_unit_ids:
        accumulator = unit_accumulators[uid]
        if accumulator:
            n_samples = next(iter(accumulator.values()))[0].shape[0]
            break
    if n_samples is None:
        raise ValueError(
            f"merge_segment_templates: not one unit had a single spike across any "
            f"of the {n_segments_used} segment(s) given; nothing to merge"
        )

    channel_locations_xy = np.zeros((n_channels_union, 2), dtype=np.float64)
    for channel_id, xy in channel_locations.items():
        channel_locations_xy[channel_index[channel_id]] = xy

    templates = np.full((len(seen_unit_ids), n_channels_union, n_samples), np.nan, dtype=np.float64)
    contributing_weight = np.zeros((len(seen_unit_ids), n_channels_union), dtype=np.float64)

    for u_idx, uid in enumerate(seen_unit_ids):
        channel_ids, template_ch_by_t, _locations_xy, weights = finalize_unit_accumulator(
            unit_accumulators[uid]
        )
        for c_local, channel_id in enumerate(channel_ids):
            c_global = channel_index[channel_id]
            templates[u_idx, c_global, :] = template_ch_by_t[c_local, :]
            contributing_weight[u_idx, c_global] = weights[c_local]

    logger.info(
        "merge_segment_templates%s: merged %d segment(s) -> %d unit(s) x "
        "%d channel(s) union x %d sample(s) (weighting=%s)",
        well_suffix, n_segments_used, len(seen_unit_ids), n_channels_union,
        n_samples, weighting,
    )

    return {
        "well": well,
        "unit_ids": seen_unit_ids,
        "channel_ids": union_channel_ids,
        "channel_locations_xy": channel_locations_xy,
        "templates": templates,
        "contributing_weight": contributing_weight,
        "n_segments": n_segments_used,
        "weighting": weighting,
        "ms_before": float(ms_before),
        "ms_after": float(ms_after),
    }


__all__ = [
    "WEIGHTING_MODES",
    "DEFAULT_MS_BEFORE",
    "DEFAULT_MS_AFTER",
    "DEFAULT_MAX_SPIKES_PER_UNIT",
    "DEFAULT_SEED",
    "new_unit_accumulator",
    "accumulate_channel_contributions",
    "finalize_unit_accumulator",
    "merge_segment_templates",
]
