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

import json
import logging
import math
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

# Supported values for `averaging_method` — how contributions from segments
# sharing a (unit, channel) pair are averaged. The parameter was named
# `weighting` until Round 2; the rename is cosmetic, the arithmetic identical.
#
# "spike_count" is the DEFAULT (Round 2, plan §6 R14): weighting each
# segment's contribution by its selected spike count makes the merged value
# equal a re-derivation from the pooled spikes — which in turn is what makes
# the downstream closed-form dense merge of curated units EXACT (capsule
# `18 dense_merge_apply`: T_merged[c] = Σ tᵤ[c]·wᵤ[c] / Σ wᵤ[c] over the
# stitched outputs — exact only when w IS the spike count). "uniform" (every
# contributing segment counts equally — a plain mean) is kept for parity with
# the published per-configuration protocol (Buccino et al. 2022) and with the
# KCNT1 reference overnight run, whose manifest records `uniform`.
#
# Documented-but-NOT-implemented future methods, deliberately excluded from
# this tuple so asking for one fails loudly instead of silently falling back:
# "amplitude_max" (keep the largest-amplitude segment's waveform per channel,
# no averaging) and "snr_weighted" (weight by per-segment SNR). A "plain
# mean" is not a future method — it is exactly what "uniform" computes.
AVERAGING_METHODS = ("spike_count", "uniform")
DEFAULT_AVERAGING_METHOD = "spike_count"
# Backward-compatible alias (the pre-Round-2 name); same tuple on purpose.
WEIGHTING_MODES = AVERAGING_METHODS


def _jsonable_id(value):
    """A channel/unit id as something `json.dumps` accepts.

    SpikeInterface hands ids back as numpy scalar types (`int64` / `str_`),
    which `json.dumps` raises on; `.item()` unwraps those, and anything
    already JSON-safe passes through untouched.
    """
    return value.item() if hasattr(value, "item") else value


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
    averaging_method=None,
    weighting=None,
    segment_retention_dir=None,
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

    `averaging_method` (`None` resolves to `DEFAULT_AVERAGING_METHOD`,
    which is `"spike_count"`; see `AVERAGING_METHODS` for the rationale and
    the documented-but-unimplemented future methods):

    * `"spike_count"` — THE DEFAULT (Round 2, plan §6 R14). A segment's
      contribution is weighted by how many of that unit's spikes actually
      went into ITS template average there (the `random_spikes`-selected
      count, read via `mea_modules.postprocess.unit_random_spike_count`), so
      a segment that saw the unit fire 400 times counts for more than one
      where it fired twice. Under this weighting the merged value equals a
      re-derivation from the pooled selected spikes — the property that makes
      the downstream closed-form dense merge of curated units exact.
    * `"uniform"` — every contributing segment counts equally (weight 1.0),
      i.e. a plain mean across segments that saw the unit on that channel.
      This is what the published per-configuration protocol this pipeline
      otherwise follows uses (Buccino et al. 2022: per-config estimate, then
      an UNWEIGHTED mean across configs), and what the KCNT1 reference
      overnight run used — it was the default until Round 2 flipped it.

    `weighting` is the DEPRECATED pre-Round-2 alias for `averaging_method`
    (same values, same arithmetic). Passing both with different values
    raises; passing `weighting` alone still works but logs a warning.

    `segment_retention_dir` (default `None` = retain nothing, plan §6 R14②):
    a directory to persist each segment's PER-UNIT template contribution into
    as the merge streams, for `post_stitch_diagnostics`' cross-segment
    consistency checks — without it those inputs die with the
    `finally: del analyzer` below and can only be re-derived by re-opening
    every segment analyzer a second time. Layout, all unit-axis-aligned to
    the returned `unit_ids` (also written there as `unit_ids.json`):

        <dir>/segments_index.json          [{index, dirname, analyzer_dir,
                                             n_channels,
                                             n_units_contributing}, ...]
        <dir>/unit_ids.json
        <dir>/segment_<NNN>/templates.npy      (n_units, n_channels_seg,
                                                n_samples) float32; a unit
                                                with no spikes in this
                                                segment is an all-NaN row,
                                                mirroring the merge's own
                                                skip of that contribution
        <dir>/segment_<NNN>/spike_counts.npy   (n_units,) int64 — the
                                                selected counts, i.e. the
                                                exact spike_count weights
        <dir>/segment_<NNN>/channel_ids.json   this segment's own channels
        <dir>/segment_<NNN>/locations_xy.npy   (n_channels_seg, 2) float64

    Cost, documented so the flag is used knowingly: disk is
    `n_units x n_channels_seg x n_samples x 4` bytes per segment (float32) —
    on the KCNT1 reference well (780 units x ~1,000 ch x 60 samples x 21
    segments) about 190 MB per segment, ~3.9 GB per well, roughly the same
    footprint as the per-segment analyzers themselves. Peak memory is
    unchanged in order: one transient float32 copy of the one open segment's
    dense templates buffer (~190 MB there), made and released inside the
    loop, so the streaming invariant (never more than one segment resident)
    still holds.

    A unit with zero spikes in a given segment contributes NOTHING from that
    segment, for ANY channel, regardless of `averaging_method` — the template row
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
            "averaging_method": averaging_method,
            "ms_before": float,
            "ms_after": float,
            "sampling_frequency_hz": float,
        }

    `sampling_frequency_hz` is read off each segment's own analyzer
    (`analyzer.sampling_frequency`, which SpikeInterface 0.103.2's
    `SortingAnalyzer` forwards from `analyzer.sorting.get_sampling_frequency()`
    — the same attribute `mea_modules.postprocess.analyzer` already reads it
    from) and checked for agreement across every segment opened: every segment
    merged for one well comes from the same physical Maxwell recording, so a
    disagreement is a data-integrity problem, not a rate to average over — see
    the `ValueError` below.

    Raises `ValueError` if `segment_analyzer_dirs` is empty, if any analyzer is
    sparse (`register_segment`'s contract is `sparse=False` — see its module
    docstring; a sparsity mask would silently hide channels this merge needs to
    see), if two segments disagree on the unit id set (a sign they were not
    registered from the same backbone sort — see
    `mea_modules.registration.segments.register_sorting_to_segment`), if two
    segments disagree on `sampling_frequency_hz`, or if not one unit had a
    single spike across every segment given.
    """
    if weighting is not None:
        if averaging_method is not None and averaging_method != weighting:
            raise ValueError(
                f"averaging_method={averaging_method!r} and its deprecated alias "
                f"weighting={weighting!r} disagree; pass averaging_method only"
            )
        logger.warning(
            "merge_segment_templates: `weighting=` is the deprecated pre-Round-2 "
            "name; pass `averaging_method=` instead"
        )
        averaging_method = weighting
    if averaging_method is None:
        averaging_method = DEFAULT_AVERAGING_METHOD
    if averaging_method not in AVERAGING_METHODS:
        raise ValueError(
            f"averaging_method must be one of {AVERAGING_METHODS}, got "
            f"{averaging_method!r} (amplitude_max / snr_weighted are documented "
            "future methods, not implemented)"
        )

    segment_analyzer_dirs = [Path(p) for p in segment_analyzer_dirs]
    if not segment_analyzer_dirs:
        raise ValueError("merge_segment_templates: no segment analyzer directories given")
    if segment_retention_dir is not None:
        segment_retention_dir = Path(segment_retention_dir)
        segment_retention_dir.mkdir(parents=True, exist_ok=True)
    retained_segments = []

    # Deferred on purpose: this is the only function in the module that needs
    # SpikeInterface / mea_modules.postprocess. The pure accumulator functions
    # above stay importable -- and unit-testable -- without either.
    from mea_modules.postprocess import load_analyzer, unit_random_spike_count, unit_template

    well_suffix = f" well={well}" if well else ""
    # Always discovered from the first segment opened, never caller-supplied:
    # an earlier version accepted a `unit_ids=` override, but nothing ever
    # populated `unit_accumulators` for a caller-supplied id (only the
    # "discovered from the first segment" branch below did), so it raised
    # KeyError on first use -- and even fixed, restricting to a SUBSET would
    # have tripped the full-set consistency check below on every segment. A
    # units filter is straightforward to add back later with its own test;
    # today nothing calls this with anything but the full set.
    seen_unit_ids = None
    sampling_frequency_hz = None
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

            # Every segment analyzer merged for one well is built off the same
            # physical Maxwell recording (mea_modules.io.load_segment), so they
            # must all report the same rate. `analyzer.sampling_frequency`
            # forwards `analyzer.sorting.get_sampling_frequency()` (confirmed
            # against SpikeInterface 0.103.2's SortingAnalyzer -- the same
            # attribute mea_modules.postprocess.analyzer already reads). Checked
            # with a tight relative tolerance rather than bare `==` only to
            # absorb harmless float round-trip noise, not to paper over a real
            # disagreement -- see the module's own no-silent-averaging stance on
            # channel/unit disagreement above.
            this_fs = float(analyzer.sampling_frequency)
            if sampling_frequency_hz is None:
                sampling_frequency_hz = this_fs
            elif not math.isclose(this_fs, sampling_frequency_hz, rel_tol=1e-9):
                raise ValueError(
                    f"{seg_dir}: sampling_frequency_hz={this_fs} does not match "
                    f"{sampling_frequency_hz} seen in earlier segments -- every "
                    "segment merged for one well is expected to share one "
                    "sampling rate (same physical recording); a real "
                    "disagreement here is a data-integrity problem worth "
                    "surfacing loudly, not silently averaging over"
                )

            channel_ids = list(analyzer.channel_ids)
            locations_xy = np.asarray(analyzer.get_channel_locations(), dtype=np.float64)[:, :2]
            # `setdefault` keeps the FIRST segment's xy for a channel routed
            # by more than one -- unlike the n_samples mismatch above, this is
            # never cross-checked, because electrode position is a fixed
            # property of the physical probe, not a per-recording measurement:
            # two segments disagreeing on where electrode 42 sits would mean
            # they were not built from the same probe geometry, which is a
            # deeper problem than this function can detect from templates alone.
            for channel_id, xy in zip(channel_ids, locations_xy):
                channel_locations.setdefault(channel_id, xy)

            skipped = 0
            unit_spike_counts = []  # aligned to these_unit_ids; retention reuses it
            for uid in these_unit_ids:
                spike_count = unit_random_spike_count(analyzer, uid)
                unit_spike_counts.append(int(spike_count))
                if spike_count <= 0:
                    # No spikes here -> the stored template row is undefined,
                    # not a real zero. Skipping keeps the eventual per-channel
                    # divisor exactly the weight of segments that actually
                    # measured this unit here (see the module docstring).
                    skipped += 1
                    continue
                weight = 1.0 if averaging_method == "uniform" else float(spike_count)
                template_ch_by_t = np.asarray(unit_template(analyzer, uid), dtype=np.float64).T
                accumulate_channel_contributions(
                    unit_accumulators[uid],
                    channel_ids=channel_ids,
                    locations_xy=locations_xy,
                    template_ch_by_t=template_ch_by_t,
                    weight=weight,
                )
            if segment_retention_dir is not None:
                retained_segments.append(
                    _write_segment_contribution(
                        segment_retention_dir,
                        index=index - 1,
                        seg_dir=seg_dir,
                        analyzer=analyzer,
                        spike_counts=unit_spike_counts,
                        channel_ids=channel_ids,
                        locations_xy=locations_xy,
                    )
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

    # seen_unit_ids and sampling_frequency_hz cannot still be None here: both
    # are set on the loop's first iteration, and segment_analyzer_dirs was
    # already checked non-empty above, and any exception raised inside the
    # loop propagates immediately rather than falling through to this line --
    # so the loop body ran at least once and set them. Asserted, not re-raised
    # as a second ValueError, precisely because it should be unreachable.
    assert seen_unit_ids is not None
    assert sampling_frequency_hz is not None

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

    if segment_retention_dir is not None:
        # Written LAST, after every segment survived the loop: a complete
        # segments_index.json is the marker that the retention set beside it
        # is whole, so a killed run leaves no index rather than a lying one.
        (segment_retention_dir / "unit_ids.json").write_text(
            json.dumps([_jsonable_id(u) for u in seen_unit_ids], indent=2)
        )
        (segment_retention_dir / "segments_index.json").write_text(
            json.dumps(retained_segments, indent=2)
        )
        logger.info(
            "merge_segment_templates%s: retained %d segment contribution(s) "
            "under %s", well_suffix, len(retained_segments), segment_retention_dir,
        )

    logger.info(
        "merge_segment_templates%s: merged %d segment(s) -> %d unit(s) x "
        "%d channel(s) union x %d sample(s) (averaging_method=%s)",
        well_suffix, n_segments_used, len(seen_unit_ids), n_channels_union,
        n_samples, averaging_method,
    )

    return {
        "well": well,
        "unit_ids": seen_unit_ids,
        "channel_ids": union_channel_ids,
        "channel_locations_xy": channel_locations_xy,
        "templates": templates,
        "contributing_weight": contributing_weight,
        "n_segments": n_segments_used,
        "averaging_method": averaging_method,
        "ms_before": float(ms_before),
        "ms_after": float(ms_after),
        "sampling_frequency_hz": sampling_frequency_hz,
    }


def _write_segment_contribution(
    retention_dir, *, index, seg_dir, analyzer, spike_counts, channel_ids, locations_xy,
):
    """Persist ONE segment's per-unit template contribution (see
    `merge_segment_templates`' `segment_retention_dir` docs for layout, cost,
    and why: these arrays are otherwise dropped with the analyzer at the end
    of each loop iteration, and cross-segment consistency diagnostics need
    them back).

    Reads the analyzer's dense `templates` buffer once — (n_units, n_samples,
    n_channels) — transposes to the (n_units, n_channels, n_samples) axis
    order every stitched artifact uses, casts to float32 (these are
    diagnostic inputs; the merge itself keeps accumulating in float64), and
    NaN-fills the rows of units with no spikes in this segment, mirroring the
    merge's own skip of those contributions (their stored template rows are
    undefined, not real zeros).

    Returns this segment's `segments_index.json` entry.
    """
    seg_out = retention_dir / f"segment_{index:03d}"
    seg_out.mkdir(parents=True, exist_ok=True)

    dense = np.asarray(
        analyzer.get_extension("templates").get_data(operator="average"), dtype=np.float32,
    )
    contribution = np.ascontiguousarray(np.transpose(dense, (0, 2, 1)))
    counts = np.asarray(spike_counts, dtype=np.int64)
    contribution[counts <= 0, :, :] = np.nan

    np.save(seg_out / "templates.npy", contribution)
    np.save(seg_out / "spike_counts.npy", counts)
    np.save(seg_out / "locations_xy.npy", np.asarray(locations_xy, dtype=np.float64))
    (seg_out / "channel_ids.json").write_text(
        json.dumps([_jsonable_id(c) for c in channel_ids], indent=2)
    )
    return {
        "index": int(index),
        "dirname": seg_out.name,
        "analyzer_dir": str(seg_dir),
        "n_channels": len(channel_ids),
        "n_units_contributing": int((counts > 0).sum()),
    }


__all__ = [
    "AVERAGING_METHODS",
    "DEFAULT_AVERAGING_METHOD",
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
