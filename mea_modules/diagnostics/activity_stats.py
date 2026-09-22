"""Per-segment activity statistics across a concatenated well.

The summary the concatenated well's collector caches -- per-electrode rates per
segment, their stability and the optimistic comparison tests. The figure that
draws it (:mod:`.segment_event_rates`) imports it back.
"""

import logging

# the detector's own labelling rule: the per-electrode rows are built from the
# event labels it produced, so one rule, never a second that could drift
from ..quality.detection import channel_labels as _channel_labels
from .gap_table import join_marks

logger = logging.getLogger(__name__)


PER_SEGMENT_ONLY = (
    "Each segment's number is computed from its own samples and its own recorded "
    "duration; nothing is computed across a join, because the gap between one "
    "segment's last spike and the next segment's first is microseconds in the "
    "file but minutes in real time — any rate, interval, correlation or slope "
    "spanning a join would be inventing structure out of the stitching."
)


_COMPARISON_ALPHA = 0.05


# The notes travel with the numbers: they are what stops the next reader turning
# a rho into a verdict or a p value into a pass/fail.
_STABILITY_NOTE = (
    "Headline stability: consecutive-segment Spearman rho of the per-electrode "
    "rates (did the same electrodes stay the busy ones) and the coefficient of "
    "variation of the segment means (how far the average moved). Descriptive; "
    "no threshold is applied."
)


_COMPARISON_NOTE = (
    "Secondary and optimistic: a repeated-measures test over the same electrodes, "
    "but electrodes share network bursts and units, so they are not independent "
    "replicates and this p value overstates the evidence. A flag, never a gate."
)


_NO_COMPARISON_NOTE = (
    "No per-channel rates in this summary, so no comparison was run: pass "
    "event_channel_labels and channel_ids to segment_event_rate_summary to get "
    "one."
)


def _rate_span(rows):
    """Lowest and highest measured rate, as a phrase for the log line."""
    rates = [row["events_per_s"] for row in rows if row["events_per_s"] is not None]
    if not rates:
        return "no measurable rate"
    return f"{min(rates):.2f}-{max(rates):.2f} /s"


def _per_channel_counts(placed, event_channel_labels, channel_ids, n_segments):
    """``(channel_axis, counts)`` with one count per (segment, channel) cell.

    `counts` is a ``(n_segments, n_channels)`` integer array over the WHOLE
    channel axis detection covered, so an electrode that fired nowhere is a row
    of zeros rather than an absence. That distinction is the whole point: a
    silent electrode is a measurement, and dropping it would raise every mean
    and narrow every error bar by exactly the channels a reviewer most wants
    counted.

    Parameters
    ----------
    placed : numpy.ndarray
        Segment index per event, as :func:`segment_event_rate_summary` computed
        it.
    event_channel_labels : array-like
        One channel label per event: the ``labels`` of
        :func:`mea_modules.quality.detect_events`.
    channel_ids : sequence
        The channels detection ran over, in the order it was given them.
    n_segments : int
        How many rows the count array needs, including any segment nothing fell
        in.

    Raises
    ------
    ValueError
        Labels and times of different lengths, an empty channel list, or a label
        that is not on the channel axis — each of which means the labels came
        from a different detection pass than the channel list did.
    """
    import numpy as np

    axis = _channel_labels(channel_ids)
    if not axis.size:
        raise ValueError("channel_ids is empty, so there is no channel axis to decompose onto")

    labels = np.asarray(event_channel_labels, dtype=np.int64).reshape(-1)
    if labels.size != placed.size:
        raise ValueError(
            f"{labels.size} channel label(s) for {placed.size} event time(s); detection "
            "returns them paired and they must stay paired"
        )

    unique_axis = np.unique(axis)
    if unique_axis.size != axis.size:
        logger.warning(
            "channel_ids repeats %d id(s); their events are all counted against the "
            "first occurrence and the repeats stay at zero",
            int(axis.size - unique_axis.size),
        )

    # searchsorted over the sorted axis rather than a dict lookup per event:
    # this runs over millions of events on a full well.
    order = np.argsort(axis, kind="stable")
    sorted_axis = axis[order]
    slot = np.searchsorted(sorted_axis, labels)
    on_axis = (slot < sorted_axis.size) & (
        sorted_axis[np.clip(slot, 0, max(0, sorted_axis.size - 1))] == labels
    )
    if not bool(on_axis.all()):
        strays = np.unique(labels[~on_axis])
        raise ValueError(
            f"{int((~on_axis).sum())} event(s) carry a channel label that is not in "
            f"channel_ids (e.g. {strays[:5].tolist()}); pass the same channel list the "
            "detection pass ran over"
        )

    # One bincount over a flattened (segment, channel) index: the loop-free form
    # of "count every event into its own cell".
    channel_index = order[slot]
    cells = placed.astype(np.int64) * axis.size + channel_index
    flat = np.bincount(cells, minlength=n_segments * axis.size)
    return axis, flat[: n_segments * axis.size].reshape(n_segments, axis.size)


def _per_channel_row(counts, recorded_s):
    """The per-electrode keys for one segment, from its own counts and duration.

    A segment with no recorded duration has no per-channel rates either, for the
    same reason it has no rate at all: dividing by a zero duration is not zero
    events per second, it is unmeasured, and the two mean opposite things to a
    reviewer.

    The spread is the SAMPLE standard deviation (``ddof=1``) — these electrodes
    are a sample of the array, not the population of interest — and the SEM is
    that over the square root of the count. A single electrode has no spread to
    report, so both come back as zero rather than as a NaN that would propagate
    into the figure.

    Both centres are reported. The mean is what the bar draws and what the SEM
    belongs to; the median is the honest one for this distribution, which is
    bounded at zero with a long right tail (a handful of electrodes produce most
    of the crossings), so a mean well above the median means the segment activity
    sits in a few electrodes rather than across the array.
    """
    import numpy as np

    if not recorded_s:
        return {
            "per_channel_events_per_s": None,
            "mean_events_per_s_per_channel": None,
            "median_events_per_s_per_channel": None,
            "sem_events_per_s_per_channel": None,
            "std_events_per_s_per_channel": None,
            "n_channels_measured": 0,
        }

    rates = np.asarray(counts, dtype=float) / float(recorded_s)
    std = float(np.std(rates, ddof=1)) if rates.size > 1 else 0.0
    return {
        "per_channel_events_per_s": [float(value) for value in rates],
        "mean_events_per_s_per_channel": float(np.mean(rates)),
        "median_events_per_s_per_channel": float(np.median(rates)),
        "sem_events_per_s_per_channel": std / float(np.sqrt(rates.size)),
        "std_events_per_s_per_channel": std,
        "n_channels_measured": int(rates.size),
    }


def _measured_groups(rows):
    """``(recs, groups)``: the per-electrode rate arrays of the segments that have one."""
    import numpy as np

    kept = [row for row in rows if row.get("per_channel_events_per_s")]
    return (
        [row.get("rec") for row in kept],
        [np.asarray(row["per_channel_events_per_s"], dtype=float) for row in kept],
    )


def _finite_or_none(value):
    import numpy as np

    return float(value) if value is not None and np.isfinite(value) else None


def _stability(rows):
    """The headline: consecutive-segment rank correlation and the CV of the means.

    Spearman rho between each segment's per-electrode rates and the next
    segment's -- the same electrodes, paired -- answers "did the busy
    electrodes stay the busy ones". A pair where one segment's rates carry no
    variation at all has no rank correlation, and is reported as ``None``
    rather than a number it does not have.

    The coefficient of variation is the sample SD of the segment means over
    their mean: one dispersion figure for "how far did the average move",
    scaled to its own size (the regularity measure MEA work reports over
    developmental time, e.g. Cotterill et al. 2016).
    """
    import numpy as np

    recs, groups = _measured_groups(rows)
    block = {
        "pairs": [],
        "consecutive_spearman_rho": [],
        "median_consecutive_rho": None,
        "min_consecutive_rho": None,
        "segment_means": [float(np.mean(group)) for group in groups],
        "cv_of_segment_means": None,
        "n_segments_compared": len(groups),
        "note": _STABILITY_NOTE,
    }
    if len(groups) >= 2:
        from scipy import stats

        for index in range(len(groups) - 1):
            a, b = groups[index], groups[index + 1]
            if np.ptp(a) == 0 or np.ptp(b) == 0:
                rho = None
            else:
                rho = _finite_or_none(stats.spearmanr(a, b).statistic)
            block["pairs"].append([recs[index], recs[index + 1]])
            block["consecutive_spearman_rho"].append(rho)
        defined = [rho for rho in block["consecutive_spearman_rho"] if rho is not None]
        if defined:
            block["median_consecutive_rho"] = float(np.median(defined))
            block["min_consecutive_rho"] = float(np.min(defined))
        means = np.asarray(block["segment_means"], dtype=float)
        if means.mean() > 0:
            block["cv_of_segment_means"] = float(np.std(means, ddof=1) / means.mean())
    return block


def _activity_comparison(rows, alpha=_COMPARISON_ALPHA):
    """The secondary, caveated test: repeated measures over the same electrodes.

    Friedman (electrode as the block) with Kendall's W for three or more
    segments; Wilcoxon signed-rank with the matched-pairs rank-biserial
    correlation for two. Both use electrode identity, which the unpaired tests
    this replaced threw away -- and with it the between-electrode baseline,
    which is most of the spread and has nothing to do with the segments.

    Returns ``test``, ``statistic``, ``p_value``, ``effect_size_name``,
    ``effect_size``, ``n_groups``, ``n_electrodes``, ``significant``, ``alpha``
    and ``note``, always all present. ``significant`` is ``None`` when nothing
    was tested rather than ``False``, which would claim the segments had been
    compared and found alike.
    """
    import numpy as np

    _recs, groups = _measured_groups(rows)
    block = {
        "test": None,
        "statistic": None,
        "p_value": None,
        "effect_size_name": None,
        "effect_size": None,
        "n_groups": len(groups),
        "n_electrodes": int(groups[0].size) if groups else 0,
        "significant": None,
        "alpha": float(alpha),
        "note": _NO_COMPARISON_NOTE,
    }
    if len(groups) < 2:
        return block
    if len({group.size for group in groups}) != 1:
        block["note"] = "segments carry different electrode counts, so they cannot be paired"
        return block

    try:
        from scipy import stats
    except ImportError:
        block["note"] = "scipy is not installed, so the between-segment comparison was skipped."
        logger.warning("scipy unavailable; segments were not compared")
        return block

    try:
        if len(groups) == 2:
            name, effect_name = "Wilcoxon signed-rank", "matched-pairs rank-biserial"
            result = stats.wilcoxon(groups[0], groups[1])
            diff = groups[0] - groups[1]
            diff = diff[diff != 0]
            ranks = stats.rankdata(np.abs(diff))
            total = ranks.sum()
            effect = (
                float((ranks[diff > 0].sum() - ranks[diff < 0].sum()) / total)
                if total else None
            )
        else:
            name, effect_name = "Friedman", "Kendall's W"
            result = stats.friedmanchisquare(*groups)
            n, k = groups[0].size, len(groups)
            effect = float(result.statistic / (n * (k - 1))) if n and k > 1 else None
    except ValueError as error:
        # scipy refuses a comparison with no variation anywhere in it. That is
        # not a failure: it is the cleanest possible "these do not differ".
        block["note"] = f"no test was run ({error}); {_COMPARISON_NOTE}"
        return block

    if not (np.isfinite(result.statistic) and np.isfinite(result.pvalue)):
        block["note"] = (
            "no test was run (the rates carry no variation to test, so the "
            f"statistic is undefined); {_COMPARISON_NOTE}"
        )
        return block

    p_value = float(result.pvalue)
    block.update(
        {
            "test": name,
            "statistic": float(result.statistic),
            "p_value": p_value,
            "effect_size_name": effect_name,
            "effect_size": _finite_or_none(effect),
            "significant": bool(p_value < float(alpha)),
            "note": _COMPARISON_NOTE,
        }
    )
    logger.info(
        "%s across %d segment(s): p = %.3g, %s = %s (secondary, optimistic)",
        name, len(groups), p_value, effect_name,
        "n/a" if block["effect_size"] is None else f"{block['effect_size']:.3f}",
    )
    return block


def segment_event_rate_summary(
    segments,
    event_times_s,
    fs_hz,
    *,
    stitch_frames=(),
    n_channels=None,
    threshold_factor=None,
    event_channel_labels=None,
    channel_ids=None,
):
    """Per-segment event counts and rates; return a JSON-able dict.

    Events are placed by where they fall between the joins, then each segment's
    count is divided by that segment's own recorded duration
    (``n_samples / fs_hz``). An event landing exactly on a join belongs to the
    LATER segment, matching how the concatenation itself assigns that sample.

    Given `event_channel_labels` and `channel_ids` the same events are also
    split per electrode, which is what turns each segment's single rate into a
    distribution the figure can show a spread for and
    :func:`_activity_comparison` can test between.

    Parameters
    ----------
    segments : sequence of dict
        One entry per segment, in concatenated order, each carrying ``rec`` (the
        label) and ``n_samples`` (its own length in frames). Extra keys are
        ignored, so a manifest row can be passed straight through.
    event_times_s : array-like
        Event times in seconds on the CONCATENATED timeline, e.g. the
        ``frames`` of :func:`mea_modules.quality.detect_events` over the
        sampling rate.
    fs_hz : float
        Sampling rate, used for both the join positions and the durations.
    stitch_frames : sequence of int
        The joins, as frame offsets on the concatenated timeline — the one
        convention across every emitter, see
        :func:`mea_modules.diagnostics.timebase.join_marks`. There must be
        exactly one fewer than there are segments: that is what makes the
        placement unambiguous, and a mismatch would silently pile every event
        into the first segment.
    n_channels : int or None
        How many channels the detection covered, recorded so the rates can be
        compared across wells whose channel counts differ. ``None`` leaves
        ``events_per_s_per_channel`` unset rather than guessing a denominator —
        unless `channel_ids` was given, in which case its length is the
        denominator, since it is the same count measured rather than a guess.
    threshold_factor : float or None
        The MAD-sigma multiple the detection used. Provenance only — nothing
        here re-thresholds — but a rate is unreadable without it, so it travels
        with the numbers.
    event_channel_labels : array-like or None
        One channel label per entry of `event_times_s`, i.e. the ``labels`` of
        :func:`mea_modules.quality.detect_events`.
        With `channel_ids`, this adds the per-electrode breakdown. Without both,
        the per-channel keys stay absent and the summary is exactly what it was.
    channel_ids : sequence or None
        The channels that detection covered, in the order it was given them.
        Every one of them appears in every segment's per-channel array, so a
        channel that fired nowhere contributes zeros rather than nothing.

    Returns
    -------
    dict
        ``threshold_factor``, ``n_channels``, ``total_events``, ``note``,
        ``zero_event_segments`` (the labels of segments nothing crossed in, the
        dead-configuration check), ``stability`` (the headline, see
        :func:`_stability`), ``activity_comparison`` (the secondary test, see
        :func:`_activity_comparison` — a flag, never a gate) and
        ``segments``, one row each with ``segment_index``, ``rec``,
        ``n_events``, ``recorded_s``, ``events_per_s`` and
        ``events_per_s_per_channel``.

        With the per-channel inputs each row also carries
        ``per_channel_events_per_s`` (one rate per electrode, zeros included),
        ``mean_events_per_s_per_channel``, ``sem_events_per_s_per_channel``,
        ``std_events_per_s_per_channel`` and ``n_channels_measured``. The mean
        is the same number as ``events_per_s_per_channel`` by construction —
        both are the segment's rate over the same channel count — and the pair
        is a cheap consistency check on any consumer that computes one from the
        other.

    Raises
    ------
    ValueError
        No segments, a non-positive `fs_hz`, a `stitch_frames` count that does
        not separate the segments given, or per-channel inputs that do not line
        up with the events they are supposed to label.
    """
    import numpy as np

    segments = list(segments or ())
    if not segments:
        raise ValueError("no segments to summarize")
    fs_hz = float(fs_hz or 0.0)
    if fs_hz <= 0.0:
        raise ValueError(f"fs_hz must be positive to turn frames into seconds; got {fs_hz}")

    stitch_frames = list(stitch_frames or ())
    if len(stitch_frames) != len(segments) - 1:
        raise ValueError(
            f"{len(stitch_frames)} join(s) cannot separate {len(segments)} segment(s); "
            "expected exactly one fewer join than there are segments"
        )

    times = np.asarray(event_times_s, dtype=float).reshape(-1)
    # The joins come from the shared implementation rather than a local
    # frame/fs loop. On the file timeline a join is an instant, so the pair's
    # two halves are equal and either one is the boundary.
    boundaries = np.asarray(
        [start for start, _stop in join_marks(stitch_frames, fs_hz)], dtype=float
    )
    # side="right" puts an event landing exactly on a join into the segment that
    # starts there, which is where that sample lives in the concatenation.
    placed = np.searchsorted(boundaries, times, side="right")

    counts = None
    if event_channel_labels is not None and channel_ids is not None:
        axis, counts = _per_channel_counts(
            placed, event_channel_labels, channel_ids, len(segments)
        )
        measured_channels = int(axis.size)
        if n_channels is None:
            n_channels = measured_channels
        elif int(n_channels) != measured_channels:
            logger.warning(
                "n_channels says %d but channel_ids carries %d; keeping the explicit "
                "n_channels for events_per_s_per_channel",
                int(n_channels), measured_channels,
            )
    elif event_channel_labels is not None or channel_ids is not None:
        logger.warning(
            "the per-channel breakdown needs BOTH event_channel_labels and channel_ids; "
            "got only one, so no per-channel rates were computed"
        )

    rows = []
    for index, entry in enumerate(segments):
        n_events = int(np.count_nonzero(placed == index))
        recorded_s = float(entry.get("n_samples", 0)) / fs_hz
        rate = (n_events / recorded_s) if recorded_s else None
        row = {
            "segment_index": index,
            "rec": entry.get("rec"),
            "n_events": n_events,
            "recorded_s": recorded_s,
            "events_per_s": rate,
            "events_per_s_per_channel": (
                None
                if rate is None or not n_channels
                else rate / float(n_channels)
            ),
        }
        if counts is not None:
            row.update(_per_channel_row(counts[index], recorded_s))
        rows.append(row)

    dead = [row["rec"] for row in rows if not row["n_events"]]
    if dead:
        logger.warning(
            "%d of %d segment(s) recorded ZERO threshold crossings on the channels "
            "detected over: %s",
            len(dead), len(rows), dead,
        )
    logger.info(
        "segment event rates: %d event(s) over %d segment(s); rates span %s",
        int(times.size),
        len(rows),
        _rate_span(rows),
    )

    return {
        "threshold_factor": None if threshold_factor is None else float(threshold_factor),
        "n_channels": None if n_channels is None else int(n_channels),
        "total_events": int(times.size),
        "note": PER_SEGMENT_ONLY,
        "zero_event_segments": dead,
        "stability": _stability(rows),
        "activity_comparison": _activity_comparison(rows),
        "segments": rows,
    }
