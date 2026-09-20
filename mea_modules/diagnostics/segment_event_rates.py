"""How much fired in each segment — the blank raster band, turned into a number.

A threshold raster across a concatenated well shows a segment that recorded
nothing as a blank vertical stretch, which is a real read but a subjective one:
whether a band is blank or merely sparse is a squint. This module divides the
same detected crossings into segments and reports a rate for each, so a dead
recording configuration is a zero rather than an impression.

Every rate divides one segment's own event count by that segment's own recorded
duration. Nothing is computed across a join, and the reason is
:data:`mea_modules.diagnostics.figure_text.PER_SEGMENT_ONLY`: the file's gap
between one segment's last sample and the next segment's first is microseconds,
while the real gap is minutes, so any rate spanning a join would be measuring
the stitching.

A segment's activity is a distribution, not only a total. Given the per-event
channel labels the detection already returns, each segment is also decomposed
into one rate per electrode — a silent electrode counted as a real zero, never
dropped, because a dead channel is evidence about the segment and omitting it
would raise every mean and shrink every spread. That decomposition is what the
figure's error bars and its scatter of individual electrodes are drawn from, and
what makes a tall bar readable as a shifted population rather than one loud
electrode.

Whether the segments differ at all is reported as ``activity_comparison``: a
non-parametric test over those per-electrode arrays (Mann-Whitney U for two
segments, Kruskal-Wallis H for three or more — a per-channel rate is bounded
below by zero and has a long right tail, which is the shape a t-test handles
worst). It is a WARNING FLAG and nothing else. Segments of one well would
normally be expected to behave similarly, so a significant difference is worth
looking at; it is never a failure, nothing here raises on it, and no gate reads
it. The figure states this result whether or not it fires — a flag that only
speaks up when significant is one a reader cannot tell from a test that was
never run.

The number and the picture are deliberately separable:

* :func:`segment_event_rate_summary` — segments + event times -> a JSON-able
  dict. Plain arithmetic, no matplotlib, so it can be asserted on directly and
  can become a pass/fail check without dragging a figure along.
* :func:`plot_segment_event_rates` — that dict -> one figure: bars sitting edge
  to edge so the row reads as one block of segments, one colour each so
  neighbours are still told apart, SD whiskers by default, and the electrodes
  themselves as jittered points. Reads the summary and nothing else, so the
  figure can never disagree with the number it draws.

This is the un-sorted counterpart to
:func:`mea_modules.postprocess.segment_activity.segment_activity_summary`, which
answers the same per-segment question about a SORT's units. This one needs no
sorter: it runs on threshold crossings, before hours are spent.

Detection itself lives in :mod:`mea_modules.diagnostics.raster`
(:func:`~mea_modules.diagnostics.raster.estimate_channel_thresholds` and
:func:`~mea_modules.diagnostics.raster.detect_threshold_crossings`); this module
takes the times and labels those return and never detects anything of its own.

Pure library: no argparse, no printing, no ``__main__``.
"""

import logging

from .channel_layout import (
    _add_caption,
    _fold_caption,
    _legend_dot,
    _legend_line,
    _new_figure,
    _save_and_release,
)
from .figure_style import legend_corner
from .figure_text import (
    PER_ELECTRODE,
    PER_SEGMENT_ONLY,
    SEGMENT_BAND,
    acronym_note,
    dispersion_key,
)
# The event labels this module decomposes are produced by raster's own rule, so
# the channel axis has to be built by that same rule or the two would not line
# up; importing it is what keeps there being one rule rather than two.
from .raster import _channel_labels
from .timebase import join_marks

logger = logging.getLogger(__name__)

# The tuned reference point the dynamic canvas (`_rates_figure_size`) is built
# from, not a figsize handed straight to the figure any more. Its height is
# taller than the 4.8 in this figure used to be — the sheet carries a caption
# several fragments longer, a legend and a scatter, and at the old height the
# bottom matter took nearly half of it and squeezed the axes until the y label
# ran off the top of its own axis — and it is kept as a live default so a
# caller that still passes `figsize=_RATES_FIGSIZE` explicitly sees no change.
_RATES_FIGSIZE = (12.0, 6.0)
_RATES_DPI = 180

# Segment count controls WIDTH here, the mirror of segment_boundary_map's
# rows-control-height pattern on the other axis (read that module for the
# house rule this follows): there each segment is a horizontal ROW, here each
# segment is a vertical COLUMN. `_RATES_FIGSIZE`'s width divided by this many
# reference segments is the per-segment allowance, so a well with exactly this
# many segments reproduces the historical canvas exactly.
_RATES_REFERENCE_SEGMENTS = 10.0
# Floor under the computed width so two segments do not shrink to a sliver
# next to a fixed-height sheet that still has to hold the caption, the legend
# and the scatter cloud (Adam, 2026-09-19): at few segments the canvas comes
# out clearly taller than it is wide, which is the point.
_RATES_MIN_WIDTH_IN = 3.6

_SEGMENT_LABEL_FONTSIZE = 6

# Bars are drawn edge to edge, so the thing that separates one segment from the
# next is its colour plus a hairline. A qualitative map, not a sequential one:
# adjacent samples of a sequential map are nearly the same colour, which is the
# opposite of what adjacent bars need.
_BAR_EDGE_COLOR = "white"
_BAR_EDGE_WIDTH = 0.4
# Pale on purpose. Most electrodes sit BELOW the mean, so most of the scatter
# falls inside the bar rather than above it; an opaque bar swallows exactly the
# points the bar is there to summarize. The fill now reads as a backdrop and the
# cloud reads on top of it.
_BAR_ALPHA = 0.35

_ERROR_COLOR = "0.15"
_ERROR_CAPSIZE = 3.0
_ERROR_LINEWIDTH = 1.4

# Which summary key the whisker is drawn from, by `dispersion`. Default SD, not
# SEM (Adam, 2026-09-19): the scatter already shows the electrode POPULATION,
# and at n in the hundreds SEM = SD / sqrt(n) is roughly a sixteenth of SD, so
# it draws as a misleadingly tight whisker sitting inside a visibly wide cloud
# of points. SD describes the spread the scatter is already showing; SEM
# describes how precisely the segment MEAN is known, which is a real question
# but not the one this figure is answering with a cloud of individual
# electrodes. `segment_event_rate_summary` keeps computing and reporting both
# in the JSON regardless of which one gets a whisker here.
_DISPERSION_SUMMARY_KEYS = {
    "sd": "std_events_per_s_per_channel",
    "sem": "sem_events_per_s_per_channel",
}
_DISPERSION_ACRONYMS = {"sd": "SD", "sem": "SEM"}

_SCATTER_COLOR = "0.10"
_SCATTER_SIZE = 4.0
_SCATTER_ALPHA = 0.55
# Half the horizontal spread of the jitter, in bar widths. Wide enough that a
# few hundred points read as a cloud, narrow enough that no point strays over
# the bar it belongs to (bars are one unit wide, so the edge is at 0.5).
_JITTER_HALF_WIDTH = 0.3

# Fixed, never a parameter. Determinism is this pipeline's one hard global bar:
# two renders of the same summary must produce the same PNG, and a jitter drawn
# from fresh entropy would break that on every re-run.
_JITTER_SEED = 0

# Past this many electrodes in one segment the cloud is a solid block and the
# render slows down for detail nobody can see. The cap travels in the returned
# manifest so a thinned cloud is never a silent thinning.
_MAX_SCATTER_POINTS = 300

# Where the y axis is cut when electrodes are scattered on it. A per-electrode
# rate distribution is heavy-tailed — a handful of loud electrodes sit an order
# of magnitude above the rest — and an autoscale that reaches the loudest one
# flattens every bar to a hairline, which is the one comparison this figure
# exists to make. The cut never goes below the tallest bar and its whisker, and
# how many points are above it is reported rather than dropped.
_SCATTER_BULK_PERCENTILE = 99.0
_Y_HEADROOM = 1.08
# ...and a floor under that legibility: however wide the cloud, the axis never
# rises to more than this multiple of the tallest bar, so the bar row always
# owns a readable fraction of the sheet. Inert on an ordinary well, where the
# bulk of the electrodes already sits inside it.
_MAX_AXIS_OVER_BARS = 5.0

# Explicit and small rather than matplotlib's own default sizes (title ~12 pt,
# axis labels ~10 pt): the title is generated text of variable length (it names
# the threshold and the channel count) and the x label is a full clause, and
# the canvas this sits on can now be as narrow as `_RATES_MIN_WIDTH_IN`, where
# the default sizes overflow past the canvas edge and are cropped out of the
# written PNG rather than merely wrapping.
_TITLE_FONTSIZE = 9
_AXIS_LABEL_FONTSIZE = 8

# The comparison mark's own geometry. Small and out of the way on purpose: it
# is a result, not chrome, and it must never read as loud as the bars it sits
# above (Adam, 2026-09-19: "not seeing any indication of significance testing
# ... should still be something indicating that").
_MARK_FONTSIZE = 6.5
_MARK_COLOR = "0.15"
# Fractions of the axis's own top (`ax.get_ylim()[1]`), not of the axes box —
# drawn in plain DATA coordinates rather than a blended axes-fraction
# transform, deliberately: `legend_corner`'s corner scan reads artist
# coordinates through `transData`, so a mark placed any other way is invisible
# to it and the legend can land right on top of this mark instead of dodging
# it. In data coordinates the same fractions keep the mark at the same
# printed height above the bars whatever the y axis happens to span, from a
# well with no scatter at all to one cut at the 99th percentile, since a
# fraction of the axis top scales with it exactly as the axes-fraction version
# would have.
_BRACKET_Y = 0.93
_BRACKET_TICK = 0.03
_BRACKET_TEXT_PAD = 0.015

# scipy's own name for the k-sample test carries a trailing statistic letter
# ("Kruskal-Wallis H") that belongs in the JSON, which is provenance, but reads
# as noise in three words of figure text; the two-sample test's own "U" stays,
# since that is how it is conventionally cited. Display-only: `test` in the
# returned `activity_comparison` block is never touched.
_TEST_DISPLAY_NAMES = {"Kruskal-Wallis H": "Kruskal-Wallis"}

_COMPARISON_ALPHA = 0.05

# The note travels with the number because the number IS a flag and nothing
# more: it is what stops the next reader turning a p value into a pass/fail.
_COMPARISON_NOTE = (
    "Warning flag only, never a failure: segments would normally be expected to "
    "behave similarly, so a significant difference is worth looking at, not a "
    "reason to reject anything. Nothing in this pipeline gates on it."
)
_NO_COMPARISON_NOTE = (
    "No per-channel rates in this summary, so no comparison was run: pass "
    "event_channel_labels and channel_ids to segment_event_rate_summary to get "
    "one."
)

# Short enough to fit the sheet's axes without overhanging them — a y label is
# drawn along the axis and cannot wrap. Every bar IS one segment, so "within
# one segment" would only restate the x axis, and the caption carries the rule
# in full. Which of the two is used says which quantity the bars are.
#
# Publication-terse (Adam, 2026-09-19): a legend key is not a clause, and
# neither is an axis label. "crossings per second, per electrode" said the
# same thing in more words than "crossings/s per electrode" does.
_RATE_YLABEL = "threshold crossings per second"
_PER_CHANNEL_YLABEL = "crossings/s per electrode"


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
        One channel label per event, the second return of
        :func:`mea_modules.diagnostics.raster.detect_threshold_crossings`.
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
    """
    import numpy as np

    if not recorded_s:
        return {
            "per_channel_events_per_s": None,
            "mean_events_per_s_per_channel": None,
            "sem_events_per_s_per_channel": None,
            "std_events_per_s_per_channel": None,
            "n_channels_measured": 0,
        }

    rates = np.asarray(counts, dtype=float) / float(recorded_s)
    std = float(np.std(rates, ddof=1)) if rates.size > 1 else 0.0
    return {
        "per_channel_events_per_s": [float(value) for value in rates],
        "mean_events_per_s_per_channel": float(np.mean(rates)),
        "sem_events_per_s_per_channel": std / float(np.sqrt(rates.size)),
        "std_events_per_s_per_channel": std,
        "n_channels_measured": int(rates.size),
    }


def _activity_comparison(rows, alpha=_COMPARISON_ALPHA):
    """Do the segments' per-channel rate distributions differ? A flag, not a gate.

    Non-parametric on purpose. A per-channel rate is bounded below by zero and
    carries a long right tail — a handful of electrodes produce most of the
    crossings — so the normality a t-test or an ANOVA assumes is exactly what
    these numbers do not have. Two segments get Mann-Whitney U, three or more
    Kruskal-Wallis H.

    Returns the ``activity_comparison`` block: ``test``, ``statistic``,
    ``p_value``, ``n_groups``, ``significant``, ``alpha`` and ``note``. The keys
    are always present; the ones a test did not produce are ``None`` rather than
    missing, so a consumer never has to branch on a key's existence, and
    ``significant`` is ``None`` when nothing was tested rather than ``False``,
    which would claim the segments had been compared and found alike.

    scipy is imported here rather than at module scope, as the rest of the
    package does it: the arithmetic above must stay importable without it.
    """
    import numpy as np

    groups = [
        np.asarray(row["per_channel_events_per_s"], dtype=float)
        for row in rows
        if row.get("per_channel_events_per_s")
    ]
    block = {
        "test": None,
        "statistic": None,
        "p_value": None,
        "n_groups": len(groups),
        "significant": None,
        "alpha": float(alpha),
        "note": _NO_COMPARISON_NOTE,
    }
    if len(groups) < 2:
        return block

    try:
        from scipy import stats
    except ImportError:
        block["note"] = "scipy is not installed, so the between-segment comparison was skipped."
        logger.warning("scipy unavailable; segments were not compared")
        return block

    try:
        if len(groups) == 2:
            name = "Mann-Whitney U"
            result = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
        else:
            name = "Kruskal-Wallis H"
            result = stats.kruskal(*groups)
    except ValueError as error:
        # scipy refuses a comparison with no variation anywhere in it (every
        # electrode of every segment on the same rate). That is not a failure:
        # it is the cleanest possible "these do not differ", so it is recorded
        # as a note and the rest of the summary stands.
        block["note"] = f"no test was run ({error}); {_COMPARISON_NOTE}"
        return block

    p_value = float(result.pvalue)
    block.update(
        {
            "test": name,
            "statistic": float(result.statistic),
            "p_value": p_value,
            "significant": bool(p_value < float(alpha)),
            "note": _COMPARISON_NOTE,
        }
    )
    if block["significant"]:
        logger.warning(
            "%s across %d segment(s): p = %.3g < %g — activity differs between "
            "segments. Worth a look, not a failure.",
            name, len(groups), p_value, float(alpha),
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
        Event times in seconds on the CONCATENATED timeline, e.g. the first
        return of
        :func:`mea_modules.diagnostics.raster.detect_threshold_crossings`.
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
        One channel label per entry of `event_times_s`, i.e. the SECOND return
        of :func:`~mea_modules.diagnostics.raster.detect_threshold_crossings`.
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
        dead-configuration check), ``activity_comparison`` (see
        :func:`_activity_comparison` — a warning flag, never a gate) and
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
        "activity_comparison": _activity_comparison(rows),
        "segments": rows,
    }


def _default_title(summary):
    """Title naming the detection the rates came from, skipping what is unknown."""
    parts = []
    factor = summary.get("threshold_factor")
    if factor is not None:
        parts.append(f"{float(factor):g} × MAD-sigma")
    n_channels = summary.get("n_channels")
    if n_channels:
        parts.append(f"{int(n_channels)} channels")
    detail = f" ({', '.join(parts)})" if parts else ""
    return f"activity per segment{detail}"


def _format_p(p_value):
    """A p value as figure text. A tiny value prints as a bound, not a float.

    scipy returns exactly 0.0 once the true value drops below the smallest
    double, and "p = 0" on a figure claims an impossibility. Above that floor
    but still small, a long float (``p = 4.1e-07``) reads as more precision
    than the point being made needs; "p < 0.001" is what a reader actually
    takes from either number.
    """
    p_value = float(p_value)
    if p_value <= 0.0:
        return "p < 1e-300"
    if p_value < 0.001:
        return "p < 0.001"
    return f"p = {p_value:.2g}"


def _comparison_label(comparison):
    """One line of figure text for `comparison`, or None if nothing was tested.

    Same string whether the bracket form (two groups) or the single-line form
    (three or more) draws it — see :func:`_TEST_DISPLAY_NAMES` for the one
    place the test's name is shortened for the figure. The verdict is
    ``", n.s."`` when the comparison did not clear alpha and nothing extra when
    it did: the number already says how significant, and "significant" printed
    next to a p value would only repeat it.
    """
    test = comparison.get("test")
    p_value = comparison.get("p_value")
    if not test or p_value is None:
        return None
    name = _TEST_DISPLAY_NAMES.get(test, test)
    verdict = "" if comparison.get("significant") else ", n.s."
    return f"{name}, {_format_p(p_value)}{verdict}"


def _bar_colors(n_bars):
    """One colour per bar, cycled from a qualitative map; deterministic.

    The bars touch, so colour is what separates one segment from the next. Ten
    or fewer take ``tab10``, whose colours are the furthest apart matplotlib
    ships; past that ``tab20``, which trades some of that separation for twice
    as many steps before the cycle repeats. Position alone picks the colour, so
    the same summary always draws the same bars.
    """
    from matplotlib import colormaps

    n_bars = max(0, int(n_bars))
    palette = colormaps["tab10" if n_bars <= 10 else "tab20"].colors
    return [palette[index % len(palette)] for index in range(n_bars)]


def _rates_figure_size(n_segments):
    """Canvas for `n_segments` bars, in inches: width scales, height does not.

    Mirrors :func:`mea_modules.diagnostics.segment_boundary_map._figure_size`
    on the other axis — there each segment is a horizontal ROW and segment
    count drives height; here each segment is a vertical COLUMN and segment
    count drives width. ``_RATES_FIGSIZE`` is the tuned reference point: its
    width divided by :data:`_RATES_REFERENCE_SEGMENTS` is the per-segment
    allowance, so a well with exactly that many segments reproduces the
    historical canvas exactly, and the floor keeps a two-segment well from
    coming out as a sliver — it comes out clearly taller than it is wide
    instead (Adam, 2026-09-19).
    """
    width_per_segment = _RATES_FIGSIZE[0] / _RATES_REFERENCE_SEGMENTS
    width_in = max(_RATES_MIN_WIDTH_IN, float(n_segments) * width_per_segment)
    return (width_in, _RATES_FIGSIZE[1])


def _draw_activity_comparison(ax, comparison, grouped_positions, all_positions):
    """State the between-segment test on `ax`, significant or not.

    A warning flag has to say so even when it is quiet (Adam, 2026-09-19): a
    reader seeing no mark at all cannot tell "not significant" from "never
    tested". Exactly two groups with two known bar positions draw a BRACKET
    spanning them, so the mark visibly belongs to that pair rather than to the
    figure in general; anything else (three or more groups, or a two-group
    comparison whose bar positions could not be pinned down) draws a plain
    RULE spanning every bar instead, since the omnibus test speaks for the
    whole row rather than one pair of them.

    Both forms draw an actual line, not just text, and that is deliberate
    rather than decorative: :func:`mea_modules.diagnostics.figure_style.legend_corner`
    scores a corner by the artist points it finds there through ``transData``,
    and text is invisible to that scan. A line spanning every bar reaches into
    BOTH upper corners, which is what pushes the legend down into whichever
    lower corner the scatter leaves emptiest instead of letting it land on top
    of this mark — exactly the ordering callers rely on by drawing this before
    the legend.

    Plain DATA coordinates throughout, scaled off the axis's OWN current top
    (`ax.get_ylim()[1]`, set by the caller's scatter-cut before this runs) —
    see :data:`_BRACKET_Y` for why. Returns nothing.
    """
    label = _comparison_label(comparison)
    if label is None:
        return

    axis_top = float(ax.get_ylim()[1])
    y_bar = _BRACKET_Y * axis_top
    if len(grouped_positions) == 2:
        x0, x1 = sorted(grouped_positions)
        y_tick = y_bar - _BRACKET_TICK * axis_top
        ax.plot(
            [x0, x0, x1, x1],
            [y_tick, y_bar, y_bar, y_tick],
            color=_MARK_COLOR,
            linewidth=1.0,
            solid_capstyle="butt",
        )
    else:
        x0, x1 = min(all_positions), max(all_positions)
        ax.plot([x0, x1], [y_bar, y_bar], color=_MARK_COLOR, linewidth=1.0)
    ax.text(
        (x0 + x1) / 2.0,
        y_bar + _BRACKET_TEXT_PAD * axis_top,
        label,
        ha="center",
        va="bottom",
        fontsize=_MARK_FONTSIZE,
        color=_MARK_COLOR,
    )


def _scatter_values(row, cap=_MAX_SCATTER_POINTS):
    """One segment's per-channel rates, thinned to `cap` points; ``(values, cut)``.

    `cut` says whether the cap bit. The subsample is every k-th entry of
    the channel axis, which is ordered by channel id and therefore unordered
    with respect to rate, so the thinned cloud has the same shape as the full
    one — and it costs no random state, leaving the jitter the only thing the
    generator is spent on.
    """
    import numpy as np

    values = row.get("per_channel_events_per_s")
    if not values:
        return None, False

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return None, False
    if values.size <= int(cap):
        return values, False

    keep = np.unique(np.linspace(0, values.size - 1, int(cap)).astype(np.int64))
    return values[keep], True


def plot_segment_event_rates(
    summary,
    out_path=None,
    *,
    title=None,
    ax=None,
    figsize=None,
    dpi=_RATES_DPI,
    annotate=True,
    dispersion="sd",
    show_comparison=True,
):
    """Draw a summary's per-segment rates as bars; return a manifest dict.

    One bar per segment, in concatenated order, labelled by recording name. The
    bars are a full unit wide and the axes are tightened to them, so the row
    reads as one continuous block of segments with no dead margin at either end;
    each takes its own colour from a qualitative map, which is what tells
    touching bars apart.

    With the per-channel breakdown in the summary the bar also carries its
    spread — SD whiskers by default (see `dispersion`) — and the electrodes
    themselves scattered over the bar with a deterministic horizontal jitter.
    Segments with more than :data:`_MAX_SCATTER_POINTS` electrodes draw an
    evenly spaced subsample, reported in the returned manifest; the bar and the
    whiskers always use every channel.

    That breakdown also decides what the bar IS, which the manifest reports as
    ``bar_quantity``. With it, the bar is the mean rate PER ELECTRODE, because
    whiskers, points and the between-segment test are all per-electrode and a
    whole-array bar would put them on an axis a factor of the electrode count
    away — every point flat against the baseline. Without it, the bar is the
    whole-segment rate, unchanged, and nothing else is on the axis to disagree
    with it. Either way the y label says which, and the whole-array rate is the
    electrode mean times the electrode count.

    A segment with no measurable rate (no recorded duration) draws at zero and
    is counted in the returned manifest, so an empty bar is never mistaken for a
    measured silence.

    Parameters
    ----------
    summary : dict
        Exactly what :func:`segment_event_rate_summary` returns. Nothing is
        recomputed here — the figure and the number cannot drift apart.
    out_path : path-like or None
        Where to write the PNG. Required unless `ax` is given.
    title : str or None
        Overrides the generated title, which names the detection the rates came
        from. A caller that knows the well usually leads with it.
    ax : matplotlib axes or None
        Draw into the caller's axes instead of building a figure. With `ax`
        given no caption is added and no file is written.
    figsize : tuple or None
        Left None the canvas is computed from the segment count (see
        :func:`_rates_figure_size`), which is what keeps a two-segment well
        from coming out wider than it is tall. An explicit tuple overrides
        that and is used exactly as given. Ignored when `ax` is given.
    dpi : float
        Figure resolution. Ignored when `ax` is given.
    annotate : bool
        Controls the explanatory chrome. True keeps the title and the caption
        block. False draws neither, leaving the axes, the units, the legend and
        the significance mark — the presentation-plot register, where the
        title's information lives in the filename instead. The mark stays
        because it is a measurement rather than chrome.
    dispersion : {"sd", "sem"}
        Which of the summary's two spread keys draws the whisker and the
        legend key: the sample standard deviation (the default) or the
        standard error of the mean. `segment_event_rate_summary` always
        computes and reports both regardless of this choice — only which one
        is DRAWN changes.
    show_comparison : bool
        True (the default) states the between-segment test on the axes
        whether or not it is significant — a bracket over the two bars for a
        two-group comparison, one line for three or more — because a warning
        flag that only speaks up when it fires is one a reader cannot tell
        from a test that was never run. Drawn regardless of `annotate`, since
        it is a result rather than chrome. False suppresses it entirely.

    Returns
    -------
    dict
        The file written (``None`` when drawing into `ax`), the bar count, how
        many of those bars have no measurable rate, how many carry a spread,
        which quantity the bars are (``bar_quantity``), which dispersion was
        drawn, the scatter cap with how many segments it thinned and how many
        drawn points sit above the y axis, and the summary's
        ``activity_comparison`` block as drawn.

    Raises
    ------
    ValueError
        The summary holds no segments, neither `out_path` nor `ax` was given,
        or `dispersion` is not ``"sd"`` or ``"sem"``.
    """
    import numpy as np

    rows = list((summary or {}).get("segments") or ())
    if not rows:
        raise ValueError("summary holds no segments to plot")
    if ax is None and out_path is None:
        raise ValueError("pass out_path to write a figure, or ax to draw into one")
    dispersion = str(dispersion).lower()
    if dispersion not in _DISPERSION_SUMMARY_KEYS:
        raise ValueError(f"dispersion must be 'sd' or 'sem'; got {dispersion!r}")

    fig = None
    if ax is None:
        size = _rates_figure_size(len(rows)) if figsize is None else figsize
        fig = _new_figure(size, dpi)
        ax = fig.subplots()

    positions = [row["segment_index"] for row in rows]
    # A row with no measurable rate draws at zero; the manifest counts them so a
    # caller can say which flat bars were measured and which were not.
    unmeasured = sum(1 for row in rows if row["events_per_s"] is None)

    # The bar has to be in the same units as the whiskers and the points, or the
    # three would be three different quantities stacked on one axis: the
    # electrode mean is ~1/n of the whole-segment rate, so scattering electrodes
    # over a whole-segment bar pins every point to the baseline. Where the
    # per-channel breakdown exists the bar is therefore the electrode mean —
    # which is also the quantity the comparison tested, so the p mark annotates
    # what is drawn. Without it, the bar is the whole-segment rate exactly as
    # before, and there is nothing else on the axis to disagree with.
    bar_key = (
        "mean_events_per_s_per_channel"
        if any(row.get("mean_events_per_s_per_channel") is not None for row in rows)
        else "events_per_s"
    )

    # NaN rather than 0.0 for a segment with no spread to report: matplotlib
    # draws no whisker there, where a zero would assert a measured spread of
    # zero on a bar that was never measured at all.
    dispersion_key_name = _DISPERSION_SUMMARY_KEYS[dispersion]
    spreads = [row.get(dispersion_key_name) for row in rows]
    with_spread = sum(1 for value in spreads if value is not None)
    yerr = (
        [np.nan if value is None else float(value) for value in spreads]
        if with_spread
        else None
    )

    heights = [float(row.get(bar_key) or 0.0) for row in rows]
    ax.bar(
        positions,
        heights,
        width=1.0,
        color=_bar_colors(len(rows)),
        # Slightly translucent so the electrodes scattered over a bar stay
        # legible against it, whatever colour the cycle gave that bar.
        alpha=_BAR_ALPHA,
        edgecolor=_BAR_EDGE_COLOR,
        linewidth=_BAR_EDGE_WIDTH,
        yerr=yerr,
        ecolor=_ERROR_COLOR,
        capsize=_ERROR_CAPSIZE,
        error_kw={"elinewidth": _ERROR_LINEWIDTH, "zorder": 3},
    )

    # One generator for the whole figure, seeded fixed and consumed in segment
    # order, so the jitter is a pure function of the summary.
    rng = np.random.default_rng(_JITTER_SEED)
    jitter_x = []
    jitter_y = []
    thinned = 0
    for position, row in zip(positions, rows):
        values, cut = _scatter_values(row)
        if values is None:
            continue
        thinned += int(cut)
        jitter_x.append(
            position + rng.uniform(-_JITTER_HALF_WIDTH, _JITTER_HALF_WIDTH, size=values.size)
        )
        jitter_y.append(values)

    if jitter_x:
        # Rasterized like the raster's own dots: a full well is tens of
        # thousands of markers, which as vectors make a PDF unopenable and carry
        # no detail worth keeping vector.
        ax.scatter(
            np.concatenate(jitter_x),
            np.concatenate(jitter_y),
            s=_SCATTER_SIZE,
            c=_SCATTER_COLOR,
            marker=".",
            linewidths=0,
            alpha=_SCATTER_ALPHA,
            zorder=4,
            rasterized=True,
        )
    if thinned:
        logger.info(
            "scatter thinned to %d point(s) on %d of %d segment(s)",
            _MAX_SCATTER_POINTS, thinned, len(rows),
        )

    # Cut the axis just above the bulk of the points rather than above the
    # loudest electrode; see _SCATTER_BULK_PERCENTILE. Never below the tallest
    # bar with its whisker, so no bar is ever cropped by this.
    above_axis = 0
    if jitter_y:
        drawn = np.concatenate(jitter_y)
        # A bar with no whisker (NaN) contributes its own height, not a gap.
        tops = [
            height + (0.0 if error is None or np.isnan(error) else float(error))
            for height, error in zip(heights, yerr or [None] * len(heights))
        ]
        tallest = max(tops)
        bulk = float(np.percentile(drawn, _SCATTER_BULK_PERCENTILE))
        top = max(tallest, min(bulk, _MAX_AXIS_OVER_BARS * tallest))
        if top > 0.0:
            top *= _Y_HEADROOM
            above_axis = int(np.count_nonzero(drawn > top))
            ax.set_ylim(0.0, top)
    if above_axis:
        logger.info(
            "%d drawn electrode(s) sit above the top of the y axis at the %g-th "
            "percentile cut",
            above_axis, _SCATTER_BULK_PERCENTILE,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [str(row["rec"]) for row in rows],
        rotation=90,
        fontsize=_SEGMENT_LABEL_FONTSIZE,
    )
    # Exactly the block of bars and nothing else: at width 1.0 the first bar's
    # left edge is half a unit before its centre and the last bar's right edge
    # half a unit after, so anything wider is dead canvas.
    ax.set_xlim(min(positions) - 0.5, max(positions) + 0.5)
    ax.set_ylabel(
        _PER_CHANNEL_YLABEL if bar_key != "events_per_s" else _RATE_YLABEL,
        fontsize=_AXIS_LABEL_FONTSIZE,
    )
    # `wrap=True`: this label is a full clause rather than a word or two, and
    # the canvas it sits under can be as narrow as `_RATES_MIN_WIDTH_IN`.
    ax.set_xlabel(
        "segment (one recording configuration, in the order they were recorded)",
        fontsize=_AXIS_LABEL_FONTSIZE,
        wrap=True,
    )

    heading = title or _default_title(summary)
    if annotate and heading:
        # `wrap=True` is a safety net, not the normal path: at the reference
        # segment count or above this fits on one line at this font size, and
        # only a well with an unusually long generated title on a narrow,
        # few-segment canvas ever needs the wrap to fire.
        ax.set_title(heading, fontsize=_TITLE_FONTSIZE, wrap=True)

    comparison = dict((summary or {}).get("activity_comparison") or {})
    p_value = comparison.get("p_value")
    if show_comparison:
        # Same rows/order `_activity_comparison` filtered to build its groups,
        # so a two-group bracket spans the two bars that were actually tested
        # rather than assuming they are the first two on the axis.
        grouped_positions = [
            row["segment_index"] for row in rows if row.get("per_channel_events_per_s")
        ]
        # A measurement, not chrome, so it is drawn regardless of `annotate`
        # and whether or not it is significant — see `_draw_activity_comparison`.
        _draw_activity_comparison(ax, comparison, grouped_positions, positions)

    # Legend every encoding: a whisker and a dot are not self-evident, and a
    # reader cannot otherwise tell the spread of the electrodes from the error
    # on the mean. Drawn AFTER the comparison mark so the emptiest-corner scan
    # sees it and cannot choose a corner the mark already sits in.
    handles = []
    if yerr is not None:
        handles.append(
            _legend_line(_ERROR_COLOR, dispersion_key(dispersion), lw=_ERROR_LINEWIDTH)
        )
    if jitter_x:
        handles.append(_legend_dot(_SCATTER_COLOR, PER_ELECTRODE, size=4.0))
    if handles:
        legend_corner(ax, handles=handles)

    written = None
    if fig is not None:
        caption = ""
        if annotate:
            # The dispersion acronym is only ON the figure (in the legend key)
            # when a whisker was actually drawn, so it is only defined here
            # when one was — expanding an acronym the figure never printed
            # would fail the "printed here at least once" half of the rule.
            acronyms = ("MAD", _DISPERSION_ACRONYMS[dispersion]) if yerr is not None else ("MAD",)
            parts = [
                acronym_note(*acronyms),
                SEGMENT_BAND,
                PER_SEGMENT_ONLY,
                "A crossing is not an identified neuron: this counts downward "
                "threshold crossings on individual electrodes, so compare the bars "
                "against each other rather than reading an absolute firing rate off "
                "one of them.",
            ]
            if bar_key != "events_per_s":
                parts.append(
                    "Each bar is the mean over the electrodes detection covered, so "
                    "multiply by that electrode count for the segment's whole-array "
                    "rate. " + (
                        "Whiskers are the standard deviation (SD) across those "
                        "electrodes — the spread the scatter is already showing, not "
                        "the precision of the mean, which at electrode counts in the "
                        "hundreds is a much narrower quantity than the spread and would "
                        "read as a misleadingly tight bar next to a wide cloud."
                        if dispersion == "sd"
                        else
                        "Whiskers are the standard error of the mean (SEM) over those "
                        "electrodes — the spread of the segment's average, which is the "
                        "quantity being compared, not the spread of its electrodes."
                    )
                )
            if jitter_x:
                parts.append(
                    "Each point is one electrode's own rate in that segment, jittered "
                    "sideways so overlapping ones stay countable, and an electrode that "
                    "crossed nothing sits at zero."
                )
            if thinned:
                parts.append(
                    f"Segments with more than {_MAX_SCATTER_POINTS} electrodes show an "
                    f"evenly spaced {_MAX_SCATTER_POINTS}-electrode subsample; the bars "
                    "and the whiskers use every electrode."
                )
            if above_axis:
                parts.append(
                    f"{above_axis} of the electrodes drawn fire above the top of the "
                    "axis and are off the sheet: the axis is cut above the bulk of them "
                    "so the bars stay comparable, and every rate is in the summary "
                    "beside this figure."
                )
            if comparison.get("test") and p_value is not None:
                parts.append(
                    f"{comparison['test']} across segments: {_format_p(p_value)} "
                    f"(alpha {float(comparison.get('alpha') or _COMPARISON_ALPHA):g}). "
                    f"{_COMPARISON_NOTE}"
                )
            # `_fold_caption`'s own default wrap width is tuned for the module
            # constant's width; scaled down here so a narrow, few-segment
            # canvas wraps its lines short enough to stay on the page instead
            # of running past its right edge. The floor keeps a very narrow
            # canvas from wrapping to a column so thin the caption grows to
            # dozens of lines.
            caption_width = max(
                60, round(118 * fig.get_figwidth() / _RATES_FIGSIZE[0])
            )
            caption = _fold_caption(parts, width=caption_width)
        _add_caption(fig, caption)
        written = _save_and_release(fig, out_path)

    logger.info(
        "wrote segment event rates: %s (%d segment(s), %d without a measurable rate, "
        "%d with a spread)",
        written, len(rows), unmeasured, with_spread,
    )
    return {
        "files": {"png": None if written is None else str(written)},
        "n_segments": len(rows),
        "n_segments_unmeasured": unmeasured,
        "n_segments_with_spread": with_spread,
        "bar_quantity": bar_key,
        "dispersion": dispersion if yerr is not None else None,
        "scatter_points_per_segment_cap": _MAX_SCATTER_POINTS,
        "n_segments_scatter_thinned": thinned,
        "n_scatter_points_above_axis": above_axis,
        "activity_comparison": comparison or None,
    }


__all__ = ["segment_event_rate_summary", "plot_segment_event_rates"]
