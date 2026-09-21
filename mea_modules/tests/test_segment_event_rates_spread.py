"""The per-electrode half of `mea_modules.diagnostics.segment_event_rates`.

The arithmetic that turns one rate per segment into one rate per electrode per
segment, the between-segment comparison built on it, and the figure's
reproducibility. The segment-level arithmetic itself (placement across a join,
per-segment durations, the zero-event check) is covered in
`mea_modules/tests/test_segment_event_rates_pure.py` and is not re-tested here.

The synthetic well: 100 Hz, four electrodes '0'..'3', two segments with the join
at frame 1000 (= 10.0 s), segment A twice as long as segment B. Electrode '3'
crosses nothing anywhere -- it is the silent channel that has to come back as a
real zero rather than as an absent row, because it is the one an average would
quietly be wrong without.

No recording, no detection, no SpikeInterface: the (times, labels) pair these
functions consume is what `mea_modules.quality.detect_events` yields (its
frames over the sampling rate, and its labels), written out by hand so a
failure is a real disagreement.
"""

import hashlib

import numpy as np
import pytest

from mea_modules.diagnostics.segment_event_rates import (
    plot_segment_event_rates,
    segment_event_rate_summary,
)

FS_HZ = 100.0

SEGMENTS = [
    {"rec": "rec0001", "n_samples": 1000},  # 10.0 s
    {"rec": "rec0002", "n_samples": 500},  # 5.0 s
]
STITCH_FRAMES = [1000]
CHANNEL_IDS = ["0", "1", "2", "3"]

#                 |------------- segment A -------------|  |--- segment B ---|
EVENT_TIMES_S = [0.5, 1.0, 2.0, 3.0, 9.999, 10.0, 11.0, 14.9]
EVENT_LABELS = [0, 0, 1, 1, 2, 0, 2, 2]


def _summary(**kwargs):
    """The synthetic well, summarized with the per-electrode breakdown on."""
    return segment_event_rate_summary(
        SEGMENTS,
        EVENT_TIMES_S,
        FS_HZ,
        stitch_frames=STITCH_FRAMES,
        event_channel_labels=EVENT_LABELS,
        channel_ids=CHANNEL_IDS,
        **kwargs,
    )


def test_the_per_channel_rates_add_back_up_to_the_segment_totals():
    """The decomposition is a decomposition: multiply each segment's per-channel
    rates by its own duration and the electrodes sum to that segment's
    `n_events`. Anything else means events were dropped or double-counted on the
    way onto the channel axis, which nothing downstream would notice."""
    rows = _summary()["segments"]
    assert [row["n_events"] for row in rows] == [5, 3]
    for row in rows:
        counted = sum(row["per_channel_events_per_s"]) * row["recorded_s"]
        assert counted == pytest.approx(row["n_events"])


def test_the_channel_mean_is_the_segment_rate_over_the_channel_count():
    """`mean_events_per_s_per_channel` and the older `events_per_s_per_channel`
    are the same quantity reached two ways -- one averaged over the electrodes,
    one divided by their count -- so a drift between them is a decomposition
    bug."""
    for row in _summary()["segments"]:
        assert row["mean_events_per_s_per_channel"] == pytest.approx(
            row["events_per_s_per_channel"]
        )


def test_a_channel_that_crossed_nothing_is_a_zero_and_not_an_absent_row():
    """Electrode '3' fires nowhere and electrode '2' fires only in B. Both are
    rows of the array in every segment: a silent electrode is a measurement, and
    dropping it would raise every mean and shrink every SEM by exactly the
    channels a reviewer is looking for."""
    rows = _summary()["segments"]
    assert [row["n_channels_measured"] for row in rows] == [4, 4]
    # A: 2, 2, 1, 0 events over 10.0 s. B: 1, 0, 2, 0 events over 5.0 s.
    np.testing.assert_allclose(rows[0]["per_channel_events_per_s"], [0.2, 0.2, 0.1, 0.0])
    np.testing.assert_allclose(rows[1]["per_channel_events_per_s"], [0.2, 0.0, 0.4, 0.0])
    assert len(rows[0]["per_channel_events_per_s"]) == len(CHANNEL_IDS)


def test_the_spread_is_the_sample_std_and_the_sem_that_follows_from_it():
    """Reported rather than recomputed downstream, so the figure's whiskers and
    any consumer's own number cannot disagree."""
    row = _summary()["segments"][0]
    rates = np.asarray(row["per_channel_events_per_s"], dtype=float)
    assert row["std_events_per_s_per_channel"] == pytest.approx(np.std(rates, ddof=1))
    assert row["sem_events_per_s_per_channel"] == pytest.approx(
        np.std(rates, ddof=1) / np.sqrt(rates.size)
    )


def test_a_zero_length_segment_has_no_per_channel_rates_either():
    """Unmeasured, not zero -- the same distinction the segment-level rate makes,
    carried into the per-electrode keys so a figure cannot draw a measured
    silence where nothing was recorded."""
    segments = [{"rec": "rec0001", "n_samples": 1000}, {"rec": "rec0002", "n_samples": 0}]
    summary = segment_event_rate_summary(
        segments,
        EVENT_TIMES_S,
        FS_HZ,
        stitch_frames=STITCH_FRAMES,
        event_channel_labels=EVENT_LABELS,
        channel_ids=CHANNEL_IDS,
    )
    empty = summary["segments"][1]
    assert empty["per_channel_events_per_s"] is None
    assert empty["sem_events_per_s_per_channel"] is None
    assert empty["n_channels_measured"] == 0


def test_without_the_channel_inputs_the_summary_is_exactly_what_it_was():
    """The new keys are absent rather than None, so an existing consumer sees no
    change at all -- and the comparison block says why it is empty instead of
    going missing."""
    summary = segment_event_rate_summary(
        SEGMENTS, EVENT_TIMES_S, FS_HZ, stitch_frames=STITCH_FRAMES
    )
    for row in summary["segments"]:
        assert "per_channel_events_per_s" not in row
        assert "sem_events_per_s_per_channel" not in row
    block = summary["activity_comparison"]
    assert block["test"] is None
    assert block["significant"] is None
    assert "event_channel_labels" in block["note"]


def test_labels_that_do_not_pair_with_the_events_raise_rather_than_mis_bin():
    """Both failures are silent otherwise: a short label array would shift every
    event onto the wrong electrode, and a stray label would land on whichever
    channel happened to sort next to it."""
    with pytest.raises(ValueError, match="must stay paired"):
        segment_event_rate_summary(
            SEGMENTS,
            EVENT_TIMES_S,
            FS_HZ,
            stitch_frames=STITCH_FRAMES,
            event_channel_labels=EVENT_LABELS[:-1],
            channel_ids=CHANNEL_IDS,
        )
    with pytest.raises(ValueError, match="not in channel_ids"):
        segment_event_rate_summary(
            SEGMENTS,
            EVENT_TIMES_S,
            FS_HZ,
            stitch_frames=STITCH_FRAMES,
            event_channel_labels=EVENT_LABELS,
            channel_ids=["0", "1"],
        )


def _many_channel_well(n_segments, per_segment_rates, n_channels=12, seed=0):
    """A well of `n_segments` 10 s segments at a given mean rate each.

    Built from a seeded generator so the comparison tests assert on a fixed
    number rather than on whatever the machine's entropy produced.
    """
    rng = np.random.default_rng(seed)
    segments = [{"rec": f"rec{i:04d}", "n_samples": 1000} for i in range(n_segments)]
    stitch = [1000 * i for i in range(1, n_segments)]
    times = []
    labels = []
    for index, rate in enumerate(per_segment_rates):
        for channel in range(n_channels):
            for _ in range(int(rng.poisson(rate * 10.0))):
                times.append(index * 10.0 + float(rng.uniform(0.0, 9.99)))
                labels.append(channel)
    order = np.argsort(np.asarray(times), kind="stable")
    return segment_event_rate_summary(
        segments,
        np.asarray(times)[order],
        FS_HZ,
        stitch_frames=stitch,
        event_channel_labels=np.asarray(labels)[order],
        channel_ids=[str(channel) for channel in range(n_channels)],
    )


def test_two_segments_get_wilcoxon_and_three_or_more_get_friedman():
    """The test is chosen by group count, not by the caller, and both are
    repeated-measures tests: the same electrodes are measured in every
    segment, and an unpaired test would throw that pairing away."""
    two = _many_channel_well(2, [1.0, 1.0])["activity_comparison"]
    assert two["test"] == "Wilcoxon signed-rank"
    assert two["effect_size_name"] == "matched-pairs rank-biserial"
    assert two["n_groups"] == 2

    three = _many_channel_well(3, [1.0, 1.0, 1.0])["activity_comparison"]
    assert three["test"] == "Friedman"
    assert three["effect_size_name"] == "Kendall's W"
    assert three["n_groups"] == 3

    for block in (two, three):
        assert block["alpha"] == 0.05
        assert block["p_value"] is not None
        assert block["n_electrodes"] == 12
        assert isinstance(block["significant"], bool)
        assert -1.0 <= block["effect_size"] <= 1.0


def test_kendalls_w_is_friedmans_statistic_over_n_times_k_minus_one():
    """W = chi2_F / (n (k - 1)): 0 when the segments rank the electrodes'
    rates no differently, 1 when every electrode moves the same way."""
    from scipy import stats

    summary = _many_channel_well(3, [0.4, 2.0, 4.0], n_channels=24)
    groups = [np.asarray(row["per_channel_events_per_s"]) for row in summary["segments"]]
    expected = stats.friedmanchisquare(*groups).statistic / (24 * 2)
    assert summary["activity_comparison"]["effect_size"] == pytest.approx(expected)
    assert 0.0 <= expected <= 1.0


def test_a_significant_difference_is_a_flag_and_says_so_rather_than_raising():
    """Segments five-fold apart in activity must come back significant -- and
    must come back, not raise: nothing in this pipeline gates on this number,
    and the note is what stops the next reader treating it as a gate."""
    block = _many_channel_well(3, [0.4, 2.0, 4.0], n_channels=24)["activity_comparison"]
    assert block["significant"] is True
    assert block["p_value"] < block["alpha"]
    assert "never a gate" in block["note"]
    assert "optimistic" in block["note"]


def test_segments_drawn_from_the_same_activity_are_not_flagged():
    """The other half of the flag: it has to stay quiet on a well behaving
    normally, or it is noise a reviewer learns to ignore."""
    block = _many_channel_well(3, [1.0, 1.0, 1.0], n_channels=24)["activity_comparison"]
    assert block["significant"] is False


def _structured_well(channel_rates, segment_scales, seed=0):
    """Per-electrode rates that differ strongly between electrodes, scaled per
    segment: the busy electrodes stay the busy ones, the average moves."""
    rng = np.random.default_rng(seed)
    n_segments = len(segment_scales)
    segments = [{"rec": f"rec{i:04d}", "n_samples": 1000} for i in range(n_segments)]
    times, labels = [], []
    for index, scale in enumerate(segment_scales):
        for channel, rate in enumerate(channel_rates):
            for _ in range(int(rng.poisson(rate * scale * 10.0))):
                times.append(index * 10.0 + float(rng.uniform(0.0, 9.99)))
                labels.append(channel)
    order = np.argsort(np.asarray(times), kind="stable")
    return segment_event_rate_summary(
        segments,
        np.asarray(times)[order],
        FS_HZ,
        stitch_frames=[1000 * i for i in range(1, n_segments)],
        event_channel_labels=np.asarray(labels)[order],
        channel_ids=[str(channel) for channel in range(len(channel_rates))],
    )


def test_stability_reads_high_when_the_same_electrodes_stay_busy():
    """The headline: each consecutive pair of segments ranks the electrodes
    alike, so rho is high on every pair even though the mean doubles."""
    rates = np.linspace(0.5, 20.0, 24)
    stability = _structured_well(rates, [1.0, 2.0, 1.0])["stability"]
    assert stability["n_segments_compared"] == 3
    assert stability["pairs"] == [["rec0000", "rec0001"], ["rec0001", "rec0002"]]
    assert len(stability["consecutive_spearman_rho"]) == 2
    assert stability["min_consecutive_rho"] > 0.9
    assert stability["median_consecutive_rho"] >= stability["min_consecutive_rho"]


def test_the_cv_is_the_sample_sd_of_the_segment_means_over_their_mean():
    rates = np.linspace(0.5, 20.0, 24)
    stability = _structured_well(rates, [1.0, 2.0, 1.0])["stability"]
    means = np.asarray(stability["segment_means"])
    assert stability["cv_of_segment_means"] == pytest.approx(
        np.std(means, ddof=1) / means.mean()
    )
    assert stability["cv_of_segment_means"] > 0.2


def test_a_pair_with_no_variation_has_no_rho_rather_than_a_made_up_one():
    """Every electrode silent in one segment: there is nothing to rank, and a
    rho of 0 would claim the ranking changed."""
    summary = _structured_well(np.linspace(0.5, 20.0, 24), [1.0, 0.0])
    stability = summary["stability"]
    assert stability["consecutive_spearman_rho"] == [None]
    assert stability["median_consecutive_rho"] is None
    assert stability["min_consecutive_rho"] is None


def test_no_per_channel_data_leaves_the_stability_block_empty_not_absent():
    summary = segment_event_rate_summary(
        SEGMENTS, EVENT_TIMES_S, FS_HZ, stitch_frames=STITCH_FRAMES, n_channels=4
    )
    stability = summary["stability"]
    assert stability["n_segments_compared"] == 0
    assert stability["consecutive_spearman_rho"] == []
    assert stability["cv_of_segment_means"] is None


def test_the_same_input_draws_the_same_summary_and_the_same_png(tmp_path):
    """Determinism is the hard bar: the scatter's jitter comes from a fixed seed,
    so two runs over one summary are byte-identical PNGs. A figure that shifted
    between renders would make every visual diff meaningless."""
    first = _summary(threshold_factor=5.0)
    second = _summary(threshold_factor=5.0)
    assert first == second

    paths = []
    for name in ("first.png", "second.png"):
        out_path = tmp_path / name
        plot_segment_event_rates(first, out_path)
        paths.append(hashlib.sha256(out_path.read_bytes()).hexdigest())
    assert paths[0] == paths[1]


def test_the_figure_reports_the_test_and_the_scatter_cap_it_drew_with(tmp_path):
    """The manifest is how a caller knows what the picture is: which quantity the
    bars are, that a cap exists at all, and the comparison the figure marked."""
    summary = _many_channel_well(3, [0.4, 2.0, 4.0], n_channels=24)
    manifest = plot_segment_event_rates(summary, tmp_path / "activity.png")
    assert manifest["bar_quantity"] == "mean_events_per_s_per_channel"
    assert manifest["scatter_points_per_segment_cap"] > 0
    assert manifest["n_segments_scatter_thinned"] == 0  # 24 channels is under the cap
    assert manifest["n_segments_with_spread"] == 3
    assert manifest["activity_comparison"]["test"] == "Friedman"
    assert manifest["stability"]["n_segments_compared"] == 3


def test_a_summary_with_no_per_channel_data_still_draws_the_old_bars(tmp_path):
    """The emitter degrades rather than refusing: without the breakdown the bars
    are the whole-segment rate exactly as before, with no whiskers and no
    points, and the manifest says which quantity was drawn."""
    summary = segment_event_rate_summary(
        SEGMENTS, EVENT_TIMES_S, FS_HZ, stitch_frames=STITCH_FRAMES, n_channels=4
    )
    manifest = plot_segment_event_rates(summary, tmp_path / "activity.png")
    assert manifest["bar_quantity"] == "events_per_s"
    assert manifest["n_segments_with_spread"] == 0
    assert manifest["activity_comparison"]["test"] is None


def test_annotate_false_drops_the_title_and_caption_and_keeps_the_axes():
    """The presentation register: no title, no caption band, but the units, the
    legend and the significance mark stay -- they are measurements, not chrome."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    summary = _many_channel_well(3, [0.4, 2.0, 4.0], n_channels=24)
    drawn = {}
    for annotate in (True, False):
        fig = Figure(figsize=(6.0, 4.0))
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        plot_segment_event_rates(summary, ax=ax, title="a title", annotate=annotate)
        drawn[annotate] = (
            ax.get_title(),
            ax.get_ylabel(),
            ax.get_legend() is not None,
            [text.get_text() for text in ax.texts],
        )

    assert drawn[True][0] == "a title"
    assert drawn[False][0] == ""
    assert drawn[False][1] == drawn[True][1]
    assert drawn[False][2] is drawn[True][2] is True
    assert drawn[False][3] == drawn[True][3]
    assert any("p" in text for text in drawn[False][3]), "the significance mark is data"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
