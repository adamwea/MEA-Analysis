"""Pure-arithmetic tests for `mea_modules.diagnostics.segment_event_rates`.

The whole point of splitting `segment_event_rate_summary` out of the rendering
is that the numbers can be asserted without a figure, so these tests hand it a
handful of event times on a made-up sampling rate and check the rates by hand.
No recording, no detection, no SpikeInterface — the detection pass that produces
`event_times_s` lives in `mea_modules.diagnostics.raster` and is not re-tested
here.

The synthetic well throughout: 100 Hz, two segments, the join at frame 1000
(= 10.0 s). Two segments rather than a long scan because everything these
functions do is per-segment arithmetic, and two is the smallest number that has
a join in it. Rates are chosen to come out as exact decimals, so a failure is a
real disagreement and not a floating-point tolerance argument.
"""

import numpy as np
import pytest

from mea_modules.diagnostics.segment_event_rates import (
    plot_segment_event_rates,
    segment_event_rate_summary,
)

FS_HZ = 100.0

# Segment A is twice as long as segment B, which is what makes "divide each
# segment by its OWN duration" distinguishable from any pooled average.
SEGMENTS = [
    {"rec": "rec0001", "n_samples": 1000},  # 10.0 s
    {"rec": "rec0002", "n_samples": 500},  # 5.0 s
]
STITCH_FRAMES = [1000]

# Five events inside A, three inside B.
EVENT_TIMES_S = [0.5, 1.0, 2.0, 3.0, 9.999, 10.0, 11.0, 14.9]


def test_each_segment_rate_uses_its_own_duration_not_the_pooled_one():
    """A: 5 events / 10.0 s = 0.5 /s. B: 3 events / 5.0 s = 0.6 /s.

    The pooled rate would be 8 / 15.0 = 0.5333 /s for both, which is exactly the
    number a rate computed across the join would produce -- and it matches
    neither segment.
    """
    summary = segment_event_rate_summary(
        SEGMENTS, EVENT_TIMES_S, FS_HZ, stitch_frames=STITCH_FRAMES
    )
    rows = summary["segments"]
    assert [row["rec"] for row in rows] == ["rec0001", "rec0002"]
    assert [row["n_events"] for row in rows] == [5, 3]
    assert [row["recorded_s"] for row in rows] == [10.0, 5.0]
    np.testing.assert_allclose([row["events_per_s"] for row in rows], [0.5, 0.6])
    assert not np.allclose(
        [row["events_per_s"] for row in rows], 8.0 / 15.0
    ), "landed on the pooled rate, i.e. a rate computed across the join"
    assert summary["total_events"] == 8


def test_an_event_exactly_on_the_join_belongs_to_the_later_segment():
    """Frame 1000 is segment B's first sample, so 10.0 s is B's event.

    Off-by-one here is silent: the count moves by one and nothing complains.
    """
    summary = segment_event_rate_summary(
        SEGMENTS, [10.0], FS_HZ, stitch_frames=STITCH_FRAMES
    )
    assert [row["n_events"] for row in summary["segments"]] == [0, 1]
    # One sample earlier is still A's.
    summary = segment_event_rate_summary(
        SEGMENTS, [10.0 - 1.0 / FS_HZ], FS_HZ, stitch_frames=STITCH_FRAMES
    )
    assert [row["n_events"] for row in summary["segments"]] == [1, 0]


def test_per_channel_rate_divides_by_the_channel_count():
    """0.5 /s over 4 channels is 0.125 /s/channel; 0.6 /s is 0.15."""
    summary = segment_event_rate_summary(
        SEGMENTS, EVENT_TIMES_S, FS_HZ, stitch_frames=STITCH_FRAMES, n_channels=4
    )
    np.testing.assert_allclose(
        [row["events_per_s_per_channel"] for row in summary["segments"]], [0.125, 0.15]
    )
    assert summary["n_channels"] == 4


def test_unknown_channel_count_leaves_the_per_channel_rate_unset():
    """No denominator is reported rather than a guessed one -- a per-channel rate
    divided by the wrong channel count is worse than an absent field, because it
    is comparable-looking across wells and wrong."""
    summary = segment_event_rate_summary(
        SEGMENTS, EVENT_TIMES_S, FS_HZ, stitch_frames=STITCH_FRAMES
    )
    assert summary["n_channels"] is None
    assert [row["events_per_s_per_channel"] for row in summary["segments"]] == [None, None]


def test_a_segment_nothing_crossed_in_is_a_zero_and_a_named_check():
    """The dead-configuration read: a rate of exactly 0.0, and the segment's own
    name in `zero_event_segments` so a caller can gate on it without re-scanning
    the rows."""
    summary = segment_event_rate_summary(
        SEGMENTS, [0.5, 1.0], FS_HZ, stitch_frames=STITCH_FRAMES
    )
    rows = summary["segments"]
    assert rows[1]["n_events"] == 0
    assert rows[1]["events_per_s"] == 0.0
    assert summary["zero_event_segments"] == ["rec0002"]
    # ... and a well where everything fired names nothing.
    summary = segment_event_rate_summary(
        SEGMENTS, EVENT_TIMES_S, FS_HZ, stitch_frames=STITCH_FRAMES
    )
    assert summary["zero_event_segments"] == []


def test_a_zero_length_segment_has_no_rate_rather_than_a_zero_one():
    """Dividing by a zero duration is not 0.0 /s, it is unmeasured -- and the two
    mean opposite things to a reviewer (nothing fired vs nothing was recorded)."""
    segments = [{"rec": "rec0001", "n_samples": 1000}, {"rec": "rec0002", "n_samples": 0}]
    summary = segment_event_rate_summary(
        segments, [0.5, 1.0], FS_HZ, stitch_frames=[1000], n_channels=4
    )
    empty = summary["segments"][1]
    assert empty["recorded_s"] == 0.0
    assert empty["events_per_s"] is None
    assert empty["events_per_s_per_channel"] is None


def test_join_count_must_separate_the_segments():
    """Without this the placement silently piles every event into segment 0 and
    reports a plausible-looking table of zeros for the rest."""
    with pytest.raises(ValueError, match="cannot separate"):
        segment_event_rate_summary(SEGMENTS, EVENT_TIMES_S, FS_HZ, stitch_frames=())
    with pytest.raises(ValueError, match="cannot separate"):
        segment_event_rate_summary(
            SEGMENTS, EVENT_TIMES_S, FS_HZ, stitch_frames=[400, 1000]
        )


def test_a_single_segment_needs_no_join_and_takes_every_event():
    """One segment has no join, so `stitch_frames` is legitimately empty."""
    summary = segment_event_rate_summary(
        [{"rec": "rec0001", "n_samples": 1000}], [0.5, 1.0, 2.0, 3.0, 9.999], FS_HZ
    )
    assert summary["segments"][0]["n_events"] == 5
    np.testing.assert_allclose(summary["segments"][0]["events_per_s"], 0.5)


def test_empty_inputs_raise_rather_than_returning_an_empty_table():
    with pytest.raises(ValueError, match="no segments"):
        segment_event_rate_summary([], EVENT_TIMES_S, FS_HZ)
    with pytest.raises(ValueError, match="fs_hz must be positive"):
        segment_event_rate_summary(SEGMENTS, EVENT_TIMES_S, 0.0, stitch_frames=STITCH_FRAMES)


def test_threshold_factor_travels_with_the_numbers_as_provenance():
    """A rate has no meaning without the threshold that produced it, so the
    factor rides in the summary rather than only in the caller's own log."""
    summary = segment_event_rate_summary(
        SEGMENTS, EVENT_TIMES_S, FS_HZ, stitch_frames=STITCH_FRAMES, threshold_factor=5.0
    )
    assert summary["threshold_factor"] == 5.0
    assert "own recorded duration" in summary["note"]


def test_the_figure_reads_the_summary_and_reports_what_it_drew(tmp_path):
    """The renderer recomputes nothing: one bar per row, and the rows with no
    measurable rate are counted so a flat bar is never read as a measured zero."""
    segments = [{"rec": "rec0001", "n_samples": 1000}, {"rec": "rec0002", "n_samples": 0}]
    summary = segment_event_rate_summary(
        segments, [0.5, 1.0], FS_HZ, stitch_frames=[1000], n_channels=4,
        threshold_factor=5.0,
    )
    out_path = tmp_path / "segment_activity.png"
    manifest = plot_segment_event_rates(summary, out_path)
    assert manifest["n_segments"] == 2
    assert manifest["n_segments_unmeasured"] == 1
    assert manifest["files"]["png"] == str(out_path)
    assert out_path.is_file() and out_path.stat().st_size > 0


def test_the_figure_can_borrow_a_caller_axes_and_writes_nothing_then():
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    summary = segment_event_rate_summary(
        SEGMENTS, EVENT_TIMES_S, FS_HZ, stitch_frames=STITCH_FRAMES
    )
    fig = Figure(figsize=(4.0, 3.0))
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    manifest = plot_segment_event_rates(summary, ax=ax)
    assert manifest["files"]["png"] is None
    assert len(ax.patches) == 2  # one bar per segment, drawn on the caller's axes


def test_the_figure_refuses_a_summary_with_nothing_in_it():
    with pytest.raises(ValueError, match="no segments to plot"):
        plot_segment_event_rates({"segments": []}, "unused.png")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
