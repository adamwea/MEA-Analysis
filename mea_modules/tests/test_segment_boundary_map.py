"""Tests for `mea_modules.diagnostics.segment_boundary_map`.

A figure cannot be asserted pixel by pixel without pinning matplotlib's own
rendering, so what is checked here is everything the figure is a picture OF: the
bar geometry the two panels place, which panels got drawn, and the span each
timeline reports. Those are the numbers a caller records and a reviewer reads
off the axes, and they are wrong in exactly the ways that matter -- a segment at
the wrong offset, a wall-clock panel silently missing.

The synthetic well: 100 Hz, two segments, the join at frame 1000 (= 10.0 s).
Segment A runs 0-10 s of file time and 0-10 s of real time; segment B follows it
in the file immediately but only starts on the clock 30 s later, which is the
gap the lower panel exists to show.

That 30 s is also what the join marking is checked against: on the file timeline
the join is the single instant 10.0 s, and on the clock it is the PAIR 10.0 s /
40.0 s, because the instrument was not recording in between.
"""

import pytest

from mea_modules.diagnostics.figure_text import FILE_TIME_AXIS, REAL_TIME_AXIS
from mea_modules.diagnostics.segment_boundary_map import plot_segment_boundary_map
from mea_modules.diagnostics.timebase import JOIN_LABEL_INSTANT, JOIN_LABEL_SPANNING

FS_HZ = 100.0

SEGMENTS = [
    {
        "rec": "rec0001",
        "n_samples": 1000,  # 10.0 s
        "start_frame": 0,
        "start_time": 0.0,
        "stop_time": 10.0,
        "gap_before_s": None,
    },
    {
        "rec": "rec0002",
        "n_samples": 500,  # 5.0 s
        "start_frame": 1000,
        "start_time": 40.0,
        "stop_time": 45.0,
        "gap_before_s": 30.0,
    },
]
STITCH_FRAMES = [1000]


def _new_axes(n):
    """`n` axes on a headless figure, the way a composing caller would supply them."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(6.0, 4.0))
    FigureCanvasAgg(fig)
    panels = fig.subplots(n, 1)
    return fig, ([panels] if n == 1 else list(panels))


def _many_segments(n):
    """`n` back-to-back segments, each 10 s long with a 30 s gap before the next.

    The 21-segment case is the real scan's shape, and the count the geometry has
    to stay legible at from the other end.
    """
    segments = []
    for index in range(n):
        start_time = index * 40.0
        segments.append(
            {
                "rec": f"rec{index:04d}",
                "n_samples": 1000,
                "start_frame": index * 1000,
                "start_time": start_time,
                "stop_time": start_time + 10.0,
                "gap_before_s": None if index == 0 else 30.0,
            }
        )
    return segments


def _without_timestamps(segments):
    """The same segments as a source that records no wall clock would hand over."""
    return [
        {key: value for key, value in entry.items() if key not in ("start_time", "stop_time")}
        for entry in segments
    ]


def _capture(monkeypatch):
    """Record what the assembled figure holds, just before it is written.

    `_save_and_release` clears the figure the moment it lands on disk, so nothing
    a test wants to read off the real (non-`axes`) path survives the call. The
    bottom-matter step is the last moment the figure is whole, so the recording
    happens there. Returns a dict the test reads after the call.
    """
    from mea_modules.diagnostics import segment_boundary_map as module

    seen = {}

    def spy(fig, text, **kwargs):
        seen["caption"] = text
        seen["titles"] = [axis.get_title() for axis in fig.axes]
        seen["xlabels"] = [axis.get_xlabel() for axis in fig.axes]
        seen["legend_labels"] = [
            handle.get_label() for handle in (kwargs.get("legend_handles") or ())
        ]
        seen["height_in"] = float(fig.get_figheight())
        return None

    monkeypatch.setattr(module, "_add_caption", spy)
    return seen


def _legend_labels(ax):
    """The labels on an axes' own legend, which is what a supplied-axes call keys."""
    legend = ax.get_legend()
    return [] if legend is None else [text.get_text() for text in legend.get_texts()]


def test_both_timelines_are_drawn_and_each_reports_its_own_span(tmp_path):
    """15.0 s in the file, 45.0 s on the clock -- the 30 s difference IS the
    figure's argument, so the two spans must not be the same number."""
    out_path = tmp_path / "segment_boundary_map.png"
    manifest = plot_segment_boundary_map(
        SEGMENTS, FS_HZ, out_path, stitch_frames=STITCH_FRAMES
    )
    assert manifest["panels"] == ["concatenated", "wall_clock"]
    assert manifest["n_segments"] == 2
    assert manifest["concatenated_span_s"] == 15.0
    assert manifest["wall_clock_span_s"] == 45.0
    assert manifest["note"] is None
    assert manifest["files"]["png"] == str(out_path)
    assert out_path.is_file() and out_path.stat().st_size > 0


def test_bars_land_at_the_offsets_the_two_timelines_actually_have():
    """Segment B abuts A in the file (left = 10.0 s) but starts 40.0 s into the
    recording session. Plotting the file offset on the clock panel is the bug
    this figure exists to make visible, so it must not be the bug in the figure.
    """
    fig, (upper, lower) = _new_axes(2)
    plot_segment_boundary_map(
        SEGMENTS, FS_HZ, axes=(upper, lower), stitch_frames=STITCH_FRAMES
    )

    # barh draws each bar as a Rectangle whose x/width are the timeline extent.
    upper_bars = sorted((patch.get_x(), patch.get_width()) for patch in upper.patches)
    assert upper_bars == [(0.0, 10.0), (10.0, 5.0)]

    lower_bars = sorted((patch.get_x(), patch.get_width()) for patch in lower.patches)
    assert lower_bars == [(0.0, 10.0), (40.0, 5.0)]
    assert fig is not None


def test_the_join_is_drawn_once_at_its_own_time():
    """Frame 1000 at 100 Hz is 10.0 s. Drawing the raw frame number instead is
    the classic frames-where-seconds-were-expected slip, and it puts the rule
    900 s off the end of the axes where nobody sees it missing."""
    _fig, (upper,) = _new_axes(1)
    plot_segment_boundary_map(SEGMENTS, FS_HZ, axes=upper, stitch_frames=STITCH_FRAMES)
    seam_x = [line.get_xdata()[0] for line in upper.lines]
    assert seam_x == [10.0]


def test_a_source_without_timestamps_still_gets_the_file_panel(tmp_path):
    """The concatenated panel needs no clock, so losing the timestamps must cost
    the lower panel and nothing else -- with a note saying so, because a figure
    that quietly arrives with one panel reads as a figure with one panel."""
    segments = [
        {key: value for key, value in entry.items() if key not in ("start_time", "stop_time")}
        for entry in SEGMENTS
    ]
    out_path = tmp_path / "segment_boundary_map.png"
    manifest = plot_segment_boundary_map(
        segments, FS_HZ, out_path, stitch_frames=STITCH_FRAMES
    )
    assert manifest["panels"] == ["concatenated"]
    assert manifest["wall_clock_span_s"] is None
    assert "no usable segment timestamps" in manifest["note"]
    assert out_path.is_file()


def test_one_missing_timestamp_drops_the_panel_rather_than_placing_the_rest():
    """Partial timestamps are worse than none: the remaining segments would be
    offset from a first segment whose own start is unknown."""
    segments = [dict(SEGMENTS[0]), dict(SEGMENTS[1])]
    segments[1]["stop_time"] = None
    _fig, panels = _new_axes(2)
    manifest = plot_segment_boundary_map(
        segments, FS_HZ, axes=panels, stitch_frames=STITCH_FRAMES
    )
    assert manifest["panels"] == ["concatenated"]
    assert len(panels[1].patches) == 0


def test_one_supplied_axes_draws_the_file_panel_and_says_why():
    _fig, (only,) = _new_axes(1)
    manifest = plot_segment_boundary_map(
        SEGMENTS, FS_HZ, axes=only, stitch_frames=STITCH_FRAMES
    )
    assert manifest["panels"] == ["concatenated"]
    assert "one axes" in manifest["note"]
    assert manifest["files"]["png"] is None


def test_the_clock_panel_brackets_each_join_with_two_marks():
    """A join on the clock is not an instant. One rule there would pin it to one
    edge of the gap and leave the reader to read the other edge as nothing --
    the missing marking this panel was reviewed for."""
    _fig, (upper, lower) = _new_axes(2)
    plot_segment_boundary_map(
        SEGMENTS, FS_HZ, axes=(upper, lower), stitch_frames=STITCH_FRAMES
    )
    assert [line.get_xdata()[0] for line in upper.lines] == [10.0]
    # A stops recording 10.0 s in, B starts 40.0 s in; the 30 s between them is
    # the chip re-routing, and both edges of it are drawn.
    assert [line.get_xdata()[0] for line in lower.lines] == [10.0, 40.0]


def test_each_panel_names_its_join_in_the_shared_vocabulary():
    """The two panels mean different things by "join", so they must not use the
    same words for it -- and the words are the shared ones, not local prose."""
    _fig, (upper, lower) = _new_axes(2)
    plot_segment_boundary_map(
        SEGMENTS, FS_HZ, axes=(upper, lower), stitch_frames=STITCH_FRAMES
    )
    assert JOIN_LABEL_INSTANT in _legend_labels(upper)
    assert JOIN_LABEL_SPANNING in _legend_labels(lower)


def test_the_gap_label_sits_inside_the_gap_it_describes():
    """The label used to land on the x of the segment AFTER the gap -- reading as
    that segment's own annotation rather than a measurement of what came before
    it. It must sit at the gap's own midpoint, on the same row, instead."""
    _fig, (_upper, lower) = _new_axes(2)
    plot_segment_boundary_map(
        SEGMENTS, FS_HZ, axes=(_upper, lower), stitch_frames=STITCH_FRAMES
    )
    gap_texts = [t for t in lower.texts if "gap" in t.get_text()]
    assert len(gap_texts) == 1
    x, y = gap_texts[0].get_position()
    # rec0002's row: it stops at file end but the gap it reports runs from
    # rec0001's stop (10.0 s) to its own start (40.0 s) -- the midpoint is 25.0.
    assert x == pytest.approx(25.0)
    assert y == 1
    assert gap_texts[0].get_ha() == "center"


def test_a_gap_too_narrow_for_its_label_moves_above_the_row_not_onto_a_bar():
    """A gap a few hundredths of a second wide, next to two 600 s gaps, cannot
    hold "+0.1s gap" without touching a bar. The fallback drops it into the
    empty band above its row instead of leaving it to overlap one."""
    segments = [
        {
            "rec": "rec0000", "n_samples": 1000, "start_frame": 0,
            "start_time": 0.0, "stop_time": 10.0, "gap_before_s": None,
        },
        {
            "rec": "rec0001", "n_samples": 1000, "start_frame": 1000,
            "start_time": 610.0, "stop_time": 620.0, "gap_before_s": 600.0,
        },
        {
            "rec": "rec0002", "n_samples": 1000, "start_frame": 2000,
            "start_time": 620.05, "stop_time": 630.05, "gap_before_s": 0.05,
        },
    ]
    _fig, (only,) = _new_axes(1)
    plot_segment_boundary_map(segments, FS_HZ, axes=only, panels="clock")
    by_text = {t.get_text(): t.get_position()[1] for t in only.texts}
    assert by_text["+600.0s gap"] == 1  # wide enough: stays on rec0001's own row
    assert by_text["+0.1s gap"] == pytest.approx(1.5)  # too narrow: falls back above


def test_legend_keys_are_short_and_stay_distinct_between_panels():
    """Adam's ruling: a legend key is two or three words, not a clause -- and the
    two coloured keys must stay distinct, or the two-panel figure's deduped
    legend would drop one colour's key outright."""
    _fig, (upper, lower) = _new_axes(2)
    plot_segment_boundary_map(
        SEGMENTS, FS_HZ, axes=(upper, lower), stitch_frames=STITCH_FRAMES
    )
    assert _legend_labels(upper) == ["segment (file time)", JOIN_LABEL_INSTANT]
    assert _legend_labels(lower) == ["segment (elapsed time)", JOIN_LABEL_SPANNING]


def test_axis_labels_use_the_shared_shorthand():
    """`_CONCATENATED_XLABEL` / `_WALL_CLOCK_XLABEL` are retired in favour of the
    shared vocabulary -- the filename token now carries the distinction the long
    wording used to spell out."""
    _fig, (upper, lower) = _new_axes(2)
    plot_segment_boundary_map(
        SEGMENTS, FS_HZ, axes=(upper, lower), stitch_frames=STITCH_FRAMES
    )
    assert upper.get_xlabel() == FILE_TIME_AXIS
    assert lower.get_xlabel() == REAL_TIME_AXIS


def test_a_join_with_no_real_gap_is_one_mark_on_both_panels():
    """Back-to-back acquisitions do exist. The two-mark treatment is a property
    of the gap, not of the panel, so an absent gap must collapse to one rule and
    say so in the legend."""
    segments = [dict(entry) for entry in SEGMENTS]
    segments[1]["start_time"] = 10.0
    segments[1]["stop_time"] = 15.0
    segments[1]["gap_before_s"] = 0.0
    _fig, (upper, lower) = _new_axes(2)
    plot_segment_boundary_map(
        segments, FS_HZ, axes=(upper, lower), stitch_frames=STITCH_FRAMES
    )
    assert [line.get_xdata()[0] for line in lower.lines] == [10.0]
    assert JOIN_LABEL_INSTANT in _legend_labels(lower)


def test_each_panel_selector_writes_one_figure_of_its_own(tmp_path):
    """Three calls, three files, one panel each where asked. The spans describe
    the data rather than the drawing, so the three stay comparable."""
    manifests = {}
    for choice, expected in (
        ("both", ["concatenated", "wall_clock"]),
        ("file", ["concatenated"]),
        ("clock", ["wall_clock"]),
    ):
        out_path = tmp_path / f"segment_boundary_map_{choice}.png"
        manifest = plot_segment_boundary_map(
            SEGMENTS, FS_HZ, out_path, stitch_frames=STITCH_FRAMES, panels=choice
        )
        assert manifest["panels"] == expected
        assert manifest["note"] is None
        assert manifest["concatenated_span_s"] == 15.0
        assert manifest["wall_clock_span_s"] == 45.0
        assert out_path.is_file() and out_path.stat().st_size > 0
        manifests[choice] = manifest
    assert len({manifest["files"]["png"] for manifest in manifests.values()}) == 3


def test_the_clock_selector_fills_the_one_axes_it_is_handed():
    """A single-panel selector with a single axes is not the degraded two-panel
    case: the requested panel goes in, and there is nothing to note."""
    _fig, (only,) = _new_axes(1)
    manifest = plot_segment_boundary_map(SEGMENTS, FS_HZ, axes=only, panels="clock")
    assert manifest["panels"] == ["wall_clock"]
    assert manifest["note"] is None
    bars = sorted((patch.get_x(), patch.get_width()) for patch in only.patches)
    assert bars == [(0.0, 10.0), (40.0, 5.0)]


def test_the_file_selector_needs_no_clock_and_raises_no_note(tmp_path):
    """Asking for the file panel on a source with no timestamps is not a
    degraded figure -- nothing was dropped, so nothing should be reported."""
    out_path = tmp_path / "file_only.png"
    manifest = plot_segment_boundary_map(
        _without_timestamps(SEGMENTS), FS_HZ, out_path, panels="file"
    )
    assert manifest["panels"] == ["concatenated"]
    assert manifest["note"] is None
    assert manifest["wall_clock_span_s"] is None


def test_annotate_false_drops_the_title_and_caption_and_keeps_the_rest(tmp_path, monkeypatch):
    """The presentation register: the title's information moves to the filename,
    but the axes, their units and the key are what make the picture readable at
    all and must survive."""
    seen = _capture(monkeypatch)
    plot_segment_boundary_map(
        SEGMENTS, FS_HZ, tmp_path / "plain.png", stitch_frames=STITCH_FRAMES, annotate=False
    )
    assert seen["titles"] == ["", ""]
    assert seen["caption"] == ""
    assert seen["legend_labels"]
    assert all(seen["xlabels"])


def test_annotate_true_is_still_the_titled_captioned_figure(tmp_path, monkeypatch):
    """The default must not have moved: every caller outside this migration gets
    what it always got."""
    seen = _capture(monkeypatch)
    plot_segment_boundary_map(
        SEGMENTS, FS_HZ, tmp_path / "annotated.png", stitch_frames=STITCH_FRAMES
    )
    assert all(seen["titles"])
    assert "Segment join" in seen["caption"]
    assert seen["legend_labels"]


def test_a_title_override_exists_for_each_panel(tmp_path, monkeypatch):
    """Both panels are titled, so both need an override -- otherwise a
    single-panel clock figure cannot be named by its caller."""
    seen = _capture(monkeypatch)
    plot_segment_boundary_map(
        SEGMENTS,
        FS_HZ,
        tmp_path / "titled.png",
        stitch_frames=STITCH_FRAMES,
        title="well A — file",
        wall_clock_title="well A — clock",
    )
    assert seen["titles"] == ["well A — file", "well A — clock"]


def test_the_canvas_follows_the_segment_count_instead_of_a_fixed_sheet(tmp_path, monkeypatch):
    """The retired figure was 7.5 in tall at two segments and at twenty-one. A
    height built from the row count is what makes a two-segment map tight and a
    twenty-one-segment one still legible."""
    seen = _capture(monkeypatch)

    plot_segment_boundary_map(SEGMENTS, FS_HZ, tmp_path / "two.png")
    two_both = seen["height_in"]
    plot_segment_boundary_map(_many_segments(21), FS_HZ, tmp_path / "many.png")
    many_both = seen["height_in"]
    plot_segment_boundary_map(SEGMENTS, FS_HZ, tmp_path / "one_panel.png", panels="file")
    two_file = seen["height_in"]
    plot_segment_boundary_map(
        SEGMENTS, FS_HZ, tmp_path / "bare.png", panels="file", annotate=False
    )
    two_file_bare = seen["height_in"]

    assert many_both > two_both
    assert two_both < 7.5
    assert two_file < two_both
    # No caption means no band reserved for one.
    assert two_file_bare < two_file
    # An explicit figsize still wins, for a caller composing a sheet by hand.
    plot_segment_boundary_map(SEGMENTS, FS_HZ, tmp_path / "fixed.png", figsize=(9.0, 6.0))
    assert seen["height_in"] == 6.0


def test_the_rows_keep_their_pitch_and_the_bars_stay_thin():
    """Bars must not grow to fill the panel when there are few segments: a row
    box of at least three rows is what stops two segments drawing as slabs, and
    a bar under half a row is what stops neighbouring rows touching."""
    _fig, (upper,) = _new_axes(1)
    plot_segment_boundary_map(SEGMENTS, FS_HZ, axes=upper, panels="file")
    assert upper.get_ylim() == (2.5, -0.5)
    assert all(patch.get_height() < 0.5 for patch in upper.patches)

    _fig, (wide,) = _new_axes(1)
    plot_segment_boundary_map(_many_segments(21), FS_HZ, axes=wide, panels="file")
    assert wide.get_ylim() == (20.5, -0.5)
    assert all(patch.get_height() < 0.5 for patch in wide.patches)


def test_it_refuses_input_it_cannot_place():
    with pytest.raises(ValueError, match="no segments"):
        plot_segment_boundary_map([], FS_HZ, "unused.png")
    with pytest.raises(ValueError, match="fs_hz must be positive"):
        plot_segment_boundary_map(SEGMENTS, 0.0, "unused.png")
    with pytest.raises(ValueError, match="out_path.*or axes"):
        plot_segment_boundary_map(SEGMENTS, FS_HZ)
    with pytest.raises(ValueError, match="panels must be one of"):
        plot_segment_boundary_map(SEGMENTS, FS_HZ, "unused.png", panels="upper")


def test_a_clock_only_figure_refuses_a_source_with_no_clock():
    """There is no honest half of a wall-clock figure. The two-panel call still
    degrades to a note, because it has another panel to show."""
    segments = _without_timestamps(SEGMENTS)
    with pytest.raises(ValueError, match="no usable per-segment timestamps"):
        plot_segment_boundary_map(segments, FS_HZ, "unused.png", panels="clock")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
