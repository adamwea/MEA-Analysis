"""Legend/annotation contract for the core review figures.

Adam's ruling (2026-08-11): every diagnostic figure must carry a legend, or an
equivalent on-figure annotation, explaining every visual encoding it uses. He
hit it on capsule 05's ``channel_layout.png`` — grey dots and red dots with
nothing on the figure saying which is which.

These tests assert on the REAL figure. Each emitter writes through
``_save_and_release``, so the tests patch that one function to capture the
``Figure`` before its artists are cleared: the public entry point is still
exercised end to end (the PNG is written and asserted non-trivial), but the
assertions run against actual legend handles, axis labels and colorbar labels
rather than against a string constant that could drift from what is drawn.

Label text is soft-wrapped by ``_wrap_label``/``_fold_caption``, so every
comparison is whitespace-normalised.
"""

import re

import numpy as np
import pytest

from mea_modules.diagnostics import channel_layout as cl
from mea_modules.diagnostics import figure_text as ft
from mea_modules.diagnostics import raster as ra
from mea_modules.diagnostics import traces as tr
from mea_modules.reconstruction import plots as rp


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------

def _norm(text):
    """Collapse whitespace so wrapped labels compare equal to their source."""
    return re.sub(r"\s+", " ", str(text)).strip()


class _Captured:
    def __init__(self):
        self.figure = None

    @property
    def legend_labels(self):
        out = []
        for axis in self.figure.get_axes():
            legend = axis.get_legend()
            if legend is not None:
                out.extend(_norm(t.get_text()) for t in legend.get_texts())
        for legend in getattr(self.figure, "legends", ()):
            out.extend(_norm(t.get_text()) for t in legend.get_texts())
        return out

    @property
    def all_text(self):
        """Every string drawn anywhere on the figure, normalised."""
        pieces = [_norm(t.get_text()) for t in self.figure.texts]
        for axis in self.figure.get_axes():
            pieces.append(_norm(axis.get_xlabel()))
            pieces.append(_norm(axis.get_ylabel()))
            pieces.append(_norm(axis.get_title()))
            pieces.extend(_norm(t.get_text()) for t in axis.texts)
            legend = axis.get_legend()
            if legend is not None:
                pieces.extend(_norm(t.get_text()) for t in legend.get_texts())
        for legend in getattr(self.figure, "legends", ()):
            pieces.extend(_norm(t.get_text()) for t in legend.get_texts())
        return " || ".join(p for p in pieces if p)


@pytest.fixture
def capture(monkeypatch):
    """Capture the Figure each emitter saves, without changing what it writes."""
    captured = _Captured()
    for module in (cl, tr, ra):
        real = module._save_and_release

        def wrapper(fig, out_path, _real=real, _cap=captured):
            _cap.figure = fig
            # Snapshot before the real call clears the artists.
            _cap._frozen_legends = _cap.legend_labels
            _cap._frozen_text = _cap.all_text
            return _real(fig, out_path)

        monkeypatch.setattr(module, "_save_and_release", wrapper)
    return captured


class FakeRecording:
    """Minimal SpikeInterface-shaped recording on a clumped Maxwell-ish layout."""

    def __init__(self, n_clusters=12, per_cluster=3, fs=20_000.0, n_samples=40_000):
        rng = np.random.default_rng(0)
        xs, ys, ids = [], [], []
        index = 0
        for cluster in range(n_clusters):
            cx = 100.0 * (cluster % 4)
            cy = 100.0 * (cluster // 4)
            for member in range(per_cluster):
                xs.append(cx + 17.5 * member)
                ys.append(cy + 17.5 * (member % 2))
                ids.append(f"ch{index}")
                index += 1
        self._ids = ids
        self._loc = np.c_[xs, ys].astype(float)
        self._fs = float(fs)
        self._n = int(n_samples)
        data = rng.normal(0.0, 5.0, size=(self._n, len(ids)))
        # A few real deflections so the raster detects something.
        for channel in range(0, len(ids), 5):
            data[rng.integers(10, self._n - 10, size=40), channel] -= 90.0
        self._data = data.astype("float32")

    def get_channel_ids(self):
        return list(self._ids)

    def get_channel_locations(self):
        return self._loc

    def get_sampling_frequency(self):
        return self._fs

    def get_num_samples(self):
        return self._n

    def has_scaleable_traces(self):
        return True

    def has_time_vector(self):
        return False

    def get_traces(self, start_frame=0, end_frame=None, channel_ids=None, return_in_uV=True):
        end_frame = self._n if end_frame is None else int(end_frame)
        wanted = list(channel_ids) if channel_ids is not None else self._ids
        cols = [self._ids.index(str(c)) for c in wanted]
        return self._data[int(start_frame):end_frame][:, cols]


@pytest.fixture
def recording():
    return FakeRecording()


# --------------------------------------------------------------------------
# channel_layout — Adam's trigger case
# --------------------------------------------------------------------------

def test_channel_layout_legend_names_both_colours(capture, recording, tmp_path):
    out = cl.plot_channel_layout(
        recording, tmp_path / "channel_layout.png",
        highlight_channel_ids=["ch0", "ch3"],
        highlight_label="representative channels — traced in traces.png",
    )
    assert out.exists() and out.stat().st_size > 5_000
    labels = " || ".join(capture._frozen_legends)
    # Grey is explained, with its own count.
    assert "routed electrodes" in labels
    # Red is explained AND names the sibling figure it feeds.
    assert "representative channels" in labels
    assert "traces.png" in labels
    assert "(n=2)" in labels


def test_channel_layout_legend_names_several_downstream_files(capture, recording, tmp_path):
    """A layout serving more than one sibling names them all."""
    cl.plot_channel_layout(
        recording, tmp_path / "layout.png",
        highlight_channel_ids=["ch0"],
        highlight_label="representative channels — traced in traces.png, traces_realtime.png, psd.png",
    )
    labels = " || ".join(capture._frozen_legends)
    for name in ("traces.png", "traces_realtime.png", "psd.png"):
        assert name in labels, f"{name} missing from the legend"


def test_channel_layout_legend_survives_no_highlight(capture, recording, tmp_path):
    """Grey still needs explaining when nothing is highlighted."""
    cl.plot_channel_layout(recording, tmp_path / "layout.png")
    labels = " || ".join(capture._frozen_legends)
    assert "routed electrodes" in labels
    assert len(capture._frozen_legends) == 1


def test_channel_layout_states_geometry_units(capture, recording, tmp_path):
    cl.plot_channel_layout(recording, tmp_path / "layout.png")
    text = capture._frozen_text
    assert "x (µm)" in text and "y (µm)" in text
    assert "(um)" not in text


def test_channel_layout_caption_reaches_the_figure(capture, recording, tmp_path):
    cl.plot_channel_layout(
        recording, tmp_path / "layout.png",
        highlight_channel_ids=["ch0"],
        highlight_label="representative channels — traced in traces.png",
        caption=ft.REPRESENTATIVE_CHANNELS,
    )
    assert "one channel is taken from each clump" in capture._frozen_text


def test_channel_layout_is_deterministic(recording, tmp_path):
    first = cl.plot_channel_layout(
        recording, tmp_path / "a.png", highlight_channel_ids=["ch0"],
        highlight_label="representative channels — traced in traces.png")
    second = cl.plot_channel_layout(
        recording, tmp_path / "b.png", highlight_channel_ids=["ch0"],
        highlight_label="representative channels — traced in traces.png")
    assert first.read_bytes() == second.read_bytes()


# --------------------------------------------------------------------------
# traces
# --------------------------------------------------------------------------

def test_traces_legend_keys_every_encoding(capture, recording, tmp_path):
    tr.plot_traces(
        recording, tmp_path / "traces.png",
        channel_ids=["ch0", "ch5"], duration_s=1.0,
        stitch_frames=(10_000,), title="segment traces",
    )
    labels = " || ".join(capture._frozen_legends)
    assert "one row per channel" in labels
    assert "µV" in labels
    assert "segment join" in labels


def test_traces_caption_defines_a_segment_join_in_plain_language(capture, recording, tmp_path):
    tr.plot_traces(
        recording, tmp_path / "traces.png", channel_ids=["ch0"],
        duration_s=1.0, stitch_frames=(10_000,),
    )
    text = capture._frozen_text
    assert "the file looks continuous there" in text.lower()
    assert "re-routed electrodes" in text
    # And it must not lean on our jargon.
    assert "seam" not in text.lower()


def test_traces_caption_takes_the_callers_cross_reference(capture, recording, tmp_path):
    tr.plot_traces(
        recording, tmp_path / "traces.png", channel_ids=["ch0"], duration_s=1.0,
        caption_extra="These are the channels marked red in channel_layout.png.",
    )
    assert "channel_layout.png" in capture._frozen_text


def test_traces_states_amplitude_units(capture, recording, tmp_path):
    tr.plot_traces(recording, tmp_path / "traces.png", channel_ids=["ch0"], duration_s=1.0)
    assert "amplitude (µV)" in capture._frozen_text


def test_traces_real_elapsed_axis_explains_the_shading(capture, recording, tmp_path):
    gaps = {"break_sample_indices": [5_000], "break_gap_frames": [40_000]}
    tr.plot_traces(
        recording, tmp_path / "traces_realtime.png", channel_ids=["ch0"],
        duration_s=1.0, time_gaps=gaps,
    )
    labels = " || ".join(capture._frozen_legends)
    assert "no data recorded" in labels
    assert "real elapsed time" in capture._frozen_text


# --------------------------------------------------------------------------
# raster
# --------------------------------------------------------------------------

def test_raster_legend_states_threshold_and_units(capture, recording, tmp_path):
    ra.plot_raster_threshold(
        recording, tmp_path / "raster_threshold.png",
        duration_s=1.0, segment_boundaries=(0.2, 0.6),
    )
    labels = " || ".join(capture._frozen_legends)
    assert "threshold crossing" in labels
    assert "MAD-sigma" in labels
    assert "refractory" in labels
    assert "segment join" in labels


def test_raster_caption_expands_MAD(capture, recording, tmp_path):
    ra.plot_raster_threshold(recording, tmp_path / "raster.png", duration_s=1.0)
    text = capture._frozen_text
    assert "median absolute deviation" in text
    assert "0.6745" in text


def test_raster_caption_says_it_is_not_spike_sorting(capture, recording, tmp_path):
    """The 'proxy not a model' caveat belongs on the figure."""
    ra.plot_raster_threshold(recording, tmp_path / "raster.png", duration_s=1.0)
    assert "not spike sorting" in capture._frozen_text


# --------------------------------------------------------------------------
# footprint diagnostic colorbars (the unlabelled-colorbar defect)
# --------------------------------------------------------------------------

def _footprint_inputs(n_channels=40, n_samples=60):
    rng = np.random.default_rng(1)
    template = rng.normal(0.0, 3.0, size=(n_channels, n_samples))
    template[7, 30] = -180.0
    locations = np.c_[
        np.repeat(np.arange(8) * 17.5, 5)[:n_channels],
        np.tile(np.arange(5) * 17.5, 8)[:n_channels],
    ].astype(float)
    return template, locations


def test_footprint_diagnostic_colorbars_are_labelled_with_units(tmp_path):
    import matplotlib.pyplot as plt

    template, locations = _footprint_inputs()
    out = rp.plot_unit_footprint_diagnostic(
        template, locations, tmp_path / "footprint_diagnostic.png",
        unit_id=7, fs=20_000.0,
    )
    assert out.exists() and out.stat().st_size > 5_000

    # Re-render under a captured figure to read the colorbar labels back.
    captured = {}
    real_close = plt.close

    def spy(fig):
        captured.setdefault("fig", fig)
        labels = []
        for axis in fig.get_axes():
            labels.append(_norm(axis.get_ylabel()))
            labels.append(_norm(axis.get_xlabel()))
        captured["labels"] = labels
        captured["texts"] = [_norm(t.get_text()) for t in fig.texts]
        return real_close(fig)

    plt.close = spy
    try:
        rp.plot_unit_footprint_diagnostic(
            template, locations, tmp_path / "again.png", unit_id=7, fs=20_000.0)
    finally:
        plt.close = real_close

    joined = " || ".join(captured["labels"])
    assert "peak |amplitude| (µV)" in joined, joined
    assert "peak time relative to the largest channel (ms)" in joined, joined
    assert "x (µm)" in joined and "y (µm)" in joined
    assert "(um)" not in joined
    caption = " ".join(captured["texts"])
    assert "One dot per electrode" in caption
    assert "Latency 0" in caption


def test_footprint_diagnostic_says_samples_without_a_sampling_rate(tmp_path):
    import matplotlib.pyplot as plt

    template, locations = _footprint_inputs()
    captured = {}
    real_close = plt.close

    def spy(fig):
        captured["labels"] = [_norm(a.get_ylabel()) for a in fig.get_axes()]
        return real_close(fig)

    plt.close = spy
    try:
        rp.plot_unit_footprint_diagnostic(
            template, locations, tmp_path / "nofs.png", unit_id=7, fs=None)
    finally:
        plt.close = real_close

    joined = " || ".join(captured["labels"])
    assert "(samples)" in joined, joined
    assert "(ms)" not in joined, "claims milliseconds without a sampling rate"


# --------------------------------------------------------------------------
# footprint reconstruction ("circle" plot) — presentation vs diagnostic style
# --------------------------------------------------------------------------

class _FakeGtr:
    """Minimal stand-in for an `axon_velocity` GraphAxonTracking result.

    `plot_unit_footprint_reconstruction` reads only `.template`, `.locations`,
    `.fs` and `.branches` (`branches` via `_branch_channel_paths`, which wants
    a list of dicts each carrying a `'channels'` array) — so a plain object
    with those four attributes drives the whole render without SpikeInterface
    or the real tracker.
    """

    def __init__(self, template, locations, fs, branches):
        self.template = template
        self.locations = locations
        self.fs = fs
        self.branches = branches


def _recon_gtr():
    template, locations = _footprint_inputs(n_channels=40, n_samples=60)
    branches = [
        {"channels": np.array([0, 1, 2, 3], dtype=int)},
        {"channels": np.array([10, 11, 12], dtype=int)},
    ]
    return _FakeGtr(template, locations, 20_000.0, branches)


def _render_recon_capturing(tmp_path, style):
    """Render the reconstruction plot and snapshot the figure at close time
    (this family saves with `fig.savefig` + `plt.close`, not `_save_and_release`)."""
    import matplotlib.pyplot as plt

    cap = {}
    real_close = plt.close

    def spy(fig):
        ax = fig.axes[0]
        cap["title"] = _norm(ax.get_title())
        cap["legend"] = (
            [_norm(t.get_text()) for t in ax.get_legend().get_texts()]
            if ax.get_legend() is not None
            else []
        )
        cap["ax_texts"] = [_norm(t.get_text()) for t in ax.texts]
        cap["cbar"] = [_norm(a.get_ylabel()) for a in fig.get_axes() if _norm(a.get_ylabel())]
        return real_close(fig)

    plt.close = spy
    try:
        rp.plot_unit_footprint_reconstruction(
            _recon_gtr(), tmp_path / f"recon_{style}.png",
            unit_id="7", dpi=110, style=style,
        )
    finally:
        plt.close = real_close
    return cap


def test_footprint_reconstruction_presentation_strips_stats_legend_and_numbers(tmp_path):
    """style="presentation" (Adam, 2026-08-12): title is the unit id alone
    with no per-unit stats, no framed branch legend, the scale circle carries
    no per-unit µV number, and the colour bar gets the compact label."""
    pres = _render_recon_capturing(tmp_path, "presentation")

    assert pres["title"] == "Unit 7", pres["title"]
    for banned in ("branch", "electrode", "amplitude", "pitch"):
        assert banned not in pres["title"], pres["title"]
    # The branch legend box is gone.
    assert pres["legend"] == [], pres["legend"]
    # No per-unit peak-amplitude number baked into the scale circle (or anywhere).
    assert not any("µV" in t for t in pres["ax_texts"]), pres["ax_texts"]
    # Compact colour-bar label.
    assert any(label == "Latency (ms)" for label in pres["cbar"]), pres["cbar"]


def test_footprint_reconstruction_diagnostic_default_is_unchanged(tmp_path):
    """The diagnostic default keeps the verbose review figure verbatim: the
    stats title, the branch legend, the µV scale-circle reference and the long
    self-documenting colour-bar label all stay."""
    diag = _render_recon_capturing(tmp_path, "diagnostic")

    assert "branch(es)" in diag["title"] and "electrode(s)" in diag["title"], diag["title"]
    assert diag["legend"] and any("branch" in x for x in diag["legend"]), diag["legend"]
    assert any("µV" in t for t in diag["ax_texts"]), diag["ax_texts"]
    assert any(
        "peak time relative to the largest electrode" in label for label in diag["cbar"]
    ), diag["cbar"]


# --------------------------------------------------------------------------
# figure_text — the shared glossary is the single source of truth
# --------------------------------------------------------------------------

def test_acronym_note_expands_only_what_was_asked_for():
    note = ft.acronym_note("MAD", "RMS")
    assert "median absolute deviation" in note
    assert "root mean square" in note
    assert "power spectral density" not in note


def test_acronym_note_skips_unknown_names():
    assert ft.acronym_note("MAD", "NOT_AN_ACRONYM") == ft.ACRONYMS["MAD"]


def test_every_acronym_entry_expands_itself():
    for name, text in ft.ACRONYMS.items():
        assert text.startswith(f"{name} ="), f"{name} does not expand itself: {text}"


def test_plain_language_strings_carry_no_jargon():
    """The glossary cannot itself lean on the terms it exists to replace."""
    banned = ("seam", "segment band", "backbone", "realtime twin", "fake timeline")
    for name in ("SEAM", "PER_SEGMENT_ONLY", "REAL_ELAPSED_AXIS", "CONTIGUOUS_AXIS"):
        text = getattr(ft, name).lower()
        for term in banned:
            assert term not in text, f"{name} still uses {term!r}"


# --------------------------------------------------------------------------
# Adam's 2026-08-11 plotting rulings
# --------------------------------------------------------------------------

def test_cluster_size_warning_is_gone(recording, caplog):
    """Ruling A: cluster size is dynamic by design; the warning was pure noise."""
    import logging

    xs = recording.get_channel_locations()[:, 0]
    ys = recording.get_channel_locations()[:, 1]
    with caplog.at_level(logging.WARNING, logger="mea_modules.diagnostics.channel_layout"):
        clusters = cl.detect_electrode_clusters(xs, ys)
    assert clusters
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_detect_electrode_clusters_no_longer_takes_the_warn_knob():
    import inspect

    params = inspect.signature(cl.detect_electrode_clusters).parameters
    assert "max_cluster_size_warn" not in params
    params = inspect.signature(tr.select_representative_channels).parameters
    assert "max_cluster_size_warn" not in params


def test_default_representative_channel_count_is_six():
    """Ruling B3: fewer stacked panels means taller, more legible traces."""
    assert tr._DEFAULT_MAX_CHANNELS == 6


def test_point_budget_is_a_real_ceiling(recording):
    """Ruling B1: floor division made max_points a suggestion, so it warned."""
    # 250k frames against a 150k budget used to give step 1 -> 250k points.
    step = tr._resolve_downsample_step(250_000, 20_000.0, None, 150_000)
    assert step == 2
    assert 250_000 / step <= 150_000


def test_default_render_cannot_trip_the_too_many_points_warning(recording, caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="mea_modules.diagnostics.traces"):
        tr.plot_traces(recording, "/tmp/_unused_points_warn.png",
                       channel_ids=["ch0"], duration_s=2.0)
    assert not [r for r in caplog.records if "points/channel" in r.getMessage()]


def test_plot_quality_presets_default_to_draft():
    """Ruling B2: high definition is available, but this pass ships draft."""
    assert tr.DEFAULT_PLOT_QUALITY == "draft"
    draft = tr.resolve_plot_quality(None)
    high = tr.resolve_plot_quality("high")
    assert draft == tr.resolve_plot_quality("draft")
    assert high["dpi"] > draft["dpi"]
    assert high["max_points"] > draft["max_points"]


def test_unknown_plot_quality_falls_back_rather_than_raising(caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="mea_modules.diagnostics.traces"):
        preset = tr.resolve_plot_quality("ultra")
    assert preset == tr.resolve_plot_quality("draft")
    assert any("unknown plot quality" in r.getMessage() for r in caplog.records)


def test_explicit_dpi_beats_the_quality_preset(recording, tmp_path):
    """A caller that named a dpi is not overridden by a quality flag."""
    from PIL import Image

    out = tr.plot_traces(recording, tmp_path / "explicit.png", channel_ids=["ch0"],
                         duration_s=0.5, dpi=100, quality="high")
    width, _height = Image.open(out).size
    assert width == pytest.approx(round(tr._TRACE_FIGSIZE[0] * 100), abs=2)


def test_high_quality_raises_the_rendered_resolution(recording, tmp_path):
    from PIL import Image

    draft = tr.plot_traces(recording, tmp_path / "draft.png", channel_ids=["ch0"],
                           duration_s=0.5, quality="draft")
    high = tr.plot_traces(recording, tmp_path / "high.png", channel_ids=["ch0"],
                          duration_s=0.5, quality="high")
    assert Image.open(high).size[0] > Image.open(draft).size[0]


# --------------------------------------------------------------------------
# bottom matter: caption and legend share one margin and may never collide
# --------------------------------------------------------------------------
#
# The defect these guard: on capsule 05's regenerated traces.png the legend box
# sat on top of the plain-language caption, hiding a whole line of it. The
# caption was written bottom-left by `_add_caption` and the legend pinned
# bottom-centre by `plot_traces`, with nothing reconciling the two.
#
# These assert on GEOMETRY, not on placement constants: they measure the drawn
# artists and demand the boxes be disjoint. A future change that reintroduces
# the overlap by any route — a second legend, a longer caption, a different
# figure size — fails here.

def _caption_texts(fig):
    """Figure-level caption texts, excluding the sup* labels."""
    special = {getattr(fig, name, None) for name in ("_suptitle", "_supxlabel", "_supylabel")}
    return [t for t in fig.texts if t not in special and str(t.get_text()).strip()]


def _fraction_box(fig, artist, renderer):
    """`artist`'s drawn extent in figure fractions (0-1 of width/height)."""
    return artist.get_window_extent(renderer).transformed(fig.transFigure.inverted())


def _axes_bottom(fig, renderer):
    """Lowest edge of anything the axes draw, in figure fractions.

    The TIGHT bbox, not the axes rectangle: an x label hanging below the frame
    is part of the plot, and covering it is the same defect as covering data.
    """
    return min(
        axis.get_tightbbox(renderer).transformed(fig.transFigure.inverted()).y0
        for axis in fig.get_axes()
        if axis.get_visible()
    )


def _bottom_matter(fig):
    """(caption box, legend box, lowest drawn axes edge) in figure fractions."""
    renderer = fig.canvas.get_renderer()
    captions = _caption_texts(fig)
    assert len(captions) == 1, f"expected exactly one caption, got {len(captions)}"
    caption_box = _fraction_box(fig, captions[0], renderer)

    legends = list(fig.legends)
    legend_box = _fraction_box(fig, legends[0], renderer) if legends else None

    return caption_box, legend_box, _axes_bottom(fig, renderer)


@pytest.fixture
def geometry(monkeypatch):
    """Measure the bottom matter of whatever figure the emitter saves."""
    seen = {}
    for module in (cl, tr, ra):
        real = module._save_and_release

        def wrapper(fig, out_path, _real=real):
            caption_box, legend_box, lowest = _bottom_matter(fig)
            seen.update(caption=caption_box, legend=legend_box, axes_bottom=lowest)
            return _real(fig, out_path)

        monkeypatch.setattr(module, "_save_and_release", wrapper)
    return seen


def _assert_disjoint(seen):
    caption, legend, axes_bottom = seen["caption"], seen["legend"], seen["axes_bottom"]
    assert legend is not None, "the figure legend went missing"
    assert not caption.overlaps(legend), (
        f"legend {legend.bounds} covers the caption {caption.bounds}"
    )
    # Stacked, not merely non-overlapping: caption at the bottom edge, legend
    # above it, axes above that. An accidental side-by-side layout would pass
    # the overlap check while still being unreadable at another caption length.
    assert legend.y0 >= caption.y1, "the legend must sit above the caption"
    assert axes_bottom >= legend.y1, "the legend must sit below the axes"


@pytest.mark.parametrize(
    "figsize",
    [
        tr._TRACE_FIGSIZE,  # what the capsules actually render at
        (6.5, 4.5),         # a small sheet: the band has to shrink to fit
        (20.0, 11.0),       # a large one: it must not balloon into dead space
        (7.0, 3.2),         # short and wide, where a fixed fraction ran out
    ],
)
def test_traces_legend_never_covers_the_caption(geometry, recording, tmp_path, figsize):
    """The capsule-05 defect, at every size these figures are rendered at."""
    tr.plot_traces(
        recording,
        tmp_path / "traces.png",
        channel_ids=["ch0", "ch5"],
        duration_s=1.0,
        stitch_frames=(10_000,),
        figsize=figsize,
        caption_extra=(
            "These are the channels marked red in channel_layout.png, and this "
            "sentence is here to push the caption onto several lines so the "
            "legend has something to collide with."
        ),
    )
    _assert_disjoint(geometry)


def test_traces_realtime_legend_never_covers_the_caption(geometry, recording, tmp_path):
    """The realtime twin carries one more key and one more caption paragraph."""
    tr.plot_traces(
        recording,
        tmp_path / "traces_realtime.png",
        channel_ids=["ch0"],
        duration_s=1.0,
        stitch_frames=(10_000,),
        time_gaps={"break_sample_indices": [5_000], "break_gap_frames": [40_000]},
    )
    _assert_disjoint(geometry)


def test_traces_raw_in_device_counts_keeps_its_caption_clear(geometry, tmp_path):
    """traces_raw adds the ADC acronym note — a longer caption, same contract."""
    class Unscaleable(FakeRecording):
        def has_scaleable_traces(self):
            return False

    tr.plot_traces(
        Unscaleable(),
        tmp_path / "traces_raw.png",
        channel_ids=["ch0"],
        duration_s=1.0,
        stitch_frames=(10_000,),
    )
    _assert_disjoint(geometry)


@pytest.mark.parametrize("figsize", [(16.0, 8.0), (6.5, 4.5)])
def test_raster_caption_stays_clear_of_the_axes(geometry, recording, tmp_path, figsize):
    """The raster keys live inside the axes, so only the caption uses the margin.

    It still has to clear the plot: the reserved band is measured from the
    caption, and a caption that outgrows it lands on the x axis.
    """
    ra.plot_raster_threshold(
        recording,
        tmp_path / "raster_threshold.png",
        duration_s=1.0,
        segment_boundaries=(0.2, 0.6),
        figsize=figsize,
    )
    assert geometry["legend"] is None, "the raster key belongs in its axes"
    assert geometry["axes_bottom"] >= geometry["caption"].y1


def test_channel_layout_caption_stays_clear_of_the_axes(geometry, recording, tmp_path):
    cl.plot_channel_layout(
        recording,
        tmp_path / "channel_layout.png",
        highlight_channel_ids=["ch0"],
        highlight_label="representative channels — traced in traces.png",
        caption=ft.REPRESENTATIVE_CHANNELS,
    )
    assert geometry["axes_bottom"] >= geometry["caption"].y1


@pytest.mark.parametrize("figsize", [tr._TRACE_FIGSIZE, (7.0, 3.2)])
def test_reserved_band_is_measured_not_guessed(geometry, recording, tmp_path, figsize):
    """The margin holds its contents and little else.

    The old rule reserved a fixed FRACTION per caption line, which was both too
    little on a short figure and over an inch of dead space on a tall one. The
    band is now measured, so the gap left between the bottom matter and the plot
    is a constant few tenths of an inch at any figure size.
    """
    tr.plot_traces(
        recording, tmp_path / "traces.png", channel_ids=["ch0"],
        duration_s=1.0, stitch_frames=(10_000,), figsize=figsize,
    )
    dead_inches = (geometry["axes_bottom"] - geometry["legend"].y1) * figsize[1]
    assert 0.0 <= dead_inches < 0.45, f"{dead_inches:.2f} in of empty margin"


def test_a_longer_caption_reserves_more_room(recording, tmp_path):
    """Nothing is clipped when the caption grows: the band grows with it."""
    bottoms = {}
    real = tr._save_and_release
    for name, extra in (
        ("short", None),
        ("long", " ".join(["A much longer explanatory sentence."] * 12)),
    ):
        measured = {}

        def wrapper(fig, out_path, _real=real, _store=measured):
            _store["y"] = _axes_bottom(fig, fig.canvas.get_renderer())
            return _real(fig, out_path)

        tr._save_and_release = wrapper
        try:
            tr.plot_traces(recording, tmp_path / f"{name}.png", channel_ids=["ch0"],
                           duration_s=1.0, caption_extra=extra)
        finally:
            tr._save_and_release = real
        bottoms[name] = measured["y"]

    assert bottoms["long"] > bottoms["short"], "a longer caption must reserve more"


def test_bottom_matter_is_deterministic(recording, tmp_path):
    """Measured placement must still render byte-identically twice."""
    first = tr.plot_traces(recording, tmp_path / "a.png", channel_ids=["ch0", "ch5"],
                           duration_s=1.0, stitch_frames=(10_000,))
    second = tr.plot_traces(recording, tmp_path / "b.png", channel_ids=["ch0", "ch5"],
                            duration_s=1.0, stitch_frames=(10_000,))
    assert first.read_bytes() == second.read_bytes()


def test_no_emitter_pins_its_own_figure_legend(recording, tmp_path):
    """`_add_caption` owns the bottom margin; a second owner is the whole bug.

    Guards the fix at the source level, because a private `fig.legend` in a new
    emitter reintroduces the collision without failing any per-figure test that
    happens not to cover that emitter.
    """
    import inspect
    from pathlib import Path

    import mea_modules

    root = Path(inspect.getfile(mea_modules)).parent
    offenders = sorted(
        str(path.relative_to(root))
        for path in root.rglob("*.py")
        if path.name != "channel_layout.py"
        and "fig.legend(" in path.read_text(encoding="utf-8")
    )
    assert not offenders, (
        f"{offenders} call fig.legend directly; pass handles to "
        "_add_caption(legend_handles=...) so caption and legend cannot collide"
    )
