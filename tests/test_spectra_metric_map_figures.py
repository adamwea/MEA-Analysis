"""Annotation contract for the PSD panels and the geometry metric maps.

Two conventions are asserted here, both of which are easy to regress silently
because they are about what is NOT on the figure.

1. ``annotate=False`` is the presentation register every emitter now takes: no
   title, no caption, and everything a reader still needs to decode the plot —
   axes, units, legend, panel identity — left in place. A regression here does
   not raise; it just quietly puts a title back on a figure that is about to be
   dropped into a slide under its own caption.

2. A label belongs to the axis, and panels drawn with ``sharex``/``sharey`` have
   ONE axis between them. ``psd.png`` printed "PSD (uV^2/Hz)" twice, once per
   panel, with a literal caret where the superscript should have been
   (2026-09-19). Both halves of that are asserted: the label is counted, and the
   units are matched as mathtext rather than as ASCII.

As in the other figure tests, the assertions run against the real ``Figure``:
each emitter writes through ``_save_tight``, which is patched to capture the
figure before its artists are cleared, so the PNG is still written and the
public entry point still exercised end to end.
"""

import re

import numpy as np
import pytest

from mea_modules.diagnostics import metric_maps as mm
from mea_modules.diagnostics import spectra as sp
from mea_modules.diagnostics.figure_text import ACRONYMS_SHORT


def _norm(text):
    """Collapse whitespace so a folded legend title compares to its source."""
    return re.sub(r"\s+", " ", str(text)).strip()


class _Captured:
    """The figure an emitter saved, frozen before ``_save_tight`` clears it."""

    def __init__(self):
        self.figure = None
        self.axis_labels = []
        self.titles = []
        self.legend_titles = []
        self.in_axes_text = []
        self.figure_text = []

    @staticmethod
    def _all_axes(fig):
        """Every axes on `fig`, including the colour bars drawn as insets.

        A colour bar made with ``colorbar(ax=...)`` used to land straight in
        ``fig.get_axes()``. It is now an inset on its panel -- that is what
        keeps two side-by-side panels exactly the same size, because
        ``colorbar(ax=...)`` takes its width out of the parent and takes a
        different amount on the left than on the right. An inset is a CHILD of
        its panel, so it is reached through ``child_axes`` rather than the
        figure's own list, and a collector that only walks ``fig.get_axes()``
        silently stops seeing every bar label.
        """
        seen, out = set(), []
        pending = list(fig.get_axes())
        while pending:
            axis = pending.pop(0)
            if id(axis) in seen:
                continue
            seen.add(id(axis))
            out.append(axis)
            pending.extend(getattr(axis, "child_axes", ()))
        return out

    def freeze(self, fig):
        self.figure = fig
        # Draw first. Positions are laid out lazily, and an inset colour
        # bar in particular reports its parent's pre-layout box until the
        # figure has been through a render -- which, since this fixture
        # intercepts the save, has not happened yet.
        try:
            fig.canvas.draw()
        except Exception:  # noqa: BLE001 - measurement only, never fatal
            pass
        # Colorbars are axes too, and their label is the panel's unit key, so
        # they are collected with the rest rather than filtered out.
        axes = self._all_axes(fig)
        self.axis_labels = [
            _norm(label)
            for axis in axes
            for label in (axis.get_xlabel(), axis.get_ylabel())
            if _norm(label)
        ]
        self.titles = [_norm(a.get_title()) for a in axes if _norm(a.get_title())]
        self.in_axes_text = [
            _norm(t.get_text()) for axis in axes for t in axis.texts
        ]
        self.figure_text = [_norm(t.get_text()) for t in fig.texts if _norm(t.get_text())]
        self.legend_titles = []
        self.legend_labels = []
        # Panel geometry, frozen here because `_save_tight` clears the figure
        # before a test can look at it. One entry per top-level panel: its own
        # box, and the boxes of the colour bars drawn as insets on it. This is
        # what lets a test assert the SIDE rule and the equal-size rule against
        # where things actually landed, rather than against a kwarg that an
        # implementation happens to pass.
        self.panel_boxes = [
            (axis.get_position(), [c.get_position() for c in getattr(axis, "child_axes", ())])
            for axis in fig.get_axes()
        ]
        for axis in axes:
            legend = axis.get_legend()
            if legend is None:
                continue
            if legend.get_title() is not None:
                self.legend_titles.append(_norm(legend.get_title().get_text()))
            self.legend_labels.extend(_norm(t.get_text()) for t in legend.get_texts())

    @property
    def everything(self):
        return " || ".join(
            self.axis_labels
            + self.titles
            + self.in_axes_text
            + self.figure_text
            + self.legend_titles
        )

    def count(self, needle):
        """How many separate text artists carry `needle`."""
        pieces = (
            self.axis_labels
            + self.titles
            + self.in_axes_text
            + self.figure_text
            + self.legend_titles
        )
        return sum(1 for piece in pieces if _norm(needle) in piece)


@pytest.fixture
def capture(monkeypatch):
    """Capture each figure without changing what the emitter writes."""
    captured = _Captured()
    for module in (sp, mm):
        real = module._save_tight

        def wrapper(fig, out_path, _real=real, _cap=captured):
            _cap.freeze(fig)
            return _real(fig, out_path)

        monkeypatch.setattr(module, "_save_tight", wrapper)
    return captured


class FakeRecording:
    """Minimal SpikeInterface-shaped recording on a small rectangular grid."""

    def __init__(self, n_channels=6, fs=10_000.0, n_samples=8192, scaleable=True):
        rng = np.random.default_rng(0)
        self._ids = [f"ch{i}" for i in range(n_channels)]
        self._loc = np.c_[
            [17.5 * (i % 3) for i in range(n_channels)],
            [17.5 * (i // 3) for i in range(n_channels)],
        ].astype(float)
        self._fs = float(fs)
        self._n = int(n_samples)
        self._scaleable = bool(scaleable)
        self._data = rng.normal(0.0, 5.0, size=(self._n, n_channels)).astype("float32")

    def get_channel_ids(self):
        return list(self._ids)

    def get_channel_locations(self):
        return self._loc

    def get_sampling_frequency(self):
        return self._fs

    def get_num_samples(self):
        return self._n

    def has_scaleable_traces(self):
        return self._scaleable

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


def _spectra(recording, sources=("raw", "preprocessed")):
    """One :func:`welch_spectra` result per named panel, over the same channels."""
    channels = recording.get_channel_ids()[:3]
    return {
        name: sp.welch_spectra(recording, channels, duration_s=0.4, nperseg=256)
        for name in sources
    }


# --------------------------------------------------------------------------
# spectra — one label per shared axis
# --------------------------------------------------------------------------


def test_psd_panels_share_one_y_label_between_them(capture, recording, tmp_path):
    """Review report: both panels carried an identical "PSD (...)" y label.

    The panels are drawn with ``sharey``, so there is one y axis and it may
    carry one label. Counted rather than merely asserted present, because the
    bug was a duplicate, not an absence."""
    out = sp.plot_spectra_panels(_spectra(recording), tmp_path / "psd.png", title="well 0")

    assert out.exists() and out.stat().st_size > 5_000
    assert capture.count("PSD (") == 1, capture.axis_labels + capture.figure_text
    # ...and it is a FIGURE label, not one panel's, so it is centred on the pair.
    assert any(text.startswith("PSD (") for text in capture.figure_text)
    assert not any(axis.get_ylabel() for axis in capture.figure.get_axes())


def test_psd_panels_each_carry_their_own_x_label(capture, recording, tmp_path):
    """X is NOT shared, unlike y (2026-09-19): a single `fig.supxlabel`
    over side-by-side panels only reserved a band of white space the two
    labels are worth more than, so each panel keeps its own "frequency (Hz)"."""
    sources = _spectra(recording)
    sp.plot_spectra_panels(sources, tmp_path / "psd.png")

    assert capture.count("frequency (Hz)") == len(sources)
    assert all(axis.get_xlabel() == "frequency (Hz)" for axis in capture.figure.get_axes())


def test_psd_units_are_mathtext_so_the_exponent_renders(capture, recording, tmp_path):
    """"uV^2/Hz" printed the caret literally. The axis carries mathtext now."""
    sp.plot_spectra_panels(_spectra(recording), tmp_path / "psd.png")

    label = next(text for text in capture.figure_text if text.startswith("PSD ("))
    assert "^2" not in label and "uV" not in label
    assert r"\mu" in label and "^{2}" in label and label.count("$") == 2


def test_psd_unit_recorded_in_the_result_stays_ascii(recording):
    """The label is mathtext; the UNIT is data and stays ASCII.

    ``welch_spectra``'s ``unit`` travels in a dict and through log lines, and a
    consumer would have to strip mathtext markup back out of it. The split is
    the point of the change, so it is pinned from both sides."""
    counts = FakeRecording(scaleable=False)
    assert sp.welch_spectra(recording, ["ch0"], duration_s=0.4)["unit"] == "uV^2/Hz"
    assert sp.welch_spectra(counts, ["ch0"], duration_s=0.4)["unit"] == "adc^2/Hz"


# --------------------------------------------------------------------------
# spectra — PSD defined on the figure
# --------------------------------------------------------------------------


def test_psd_is_expanded_in_the_legend(capture, recording, tmp_path):
    """The review asked for PSD defined on the plot, in the legend.

    The wording is the canonical one from ``figure_text``, not a local
    paraphrase — that table exists so the figure, its README and the report
    generator cannot drift apart."""
    sp.plot_spectra_panels(_spectra(recording), tmp_path / "psd.png")

    assert capture.legend_titles, "the legend carries no title"
    assert _norm(ACRONYMS_SHORT["PSD"]) in _norm(capture.legend_titles[0])


def test_counts_panel_also_expands_adc(capture, tmp_path):
    """A recording that cannot scale prints ADC on the axis, so it defines it.

    Every acronym on the figure is expanded at least once on the figure; the
    counts axis is the only place ADC appears, so the definition rides along
    with it rather than being printed on every figure."""
    counts = FakeRecording(scaleable=False)
    sp.plot_spectra_panels(_spectra(counts, sources=("raw",)), tmp_path / "psd.png")

    title = _norm(capture.legend_titles[0])
    assert _norm(ACRONYMS_SHORT["PSD"]) in title
    assert _norm(ACRONYMS_SHORT["ADC"]) in title


def test_microvolt_panel_does_not_define_adc(capture, recording, tmp_path):
    """The converse: a figure that never prints ADC does not define it either."""
    sp.plot_spectra_panels(_spectra(recording), tmp_path / "psd.png")

    assert _norm(ACRONYMS_SHORT["ADC"]) not in _norm(capture.legend_titles[0])


# --------------------------------------------------------------------------
# spectra — annotate
# --------------------------------------------------------------------------


def test_psd_annotate_false_drops_the_title_and_keeps_everything_else(
    capture, recording, tmp_path
):
    """The presentation register: no title, but the figure still decodes.

    Axis labels, units, the legend with its PSD expansion, and each panel's own
    identity all survive — those are what a reader needs, and none of them is
    the caption-and-title chrome ``annotate`` switches off."""
    sources = _spectra(recording)
    sp.plot_spectra_panels(sources, tmp_path / "psd.png", title="P003454 well 0", annotate=False)

    assert "P003454" not in capture.everything
    assert capture.figure._suptitle is None
    assert capture.count("PSD (") == 1
    assert capture.count("frequency (Hz)") == len(sources)
    assert _norm(ACRONYMS_SHORT["PSD"]) in _norm(capture.legend_titles[0])
    assert any("raw" in text for text in capture.in_axes_text)
    assert any("preprocessed" in text for text in capture.in_axes_text)


def test_psd_annotate_true_is_the_default_and_draws_the_title(capture, recording, tmp_path):
    """Default stays True, so callers outside this change are untouched."""
    sp.plot_spectra_panels(_spectra(recording), tmp_path / "psd.png", title="P003454 well 0")

    assert "P003454 well 0" in capture.everything


def test_psd_panel_identity_is_never_a_per_panel_title(capture, recording, tmp_path):
    """Identity moved into the axes, so no panel carries a title band at all."""
    sp.plot_spectra_panels(_spectra(recording), tmp_path / "psd.png", title="well 0")

    assert capture.titles == []
    assert any("raw (3 representative channels)" in t for t in capture.in_axes_text)


# --------------------------------------------------------------------------
# metric_maps
# --------------------------------------------------------------------------


def _metrics(recording):
    """``(noise, activity)`` shaped like the quality results the maps consume."""
    rng = np.random.default_rng(1)
    ids = recording.get_channel_ids()
    noise = {"channel_ids": ids, "noise": np.abs(rng.normal(5.0, 2.0, len(ids))), "unit": "uV"}
    activity = {
        "channel_ids": ids,
        "rate_hz": rng.random(len(ids)) * 3.0,
        "threshold_sd": 5.0,
    }
    return noise, activity


def test_metric_maps_each_panel_keeps_its_own_x_label_and_shares_one_y_label(
    capture, recording, tmp_path
):
    """RULE (2026-09-19): left/right panels genuinely share the y axis (one
    label), but round 2's single shared x label read oddly and cost more white
    space than it saved — each panel now keeps its OWN x label instead.

    AMENDED 2026-09-20: the one y label is on the LEFTMOST AXES, not on the
    figure. A figure-level label is placed against the figure's left edge,
    which is where the leftmost panel's outboard colour-bar label already is,
    so the two printed on top of each other. Anchored to the axes, matplotlib
    lays it out from the axes and its ticks and the collision cannot recur."""
    noise, activity = _metrics(recording)
    out = mm.plot_noise_activity_map(recording, noise, activity, tmp_path / "map.png")

    assert out.exists() and out.stat().st_size > 5_000
    # One x label PER PANEL (two panels), drawn on the axes themselves.
    assert capture.count(f"x ({mm._UM_LABEL})") == 2
    # One y label for the pair, shared rather than repeated.
    assert capture.count(f"y ({mm._UM_LABEL})") == 1
    # NEITHER label is figure-level: both live on axes, which is what keeps
    # the y label clear of the colour bar that sits outboard of it.
    assert f"y ({mm._UM_LABEL})" not in capture.figure_text
    assert f"x ({mm._UM_LABEL})" not in capture.figure_text


def test_metric_maps_micrometre_axes_are_real_mathtext(capture, recording, tmp_path):
    """metric_maps used to print the ASCII "um"; it is mathtext now, the way
    spectra.py renders its PSD units (2026-09-19)."""
    noise, activity = _metrics(recording)
    mm.plot_noise_activity_map(recording, noise, activity, tmp_path / "map.png")

    assert "um)" not in capture.everything
    assert r"\mu" in mm._UM_LABEL and mm._UM_LABEL.count("$") == 2


def test_metric_maps_annotate_false_drops_both_title_layers(capture, recording, tmp_path):
    """No suptitle and no panel titles; the colour bars still name each panel,
    including the estimator (MAD) and the threshold (SD) — the numbers that
    used to live only in the panel title, which annotate=False drops."""
    noise, activity = _metrics(recording)
    mm.plot_noise_activity_map(
        recording, noise, activity, tmp_path / "map.png", title="P003454 well 0", annotate=False
    )

    assert "P003454" not in capture.everything
    assert capture.figure._suptitle is None
    assert capture.titles == []
    assert "MAD noise (uV)" in capture.axis_labels
    assert "≥5 SD crossing rate (events / s)" in capture.axis_labels
    assert capture.count(f"x ({mm._UM_LABEL})") == 2
    assert capture.count(f"y ({mm._UM_LABEL})") == 1


def test_metric_maps_annotate_true_is_the_default_and_draws_both(capture, recording, tmp_path):
    """Default stays True: suptitle and per-panel titles, as before."""
    noise, activity = _metrics(recording)
    mm.plot_noise_activity_map(
        recording, noise, activity, tmp_path / "map.png", title="P003454 well 0"
    )

    assert "P003454 well 0" in capture.everything
    assert "MAD noise" in capture.titles
    assert any(t.startswith("Activity rate") for t in capture.titles)


def test_firing_rate_map_takes_annotate_through_its_kwargs(capture, recording, tmp_path):
    """The single-panel figure routes `annotate` on to the shared engine."""
    _, activity = _metrics(recording)
    mm.plot_firing_rate_map(
        recording, activity, tmp_path / "rate.png", title="P003454 well 0", annotate=False
    )

    assert "P003454" not in capture.everything
    assert capture.titles == []
    assert "≥5 SD crossing rate (events / s)" in capture.axis_labels
    assert capture.count(f"x ({mm._UM_LABEL})") == 1


def test_plot_noise_map_is_the_individual_twin_of_the_activity_map(capture, recording, tmp_path):
    """Review ruling: "make sure we have individual panel plots for each of
    these [noise and activity] in addition to the multipanel." Activity
    already had one (`plot_firing_rate_map`); this is noise's."""
    noise, _ = _metrics(recording)
    out = mm.plot_noise_map(recording, noise, tmp_path / "noise.png", annotate=False)

    assert out.exists() and out.stat().st_size > 5_000
    assert capture.titles == []
    assert "MAD noise (uV)" in capture.axis_labels
    assert capture.count(f"x ({mm._UM_LABEL})") == 1
    # A single-panel figure is drawn by the SAME routine as the composite's
    # panel — not a second copy — so it carries the same metric legend key.
    assert any("MAD noise" in label for label in capture.legend_labels)


def test_metric_map_legend_expands_its_acronym_once(capture, recording, tmp_path):
    """Review ruling: "needing legends describing metric used for noise /
    activity." MAD and SD are expanded in the legend title, the way
    spectra.py expands PSD."""
    noise, activity = _metrics(recording)
    mm.plot_noise_activity_map(recording, noise, activity, tmp_path / "map.png", annotate=False)

    assert capture.legend_titles, "no panel carries a legend title"
    joined = " || ".join(capture.legend_titles)
    assert _norm(ACRONYMS_SHORT["MAD"]) in _norm(joined)
    assert _norm(ACRONYMS_SHORT["SD"]) in _norm(joined)


def test_colorbar_side_mirrors_the_panels_column(capture, recording, tmp_path):
    """Rule (2026-09-19): a LEFT panel's colour bar sits on ITS left, a
    RIGHT panel's stays on the right — derived from column index, not a
    hardcoded per-call side.

    Asserted against where the bar actually landed rather than against the
    `location=` kwarg the old implementation passed: the bar is now an inset,
    so the kwarg is gone but the rule it encoded is unchanged, and geometry is
    the thing the rule was ever really about.
    """
    noise, activity = _metrics(recording)
    mm.plot_noise_activity_map(recording, noise, activity, tmp_path / "map.png")

    assert len(capture.panel_boxes) == 2, "expected exactly two panels"
    (left_panel, left_bars), (right_panel, right_bars) = capture.panel_boxes
    assert len(left_bars) == 1 and len(right_bars) == 1, "one colour bar per panel"
    assert left_bars[0].x1 <= left_panel.x0, "the left panel's bar is not on its left"
    assert right_bars[0].x0 >= right_panel.x1, "the right panel's bar is not on its right"


def test_single_panel_map_colorbar_defaults_to_the_right(capture, recording, tmp_path):
    """One panel is its own right half, so its bar goes on the right."""
    noise, _ = _metrics(recording)
    mm.plot_noise_map(recording, noise, tmp_path / "noise.png")

    assert len(capture.panel_boxes) == 1
    panel, bars = capture.panel_boxes[0]
    assert len(bars) == 1
    assert bars[0].x0 >= panel.x1, "a single panel's bar is not on its right"


def test_two_panels_are_exactly_the_same_size(capture, recording, tmp_path):
    """Review ruling (2026-09-20): the two panels must be identically sized.

    They were not. `colorbar(ax=ax)` takes its space out of the parent axes and
    takes a DIFFERENT amount on each side — 25% of the width at
    ``location="left"`` against 20% at ``"right"`` — so the side rule above was
    itself leaving the panels 6.25% apart. Drawing each bar as an inset leaves
    the panels untouched. A side-by-side comparison whose panels are not the
    same size silently misstates the thing it exists to compare.
    """
    noise, activity = _metrics(recording)
    mm.plot_noise_activity_map(recording, noise, activity, tmp_path / "map.png")

    (first, _), (second, _) = capture.panel_boxes
    assert first.width == pytest.approx(second.width, rel=1e-9)
    assert first.height == pytest.approx(second.height, rel=1e-9)


def test_plot_metric_maps_draws_into_callers_axes(recording, tmp_path):
    """The house pattern (2026-09-19): `axes=` lets a composed sheet reuse
    this drawing routine instead of carrying a second copy of it."""
    noise, _ = _metrics(recording)
    fig = mm._new_figure((6.0, 4.0), 100)
    ax = fig.subplots()
    panels = [mm._noise_panel(noise)]

    result = mm.plot_metric_maps(recording, list(noise["channel_ids"]), panels, axes=[ax])

    assert result is None
    assert not list(tmp_path.iterdir())
    assert ax.get_xlabel() == f"x ({mm._UM_LABEL})"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
