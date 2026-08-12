"""Every postprocess figure must be decodable from the figure alone.

Adam's two rulings (2026-08-11) are the spec these tests enforce on the five
postprocess emitters, the same way the diagnostics emitters are already held to
them:

1. every drawn encoding — colour, marker, line style, shaded span, colormap —
   carries a legend key or a labelled colour bar;
2. every quantity states its unit, every acronym is expanded once on the figure,
   and no reader-facing string uses one of our terms of art.

The emitters return a ``Path``, not a ``Figure``, so the assertions go through a
spy on each module's ``_save_and_release``: it captures the real legend texts,
axis labels and caption before the real one writes the PNG and drops the
artists. That gives assertions on actual legend handles while still exercising
the public entry point end to end, PNG included.

Every comparison is made on whitespace-stripped strings, because both
``_wrap_label`` and ``_fold_caption`` soft-wrap — including at hyphens, so
"peak-to-peak" can legitimately arrive split across two lines.

Pure fakes throughout: no SpikeInterface, no recordings on disk. These are
presentation tests; the arithmetic they draw is covered elsewhere.
"""

import inspect

import numpy as np
import pytest

from mea_modules.diagnostics import figure_text
from mea_modules.postprocess import (
    footprints,
    segment_activity,
    unit_raster,
    unit_traces,
    waveforms,
)

FS = 10_000.0


# --------------------------------------------------------------------------
# capture + matching
# --------------------------------------------------------------------------

def _flat(text):
    """Collapse the soft wrapping the legend/caption helpers insert."""
    return " ".join(str(text).split())


def _squash(text):
    """Drop whitespace entirely, so a hyphen-break cannot defeat a match."""
    return "".join(str(text).split())


def _has(haystack, needle):
    return _squash(needle) in _squash(haystack)


def _one(labels, needle):
    """The single captured label containing `needle`; fails loudly if absent."""
    hits = [label for label in labels if _has(label, needle)]
    assert hits, f"no label containing {needle!r} in {labels!r}"
    return hits[0]


def _legend_labels(fig):
    labels = []
    for legend in list(fig.legends):
        labels.extend(_flat(text.get_text()) for text in legend.get_texts())
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is not None:
            labels.extend(_flat(text.get_text()) for text in legend.get_texts())
    return labels


def _axis_labels(fig):
    """Every axis label on the figure — colour bar labels included.

    ``Colorbar.set_label`` writes the long axis label of the colour bar's own
    Axes, so sweeping ``fig.axes`` picks the colour bar up without the test
    needing a handle on it.
    """
    labels = []
    for ax in fig.axes:
        labels.extend([_flat(ax.get_xlabel()), _flat(ax.get_ylabel())])
    return [label for label in labels if label]


def _snapshot(fig):
    suptitle = getattr(fig, "_suptitle", None)
    return {
        "legend": _legend_labels(fig),
        "axes": _axis_labels(fig),
        "titles": [_flat(ax.get_title()) for ax in fig.axes if ax.get_title()],
        "suptitle": _flat(suptitle.get_text()) if suptitle is not None else "",
        # The caption is the only figure-level text these emitters add.
        "caption": " ".join(_flat(text.get_text()) for text in fig.texts),
    }


def _spy(monkeypatch, module):
    """Capture the Figure each emitter builds, just before it is written."""
    captured = {}
    real = module._save_and_release

    def spy(fig, out_path):
        captured.update(_snapshot(fig))
        return real(fig, out_path)

    monkeypatch.setattr(module, "_save_and_release", spy)
    return captured


def _blob(captured):
    """Every reader-facing string on the figure, as one lowercase haystack."""
    parts = (
        list(captured["legend"])
        + list(captured["axes"])
        + list(captured["titles"])
        + [captured["suptitle"], captured["caption"]]
    )
    return " ".join(parts).lower()


# Our terms of art. Adam explicitly did not follow these, so none of them may
# appear anywhere a reader can see them (mea_modules/diagnostics/figure_text.py).
JARGON = (
    "seam",
    "segment band",
    "within each band",
    "backbone",
    "realtime twin",
    "fake timeline",
    "cluster representative",
    "dense stitch",
    "sparsity",
    "extremum",
    "raw units",
)


def _assert_plain_language(captured):
    # Deliberately NOT squashed: squashing would fuse word boundaries and turn
    # innocent text ("these amplitudes") into a false jargon hit ("...seam...").
    blob = _blob(captured)
    for term in JARGON:
        assert term not in blob, f"insider jargon {term!r} on the figure: {blob!r}"
    # A bare "um"/"uV" is exactly the unit bug this pass exists to fix.
    assert "(um)" not in blob
    assert "uv)" not in blob


def _assert_png(path):
    assert path.is_file()
    # A truncated or blank write is a few hundred bytes; a real figure is far
    # larger even before the data is dense.
    assert path.stat().st_size > 5_000


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------

class FakeSorting:
    """The slice of a SpikeInterface Sorting the raster emitters touch."""

    def __init__(self, trains, fs=FS):
        self._trains = {str(k): np.asarray(v, dtype=np.int64) for k, v in trains.items()}
        self._fs = float(fs)

    @property
    def unit_ids(self):
        return list(self._trains)

    def get_sampling_frequency(self):
        return self._fs

    def get_unit_spike_train(self, unit_id):
        return self._trains[str(unit_id)]


class FakeRecording:
    """One channel of traces, optionally unscaleable to microvolts."""

    def __init__(self, traces, channel_ids=("0",), fs=FS, scaleable=True):
        self._traces = np.asarray(traces, dtype=float)
        self._channel_ids = [str(channel) for channel in channel_ids]
        self._fs = float(fs)
        self._scaleable = bool(scaleable)

    def get_sampling_frequency(self):
        return self._fs

    def get_num_samples(self):
        return int(self._traces.shape[0])

    def get_channel_ids(self):
        return list(self._channel_ids)

    def get_traces(self, start_frame, end_frame, channel_ids, return_in_uV):
        if return_in_uV and not self._scaleable:
            raise ValueError("recording carries no gain and offset")
        columns = [self._channel_ids.index(str(channel)) for channel in channel_ids]
        return self._traces[int(start_frame):int(end_frame)][:, columns]


class _Extension:
    def __init__(self, nbefore):
        self.nbefore = int(nbefore)


class _Waveforms(_Extension):
    def __init__(self, snippets, nbefore):
        super().__init__(nbefore)
        self._snippets = snippets

    def get_waveforms_one_unit(self, unit_id):
        return self._snippets[str(unit_id)]


class _Templates(_Extension):
    def __init__(self, dense, nbefore):
        super().__init__(nbefore)
        self._dense = np.asarray(dense, dtype=float)

    def get_data(self, operator="average"):
        return self._dense


class _RandomSpikes:
    def __init__(self, unit_indices):
        self._data = np.asarray(
            [(int(index),) for index in unit_indices], dtype=[("unit_index", "i8")]
        )

    def get_random_spikes(self):
        return self._data


# How many spikes ``random_spikes`` "kept" per unit in the fake analyzer. The
# waveform legend quotes this number, so the tests pin it.
N_KEPT = 7


class FakeAnalyzer:
    """A dense (unsparse) analyzer over the electrodes in `locations`."""

    sparsity = None

    def __init__(self, unit_ids, dense_templates, snippets, locations, nbefore=5, fs=FS):
        self.unit_ids = list(unit_ids)
        self.channel_ids = [str(index) for index in range(np.asarray(locations).shape[0])]
        self.sampling_frequency = float(fs)
        self._locations = np.asarray(locations, dtype=float)
        self._extensions = {
            "waveforms": _Waveforms(snippets, nbefore),
            "templates": _Templates(dense_templates, nbefore),
            "random_spikes": _RandomSpikes(
                [index for index, _ in enumerate(unit_ids) for _ in range(N_KEPT)]
            ),
        }

    def has_extension(self, name):
        return name in self._extensions

    def get_extension(self, name):
        return self._extensions.get(name)

    def get_channel_locations(self):
        return self._locations

    def channel_ids_to_indices(self, channel_ids):
        return [self.channel_ids.index(str(channel)) for channel in channel_ids]


def _spike(n_samples, nbefore, depth_uv):
    """A toy spike: flat, one trough at the alignment sample."""
    wave = np.zeros(n_samples, dtype=float)
    wave[nbefore] = -abs(depth_uv)
    wave[nbefore + 1] = abs(depth_uv) * 0.4
    return wave


@pytest.fixture
def analyzer():
    """2 units over 4 electrodes, 20 samples, nbefore 5, peaks 60 / 20 µV."""
    n_samples, nbefore, n_channels = 20, 5, 4
    unit_ids = ["u0", "u1"]
    dense = np.zeros((len(unit_ids), n_samples, n_channels), dtype=float)
    snippets = {}
    rng = np.random.default_rng(0)
    for unit_index, unit_id in enumerate(unit_ids):
        for channel in range(n_channels):
            depth = (60.0 if unit_index == 0 else 20.0) / (channel + 1)
            dense[unit_index, :, channel] = _spike(n_samples, nbefore, depth)
        # (n_spikes, n_samples, n_channels): the waveform-buffer layout.
        snippets[unit_id] = dense[unit_index][None, :, :] + rng.normal(
            0.0, 1.0, size=(9, n_samples, n_channels)
        )
    locations = np.array([[0.0, 0.0], [17.5, 0.0], [0.0, 17.5], [17.5, 17.5]])
    return FakeAnalyzer(unit_ids, dense, snippets, locations, nbefore=nbefore)


# --------------------------------------------------------------------------
# unit_raster.plot_unit_raster
# --------------------------------------------------------------------------

def test_unit_raster_legends_every_mark(tmp_path, monkeypatch):
    sorting = FakeSorting({"a": [10, 500, 900], "b": [20, 40], "c": [30]})
    captured = _spy(monkeypatch, unit_raster)

    out = unit_raster.plot_unit_raster(
        sorting,
        tmp_path / "unit_raster.png",
        duration_s=0.1,
        segment_boundaries=(0.05,),
        max_units=2,
        caption_extra="The same spikes on real elapsed time: unit_raster_realtime.png",
    )
    _assert_png(out)

    spikes = _one(captured["legend"], "one spike from one sorted unit")
    assert _has(spikes, "2 unit rows")
    assert _has(spikes, "5 spikes drawn")
    assert _one(captured["legend"], "segment join")

    assert "time (s)" in captured["axes"]
    y_label = _one(captured["axes"], "one row per sorted unit")
    assert _has(y_label, "events / s")
    assert _has(y_label, "fastest first")

    caption = captured["caption"]
    assert _has(caption, "Segment join:")               # figure_text.SEAM, verbatim
    assert _has(caption, figure_text.ACRONYMS["ISI"])
    assert _has(caption, figure_text.CONTIGUOUS_AXIS)
    assert _has(caption, "events per second (Hz)")
    # max_units clipped the sort: the figure has to say what it is not showing.
    assert _has(caption, "the other 1 unit(s) fire more slowly and are not drawn")
    # Cross-artifact naming: the sibling is named by its real emitted filename.
    assert _has(caption, "unit_raster_realtime.png")
    _assert_plain_language(captured)


def test_unit_raster_realtime_legends_the_shading(tmp_path, monkeypatch):
    sorting = FakeSorting({"a": np.arange(0, 20_000, 500), "b": [100, 12_000]})
    captured = _spy(monkeypatch, unit_raster)

    out = unit_raster.plot_unit_raster(
        sorting,
        tmp_path / "unit_raster_realtime.png",
        time_gaps={"segments": [{"start_sample": 5_000, "gap_before_s": 30.0}]},
    )
    _assert_png(out)

    assert _one(captured["legend"], "no data recorded")
    assert "time (s, real elapsed)" in captured["axes"]
    assert _has(captured["caption"], figure_text.REAL_ELAPSED_AXIS)
    assert _has(captured["caption"], figure_text.NO_DATA_SHADING)
    _assert_plain_language(captured)


def test_unit_raster_figure_text_is_deterministic(tmp_path, monkeypatch):
    """Same input, same figure text — no timestamps, no reordering."""
    sorting = FakeSorting({"a": [10, 500], "b": [20]})
    real = unit_raster._save_and_release
    runs = []

    def spy(fig, out_path):
        runs.append(_snapshot(fig))
        return real(fig, out_path)

    monkeypatch.setattr(unit_raster, "_save_and_release", spy)
    for index in range(2):
        unit_raster.plot_unit_raster(sorting, tmp_path / f"r{index}.png", duration_s=0.1)

    assert runs[0] == runs[1]


# --------------------------------------------------------------------------
# unit_raster.plot_firing_rate_histogram
# --------------------------------------------------------------------------

def test_firing_rate_histogram_legends_bars_and_median(tmp_path, monkeypatch):
    sorting = FakeSorting({
        "a": np.arange(0, 20_000, 100),
        "b": np.arange(0, 20_000, 1_000),
        "c": [5],
        "d": [],
    })
    captured = _spy(monkeypatch, unit_raster)

    out = unit_raster.plot_firing_rate_histogram(
        sorting,
        tmp_path / "firing_rate_hist.png",
        duration_s=2.0,
        caption_extra="These are the rows of unit_raster.png, as a distribution.",
    )
    _assert_png(out)

    bars = _one(captured["legend"], "sorted units per firing-rate bin")
    assert _has(bars, "equally spaced on the logarithmic axis")
    median = _one(captured["legend"], "median firing rate")
    assert _has(median, "events / s (Hz)")

    assert "firing rate (Hz, i.e. events / s) — logarithmic axis" in captured["axes"]
    assert "sorted units in this bin (count)" in captured["axes"]

    caption = captured["caption"]
    assert _has(caption, "events per second (Hz)")
    assert _has(caption, "fired no spikes at all")
    assert _has(caption, figure_text.PROXY_NOT_MODEL)
    assert _has(caption, "unit_raster.png")
    _assert_plain_language(captured)


# --------------------------------------------------------------------------
# waveforms
# --------------------------------------------------------------------------

def test_unit_waveform_legends_snippets_template_and_units(tmp_path, monkeypatch, analyzer):
    captured = _spy(monkeypatch, waveforms)

    out = waveforms.plot_unit_waveform(
        analyzer,
        "u0",
        tmp_path / "unit_u0.png",
        n_spikes=5,
        caption_extra="Same unit elsewhere: footprints/unit_u0.png, traces/unit_u0.png.",
    )
    _assert_png(out)

    snippet = _one(captured["legend"], "one recorded spike")
    assert _has(snippet, f"5 of the {N_KEPT} spikes stored for this unit")
    template = _one(captured["legend"], "template:")
    assert _has(template, f"average of all {N_KEPT} stored spikes")
    assert _one(captured["legend"], "zero amplitude")

    assert "amplitude (µV)" in captured["axes"]
    assert _one(captured["axes"], "time (ms,")

    caption = captured["caption"]
    assert _has(caption, "Amplitude is microvolts (µV)")
    assert _has(caption, "milliseconds (ms)")
    assert _has(caption, "central 96% of the recorded spike values")
    assert _has(caption, "footprints/unit_u0.png")
    _assert_plain_language(captured)


def test_waveform_grid_names_its_axes_in_the_caption(tmp_path, monkeypatch, analyzer):
    captured = _spy(monkeypatch, waveforms)

    out = waveforms.plot_waveform_grid(
        analyzer,
        ["u0", "u1"],
        tmp_path / "waveform_grid_000.png",
        n_cols=2,
        n_spikes=4,
        caption="Each panel has its own full-size figure at waveforms/unit_u0.png.",
    )
    _assert_png(out)

    assert _one(captured["legend"], "one recorded spike, drawn faintly")
    assert _one(captured["legend"], "template: mean of all stored spikes")
    assert _one(captured["legend"], "time zero: the template's trough")

    caption = captured["caption"]
    assert _has(caption, "milliseconds (ms)")
    assert _has(caption, "microvolts (µV)")
    assert _has(caption, "comparable in SHAPE, not in size")
    assert _has(caption, "Up to 4 spikes are drawn in each panel")
    assert _has(caption, "average of EVERY spike stored for that unit")
    assert _has(caption, "waveforms/unit_u0.png")
    _assert_plain_language(captured)


def test_waveform_grid_legend_is_on_the_figure_not_in_a_panel(tmp_path, monkeypatch, analyzer):
    """A grid legend must not sit inside a panel, where it would cover data."""
    seen = {}
    real = waveforms._save_and_release

    def spy(fig, out_path):
        seen["figure_legends"] = len(fig.legends)
        seen["axes_legends"] = sum(ax.get_legend() is not None for ax in fig.axes)
        return real(fig, out_path)

    monkeypatch.setattr(waveforms, "_save_and_release", spy)
    waveforms.plot_waveform_grid(analyzer, ["u0", "u1"], tmp_path / "g.png", n_cols=2)

    assert seen["figure_legends"] == 1
    assert seen["axes_legends"] == 0


# --------------------------------------------------------------------------
# footprints
# --------------------------------------------------------------------------

def test_unit_footprint_legends_colours_and_states_micrometres(tmp_path, monkeypatch, analyzer):
    captured = _spy(monkeypatch, footprints)

    out = footprints.plot_unit_footprint(
        analyzer,
        "u0",
        tmp_path / "unit_u0.png",
        extremum_label="loudest electrode — its waveform is waveforms/unit_u0.png",
        caption="Drawn from the templates in analyzer/.",
    )
    _assert_png(out)

    colour = _one(captured["legend"], "one miniature copy")
    assert _has(colour, "peak-to-peak (PTP) amplitude")
    assert _one(captured["legend"], "every other electrode on the array")
    # Cross-artifact naming, supplied by the caller because only it knows the name.
    assert _one(captured["legend"], "waveforms/unit_u0.png")
    scale = _one(captured["legend"], "scale bar:")
    assert _has(scale, "ms across")
    assert _has(scale, "µV top to bottom")

    assert "x (µm)" in captured["axes"]
    assert "y (µm)" in captured["axes"]
    colorbar = _one(captured["axes"], "peak-to-peak (PTP) amplitude")
    assert _has(colorbar, "(µV)")

    caption = captured["caption"]
    assert _has(caption, figure_text.ACRONYMS["PTP"])
    assert _has(caption, "axes are micrometres, µm")
    assert _has(caption, "shared microvolt (µV) scale")
    assert _has(caption, "Drawn from the templates in analyzer/.")
    assert _one(captured["titles"], "µm")
    _assert_plain_language(captured)


def test_unit_footprint_per_channel_refuses_one_microvolt_height(tmp_path, monkeypatch, analyzer):
    captured = _spy(monkeypatch, footprints)

    footprints.plot_unit_footprint(
        analyzer, "u0", tmp_path / "pc.png", normalize="per_channel"
    )

    scale = _one(captured["legend"], "scale bar:")
    assert _has(scale, "each trace's own peak")
    assert _has(scale, "not comparable between electrodes")
    assert _has(captured["caption"], "colour still carries the true amplitude")
    _assert_plain_language(captured)


def test_footprint_grid_colorbar_is_relative_and_says_so(tmp_path, monkeypatch, analyzer):
    captured = _spy(monkeypatch, footprints)

    out = footprints.plot_footprint_grid(
        analyzer,
        ["u0", "u1"],
        tmp_path / "footprint_grid_000.png",
        n_cols=2,
        caption="Each panel has its own full-size figure at footprints/unit_u0.png.",
    )
    _assert_png(out)

    colorbar = _one(captured["axes"], "peak-to-peak (PTP) amplitude")
    assert _has(colorbar, "÷ that panel's largest")
    assert _has(captured["caption"], "fraction of the largest trace in the SAME panel")

    assert _one(captured["legend"], "one trace per electrode reached")
    assert _one(captured["legend"], "every other electrode")
    assert _one(captured["legend"], "largest-signal electrode")

    caption = captured["caption"]
    assert _has(caption, "micrometres (µm)")
    assert _has(caption, "miniature copy of that unit's average waveform")
    assert _has(caption, figure_text.ACRONYMS["PTP"])
    assert _has(caption, "printed in its title")
    assert _has(caption, "footprints/unit_u0.png")

    title = _one(captured["titles"], "u0")
    assert _has(title, "electrodes")
    assert _has(title, "µV")
    _assert_plain_language(captured)


# --------------------------------------------------------------------------
# unit_traces
# --------------------------------------------------------------------------

def _trace_recording(scaleable=True, n_samples=4_000):
    rng = np.random.default_rng(0)
    trace = rng.normal(0.0, 5.0, size=(n_samples, 1))
    for frame in (500, 1_500, 2_500):
        trace[frame, 0] = -80.0
    return FakeRecording(trace, channel_ids=("0",), scaleable=scaleable)


def test_unit_trace_legends_trace_marks_and_joins(tmp_path, monkeypatch):
    recording = _trace_recording()
    sorting = FakeSorting({"u0": [500, 1_500, 2_500]})
    captured = _spy(monkeypatch, unit_traces)

    out = unit_traces.plot_unit_trace(
        recording,
        sorting,
        "u0",
        "0",
        tmp_path / "unit_u0.png",
        window_s=0.4,
        start_time_s=0.0,
        stitch_frames=(2_000,),
        caption_extra="Same unit elsewhere: waveforms/unit_u0.png.",
    )
    _assert_png(out)

    trace = _one(captured["legend"], "recorded signal on electrode 0")
    assert _has(trace, "amplitude in µV")
    mark = _one(captured["legend"], "a spike the sorter assigned to unit u0")
    assert _has(mark, "trace's own value")
    assert _has(mark, "3 in this window")
    assert _one(captured["legend"], "segment join")

    assert "time (s)" in captured["axes"]
    assert "amplitude (µV)" in captured["axes"]

    caption = captured["caption"]
    assert _has(caption, "sorter's own event times")
    assert _has(caption, "Segment join:")               # figure_text.SEAM, verbatim
    assert _has(caption, figure_text.CONTIGUOUS_AXIS)
    assert _has(caption, "waveforms/unit_u0.png")
    _assert_plain_language(captured)


def test_unit_trace_unscaleable_recording_says_device_counts(tmp_path, monkeypatch):
    recording = _trace_recording(scaleable=False)
    sorting = FakeSorting({"u0": [500, 1_500]})
    captured = _spy(monkeypatch, unit_traces)

    unit_traces.plot_unit_trace(
        recording, sorting, "u0", "0", tmp_path / "counts.png",
        window_s=0.4, start_time_s=0.0,
    )

    assert "amplitude (device counts (ADC))" in captured["axes"]
    assert _one(captured["legend"], "amplitude in device counts (ADC)")
    caption = captured["caption"]
    assert _has(caption, "the device's own counts rather than converted to microvolts")
    assert _has(caption, figure_text.ACRONYMS["ADC"])
    _assert_plain_language(captured)


# --------------------------------------------------------------------------
# segment_activity
# --------------------------------------------------------------------------

def test_spikes_per_segment_labels_both_count_axes_and_the_colour_scale(tmp_path, monkeypatch):
    counts = np.array([[120, 0, 45], [0, 0, 900], [7, 3, 0]], dtype=np.int64)
    captured = _spy(monkeypatch, segment_activity)

    out = segment_activity.plot_spikes_per_segment(
        counts,
        ["u0", "u1", "u2"],
        ["seg000", "seg001", "seg002"],
        tmp_path / "spikes_per_segment.png",
        caption_extra="The same counts as arrays: spikes_per_segment.npy.",
    )
    _assert_png(out)

    colorbar = _one(captured["axes"], "log10(1 + count) scale")
    assert _has(colorbar, "spikes in this unit × segment cell (count)")

    assert "spikes in this segment (count)" in captured["axes"]
    assert "units firing at least once (count)" in captured["axes"]
    assert "segment (one recording configuration, in file order)" in captured["axes"]

    bars = _one(captured["legend"], "spikes per segment, all units")
    assert _has(bars, "left axis")
    line = _one(captured["legend"], "units firing at least once")
    assert _has(line, "right axis")

    caption = captured["caption"]
    assert _has(caption, "one step up the colour bar is ten times as many spikes")
    assert _has(caption, "A fully dark COLUMN is a segment the sort found nothing in")
    assert _has(caption, figure_text.PER_SEGMENT_ONLY)
    assert _has(caption, figure_text.PROXY_NOT_MODEL)
    assert _has(caption, "spikes_per_segment.npy")
    _assert_plain_language(captured)


# --------------------------------------------------------------------------
# backwards compatibility
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "func, new_params",
    [
        (unit_raster.plot_unit_raster, ["caption_extra"]),
        (unit_raster.plot_firing_rate_histogram, ["caption_extra"]),
        (waveforms.plot_unit_waveform, ["caption_extra"]),
        (waveforms.plot_waveform_grid, ["caption"]),
        (footprints.plot_unit_footprint, ["extremum_label", "caption"]),
        (footprints.plot_footprint_grid, ["caption"]),
        (unit_traces.plot_unit_trace, ["caption_extra"]),
        (segment_activity.plot_spikes_per_segment, ["caption_extra"]),
    ],
)
def test_new_annotation_params_are_optional_and_last(func, new_params):
    """Presentation params default to None and sit at the END of the signature.

    Anything else would break a caller that passes figsize/dpi positionally.
    """
    parameters = inspect.signature(func).parameters
    assert list(parameters)[-len(new_params):] == new_params
    for name in new_params:
        assert parameters[name].default is None


# --------------------------------------------------------------------------
# bottom matter: the small-multiple sheets share one margin with the caption
# --------------------------------------------------------------------------
#
# These sheets are data edge to edge, so their key goes on the FIGURE, in the
# same bottom margin the caption uses. Each used to hang its legend off the
# lowest panel with its own private helper — a guess that happened to clear the
# caption at the sizes tried. Both now go through `_add_caption`, which measures
# the two and gives each a band, and these assert the boxes really are disjoint.

def _measure_bottom_matter(monkeypatch, module):
    """Capture the drawn geometry of the caption, legend and panels."""
    seen = {}
    real = module._save_and_release

    def spy(fig, out_path):
        renderer = fig.canvas.get_renderer()
        to_fraction = fig.transFigure.inverted()
        special = {getattr(fig, name, None) for name in ("_suptitle", "_supxlabel", "_supylabel")}
        captions = [t for t in fig.texts if t not in special and _flat(t.get_text())]
        assert len(captions) == 1 and len(fig.legends) == 1
        seen["caption"] = captions[0].get_window_extent(renderer).transformed(to_fraction)
        seen["legend"] = fig.legends[0].get_window_extent(renderer).transformed(to_fraction)
        # The panel RECTANGLES. These sheets draw no tick labels, so the frame
        # is the panel's real edge. (`plot_footprint_grid`'s colour-bar label is
        # longer than a one-row bar and hangs below its own axes — a separate,
        # pre-existing defect of that figure, not of this margin.)
        seen["panels_bottom"] = min(
            ax.get_position().y0 for ax in fig.axes if ax.get_visible()
        )
        return real(fig, out_path)

    monkeypatch.setattr(module, "_save_and_release", spy)
    return seen


def _assert_bands_are_disjoint(seen):
    caption, legend = seen["caption"], seen["legend"]
    assert not caption.overlaps(legend), (
        f"legend {legend.bounds} covers the caption {caption.bounds}"
    )
    assert legend.y0 >= caption.y1, "the legend must sit above the caption"
    assert seen["panels_bottom"] >= legend.y1, "the legend must sit below the panels"


@pytest.mark.parametrize("n_cols", [1, 2, 4])
def test_waveform_grid_legend_never_covers_the_caption(tmp_path, monkeypatch, analyzer, n_cols):
    seen = _measure_bottom_matter(monkeypatch, waveforms)
    waveforms.plot_waveform_grid(
        analyzer, ["u0", "u1"], tmp_path / "waveform_grid.png", n_cols=n_cols,
        caption="An extra sentence from the caller, long enough to wrap the caption "
                "onto another line and give the legend something to collide with.",
    )
    _assert_bands_are_disjoint(seen)


@pytest.mark.parametrize("n_cols", [1, 2, 4])
def test_footprint_grid_legend_never_covers_the_caption(tmp_path, monkeypatch, analyzer, n_cols):
    seen = _measure_bottom_matter(monkeypatch, footprints)
    footprints.plot_footprint_grid(
        analyzer, ["u0", "u1"], tmp_path / "footprint_grid.png", n_cols=n_cols,
        caption="An extra sentence from the caller, long enough to wrap the caption "
                "onto another line and give the legend something to collide with.",
    )
    _assert_bands_are_disjoint(seen)
