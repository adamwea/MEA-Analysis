"""Every figure these two modules emit must be decodable from the figure alone.

Adam's ruling (2026-08-11): a competent neuroscientist who has never read this
source has to be able to read the plot unaided. That makes the legend, the
colour-bar label and the caption part of the CONTRACT, not decoration — so they
are asserted here the way any other output is.

What each test does: render the figure, capture every text artist on it before
:func:`_save_and_release` clears the figure, and assert the required wording is
present. Legend labels are soft-wrapped by
:func:`mea_modules.diagnostics.channel_layout._wrap_label` and captions by
``_fold_caption``, so the blob is whitespace-normalised before matching —
wrapping is a rendering choice, the words are the contract.

Deliberately NOT asserted: pixel layout. These tests guard that the words exist
and say what we think they say; whether the legend box is 3 mm from the corner
is not a thing a test should own.
"""

import re

import numpy as np
import pytest

from mea_modules.diagnostics import recovery, stitch_wiring
from mea_modules.diagnostics.figure_text import (
    BACKBONE_CHANNELS,
    DENSE_STITCH,
    PROXY_NOT_MODEL,
    acronym_note,
)

# --------------------------------------------------------------------------- #
# A toy well: 12 electrodes on a 4 x 3 grid, 5 units, 3 segments.
# --------------------------------------------------------------------------- #

N_SAMPLES = 24
NBEFORE = 8
POSITIONS = np.array(
    [[x * 17.5, y * 17.5] for y in range(3) for x in range(4)], dtype=float
)
N_CHANNELS = POSITIONS.shape[0]
# Enough units that the measured-precision panel's coverage bins clear the ">20
# pairs per bin" floor it needs before it will draw anything — that panel's
# legend is part of what these tests guard, so the fixture has to reach it.
N_UNITS = 60
N_SEGMENTS = 3

# channels 0..3 are routed by every segment; the rest by one segment each
ROUTING = np.zeros((N_SEGMENTS, N_CHANNELS), dtype=bool)
ROUTING[:, :4] = True
for s in range(N_SEGMENTS):
    ROUTING[s, 4 + s] = True
    ROUTING[s, 7 + s] = True


def _coverage():
    """(n_units, n_channels) spike counts; the last unit is measured nowhere.

    Log-uniform between 1 and ~400 spikes, because the figures under test plot
    coverage on log axes and bin it logarithmically — a linear draw piles every
    pair into the top two bins and leaves the interesting ones empty.
    """
    rng = np.random.default_rng(7)
    routed = ROUTING.sum(axis=0) > 0
    coverage = np.zeros((N_UNITS, N_CHANNELS))
    for u in range(N_UNITS - 1):
        draw = 10.0 ** rng.uniform(0.0, 2.6, size=int(routed.sum()))
        coverage[u, routed] = np.maximum(np.round(draw), 1.0)
    return coverage


def _templates(coverage):
    """(n_units, n_samples, n_channels) in µV, zero where nothing was measured."""
    rng = np.random.default_rng(11)
    templates = rng.normal(0.0, 1.5, size=(N_UNITS, N_SAMPLES, N_CHANNELS))
    templates[:, NBEFORE + 2, :] -= 120.0
    templates[coverage[:, None, :].repeat(N_SAMPLES, axis=1) == 0] = 0.0
    return templates


COVERAGE = _coverage()
TEMPLATES = _templates(COVERAGE)
TOTALS = COVERAGE.max(axis=1)
BACKBONE_MASK = ROUTING.sum(axis=0) == N_SEGMENTS


# --------------------------------------------------------------------------- #
# Capture harness
# --------------------------------------------------------------------------- #


def _normalise(text):
    """Collapse every run of whitespace, so wrapped labels match unwrapped ones."""
    return re.sub(r"\s+", " ", str(text)).strip()


@pytest.fixture()
def figure_text(monkeypatch):
    """Call `capture(module, fn, ...)`; get back the figure's normalised text.

    The emitters release their figure as they save it, so the text has to be
    read inside `_save_and_release`. Both modules import that helper by name, so
    patching the module attribute intercepts it without touching the shared one.
    """
    def capture(module, call):
        from matplotlib.text import Text

        real = module._save_and_release
        seen = {}

        def spy(fig, out_path):
            seen["texts"] = [
                _normalise(artist.get_text())
                for artist in fig.findobj(Text)
                if _normalise(artist.get_text())
            ]
            return real(fig, out_path)

        monkeypatch.setattr(module, "_save_and_release", spy)
        out_path = call()
        monkeypatch.undo()
        assert out_path.is_file() and out_path.stat().st_size > 5_000, (
            f"{out_path} is missing or too small to be a real rendered figure"
        )
        assert "texts" in seen, "the emitter never went through _save_and_release"
        return " || ".join(seen["texts"])

    return capture


def _assert_says(blob, *phrases):
    missing = [p for p in phrases if _normalise(p) not in blob]
    assert not missing, "figure never says: " + " ;; ".join(missing)


# --------------------------------------------------------------------------- #
# The shared wording constants are what we think they are
# --------------------------------------------------------------------------- #


def test_unit_bearing_labels_carry_their_unit():
    """Every drawn quantity names its unit, or says it has none."""
    assert "µV" in recovery.AMPLITUDE_COLORBAR_LABEL
    assert "count" in recovery.SEGMENTS_ROUTED_COLORBAR_LABEL
    assert "count" in recovery.COVERAGE_WEIGHT_LABEL
    assert "count" in recovery.COVERAGE_COLORBAR_LABEL
    assert "count" in stitch_wiring.UNITS_PER_ELECTRODE_COLORBAR_LABEL
    assert "count" in stitch_wiring.COVERED_CHANNELS_AXIS_LABEL
    assert "µV" in stitch_wiring.DIFF_AXIS_LABEL
    assert "dimensionless" in stitch_wiring.DIFF_FRACTION_COLORBAR_LABEL
    # `um` is not a unit symbol; the ruling was that geometry reads µm.
    for label in (recovery.AMPLITUDE_COLORBAR_LABEL, recovery.COVERAGE_COLORBAR_LABEL):
        assert " um" not in label


def test_no_insider_jargon_in_the_reader_facing_constants():
    """Our terms of art never reach the reader undefined."""
    jargon = ("backbone", "seam", "rescale", "trust", "dense stitch")
    constants = [
        getattr(recovery, name) for name in recovery.__all__
        if name.isupper()
    ] + [
        getattr(stitch_wiring, name) for name in stitch_wiring.__all__
        if name.isupper() and name.startswith(("COVERED", "DIFF", "MULTI", "SINGLE",
                                               "STITCH", "SUBSAMPLE", "UNITS", "ZERO"))
    ]
    for text in constants:
        lowered = str(text).lower()
        for term in jargon:
            assert term not in lowered, f"{term!r} leaked into figure text: {text!r}"


def test_clipping_is_always_stated():
    """A clipped colour scale that does not say so is a lie by omission."""
    assert "1st percentile" in recovery.AMPLITUDE_FLOOR_NOTE
    assert "clipped at 2" in stitch_wiring.DIFF_FRACTION_COLORBAR_LABEL
    assert "1e-4" in stitch_wiring.DIFF_AXIS_LABEL
    assert "fixed random seed" in stitch_wiring.SUBSAMPLE_NOTE


# --------------------------------------------------------------------------- #
# recovery.py
# --------------------------------------------------------------------------- #


def test_coverage_map_legends_and_colorbar(tmp_path, figure_text):
    blob = figure_text(recovery, lambda: recovery.plot_coverage_map(
        POSITIONS, ROUTING.sum(axis=0), tmp_path / "coverage_map.png",
        routing=ROUTING, title="toy well — segment coverage",
        highlight_label="electrodes routed in every segment — compared across "
                        "segments in backbone_agreement.png",
    ))
    _assert_says(
        blob,
        "x (µm)", "y (µm)",
        recovery.SEGMENTS_ROUTED_COLORBAR_LABEL,
        "electrodes with this coverage count",
        "electrodes this segment routed (count)",
        # the cross-artifact name the caller supplied, wrapped but intact
        "compared across segments in backbone_agreement.png",
        # the plain-English gloss for our term of art
        BACKBONE_CHANNELS,
        recovery.MARKER_IS_ELECTRODE_NOTE,
    )
    assert "x (um)" not in blob and "y (um)" not in blob


def test_coverage_map_without_routing_still_labels_itself(tmp_path, figure_text):
    blob = figure_text(recovery, lambda: recovery.plot_coverage_map(
        POSITIONS, ROUTING.sum(axis=0), tmp_path / "coverage_map_2.png",
    ))
    _assert_says(blob, recovery.SEGMENTS_ROUTED_COLORBAR_LABEL, "x (µm)")
    # no per-segment panel -> no key for a rule that was never drawn
    assert recovery.BACKBONE_LEGEND_LABEL not in blob


def test_rescale_effect_stitch_trust_figure(tmp_path, figure_text):
    blob = figure_text(recovery, lambda: recovery.plot_rescale_effect(
        COVERAGE, TOTALS, tmp_path / "stitch_trust.png",
        templates=TEMPLATES, nbefore=NBEFORE,
        title="toy well — how much data backs each stitched cell",
    ))
    _assert_says(
        blob,
        recovery.COVERAGE_WEIGHT_LABEL,
        "unit-electrode pairs with this much data behind them",
        # the summary numbers ride in the legend label, not a floating text box
        "rest on a single one)",
        "reference mark at n = 1",
        "reference mark at n = 100",
        "the dashed lines are reference marks",
        "noise left in the pre-spike baseline",
        "reference curve, anchored on the first bin — drawn, not fitted",
        "share of unit-electrode pairs surviving this minimum (%)",
        "noise left after averaging (µV RMS)",
        # acronym expanded on the figure, verbatim from figure_text
        acronym_note("RMS"),
        # descriptive, not fitted
        PROXY_NOT_MODEL,
        DENSE_STITCH,
    )
    # the jargon this figure used to print
    assert "what a trust threshold costs" not in blob
    assert "(uV RMS)" not in blob


def test_rescale_effect_accepts_a_caller_supplied_weight_label(tmp_path, figure_text):
    blob = figure_text(recovery, lambda: recovery.plot_rescale_effect(
        COVERAGE, TOTALS, tmp_path / "stitch_trust_2.png",
        weight_label="spikes averaged into one unit-electrode value (count)",
    ))
    _assert_says(blob, "spikes averaged into one unit-electrode value (count)")
    # RMS is only defined when the panel that prints it was drawn
    assert acronym_note("RMS") not in blob


def test_footprint_gain_legends_and_colorbars(tmp_path, figure_text):
    blob = figure_text(recovery, lambda: recovery.plot_footprint_gain(
        POSITIONS, TEMPLATES[0], tmp_path / "footprint_gain.png",
        backbone_mask=BACKBONE_MASK, unit_id="u0", coverage=COVERAGE[0],
        title="toy well unit u0 — stitched footprint",
        highlight_label="electrodes routed in every segment — compared across "
                        "segments in backbone_agreement.png",
    ))
    _assert_says(
        blob,
        recovery.AMPLITUDE_COLORBAR_LABEL,
        recovery.COVERAGE_COLORBAR_LABEL,
        "compared across segments in backbone_agreement.png",
        f"(n={int(BACKBONE_MASK.sum())})",
        recovery.AMPLITUDE_FLOOR_NOTE,
        BACKBONE_CHANNELS,
        DENSE_STITCH,
        PROXY_NOT_MODEL,
        "x (µm)", "y (µm)",
    )
    assert "sorting backbone" not in blob
    assert "(uV, log)" not in blob


def test_footprint_gain_without_coverage_or_backbone(tmp_path, figure_text):
    blob = figure_text(recovery, lambda: recovery.plot_footprint_gain(
        POSITIONS, TEMPLATES[1], tmp_path / "footprint_gain_2.png", unit_id="u1",
    ))
    _assert_says(blob, recovery.AMPLITUDE_COLORBAR_LABEL, recovery.AMPLITUDE_FLOOR_NOTE)
    assert recovery.BACKBONE_LEGEND_LABEL not in blob
    assert recovery.COVERAGE_COLORBAR_LABEL not in blob


def test_rescale_before_after_has_a_colorbar_per_panel(tmp_path, figure_text):
    blob = figure_text(recovery, lambda: recovery.plot_rescale_before_after(
        POSITIONS, TEMPLATES[:2], COVERAGE[:2], TOTALS[:2],
        tmp_path / "before_after.png", unit_ids=["u0", "u1"],
        title="toy well — the coverage correction, before and after",
    ))
    _assert_says(
        blob,
        recovery.AMPLITUDE_COLORBAR_LABEL,
        recovery.NO_ELECTRODE_LEGEND_LABEL,
        "before the coverage correction",
        "after the coverage correction",
        "plain average over the segments that measured each electrode",
        "corrected for how many segments could see each electrode",
        recovery.AMPLITUDE_FLOOR_NOTE,
        "colour scales are NOT comparable between columns",
    )
    assert "raw SI average" not in blob
    assert "rescale · peak" not in blob


def test_template_agreement_names_both_routes_and_the_identity_line(tmp_path, figure_text):
    blob = figure_text(recovery, lambda: recovery.plot_template_agreement(
        TEMPLATES[0], TEMPLATES[0] * 1.001, tmp_path / "agreement.png",
        labels=("per-segment merge", "one pass over the union, then corrected"),
    ))
    _assert_says(
        blob,
        "per-segment merge (µV)",
        "one pass over the union, then corrected (µV)",
        "one averaged waveform value, on both routes (µV)",
        "exact agreement (the line y = x)",
        "values whose difference falls in this bin",
        "Amplitudes are in microvolts (µV) on both axes.",
    )


# --------------------------------------------------------------------------- #
# stitch_wiring.py
# --------------------------------------------------------------------------- #


def test_nan_coverage_summary_legends_and_colorbar(tmp_path, figure_text):
    blob = figure_text(stitch_wiring, lambda: stitch_wiring.plot_nan_coverage_summary(
        COVERAGE, POSITIONS, tmp_path / "nan_coverage.png",
        title="toy well — stitched coverage mask",
        highlight_label="units with no measurement on any electrode — listed by id "
                        "in nan_coverage.json",
    ))
    _assert_says(
        blob,
        stitch_wiring.UNITS_PER_ELECTRODE_COLORBAR_LABEL,
        stitch_wiring.COVERED_CHANNELS_AXIS_LABEL,
        "units whose electrode count falls in this bin",
        # the sibling artifact, named by its real emitted filename
        "listed by id in nan_coverage.json",
        "(n=1)",  # unit 4 is measured nowhere in the toy well
        DENSE_STITCH,
        "x (µm)", "y (µm)",
    )
    assert "x (um)" not in blob


def test_nan_coverage_summary_omits_the_zero_rule_when_none_apply(tmp_path, figure_text):
    covered = COVERAGE.copy()
    covered[-1, 0] = 5.0  # nobody is left with zero coverage
    blob = figure_text(stitch_wiring, lambda: stitch_wiring.plot_nan_coverage_summary(
        covered, POSITIONS, tmp_path / "nan_coverage_2.png",
    ))
    assert stitch_wiring.ZERO_COVERAGE_LEGEND_LABEL not in blob
    _assert_says(blob, stitch_wiring.UNITS_PER_ELECTRODE_COLORBAR_LABEL)


def _comparison_inputs():
    """`(summary, arrays, weight)` shaped like `compare_stitches` returns."""
    rng = np.random.default_rng(3)
    covered = COVERAGE > 0
    single = covered & (COVERAGE < 50)
    max_abs = np.where(covered, rng.random(COVERAGE.shape) * 0.5, 0.0)
    max_abs[single] = 0.0
    summary = {"labels": ["uniform", "spike_count"]}
    arrays = {
        "max_abs": max_abs,
        "peak_first": np.where(covered, 120.0, 0.0),
        "covered": covered,
        "single": single,
    }
    return summary, arrays, COVERAGE


def test_stitch_comparison_legends_colorbar_and_subsample_note(tmp_path, figure_text):
    summary, arrays, weight = _comparison_inputs()
    blob = figure_text(stitch_wiring, lambda: stitch_wiring.plot_stitch_comparison(
        summary, arrays, weight, tmp_path / "stitch_comparison.png",
        title="toy well — uniform vs spike_count",
    ))
    _assert_says(
        blob,
        stitch_wiring.SINGLE_COVERAGE_LEGEND_LABEL,
        stitch_wiring.MULTI_COVERAGE_LEGEND_LABEL,
        stitch_wiring.DIFF_AXIS_LABEL,
        stitch_wiring.DIFF_FRACTION_COLORBAR_LABEL,
        stitch_wiring.SUBSAMPLE_NOTE,
        # the weight axis names WHICH pass it belongs to
        "data behind this cell in the uniform stitch",
        "one unit-electrode cell measured by several segments",
        DENSE_STITCH,
        PROXY_NOT_MODEL,
    )
    assert "(uV, floored at 1e-4)" not in blob
    assert "|diff| / cell's own peak ||" not in blob


def test_stitch_comparison_accepts_a_caller_supplied_weight_label(tmp_path, figure_text):
    summary, arrays, weight = _comparison_inputs()
    blob = figure_text(stitch_wiring, lambda: stitch_wiring.plot_stitch_comparison(
        summary, arrays, weight, tmp_path / "stitch_comparison_2.png",
        weight_label="spikes averaged into this cell by the {label} pass (count)",
    ))
    _assert_says(blob, "spikes averaged into this cell by the uniform pass (count)")


def test_new_parameters_are_keyword_optional():
    """The capsules call these positionally; the new keys must be appended-only."""
    import inspect

    expected_tail = {
        recovery.plot_coverage_map: ["highlight_label", "caption"],
        recovery.plot_rescale_effect: ["weight_label", "caption"],
        recovery.plot_footprint_gain: ["highlight_label", "weight_label", "caption"],
        recovery.plot_rescale_before_after: ["caption"],
        recovery.plot_template_agreement: ["caption"],
        stitch_wiring.plot_nan_coverage_summary: ["highlight_label", "caption"],
        stitch_wiring.plot_stitch_comparison: ["weight_label", "caption"],
    }
    for func, tail in expected_tail.items():
        names = list(inspect.signature(func).parameters)
        assert names[-len(tail):] == tail, f"{func.__name__} signature changed shape"
        for name in tail:
            assert inspect.signature(func).parameters[name].default is None
