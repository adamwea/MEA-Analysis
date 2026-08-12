"""The unit-locations figure must be decodable from the figure alone.

Same contract, same harness as `test_diagnostics_figure_legends.py` (Adam's
2026-08-11 ruling: legend, colour-bar label and caption are part of the
CONTRACT): render the figure, capture every text artist before
`_save_and_release` clears it, and assert the required wording is present.
Pixel layout is deliberately not asserted.

Plus the two properties the pipeline's `recompute_unit_locations` capsule
leans on: NaN rows are explained rather than silently dropped, and the render
is deterministic (two identical calls produce byte-identical files — the
figure is regenerated retroactively onto review runs, so a render that
wiggles would show up as a spurious diff).
"""

import re

import numpy as np
import pytest

from mea_modules.diagnostics import unit_locations
from mea_modules.diagnostics.figure_text import acronym_note

# --------------------------------------------------------------------------- #
# A toy well: 36 electrodes on a 6 x 6 grid, 10 units.
# --------------------------------------------------------------------------- #

PITCH_UM = 17.5
CHANNELS = np.array(
    [[x * PITCH_UM, y * PITCH_UM] for y in range(6) for x in range(6)], dtype=float
)
N_UNITS = 10

RNG = np.random.default_rng(20)
MONOPOLAR = np.column_stack([
    RNG.uniform(0.0, 5 * PITCH_UM, size=N_UNITS),
    RNG.uniform(0.0, 5 * PITCH_UM, size=N_UNITS),
    RNG.uniform(5.0, 30.0, size=N_UNITS),
    RNG.uniform(100.0, 900.0, size=N_UNITS),
])
COM = MONOPOLAR[:, :2] + RNG.normal(0.0, 3.0, size=(N_UNITS, 2))
# units 8 and 9 have no triangulation fit; unit 9 has no CoM either
MONOPOLAR[8:] = np.nan
COM[9] = np.nan


def _normalise(text):
    return re.sub(r"\s+", " ", str(text)).strip()


@pytest.fixture()
def figure_text(monkeypatch):
    """Call `capture(call)`; get back the figure's normalised text blob."""
    def capture(call):
        from matplotlib.text import Text

        real = unit_locations._save_and_release
        seen = {}

        def spy(fig, out_path):
            seen["texts"] = [
                _normalise(artist.get_text())
                for artist in fig.findobj(Text)
                if _normalise(artist.get_text())
            ]
            return real(fig, out_path)

        monkeypatch.setattr(unit_locations, "_save_and_release", spy)
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


def test_legend_counts_caption_and_axes(tmp_path, figure_text):
    blob = figure_text(lambda: unit_locations.plot_unit_locations(
        CHANNELS, tmp_path / "unit_locations.png",
        monopolar=MONOPOLAR, com=COM,
        title="toy well — recomputed unit locations",
        n_zero_coverage=1, n_low_local_coverage=1,
    ))
    _assert_says(
        blob,
        "x (µm)", "y (µm)",
        # every layer names itself, with counts
        f"recording electrodes on the array (n={CHANNELS.shape[0]})",
        f"unit location, monopolar-triangulation fit (n=8 of {N_UNITS} units)",
        f"unit location, centre of mass (CoM) estimate (n=9 of {N_UNITS} units)",
        "line joining a unit's two estimates",
        # the acronym is expanded on the figure itself
        acronym_note("CoM"),
        # what a marker IS, in plain language
        "one sorted unit's estimated position",
        # missing units are counted and explained, reason by reason
        f"2 of {N_UNITS} units have no triangulation fit",
        "1 had no measured signal on any electrode",
        "1 had too few measured electrodes near the signal peak",
    )
    assert "x (um)" not in blob and "y (um)" not in blob
    # a reason with a zero count is dropped, not printed as "0 ..."
    assert "0 fit(s)" not in blob


def test_single_method_draws_no_ghost_layers(tmp_path, figure_text):
    blob = figure_text(lambda: unit_locations.plot_unit_locations(
        CHANNELS, tmp_path / "com_only.png",
        monopolar=MONOPOLAR, com=COM, methods=("com",),
    ))
    _assert_says(blob, "centre of mass (CoM) estimate")
    assert "monopolar" not in blob.lower()
    assert "line joining" not in blob
    # CoM-only view still explains its own missing unit
    _assert_says(blob, f"1 of {N_UNITS} units have no estimate")


def test_all_nan_method_layer_is_omitted_but_explained(tmp_path, figure_text):
    all_nan = np.full_like(MONOPOLAR, np.nan)
    blob = figure_text(lambda: unit_locations.plot_unit_locations(
        CHANNELS, tmp_path / "no_fits.png",
        monopolar=all_nan, com=COM, n_zero_coverage=0,
    ))
    # no key for a layer that drew nothing...
    assert "monopolar-triangulation fit" not in blob
    # ...but the caption still accounts for every unit
    _assert_says(blob, f"{N_UNITS} of {N_UNITS} units have no triangulation fit")


def test_misaligned_or_empty_inputs_raise():
    with pytest.raises(ValueError, match="row-aligned"):
        unit_locations.plot_unit_locations(
            CHANNELS, "unused.png", monopolar=MONOPOLAR, com=COM[:3],
        )
    with pytest.raises(ValueError, match="nothing to plot"):
        unit_locations.plot_unit_locations(CHANNELS, "unused.png")
    with pytest.raises(ValueError, match="unknown method"):
        unit_locations.plot_unit_locations(
            CHANNELS, "unused.png", monopolar=MONOPOLAR, methods=("centroid",),
        )


def test_render_is_deterministic(tmp_path):
    """Two identical calls, byte-identical files — the retrofit-diff guarantee."""
    a = unit_locations.plot_unit_locations(
        CHANNELS, tmp_path / "a.png", monopolar=MONOPOLAR, com=COM,
    )
    b = unit_locations.plot_unit_locations(
        CHANNELS, tmp_path / "b.png", monopolar=MONOPOLAR, com=COM,
    )
    assert a.read_bytes() == b.read_bytes()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
