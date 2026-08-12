"""Pure tests for `mea_modules.reconstruction.overlay` — the all-arbors,
one-color-per-neuron well overlay.

No `axon_velocity` needed: `unit_arbor_record` reads only plain attributes
(`branches` / `locations` / `init_channel`), so a tiny duck-typed stand-in
exercises the exact same code path an unpickled `GraphAxonTracking` does. The
pipeline repo's `tests/test_all_recon_overlay_toy.py` covers the real
capsule-on-real-`gtr` wiring; THIS file covers the drawing contract itself —
runs headless, writes real PNG/SVG bytes, one distinct color per neuron,
honest exclusion accounting, deterministic output — in the style of the other
`test_*_pure.py` files here (plain numpy, milliseconds, no fixtures beyond
tmp_path).
"""

import json
import re

import numpy as np
import pytest

from mea_modules.reconstruction import overlay as ov


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _norm(text):
    """Collapse whitespace so wrapped captions compare equal to their source —
    the same normalization `test_core_figure_legends._norm` uses."""
    return re.sub(r"\s+", " ", str(text)).strip()


class _FakeGtr:
    """The three attributes `unit_arbor_record` documents itself to read."""

    def __init__(self, locations, branches, init_channel):
        self.locations = locations
        self.branches = branches
        self.init_channel = init_channel


def _grid_locations(n_rows=6, n_cols=9, pitch=17.5):
    rows, cols = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")
    return np.stack([cols.ravel() * pitch, rows.ravel() * pitch], axis=1).astype(float)


def _three_unit_records(locations):
    """Three synthetic arbors on one grid: a horizontal run, a vertical run,
    and a two-branch unit — plus distinct init channels."""
    make = ov.unit_arbor_record
    unit_a = make(_FakeGtr(locations, [{"channels": np.arange(0, 8)}], 0), unit_id="3")
    unit_b = make(_FakeGtr(locations, [{"channels": np.arange(0, 54, 9)}], 9), unit_id="12")
    unit_c = make(
        _FakeGtr(
            locations,
            [{"channels": np.array([20, 21, 22, 23])}, {"channels": np.array([20, 29, 38])}],
            20,
        ),
        unit_id="7",
    )
    return [unit_a, unit_b, unit_c]


def test_unit_arbor_record_extracts_paths_init_and_counts():
    locations = _grid_locations()
    record = ov.unit_arbor_record(
        _FakeGtr(
            locations,
            [
                {"channels": np.array([0, 1, 2])},   # drawable
                {"channels": np.array([5])},          # single point: not drawable
                {"channels": np.array([900, 901])},   # out of range: not drawable
            ],
            init_channel=2,
        ),
        unit_id="42",
    )
    assert record["unit_id"] == "42"
    assert record["n_branches_total"] == 3
    assert len(record["branch_paths"]) == 1
    np.testing.assert_allclose(record["branch_paths"][0], locations[[0, 1, 2], :2])
    np.testing.assert_allclose(record["init_xy"], locations[2, :2])


def test_overlay_writes_png_and_svg_with_distinct_colors(tmp_path):
    locations = _grid_locations()
    records = _three_unit_records(locations)
    png = tmp_path / "all_reconstructions.png"
    svg = tmp_path / "all_reconstructions.svg"

    manifest = ov.plot_all_reconstructions(
        records, png, locations=locations, well_label="well000", svg_path=svg,
    )

    png_bytes = png.read_bytes()
    assert png_bytes[:8] == PNG_MAGIC
    assert len(png_bytes) > 1000
    svg_text = svg.read_text(errors="replace")
    assert "<svg" in svg_text[:500]

    assert manifest["n_units_drawn"] == 3
    assert manifest["n_branches_drawn"] == 4
    colors = [entry["color"] for entry in manifest["units"]]
    assert len(set(colors)) == 3, f"colors must be distinct per neuron, got {colors}"
    # Stable, unit-id-sorted assignment: numeric order 3 < 7 < 12 regardless
    # of the order the records arrived in.
    assert [entry["unit_id"] for entry in manifest["units"]] == ["3", "7", "12"]
    assert manifest["files"] == {"png": str(png), "svg": str(svg)}
    # The capsule writes this dict straight to JSON — it must serialize as-is.
    json.dumps(manifest)


def test_caption_explains_every_encoding_and_the_exclusions(tmp_path):
    locations = _grid_locations()
    records = _three_unit_records(locations)
    # One unit whose only branch is undrawable -> excluded HERE; plus two
    # upstream exclusions the capsule reports in.
    records.append(
        ov.unit_arbor_record(
            _FakeGtr(locations, [{"channels": np.array([1])}], 1), unit_id="99",
        )
    )
    manifest = ov.plot_all_reconstructions(
        records, tmp_path / "o.png", locations=locations, well_label="well000",
        n_units_excluded_upstream=2,
        excluded_upstream_reason="their unit directories carried no gtr.pkl",
    )
    caption = _norm(manifest["caption"])
    # The encodings, in plain language (Adam's legend rule).
    assert "one axonal branch" in caption
    assert "Colour identifies the neuron" in caption
    assert "initiation site" in caption
    assert "soma (cell body)" in caption
    assert "recording electrodes" in caption
    assert "micrometres" in caption and "µm" in caption
    # The accounting: 3 drawn of 6 total, both exclusion reasons stated.
    assert "3 of 6 reconstructed neuron(s) are drawn" in caption
    assert "no tracked branch spanned 2 or more electrodes" in caption
    assert "no gtr.pkl" in caption
    assert manifest["excluded_no_drawable_branch"] == ["99"]


def test_output_is_deterministic(tmp_path):
    locations = _grid_locations()
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    manifest_1 = ov.plot_all_reconstructions(
        _three_unit_records(locations), a, locations=locations, well_label="w")
    manifest_2 = ov.plot_all_reconstructions(
        list(reversed(_three_unit_records(locations))), b, locations=locations,
        well_label="w")
    assert a.read_bytes() == b.read_bytes(), "same records (any order) must render identical bytes"
    for entry_1, entry_2 in zip(manifest_1["units"], manifest_2["units"]):
        assert entry_1["color"] == entry_2["color"]


def test_distinct_unit_colors_are_unique_at_realistic_counts():
    # 43 = the validation well's real reconstructed-unit count; check well past it.
    for n in (1, 20, 43, 96):
        colors = ov.distinct_unit_colors(n)
        assert len(colors) == n
        assert len(set(colors)) == n, f"palette repeats a color at n={n}"


def test_dpi_and_style_knobs_are_honored(tmp_path):
    locations = _grid_locations()
    records = _three_unit_records(locations)
    small = ov.plot_all_reconstructions(
        records, tmp_path / "small.png", locations=locations, dpi=60)
    big = ov.plot_all_reconstructions(
        records, tmp_path / "big.png", locations=locations, dpi=300,
        linewidth=3.0, alpha=0.5)
    assert small["params"]["dpi"] == 60
    assert big["params"]["dpi"] == 300 and big["params"]["linewidth"] == 3.0
    assert (tmp_path / "big.png").stat().st_size > (tmp_path / "small.png").stat().st_size


# --------------------------------------------------------------------------
# style="presentation" — the deck-ready cut (Adam, 2026-08-12)
# --------------------------------------------------------------------------
#
# Same capture trick the per-unit footprint's presentation tests use
# (test_core_figure_legends._render_recon_capturing): this family saves with
# `fig.savefig` + `plt.close`, so patch `plt.close` to snapshot the figure
# before its artists are cleared. Assertions run on the REAL figure — title,
# legend texts, figure-level caption texts, and the drawn diamond markers.


def _render_overlay_capturing(tmp_path, records, style, **kwargs):
    import matplotlib.pyplot as plt

    cap = {}
    real_close = plt.close

    def spy(fig):
        ax = fig.axes[0]
        cap["title"] = _norm(ax.get_title())
        legend = ax.get_legend()
        cap["legend"] = (
            [_norm(t.get_text()) for t in legend.get_texts()]
            if legend is not None else []
        )
        # The multi-line prose caption is a FIGURE-level text (fig.text); the
        # title and scale-bar label are axes-level, so they never show here.
        cap["fig_texts"] = [_norm(t.get_text()) for t in fig.texts if _norm(t.get_text())]
        # The soma markers are ax.plot(marker="D") lines; branch polylines and
        # the scale bar carry no marker. A legend handle's diamond is NOT in
        # ax.lines, so this counts only real drawn somas.
        cap["n_diamonds"] = sum(1 for line in ax.lines if line.get_marker() == "D")
        return real_close(fig)

    plt.close = spy
    try:
        manifest = ov.plot_all_reconstructions(
            records, tmp_path / f"overlay_{style}.png", style=style, **kwargs)
    finally:
        plt.close = real_close
    return cap, manifest


def test_overlay_presentation_drops_caption_and_big_legend_keeps_diamonds(tmp_path):
    """Presentation: no prose caption, no 43-id legend, a standard title with
    no per-well stats — but the diamonds and one-color-per-neuron stay."""
    locations = _grid_locations()
    records = _three_unit_records(locations)  # all three carry an init marker
    cap, manifest = _render_overlay_capturing(
        tmp_path, records, "presentation", locations=locations, well_label="well000")

    # Standard title only — none of the per-well stats baked in.
    assert cap["title"] == "All reconstructions", cap["title"]
    for banned in ("neuron(s)", "branch(es)", "well000"):
        assert banned not in cap["title"], cap["title"]
    # The multi-line prose caption is gone.
    assert cap["fig_texts"] == [], cap["fig_texts"]
    # The outside-right per-unit-id legend is gone; only the compact soma key
    # remains (one entry), and it is not a unit-id list.
    assert len(cap["legend"]) == 1, cap["legend"]
    assert "soma" in cap["legend"][0].lower(), cap["legend"]
    assert not any(t.lower().startswith("unit ") for t in cap["legend"]), cap["legend"]
    # The diamonds Adam loves still draw — one soma marker per drawn neuron.
    assert cap["n_diamonds"] == 3, cap["n_diamonds"]
    # The manifest records the cut, honestly.
    assert manifest["params"]["style"] == "presentation"
    assert manifest["caption_shown"] is False
    assert manifest["unit_legend_shown"] is False
    assert manifest["soma_key_shown"] is True
    assert (tmp_path / "overlay_presentation.png").stat().st_size > 1000


def test_overlay_diagnostic_default_is_unchanged(tmp_path):
    """The diagnostic default keeps the verbose review figure verbatim: the
    stats title with the well label, the prose caption, and the per-unit-id
    legend all stay — and so do the diamonds."""
    locations = _grid_locations()
    records = _three_unit_records(locations)
    cap, manifest = _render_overlay_capturing(
        tmp_path, records, "diagnostic", locations=locations, well_label="well000")

    assert "neuron(s)" in cap["title"] and "branch(es)" in cap["title"], cap["title"]
    assert "well000" in cap["title"], cap["title"]
    # The self-documenting caption is on the figure.
    assert cap["fig_texts"] and any("axonal branch" in t for t in cap["fig_texts"]), cap["fig_texts"]
    # The per-unit-id legend is present (3 units, under the auto cap).
    assert any(t.lower().startswith("unit ") for t in cap["legend"]), cap["legend"]
    assert cap["n_diamonds"] == 3, cap["n_diamonds"]
    assert manifest["caption_shown"] is True
    assert manifest["unit_legend_shown"] is True
    assert manifest["soma_key_shown"] is False
    assert manifest["params"]["style"] == "diagnostic"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
