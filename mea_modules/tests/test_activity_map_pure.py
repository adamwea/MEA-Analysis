"""Pure tests for `mea_modules.diagnostics.activity_map` — the whole-chip
template-projected activity field and its figure.

Synthetic 2-unit world on a small regular grid: unit A is loud and fast at one
electrode, unit B is quiet and slow at another. The field must peak where the
rate-weighted amplitude is largest, obey the exact ``sum_u rate_u * ptp_u[e]``
formula, respect the covered-channel mask, and render deterministic bytes — in
the style of the other ``test_*_pure.py`` files here (plain numpy, tmp_path, no
heavy deps).
"""

import json

import numpy as np
import pytest

from mea_modules.diagnostics import activity_map as am


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _grid_locations(n_rows=5, n_cols=5, pitch=17.5):
    rows, cols = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")
    return np.stack([cols.ravel() * pitch, rows.ravel() * pitch], axis=1).astype(float)


def _two_unit_bundle(n_channels, hot_a, hot_b, *, n_samples=8):
    """Two dense templates: unit A bumps channel `hot_a`, unit B bumps `hot_b`.

    A bump is a +/- excursion giving a known peak-to-peak. Everything else is
    flat (ptp 0), so the activity at a bumped channel is exactly
    rate * that channel's ptp.
    """
    templates = np.zeros((2, n_channels, n_samples), dtype=float)
    # unit A: ptp = 40 at hot_a (peak +20, trough -20)
    templates[0, hot_a, 2] = 20.0
    templates[0, hot_a, 5] = -20.0
    # unit B: ptp = 10 at hot_b
    templates[1, hot_b, 2] = 5.0
    templates[1, hot_b, 5] = -5.0
    weight = np.full((2, n_channels), 7.0)
    return templates, weight


def test_activity_follows_the_exact_formula_and_hot_spots():
    locations = _grid_locations()
    n_ch = locations.shape[0]  # 25
    hot_a, hot_b = 12, 6  # centre, and an off-centre cell
    templates, weight = _two_unit_bundle(n_ch, hot_a, hot_b)
    rates = np.array([10.0, 1.0])  # A loud+fast, B quiet+slow

    activity = am.template_projected_activity(templates, weight, rates)

    assert activity.shape == (n_ch,)
    # Exact projection: activity[hot_a] = rate_A * ptp_A[hot_a] = 10 * 40 = 400.
    assert activity[hot_a] == pytest.approx(400.0)
    # activity[hot_b] = rate_B * ptp_B[hot_b] = 1 * 10 = 10.
    assert activity[hot_b] == pytest.approx(10.0)
    # The loud+fast unit dominates: the global maximum is at hot_a.
    assert int(np.argmax(activity)) == hot_a
    # A channel neither unit bumps carries zero activity.
    quiet = 24
    assert quiet not in (hot_a, hot_b)
    assert activity[quiet] == pytest.approx(0.0)


def test_covered_mask_zeroes_uncovered_channels():
    locations = _grid_locations()
    n_ch = locations.shape[0]
    templates, weight = _two_unit_bundle(n_ch, hot_a=12, hot_b=6)
    rates = np.array([10.0, 1.0])
    # Mask channel 12 out for BOTH units: its bump must not count.
    weight[:, 12] = 0.0
    activity = am.template_projected_activity(templates, weight, rates)
    assert activity[12] == pytest.approx(0.0)


def test_nan_templates_and_nan_rates_are_safe():
    locations = _grid_locations()
    n_ch = locations.shape[0]
    templates, weight = _two_unit_bundle(n_ch, hot_a=12, hot_b=6)
    templates[0, 0, :] = np.nan  # a NaN channel must not poison the sum
    rates = np.array([10.0, np.nan])  # unit B has no count -> contributes 0
    activity = am.template_projected_activity(templates, weight, rates)
    assert np.isfinite(activity).all()
    assert activity[6] == pytest.approx(0.0)   # unit B silenced by NaN rate
    assert activity[12] == pytest.approx(400.0)


def test_chunking_does_not_change_the_result():
    locations = _grid_locations()
    n_ch = locations.shape[0]
    templates, weight = _two_unit_bundle(n_ch, hot_a=12, hot_b=6)
    rates = np.array([10.0, 1.0])
    a1 = am.template_projected_activity(templates, weight, rates, chunk=1)
    a2 = am.template_projected_activity(templates, weight, rates, chunk=100)
    assert np.array_equal(a1, a2)


def test_shape_mismatches_raise():
    locations = _grid_locations()
    n_ch = locations.shape[0]
    templates, weight = _two_unit_bundle(n_ch, hot_a=12, hot_b=6)
    with pytest.raises(ValueError):
        am.template_projected_activity(templates, weight, np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        am.template_projected_activity(templates[:, :, 0], weight, np.array([1.0, 2.0]))


def test_plot_writes_png_and_serializable_manifest(tmp_path):
    locations = _grid_locations()
    n_ch = locations.shape[0]
    templates, weight = _two_unit_bundle(n_ch, hot_a=12, hot_b=6)
    activity = am.template_projected_activity(templates, weight, np.array([10.0, 1.0]))

    out = tmp_path / am.WHOLE_CHIP_ACTIVITY_FILENAME
    manifest = am.plot_whole_chip_activity(activity, locations, out, style="presentation")

    png = out.read_bytes()
    assert png[:8] == PNG_MAGIC and len(png) > 1000
    assert manifest["files"]["png"] == str(out)
    assert manifest["render"] in ("image", "scatter")
    assert manifest["scale"] in ("log", "linear")
    assert manifest["n_channels"] == n_ch
    # A perfectly regular grid must render as an image.
    assert manifest["render"] == "image"
    assert manifest["grid"]["n_rows"] == 5 and manifest["grid"]["n_cols"] == 5
    json.dumps(manifest)  # the capsule writes this straight to JSON


def test_render_and_scale_knobs(tmp_path):
    locations = _grid_locations()
    n_ch = locations.shape[0]
    templates, weight = _two_unit_bundle(n_ch, hot_a=12, hot_b=6)
    activity = am.template_projected_activity(templates, weight, np.array([10.0, 1.0]))

    scatter = am.plot_whole_chip_activity(
        activity, locations, tmp_path / "scatter.png", render="scatter")
    assert scatter["render"] == "scatter"
    assert (tmp_path / "scatter.png").read_bytes()[:8] == PNG_MAGIC

    forced_log = am.plot_whole_chip_activity(
        activity, locations, tmp_path / "log.png", scale="log")
    assert forced_log["scale"] == "log"
    forced_lin = am.plot_whole_chip_activity(
        activity, locations, tmp_path / "lin.png", scale="linear")
    assert forced_lin["scale"] == "linear"


def test_scale_auto_switches_on_dynamic_range(tmp_path):
    locations = _grid_locations()
    n_ch = locations.shape[0]
    # Wide range: one channel 500x the rest -> auto picks log.
    wide = np.full(n_ch, 2.0)
    wide[10] = 1000.0
    m_wide = am.plot_whole_chip_activity(wide, locations, tmp_path / "w.png", scale="auto")
    assert m_wide["scale"] == "log"
    # Narrow range: within a factor of 2 -> auto stays linear.
    narrow = np.linspace(10.0, 18.0, n_ch)
    m_narrow = am.plot_whole_chip_activity(narrow, locations, tmp_path / "n.png", scale="auto")
    assert m_narrow["scale"] == "linear"


def test_output_is_deterministic(tmp_path):
    locations = _grid_locations()
    n_ch = locations.shape[0]
    templates, weight = _two_unit_bundle(n_ch, hot_a=12, hot_b=6)
    activity = am.template_projected_activity(templates, weight, np.array([10.0, 1.0]))
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    am.plot_whole_chip_activity(activity, locations, a)
    am.plot_whole_chip_activity(activity, locations, b)
    assert a.read_bytes() == b.read_bytes(), "same field must render identical bytes"


def test_svg_is_written_when_requested(tmp_path):
    locations = _grid_locations()
    n_ch = locations.shape[0]
    activity = np.linspace(1.0, 100.0, n_ch)
    manifest = am.plot_whole_chip_activity(
        activity, locations, tmp_path / "a.png", svg_path=tmp_path / "a.svg")
    assert manifest["files"]["svg"] == str(tmp_path / "a.svg")
    assert "<svg" in (tmp_path / "a.svg").read_text(errors="replace")[:500]


def test_dpi_knob_grows_the_file(tmp_path):
    locations = _grid_locations()
    n_ch = locations.shape[0]
    activity = np.linspace(1.0, 100.0, n_ch)
    am.plot_whole_chip_activity(activity, locations, tmp_path / "small.png", dpi=60)
    am.plot_whole_chip_activity(activity, locations, tmp_path / "big.png", dpi=300)
    assert (tmp_path / "big.png").stat().st_size > (tmp_path / "small.png").stat().st_size


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
