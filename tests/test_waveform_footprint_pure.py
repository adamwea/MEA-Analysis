"""Pure toy test for :func:`mea_modules.postprocess.footprints.
plot_unit_waveform_footprint` — the waveform-footprint style rendered from
DENSE template arrays (no SortingAnalyzer, no SpikeInterface).

Covers, on a synthetic grid with a Gaussian-decay unit:

1. a real PNG is produced at the returned path, deterministically
   (two renders -> byte-identical files);
2. the channel-subset rule actually subsets — only channels clearing
   ``trace_threshold`` of the unit's peak carry a trace, the rest are
   position-only context;
3. Adam's figure rulings hold (2026-08-11): every encoding is legended, the
   omission rule is stated on the figure, PTP is expanded, axes carry µm.

Presentation assertions go through a spy on the module's
``_save_and_release`` (the ``test_postprocess_figure_legends.py`` pattern) so
they read the real legend/caption strings while the public entry point still
runs end to end, PNG included.
"""

import numpy as np
import pytest

from mea_modules.postprocess import footprints

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

FS = 20000.0
N_ROWS = 12
N_COLS = 12
PITCH_UM = 17.5
N_SAMPLES = 24
TRACE_THRESHOLD = 0.1


def _grid_locations():
    rows, cols = np.meshgrid(np.arange(N_ROWS), np.arange(N_COLS), indexing="ij")
    xs = cols.ravel().astype(float) * PITCH_UM
    ys = rows.ravel().astype(float) * PITCH_UM
    return np.stack([xs, ys], axis=1)


def _gaussian_template(locations, center_xy=(87.5, 105.0), peak_uv=120.0, spread_um=30.0):
    """(n_channels, n_samples) whose ptp decays smoothly from `center_xy`."""
    dist = np.linalg.norm(locations - np.asarray(center_xy), axis=1)
    ptp = peak_uv * np.exp(-dist / spread_um)
    template = np.zeros((locations.shape[0], N_SAMPLES))
    template[:, 10] = 0.4 * ptp
    template[:, 12] = -0.6 * ptp
    return template


def _flat(text):
    return " ".join(str(text).split())


def _squash(text):
    return "".join(str(text).split())


def _has(haystack, needle):
    return _squash(needle) in _squash(haystack)


def _spy(monkeypatch):
    """Capture legend/caption/axes strings and the trace count pre-save."""
    captured = {}
    real = footprints._save_and_release

    def spy(fig, out_path):
        from matplotlib.collections import LineCollection

        ax = fig.axes[0]
        traces = [
            c for c in ax.collections
            if isinstance(c, LineCollection) and len(c.get_paths()) > 0
        ]
        captured["n_traces"] = len(traces[0].get_paths()) if traces else 0
        # The diagnostic legend lives in `_add_caption`'s figure-level bottom
        # margin; the presentation compact key lives on the axes. Capture the
        # two SEPARATELY (the split is what a presentation assertion checks)
        # and also merged, the figure-legends suites' convention.
        fig_labels = []
        for legend in list(fig.legends):
            fig_labels.extend(_flat(t.get_text()) for t in legend.get_texts())
        ax_labels = []
        if ax.get_legend() is not None:
            ax_labels.extend(_flat(t.get_text()) for t in ax.get_legend().get_texts())
        captured["fig_legend"] = fig_labels
        captured["ax_legend"] = ax_labels
        captured["legend"] = fig_labels + ax_labels
        captured["xlabel"] = _flat(ax.get_xlabel())
        captured["ylabel"] = _flat(ax.get_ylabel())
        captured["title"] = _flat(ax.get_title())
        # Axis limits as SET before save — the crop a zoom_bbox produces
        # (aspect="equal" here is adjustable="box", so it moves the box, never
        # the data limits, and get_xlim/get_ylim read back exactly as set).
        captured["xlim"] = tuple(float(v) for v in ax.get_xlim())
        captured["ylim"] = tuple(float(v) for v in ax.get_ylim())
        # Colour-bar label is the y-label of the colour-bar's own axes.
        captured["cbar_labels"] = [
            _flat(a.get_ylabel()) for a in fig.axes if a.get_ylabel()
        ]
        captured["caption"] = " ".join(_flat(t.get_text()) for t in fig.texts)
        return real(fig, out_path)

    monkeypatch.setattr(footprints, "_save_and_release", spy)
    return captured


def test_waveform_footprint_png_subset_and_legend(tmp_path, monkeypatch):
    locations = _grid_locations()
    template = _gaussian_template(locations)
    captured = _spy(monkeypatch)

    out = footprints.plot_unit_waveform_footprint(
        template,
        locations,
        tmp_path / "waveform_footprint.png",
        unit_id="7",
        fs=FS,
        trace_threshold=TRACE_THRESHOLD,
    )
    png = out.read_bytes()
    assert png[:8] == PNG_MAGIC and len(png) > 1000

    # The subset rule: exactly the channels clearing the threshold got traces.
    ptp = np.ptp(template, axis=1)
    expected = int((ptp >= TRACE_THRESHOLD * ptp.max()).sum())
    assert captured["n_traces"] == expected
    assert 0 < expected < locations.shape[0]

    # Adam's rulings: µm axes, legended encodings, stated omission, PTP expanded.
    assert "µm" in captured["xlabel"] and "µm" in captured["ylabel"]
    legend_blob = " ".join(captured["legend"])
    assert _has(legend_blob, "average waveform")
    assert _has(legend_blob, "position only")          # grey context dots
    assert _has(legend_blob, "signal is largest")      # the extremum ring
    assert _has(legend_blob, "scale bar")              # the corner bracket
    assert _has(captured["caption"], "PTP = peak-to-peak")
    assert _has(captured["caption"], "omitted")
    assert _has(captured["caption"], f"{TRACE_THRESHOLD:.0%}")


def test_waveform_footprint_deterministic(tmp_path):
    locations = _grid_locations()
    template = _gaussian_template(locations)

    first = footprints.plot_unit_waveform_footprint(
        template, locations, tmp_path / "a.png", unit_id="7", fs=FS,
    )
    second = footprints.plot_unit_waveform_footprint(
        template, locations, tmp_path / "b.png", unit_id="7", fs=FS,
    )
    assert first.read_bytes() == second.read_bytes()


def test_waveform_footprint_presentation_drops_caption_legend_and_stats(tmp_path, monkeypatch):
    """style="presentation" is the deck cut (Adam, 2026-08-12): no prose
    caption, no framed figure legend, the title is the unit id alone with no
    per-unit stats baked in, and the colour-bar label is the compact form."""
    locations = _grid_locations()
    template = _gaussian_template(locations)
    captured = _spy(monkeypatch)

    footprints.plot_unit_waveform_footprint(
        template, locations, tmp_path / "presentation.png",
        unit_id="7", fs=FS, style="presentation",
    )

    # No multi-line prose caption block below the figure.
    assert captured["caption"].strip() == ""
    # The verbose figure-margin legend is gone; at most a compact inline key.
    assert captured["fig_legend"] == []
    fig_and_ax = " ".join(captured["legend"])
    assert not _has(fig_and_ax, "average waveform")
    assert not _has(fig_and_ax, "position only")
    # Standard title = the unit id alone, carrying no per-unit numbers.
    assert captured["title"] == "Unit 7"
    for banned in ("of", "peak", "µV", "electrodes drawn"):
        assert banned not in captured["title"], captured["title"]
    # Compact colour-bar label.
    assert any(label == "PTP (µV)" for label in captured["cbar_labels"]), captured["cbar_labels"]


def test_waveform_footprint_presentation_is_less_sparse(tmp_path, monkeypatch):
    """The sparsity fix (Adam: "can they be less sparse?"): presentation's
    lower default trace threshold draws strictly MORE traces than the
    diagnostic default on the same unit."""
    locations = _grid_locations()
    template = _gaussian_template(locations)
    captured = _spy(monkeypatch)

    footprints.plot_unit_waveform_footprint(
        template, locations, tmp_path / "diag.png", unit_id="7", fs=FS,
    )
    n_diagnostic = captured["n_traces"]
    footprints.plot_unit_waveform_footprint(
        template, locations, tmp_path / "pres.png", unit_id="7", fs=FS,
        style="presentation",
    )
    n_presentation = captured["n_traces"]

    assert n_presentation > n_diagnostic, (n_presentation, n_diagnostic)
    # And the presentation default really is the lower threshold constant.
    assert (
        footprints.DEFAULT_WAVEFORM_TRACE_THRESHOLD_PRESENTATION
        < footprints.DEFAULT_WAVEFORM_TRACE_THRESHOLD
    )


def test_waveform_footprint_diagnostic_default_is_unchanged(tmp_path, monkeypatch):
    """The diagnostic default keeps the verbose review figure verbatim — the
    prose caption, the figure-wide legend and the stats-carrying title all
    stay (the team still wants this version)."""
    locations = _grid_locations()
    template = _gaussian_template(locations)
    captured = _spy(monkeypatch)

    footprints.plot_unit_waveform_footprint(
        template, locations, tmp_path / "diagnostic.png", unit_id="7", fs=FS,
    )

    assert _has(captured["caption"], "PTP = peak-to-peak")
    assert _has(captured["caption"], "omitted")
    legend_blob = " ".join(captured["fig_legend"])
    assert _has(legend_blob, "average waveform")
    assert _has(legend_blob, "scale bar")
    # The verbose title still carries the per-unit numbers.
    assert _has(captured["title"], "waveform footprint")
    assert _has(captured["title"], "electrodes drawn")


def test_waveform_footprint_zoom_bbox_crops_axes_to_box_plus_pad(tmp_path, monkeypatch):
    """zoom_bbox crops the axes to the supplied box + the pitch-scaled margin,
    WINNING over the default drawn-trace framing (Adam, 2026-08-12 — frame each
    footprint to its arbor region so the far-field threshold-crossings drop out
    of view). The margin is the module's own zoom_pad_pitches*pitch +
    max(width_um, height_um), computed from the layout, not hard-coded."""
    locations = _grid_locations()
    template = _gaussian_template(locations)
    captured = _spy(monkeypatch)

    # A box strictly inside the grid, tighter than where the drawn traces reach,
    # so the assertion is a real crop and not coincidentally the default frame.
    bbox = (30.0, 90.0, 40.0, 100.0)
    footprints.plot_unit_waveform_footprint(
        template, locations, tmp_path / "arbor_zoom.png",
        unit_id="7", fs=FS, zoom_bbox=bbox,
    )

    pitch = footprints._dense_pitch_um(locations)
    width_um = footprints._WIDTH_IN_PITCHES * pitch
    height_um = footprints._HEIGHT_IN_PITCHES * pitch
    margin = footprints._ZOOM_MARGIN_PITCHES * pitch + max(width_um, height_um)

    xmin, xmax, ymin, ymax = bbox
    assert captured["xlim"] == pytest.approx((xmin - margin, xmax + margin))
    assert captured["ylim"] == pytest.approx((ymin - margin, ymax + margin))

    # And the crop is genuinely tighter than the default drawn-trace frame:
    # rendering the same unit WITHOUT a bbox frames a strictly wider span.
    footprints.plot_unit_waveform_footprint(
        template, locations, tmp_path / "default_zoom.png",
        unit_id="7", fs=FS,
    )
    default_span_x = captured["xlim"][1] - captured["xlim"][0]
    cropped_span_x = (xmax + margin) - (xmin - margin)
    assert cropped_span_x < default_span_x


def test_waveform_footprint_zoom_bbox_pad_scales_with_zoom_pad_pitches(tmp_path, monkeypatch):
    """The arbor crop honors zoom_pad_pitches: a larger pad widens the frame by
    exactly the extra pitches on every side, so the knob is live under a bbox."""
    locations = _grid_locations()
    template = _gaussian_template(locations)
    captured = _spy(monkeypatch)
    bbox = (30.0, 90.0, 40.0, 100.0)
    pitch = footprints._dense_pitch_um(locations)

    footprints.plot_unit_waveform_footprint(
        template, locations, tmp_path / "pad4.png",
        unit_id="7", fs=FS, zoom_bbox=bbox, zoom_pad_pitches=4.0,
    )
    span_x_4 = captured["xlim"][1] - captured["xlim"][0]
    footprints.plot_unit_waveform_footprint(
        template, locations, tmp_path / "pad8.png",
        unit_id="7", fs=FS, zoom_bbox=bbox, zoom_pad_pitches=8.0,
    )
    span_x_8 = captured["xlim"][1] - captured["xlim"][0]
    # +4 pitches of pad on each side => +8 pitches of total x-span.
    assert span_x_8 - span_x_4 == pytest.approx(8.0 * pitch)


def test_waveform_footprint_rejects_flat_template(tmp_path):
    locations = _grid_locations()
    with pytest.raises(ValueError, match="flat"):
        footprints.plot_unit_waveform_footprint(
            np.zeros((locations.shape[0], N_SAMPLES)),
            locations,
            tmp_path / "flat.png",
            unit_id="0",
            fs=FS,
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
