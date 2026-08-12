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
        # The legend lives in `_add_caption`'s figure-level bottom margin, not
        # on the axes — sweep both, the figure-legends suites' convention.
        labels = []
        for legend in list(fig.legends):
            labels.extend(_flat(t.get_text()) for t in legend.get_texts())
        if ax.get_legend() is not None:
            labels.extend(_flat(t.get_text()) for t in ax.get_legend().get_texts())
        captured["legend"] = labels
        captured["xlabel"] = _flat(ax.get_xlabel())
        captured["ylabel"] = _flat(ax.get_ylabel())
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
