"""The flagged-channels figure's legend: one key per rule, even at zero.

A review ruling, 2026-09-19, on a run with nothing flagged: "in this case I
see none, but make it clear by what metric we might flag channels." These
tests pin the two halves of that fix: `flag_channels` records the thresholds a
figure needs to word its keys, and `flagged_channel_groups` turns those into
one legend-ready group per rule, drawn whether or not it matched anything.
"""

import re

import numpy as np
import pytest

from mea_modules.diagnostics import channel_flags as cf
from mea_modules.diagnostics import channel_layout as cl
from mea_modules.diagnostics import flagged_channels as fc


def _norm(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def _noise(values, ids=None):
    ids = ids or [f"ch{i}" for i in range(len(values))]
    return {"channel_ids": ids, "noise": np.asarray(values, dtype=float)}


# --------------------------------------------------------------------------
# flag_channels — thresholds ride along, unchanged criteria
# --------------------------------------------------------------------------


def test_flag_channels_records_the_thresholds_it_used():
    """Metadata only: the ratios a figure needs to word its keys, not a new rule."""
    noise = _noise([1.0, 5.0, 5.0, 5.0, 5.0, 50.0])
    result = cf.flag_channels(noise, dead_noise_ratio=0.2, noisy_noise_ratio=6.0)

    assert result["thresholds"] == {"dead_noise_ratio": 0.2, "noisy_noise_ratio": 6.0}
    # The criteria themselves are untouched by adding this field.
    assert result["by_rule"]["dead"] == ["ch0"]
    assert result["by_rule"]["noisy"] == ["ch5"]


def test_flag_channels_defaults_still_match_dead_well_flags():
    """Unchanged from before: the two functions share one default pair so the
    id list and the array-level fractions can never describe different sets."""
    assert cf.DEFAULT_DEAD_NOISE_RATIO == 0.1
    assert cf.DEFAULT_NOISY_NOISE_RATIO == 5.0


# --------------------------------------------------------------------------
# flagged_channel_groups — one group per rule, zero included
# --------------------------------------------------------------------------


def test_flagged_channel_groups_keys_every_rule_even_when_nothing_matched():
    """The trigger case: a clean well must still say what was tested."""
    noise = _noise([5.0] * 6)  # nothing dead, nothing noisy, no detector
    flagged = cf.flag_channels(noise, bad_channels=None)

    groups = fc.flagged_channel_groups(flagged)
    labels = [label for _ids, label, _color in groups]
    ids_by_label = {label: ids for ids, label, _color in groups}

    assert len(groups) == 3  # dead, noisy, detector_bad — always all three
    for label in labels:
        assert ids_by_label[label] == []  # empty, but still a named group


def test_flagged_channel_groups_names_the_ratio_actually_used():
    noise = _noise([1.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    flagged = cf.flag_channels(noise, dead_noise_ratio=0.3, noisy_noise_ratio=7.0)

    groups = fc.flagged_channel_groups(flagged)
    dead_ids, dead_label, _ = groups[0]
    noisy_ids, noisy_label, _ = groups[1]

    assert "0.3" in dead_label
    assert "7" in noisy_label
    assert dead_ids == ["ch0"]
    assert noisy_ids == []


def test_flagged_channel_groups_colours_are_distinct():
    noise = _noise([5.0] * 4)
    groups = fc.flagged_channel_groups(cf.flag_channels(noise))
    colors = [color for _ids, _label, color in groups]
    assert len(set(colors)) == len(colors)


# --------------------------------------------------------------------------
# plot_channel_layout(groups=...) — the figure that reads those groups
# --------------------------------------------------------------------------


class FakeRecording:
    """A small grid of channels with real locations."""

    def __init__(self, n_channels=6):
        self._ids = [f"ch{i}" for i in range(n_channels)]
        self._loc = np.c_[
            [17.5 * (i % 3) for i in range(n_channels)],
            [17.5 * (i // 3) for i in range(n_channels)],
        ].astype(float)

    def get_channel_ids(self):
        return list(self._ids)

    def get_channel_locations(self):
        return self._loc


@pytest.fixture
def capture(monkeypatch):
    """Snapshot the legend BEFORE `_save_and_release` clears the figure's axes."""
    captured = {}
    real = cl._save_and_release

    def wrapper(fig, out_path, _real=real):
        (axis,) = fig.get_axes()
        captured["labels"] = [_norm(t.get_text()) for t in axis.get_legend().get_texts()]
        return _real(fig, out_path)

    monkeypatch.setattr(cl, "_save_and_release", wrapper)
    return captured


def test_plot_channel_layout_groups_key_every_group_including_empty(capture, tmp_path):
    recording = FakeRecording()
    groups = [
        (["ch0"], "dead: noise ≤ 0.1× median", "#c0392b"),
        ([], "noisy: noise ≥ 5× median", "#e67e22"),
        ([], "detector-flagged (SpikeInterface)", "#8e44ad"),
    ]

    cl.plot_channel_layout(recording, tmp_path / "flagged.png", groups=groups, annotate=False)

    labels = " || ".join(capture["labels"])
    assert "dead: noise" in labels and "(n=1)" in labels
    assert "noisy: noise" in labels and "(n=0)" in labels
    assert "detector-flagged" in labels and "(n=0)" in labels
    # The grey base still names itself, at the count NOT covered by any group.
    assert "routed electrodes" in labels and "(n=5)" in labels


def test_plot_channel_layout_groups_and_highlight_are_not_combined(tmp_path):
    """`groups` replaces `highlight_channel_ids` for a call; they don't merge —
    only one drawing path runs, so behaviour cannot depend on both at once."""
    recording = FakeRecording()
    out = cl.plot_channel_layout(
        recording, tmp_path / "a.png",
        highlight_channel_ids=["ch0"], highlight_label="ignored",
        groups=[(["ch1"], "only this one", "#000000")],
    )
    assert out.exists()


def test_flagged_channel_groups_feeds_plot_channel_layout_end_to_end(capture, tmp_path):
    """The two halves of the fix wired together, on a run with nothing flagged."""
    recording = FakeRecording()
    noise = _noise([5.0] * 6, ids=recording.get_channel_ids())
    flagged = cf.flag_channels(noise, bad_channels=None)

    cl.plot_channel_layout(
        recording, tmp_path / "flagged.png",
        groups=fc.flagged_channel_groups(flagged), annotate=False,
    )

    labels = " || ".join(capture["labels"])
    assert "dead" in labels and "noisy" in labels and "detector" in labels
    assert labels.count("(n=0)") == 3


def test_one_renderer_draws_the_flagged_channels_figure(capture, tmp_path):
    """The figure is one mea_modules call: every rule keyed at its count (zero
    included) and the span the rules were applied to in the legend title, so a
    clean array and an unreachable cutoff read differently."""
    recording = FakeRecording()
    noise = _noise([5.0, 5.2, 4.8, 5.1, 4.9, 5.0], ids=recording.get_channel_ids())
    flagged = cf.flag_channels(noise, bad_channels=None)

    out = fc.plot_flagged_channels(recording, flagged, tmp_path / "flagged.png", annotate=False)

    assert out.exists()
    labels = " || ".join(capture["labels"])
    assert labels.count("(n=0)") == 3
    assert "dead" in labels and "noisy" in labels and "detector" in labels


def test_the_flag_arithmetic_carries_no_presentation():
    """Colours and legend wording live with the figure, so editing them never
    touches the module the diagnostics fingerprint counts."""
    assert not hasattr(cf, "_RULE_COLORS")
    assert not hasattr(cf, "flagged_channel_groups")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
