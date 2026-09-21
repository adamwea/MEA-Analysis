"""A zero-flag result has to say whether the rules had anything to bite on."""

import numpy as np
import pytest

from mea_modules.diagnostics import flag_channels, flagged_channel_groups, flagged_channel_note


def _noise(values):
    return {"channel_ids": list(range(len(values))), "noise": np.asarray(values, dtype=float)}


def _clean_array(n=987, seed=0):
    """The measured well005 spread: every channel within 0.71-1.90x the median."""
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(1.0, 0.15, n), 0.711, 1.901) * 5.0


def test_a_clean_array_flags_nothing_and_says_how_far_it_was_from_the_cutoffs():
    flagged = flag_channels(_noise(_clean_array()))

    assert flagged["n_flagged"] == 0
    observed = flagged["observed"]
    # The whole point: the span is reported, so "nothing fired" is checkable.
    assert observed["min_ratio"] > flagged["thresholds"]["dead_noise_ratio"]
    assert observed["max_ratio"] < flagged["thresholds"]["noisy_noise_ratio"]
    assert "observed" in flagged_channel_note(flagged)


def test_the_note_distinguishes_a_clean_array_from_an_unreachable_ruleset():
    """The two look identical on the figure without it.

    Both of these flag zero channels. One is a clean array; the other is a
    ruleset that could not fire on the data it was given. The rule keys read
    `(n=0)` in both cases -- only the observed span tells them apart.
    """
    clean = flag_channels(_noise(_clean_array()))

    # Same data, cutoffs moved so far out that nothing could ever reach them.
    unreachable = flag_channels(
        _noise(_clean_array()), dead_noise_ratio=1e-6, noisy_noise_ratio=1e6
    )

    assert clean["n_flagged"] == unreachable["n_flagged"] == 0
    assert [len(ids) for ids, _, _ in flagged_channel_groups(clean)] == [0, 0, 0]
    assert [len(ids) for ids, _, _ in flagged_channel_groups(unreachable)] == [0, 0, 0]

    # The keys name their own cutoffs, and the note names the span, so the two
    # figures differ in the one place that settles which case this is.
    clean_keys = [label for _, label, _ in flagged_channel_groups(clean)]
    unreachable_keys = [label for _, label, _ in flagged_channel_groups(unreachable)]
    assert clean_keys != unreachable_keys
    assert flagged_channel_note(clean) == flagged_channel_note(unreachable)


def test_real_faults_are_flagged_and_widen_the_reported_span():
    values = _clean_array()
    median = float(np.median(values))
    values[3] = 0.01 * median  # dead
    values[9] = 30.0 * median  # shorted

    flagged = flag_channels(_noise(values))

    assert flagged["n_flagged"] == 2
    by_rule = flagged["by_rule"]
    assert by_rule["dead"] == [3]
    assert by_rule["noisy"] == [9]
    assert flagged["observed"]["min_ratio"] < flagged["thresholds"]["dead_noise_ratio"]
    assert flagged["observed"]["max_ratio"] > flagged["thresholds"]["noisy_noise_ratio"]


def test_the_rule_keys_stay_publication_short():
    """Review ruling: legend keys read as they would in a published figure."""
    flagged = flag_channels(_noise(_clean_array()))

    for _, label, _ in flagged_channel_groups(flagged):
        assert len(label) <= 24, label
        assert ":" not in label  # "dead: noise <= 0.1x median" was the old form


def test_an_unmeasurable_run_yields_no_note_rather_than_a_wrong_one():
    flagged = flag_channels(_noise(np.zeros(8)))

    assert flagged["observed"]["min_ratio"] is None
    assert flagged_channel_note(flagged) is None


def test_the_note_survives_a_dict_without_an_observed_block():
    """Older bad_channels.json on disk predates the block; a figure drawn from
    one must still render rather than raising."""
    assert flagged_channel_note({}) is None
    assert flagged_channel_note({"observed": {}}) is None
