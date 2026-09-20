"""Which channel ids not to trust, and which rule flagged each one.

:func:`mea_modules.quality.dead_well_flags` answers the array-level question —
what FRACTION of this recording is unusable, and is the well dead. This module
answers the id-level twin: *which* channels, under *which* rule. Same three
rules, same ratios, so the ids listed here and the fractions in that verdict
always describe the same set. The split exists because a verdict wants
fractions and a review artifact — a layout plot with the bad channels
highlighted, a list handed to a sorter — wants ids.

The rules are relative to the recording's own median noise rather than to an
absolute voltage, so they survive a change of gain, of units, or of chip
revision:

* dead — noise at or below ``dead_noise_ratio`` times the median (flat or
  disconnected); a NaN counts as dead,
* noisy — noise at or above ``noisy_noise_ratio`` times the median,
* detector — whatever :func:`mea_modules.quality.detect_bad_channels` flagged.

**The noise this reads must have been measured through a chain that filtered
in floating point.** Filtering integer traces quantizes each channel's MAD onto
multiples of the ADC least-significant bit, which collapses a thousand channels
onto three or four distinct noise values — and every rule here is a ratio to
the median of those values, so all three degenerate at once. The caller owns
that cast; this module owns what the numbers mean afterwards.

Pure arithmetic: a metrics dict in, a JSON-serializable dict out. No traces
read, no files written.
"""

import numpy as np

# Kept identical to `dead_well_flags`'s own defaults so the id list and the
# fractions cannot describe different sets when a caller passes neither.
DEFAULT_DEAD_NOISE_RATIO = 0.1
DEFAULT_NOISY_NOISE_RATIO = 5.0


def flag_channels(
    noise,
    bad_channels=None,
    dead_noise_ratio=DEFAULT_DEAD_NOISE_RATIO,
    noisy_noise_ratio=DEFAULT_NOISY_NOISE_RATIO,
):
    """Flag unusable channels by id, recording which rule fired for each.

    Parameters
    ----------
    noise : dict
        A :func:`mea_modules.quality.mad_noise` result — ``channel_ids`` and
        ``noise`` are the two keys read. A bare per-channel sequence is not
        enough here (unlike ``dead_well_flags``): ids are the whole output.
    bad_channels : dict or None
        A :func:`mea_modules.quality.detect_bad_channels` result, or None when
        the detector was not run or failed. Only ``bad_channel_ids`` is read,
        and ids are compared as strings so an int/str id space still joins.
    dead_noise_ratio : float
        Noise at or below this fraction of the median flags a channel dead.
    noisy_noise_ratio : float
        Noise at or above this multiple of the median flags it noisy. This is
        the same MAD multiple used as the spike-detection threshold, which is
        why one setting drives both.

    Returns
    -------
    dict
        ``channel_ids`` (recording order, never sorted — they may be strings),
        ``flagged_channel_ids``, ``n_flagged``, ``flagged_fraction``,
        ``by_rule`` with the ``dead`` / ``noisy`` / ``detector_bad`` id lists,
        and ``thresholds`` recording the two ratios this call actually used
        (the detector carries no threshold of its own). A channel may appear
        under more than one rule; the flagged list is their union.

        ``thresholds`` exists purely so a figure drawn from this dict — see
        :func:`flagged_channel_groups` — can print the cutoff it drew, without
        the caller having to thread `dead_noise_ratio`/`noisy_noise_ratio`
        through twice. It records what was passed in; it decides nothing.
    """
    channel_ids = list(noise["channel_ids"])
    values = np.asarray(noise["noise"], dtype=np.float64)

    median = float(np.nanmedian(values)) if np.any(np.isfinite(values)) else 0.0
    if median > 0:
        # Negated comparison rather than `<=`, so a NaN counts as dead.
        dead_mask = ~(values > dead_noise_ratio * median)
        noisy_mask = values >= float(noisy_noise_ratio) * median
    else:
        # Every channel flat or unmeasurable: there is nothing to be relative
        # to. Silent on purpose — `dead_well_flags` warns on the same dict, and
        # a caller running both would otherwise log the same fact twice.
        dead_mask = np.ones(values.size, dtype=bool)
        noisy_mask = np.zeros(values.size, dtype=bool)

    detector_ids = {str(cid) for cid in (bad_channels or {}).get("bad_channel_ids", [])}
    detector_mask = np.asarray([str(cid) in detector_ids for cid in channel_ids], dtype=bool)

    unusable = dead_mask | noisy_mask | detector_mask
    n_channels = len(channel_ids)

    def ids_where(mask):
        return [channel_ids[i] for i in np.flatnonzero(mask)]

    return {
        "channel_ids": channel_ids,
        "flagged_channel_ids": ids_where(unusable),
        "n_flagged": int(np.count_nonzero(unusable)),
        "flagged_fraction": float(np.count_nonzero(unusable) / n_channels) if n_channels else 0.0,
        "by_rule": {
            "dead": ids_where(dead_mask),
            "noisy": ids_where(noisy_mask),
            "detector_bad": ids_where(detector_mask),
        },
        "thresholds": {
            "dead_noise_ratio": float(dead_noise_ratio),
            "noisy_noise_ratio": float(noisy_noise_ratio),
        },
    }


# --- the flagged-channels figure's legend, computed from the same dict -----
#
# A figure with nothing flagged still has to say what was TESTED, or a clean
# well and a broken flagging rule look identical on the page (review ruling,
# 2026-09-19: "in this case I see none, but make it clear by what metric we
# might flag channels"). So every rule below draws a legend key at whatever
# count it actually got, including zero — the caller (`plot_channel_layout`)
# is told to key a group even when its id list is empty.

# One colour per rule, fixed here rather than beside the plotting code, so the
# figure and this module's own `by_rule` keys cannot name a criterion two
# different colours in two different places.
_RULE_COLORS = {
    "dead": "#c0392b",  # red — the historical single "flagged" colour
    "noisy": "#e67e22",  # orange
    "detector_bad": "#8e44ad",  # purple
}

# Drawing order == legend order. Fixed rather than `dict` iteration order so a
# rerun cannot silently reorder the keys.
_RULE_ORDER = ("dead", "noisy", "detector_bad")


def flagged_channel_groups(flagged):
    """``(channel_ids, label, colour)`` per rule, from a :func:`flag_channels` result.

    One group per entry in ``by_rule``, in a fixed order, each labelled with the
    criterion it tests — including the ratio this run actually used, read back
    from ``flagged["thresholds"]`` rather than re-passed by the caller. A rule
    that flagged nothing still gets a group (an empty id list), because the
    caller draws a legend key at whatever count a group has, zero included.

    Parameters
    ----------
    flagged : dict
        A :func:`flag_channels` result.

    Returns
    -------
    list of (list, str, str)
        Ready to hand to
        :func:`mea_modules.diagnostics.channel_layout.plot_channel_layout`'s
        ``groups`` argument.
    """
    thresholds = flagged.get("thresholds") or {}
    dead_ratio = thresholds.get("dead_noise_ratio")
    noisy_ratio = thresholds.get("noisy_noise_ratio")
    by_rule = flagged.get("by_rule") or {}

    labels = {
        "dead": (
            f"dead: noise ≤ {dead_ratio:g}× median"
            if dead_ratio is not None
            else "dead: noise at/below the median floor"
        ),
        "noisy": (
            f"noisy: noise ≥ {noisy_ratio:g}× median"
            if noisy_ratio is not None
            else "noisy: noise far above the median"
        ),
        "detector_bad": "detector-flagged (SpikeInterface)",
    }
    return [(by_rule.get(rule, []), labels[rule], _RULE_COLORS[rule]) for rule in _RULE_ORDER]
