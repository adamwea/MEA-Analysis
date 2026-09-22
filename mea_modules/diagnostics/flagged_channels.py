"""The flagged-channels figure: the probe with each flag rule's channels keyed.

Drawn from a :func:`mea_modules.diagnostics.channel_flags.flag_channels` result.
The rule colours and legend wording live here, with the figure, not beside the
flag arithmetic: the arithmetic is what a pipeline fingerprints as a computed
diagnostic, and a colour change is not a new number.
"""

from .channel_layout import plot_channel_layout


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
            f"dead ≤{dead_ratio:g}× median"
            if dead_ratio is not None
            else "dead at the median floor"
        ),
        "noisy": (
            f"noisy ≥{noisy_ratio:g}× median"
            if noisy_ratio is not None
            else "noisy above the median"
        ),
        "detector_bad": "detector-flagged",
    }
    return [(by_rule.get(rule, []), labels[rule], _RULE_COLORS[rule]) for rule in _RULE_ORDER]


def flagged_channel_note(flagged):
    """One line naming the spread the rules were applied to, or None.

    A figure where every rule reads ``(n=0)`` is ambiguous: it looks the same
    whether the array is genuinely clean or the cutoffs are unreachable. The
    rules are ratios to the median, so the answer is the range of those ratios
    the data actually covered -- printed beside the cutoffs, it lets a reader
    make the one comparison that settles it. A span that never approaches a
    cutoff is a clean array; a span that crosses one while nothing is flagged
    is a broken rule.

    Returns None when there is nothing to say (no `observed` block, or a run
    where nothing was measurable), so a caller can pass it straight through.
    """
    observed = (flagged or {}).get("observed") or {}
    low, high = observed.get("min_ratio"), observed.get("max_ratio")
    if low is None or high is None:
        return None
    return f"observed {low:.2f}–{high:.2f}× median (n={int(observed.get('n_measured', 0))})"


def plot_flagged_channels(recording, flagged, out_path=None, *, annotate=True, **kwargs):
    """Draw the probe with every flag rule's channels keyed, zero included.

    `recording` is anything :func:`plot_channel_layout` draws (a recording or a
    cached probe); `flagged` a :func:`flag_channels` result. Each rule gets a
    legend key at its count, and the legend title states the span of noise
    ratios the rules were applied to (:func:`flagged_channel_note`). Other
    keyword arguments go to :func:`plot_channel_layout`.
    """
    return plot_channel_layout(
        recording, out_path,
        groups=flagged_channel_groups(flagged),
        legend_title=flagged_channel_note(flagged),
        annotate=annotate,
        **kwargs,
    )
