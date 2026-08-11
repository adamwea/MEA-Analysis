"""How each unit's spikes distribute across the recording's segments.

The concatenated timeline the sorter saw is a stack of separate AxonTracking
configurations glued end to end. A unit's spike count per SEGMENT is therefore a
first-class descriptive fact about it: a unit firing in every segment is a cell
the electrodes kept seeing for the whole scan, while a unit alive in exactly one
segment is a different animal — maybe a cell the routing only covered once,
maybe drift, maybe an artifact of one configuration. The stitch downstream
averages templates across segments, so "which segments even contain this unit"
is also the provenance of every stitched template.

Two counts, deliberately both:

* :func:`spikes_per_segment` — the FULL spike trains. The honest description of
  where the unit fired.
* :func:`mea_modules.registration.selected_spikes_per_segment` — only the spikes
  the analyzer's ``random_spikes`` selection kept, i.e. what the templates were
  actually built from. The capsule emits both so a segment that fired plenty but
  contributed nothing to a template is visible.

Descriptive only: nothing here filters or scores a unit.
"""

import logging

from ..diagnostics.channel_layout import (
    _add_caption,
    _fold_caption,
    _legend_line,
    _legend_patch,
    _new_figure,
    _save_and_release,
)
from ..diagnostics.figure_text import PER_SEGMENT_ONLY, PROXY_NOT_MODEL

logger = logging.getLogger(__name__)

_FIGSIZE = (14.0, 9.0)
_DPI = 180

# Legend/caption defaults, matched to the diagnostics figures.
_LEGEND_FONTSIZE = 7
_LEGEND_FRAMEALPHA = 0.85

_BAR_COLOR = "#4a6fa5"
_UNITS_COLOR = "#c0392b"


def _segment_of(sample_indices, segment_bounds):
    """Segment index for each sample, or -1 outside every span.

    Same arithmetic as registration's placement (searchsorted on the starts,
    then an end check) — kept here rather than imported because that one is a
    private detail of the coverage module.
    """
    import numpy as np

    samples = np.asarray(sample_indices, dtype=np.int64)
    starts = np.asarray([int(start) for _rec, start, _end in segment_bounds], dtype=np.int64)
    ends = np.asarray([int(end) for _rec, _start, end in segment_bounds], dtype=np.int64)

    index = np.searchsorted(starts, samples, side="right") - 1
    valid = (index >= 0) & (index < starts.size)
    index = np.where(valid, index, 0)
    inside = valid & (samples < ends[index])
    return np.where(inside, index, -1)


def spikes_per_segment(sorting, segment_bounds):
    """``(n_units, n_segments)`` spike counts from the FULL trains.

    Rows follow ``sorting.unit_ids``; columns follow `segment_bounds`
    (``[(rec, start_frame, end_frame), ...]`` on the concatenated timeline, the
    shape :func:`mea_modules.registration.segment_bounds_from_manifest` returns).
    Spikes outside every span are counted and warned about, never silently
    dropped — they mean the bounds describe a different concatenation than the
    one that was sorted.
    """
    import numpy as np

    unit_ids = list(sorting.unit_ids)
    counts = np.zeros((len(unit_ids), len(segment_bounds)), dtype=np.int64)
    spikes = sorting.to_spike_vector()

    segment_of = _segment_of(spikes["sample_index"], segment_bounds)
    unplaced = int((segment_of < 0).sum())
    if unplaced:
        logger.warning(
            "%d spike(s) fall outside every segment span; the bounds do not "
            "match this recording", unplaced,
        )

    keep = segment_of >= 0
    np.add.at(counts, (spikes["unit_index"][keep], segment_of[keep]), 1)
    return counts


def segment_activity_summary(counts, unit_ids, segment_labels):
    """JSON-able description of a ``(n_units, n_segments)`` count matrix.

    Per unit: total spikes, how many segments it appears in at all, and the
    segment holding its largest share (with that share as a fraction) — the
    number that separates "alive throughout" from "alive in one configuration".
    Per segment: total spikes and how many units appear in it, which is the
    dead-configuration check from the sort's point of view.
    """
    import numpy as np

    counts = np.asarray(counts, dtype=np.int64)
    totals = counts.sum(axis=1)

    units = []
    for row, unit_id in enumerate(unit_ids):
        total = int(totals[row])
        dominant = int(np.argmax(counts[row])) if total else None
        units.append({
            "unit_id": str(unit_id),
            "n_spikes": total,
            "n_segments_active": int((counts[row] > 0).sum()),
            "dominant_segment": None if dominant is None else str(segment_labels[dominant]),
            "dominant_fraction": (
                None if not total else float(counts[row][dominant]) / float(total)
            ),
        })

    segments = [
        {
            "segment": str(label),
            "n_spikes": int(counts[:, column].sum()),
            "n_units_active": int((counts[:, column] > 0).sum()),
        }
        for column, label in enumerate(segment_labels)
    ]
    return {
        "n_units": len(list(unit_ids)),
        "n_segments": len(list(segment_labels)),
        "units": units,
        "segments": segments,
    }


def plot_spikes_per_segment(
    counts,
    unit_ids,
    segment_labels,
    out_path,
    title=None,
    figsize=_FIGSIZE,
    dpi=_DPI,
    caption_extra=None,
):
    """Heatmap of unit x segment counts over a per-segment roll-up; return `out_path`.

    Colour is ``log10(1 + n)`` — spike counts per cell span zero to tens of
    thousands, and a linear scale shows two units and a black wall. Rows keep
    the caller's unit order (this package ranks nothing); a fully-dark COLUMN is
    the read that matters most — a segment the sort found nothing in.

    The lower panel restates the columns as numbers: spikes per segment as bars,
    units-active per segment as a line on its own axis, which is the difference
    between "one loud unit" and "everyone was there".

    Both lower-panel encodings carry legend keys, the colour bar names its
    quantity and its scale, and every axis states its unit (Adam, 2026-08-11) —
    a bar height and a line height are both counts on this figure, of two
    completely different things, and only labels keep them apart.

    `caption_extra` appends one more caption sentence, which is where a caller
    names the sibling artifacts written beside this figure by their real emitted
    filenames, e.g.::

        caption_extra=(
            "The same counts as arrays: spikes_per_segment.npy (drawn here) and "
            "template_spikes_per_segment.npy (only the spikes the templates used)."
        )
    """
    import numpy as np

    counts = np.asarray(counts, dtype=np.int64)
    if counts.size == 0:
        raise ValueError("no counts to plot")

    fig = _new_figure(figsize, dpi)
    # The colour bar is a subplot of the SAME gridspec as the two panels rather
    # than space stolen from the heatmap by fig.colorbar(ax=...): the stolen
    # form puts the heatmap in a gridspec of its own, which the tight_layout
    # `_add_caption` re-runs cannot lay out — the bar's label ends up off the
    # page and the caption ends up on the bars.
    # No hspace/wspace here on purpose: setting them locally makes the whole
    # gridspec opaque to tight_layout (`locally_modified_subplot_params`), which
    # silently drops the caption on top of the segment labels. tight_layout
    # picks the spacing instead.
    grid = fig.add_gridspec(2, 2, height_ratios=[3.0, 1.0], width_ratios=[60.0, 1.0])
    ax_heat = fig.add_subplot(grid[0, 0])
    ax_bars = fig.add_subplot(grid[1, 0], sharex=ax_heat)
    cax = fig.add_subplot(grid[0, 1])
    # sharex alone does not hide the upper panel's tick labels the way
    # fig.subplots(sharex=True) does.
    ax_heat.tick_params(labelbottom=False)

    image = ax_heat.imshow(
        np.log10(1.0 + counts.astype(float)),
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        origin="upper",
        extent=(-0.5, counts.shape[1] - 0.5, counts.shape[0] - 0.5, -0.5),
    )
    bar = fig.colorbar(image, cax=cax)
    # Name the quantity AND the scale. What a step of the bar MEANS in spikes is
    # the half nobody reads off "log10", so it is spelled out in the caption.
    bar.set_label(
        "spikes in this unit × segment cell (count), log10(1 + count) scale",
        fontsize=8,
    )
    ax_heat.set_ylabel(f"unit (sorter order, {counts.shape[0]} units)")
    ax_heat.set_title(
        title
        or (
            f"spikes per segment - {counts.shape[0]} units x "
            f"{counts.shape[1]} segments"
        )
    )

    columns = np.arange(counts.shape[1])
    ax_bars.bar(columns, counts.sum(axis=0), color=_BAR_COLOR, width=0.8)
    ax_bars.set_ylabel("spikes in this segment (count)", color=_BAR_COLOR)
    ax_bars.set_xlabel("segment (one recording configuration, in file order)")
    ax_bars.set_xticks(columns)
    ax_bars.set_xticklabels([str(label) for label in segment_labels], rotation=90, fontsize=7)

    ax_units = ax_bars.twinx()
    ax_units.plot(columns, (counts > 0).sum(axis=0), color=_UNITS_COLOR, lw=1.2, marker=".")
    ax_units.set_ylabel("units firing at least once (count)", color=_UNITS_COLOR)
    ax_units.set_ylim(bottom=0)

    # The legend lives in the lower panel: the heatmap above is dense by design
    # and any box on it hides units, while the bars leave headroom. Its colour
    # is explained by the colour bar rather than by a key.
    ax_bars.legend(
        handles=[
            _legend_patch(_BAR_COLOR, "spikes per segment, all units (left axis)"),
            _legend_line(
                _UNITS_COLOR, "units firing at least once (right axis)", lw=1.2
            ),
        ],
        loc="best",
        fontsize=_LEGEND_FONTSIZE,
        framealpha=_LEGEND_FRAMEALPHA,
        labelspacing=0.7,
    )

    caption_parts = [
        "One row per sorted unit, one column per segment; a cell is how many "
        "spikes that unit fired in that segment, coloured on a log10(1 + count) "
        "scale — one step up the colour bar is ten times as many spikes. A fully "
        "dark COLUMN is a segment the sort found nothing in, and a row lit in one "
        "column only is a unit seen in a single recording configuration.",
        PER_SEGMENT_ONLY,
        PROXY_NOT_MODEL,
        caption_extra,
    ]
    _add_caption(fig, _fold_caption(caption_parts))

    out_path = _save_and_release(fig, out_path)
    logger.info(
        "wrote spikes-per-segment: %s (%d units x %d segments, %d spikes placed)",
        out_path, counts.shape[0], counts.shape[1], int(counts.sum()),
    )
    return out_path


__all__ = ["spikes_per_segment", "segment_activity_summary", "plot_spikes_per_segment"]
