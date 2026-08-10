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

from ..diagnostics.channel_layout import _new_figure, _save_and_release

logger = logging.getLogger(__name__)

_FIGSIZE = (14.0, 9.0)
_DPI = 180


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
):
    """Heatmap of unit x segment counts over a per-segment roll-up; return `out_path`.

    Colour is ``log10(1 + n)`` — spike counts per cell span zero to tens of
    thousands, and a linear scale shows two units and a black wall. Rows keep
    the caller's unit order (this package ranks nothing); a fully-dark COLUMN is
    the read that matters most — a segment the sort found nothing in.

    The lower panel restates the columns as numbers: spikes per segment as bars,
    units-active per segment as a line on its own axis, which is the difference
    between "one loud unit" and "everyone was there".
    """
    import numpy as np

    counts = np.asarray(counts, dtype=np.int64)
    if counts.size == 0:
        raise ValueError("no counts to plot")

    fig = _new_figure(figsize, dpi)
    ax_heat, ax_bars = fig.subplots(
        2, 1, sharex=True, height_ratios=[3.0, 1.0], gridspec_kw={"hspace": 0.08}
    )

    image = ax_heat.imshow(
        np.log10(1.0 + counts.astype(float)),
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        origin="upper",
        extent=(-0.5, counts.shape[1] - 0.5, counts.shape[0] - 0.5, -0.5),
    )
    bar = fig.colorbar(image, ax=ax_heat, fraction=0.03, pad=0.01)
    bar.set_label("log10(1 + spikes)")
    ax_heat.set_ylabel(f"unit (sorter order, {counts.shape[0]} units)")
    ax_heat.set_title(
        title
        or (
            f"spikes per segment - {counts.shape[0]} units x "
            f"{counts.shape[1]} segments"
        )
    )

    columns = np.arange(counts.shape[1])
    ax_bars.bar(columns, counts.sum(axis=0), color="#4a6fa5", width=0.8)
    ax_bars.set_ylabel("spikes", color="#4a6fa5")
    ax_bars.set_xlabel("segment")
    ax_bars.set_xticks(columns)
    ax_bars.set_xticklabels([str(label) for label in segment_labels], rotation=90, fontsize=7)

    ax_units = ax_bars.twinx()
    ax_units.plot(columns, (counts > 0).sum(axis=0), color="#c0392b", lw=1.2, marker=".")
    ax_units.set_ylabel("units active", color="#c0392b")
    ax_units.set_ylim(bottom=0)

    out_path = _save_and_release(fig, out_path)
    logger.info(
        "wrote spikes-per-segment: %s (%d units x %d segments, %d spikes placed)",
        out_path, counts.shape[0], counts.shape[1], int(counts.sum()),
    )
    return out_path


__all__ = ["spikes_per_segment", "segment_activity_summary", "plot_spikes_per_segment"]
