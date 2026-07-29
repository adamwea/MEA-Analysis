"""Correct templates computed over a zero-padded union recording.

A union recording presents every electrode any segment routed, zero-filled where
a segment did not route one. Templates averaged straight off it are wrong, but
only in their divisor: SpikeInterface sums the same snippets we want and then
divides by ALL of a unit's spikes, where the correct divisor counts only the
spikes from segments that actually routed the electrode. So

    T_true[u, c] = T_si[u, c] * N[u] / N[u, c]

with ``N[u, c]`` the number of unit u's spikes lying in segments that routed
electrode c. That factor is metadata — no traces involved — and this module
computes it.

Why the correction cannot live in the recording: a recording holds one value at
``(sample, channel)`` and knows nothing about units, but the divisor is
per-(unit, channel). The same electrode needs a different divisor for a unit
that fired mostly in segment 1 than for one that fired mostly in segment 7. A
per-sample validity mask would express it, and SpikeInterface has no operation
that consumes one — ``ComputeTemplates`` takes only ``ms_before``/``ms_after``/
``operators``, and ``Templates.sparsity_mask`` is per-unit, applied at the wrong
stage. Hence the closed form here.

The result is identical to extracting a partial template per segment and fusing
them with a spike-count-weighted mean — the same arithmetic the previous build's
``templates/core/merge.py`` performed — but it needs one pass over the recording
instead of one analyzer per segment.

**Subsampling matters.** ``random_spikes`` keeps a capped, seeded subset per
unit, so the divisor must count the SELECTED spikes, not the full trains.
:func:`selected_spikes_per_segment` reads the selection back off the analyzer
for exactly that reason; using the full counts silently rescales by the wrong
factor whenever a unit is capped.

Pure library: no argparse, no printing, no ``__main__``.
"""

import logging

logger = logging.getLogger(__name__)


def routing_table(grid_positions, segment_locations, tolerance_um=1.0):
    """``(n_segments, n_grid)`` bool — which segment routed which electrode.

    `segment_locations` is one ``(n_local, 2)`` xy array per segment, in the
    order the segments occupy the concatenated timeline. Positions are matched
    onto the grid within `tolerance_um`; an unmatched electrode is a grid that
    was not built from these segments and is reported rather than dropped.
    """
    import numpy as np

    grid = np.asarray(grid_positions, dtype=float)[:, :2]
    table = np.zeros((len(segment_locations), grid.shape[0]), dtype=bool)

    unmatched = 0
    for index, locations in enumerate(segment_locations):
        locations = np.asarray(locations, dtype=float)[:, :2]
        for point in locations:
            distances = np.sum((grid - point) ** 2, axis=1)
            nearest = int(np.argmin(distances))
            if np.sqrt(distances[nearest]) <= float(tolerance_um):
                table[index, nearest] = True
            else:
                unmatched += 1

    if unmatched:
        logger.warning(
            "%d segment electrode(s) had no grid position within %.3f um; the grid "
            "does not describe these segments", unmatched, float(tolerance_um),
        )
    logger.info(
        "routing table: %d segment(s) x %d electrode(s); %d electrode(s) routed once, "
        "%d routed by every segment",
        table.shape[0], table.shape[1],
        int((table.sum(axis=0) == 1).sum()), int((table.sum(axis=0) == table.shape[0]).sum()),
    )
    return table


def _segment_of(sample_indices, segment_bounds):
    """Segment index for each sample, or -1 when it falls outside every span."""
    import numpy as np

    samples = np.asarray(sample_indices, dtype=np.int64)
    starts = np.asarray([int(start) for _rec, start, _end in segment_bounds], dtype=np.int64)
    ends = np.asarray([int(end) for _rec, _start, end in segment_bounds], dtype=np.int64)

    # searchsorted on the starts places each sample in the last segment that
    # begins at or before it; the end check then rejects anything past the tail.
    index = np.searchsorted(starts, samples, side="right") - 1
    valid = (index >= 0) & (index < starts.size)
    index = np.where(valid, index, 0)
    inside = valid & (samples < ends[index])
    return np.where(inside, index, -1)


def selected_spikes_per_segment(analyzer, segment_bounds):
    """``(n_units, n_segments)`` counts of the spikes templates were built from.

    Reads the ``random_spikes`` selection rather than the sorting, because that
    selection is what the template average actually used. When the extension is
    absent every spike was used and the full trains are counted instead.
    """
    import numpy as np

    unit_ids = list(analyzer.unit_ids)
    counts = np.zeros((len(unit_ids), len(segment_bounds)), dtype=np.int64)
    spikes = analyzer.sorting.to_spike_vector()

    extension = analyzer.get_extension("random_spikes") if analyzer.has_extension("random_spikes") else None
    if extension is None:
        logger.info("no random_spikes extension; counting every spike")
        selected = spikes
    else:
        indices = np.asarray(extension.get_data(), dtype=np.int64).ravel()
        selected = spikes[indices]
        logger.info(
            "counting the %d spike(s) random_spikes actually selected (of %d)",
            selected.size, spikes.size,
        )

    segment_of = _segment_of(selected["sample_index"], segment_bounds)
    unplaced = int((segment_of < 0).sum())
    if unplaced:
        logger.warning(
            "%d selected spike(s) fall outside every segment span and cannot be "
            "attributed; the bounds do not match this recording", unplaced,
        )

    keep = segment_of >= 0
    np.add.at(counts, (selected["unit_index"][keep], segment_of[keep]), 1)
    return counts


def coverage_counts(selected_per_segment, routing):
    """``(n_units, n_grid)`` — spikes behind each unit/electrode average.

    One matrix product: for electrode c, sum a unit's selected spikes over the
    segments that routed c. This is the divisor SpikeInterface should have used.
    """
    import numpy as np

    return np.asarray(selected_per_segment, dtype=np.int64) @ np.asarray(routing, dtype=np.int64)


def rescale_union_templates(templates, coverage, totals=None):
    """Turn union-recording templates into correctly-normalised ones.

    `templates` is SpikeInterface's ``(n_units, n_samples, n_channels)``.
    `coverage` is :func:`coverage_counts`. `totals` defaults to each unit's row
    sum, i.e. every selected spike.

    Electrodes no segment routed for a unit stay exactly zero — there is no
    measurement to scale — which is the same convention the per-segment merge
    uses for a channel nobody reached.
    """
    import numpy as np

    templates = np.asarray(templates)
    coverage = np.asarray(coverage, dtype=float)
    if totals is None:
        totals = coverage.max(axis=1)
    totals = np.asarray(totals, dtype=float)

    if coverage.shape != (templates.shape[0], templates.shape[2]):
        raise ValueError(
            f"coverage is {coverage.shape} but templates imply "
            f"({templates.shape[0]}, {templates.shape[2]})"
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(coverage > 0, totals[:, None] / coverage, 0.0)

    corrected = templates * scale[:, None, :].astype(templates.dtype, copy=False)

    uncovered = int((coverage == 0).sum())
    logger.info(
        "rescaled %d unit(s) x %d electrode(s); %d unit-electrode pair(s) had no "
        "contributing segment and stay zero (median scale %.2f, max %.2f)",
        templates.shape[0], templates.shape[2], uncovered,
        float(np.median(scale[coverage > 0])) if (coverage > 0).any() else 0.0,
        float(scale.max()) if scale.size else 0.0,
    )
    return corrected


__all__ = [
    "routing_table",
    "selected_spikes_per_segment",
    "coverage_counts",
    "rescale_union_templates",
]
