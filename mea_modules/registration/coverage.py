"""Attribute a sort's selected spikes to the segments they fall in.

Concatenation ran the sort on one continuous timeline built from several
recording segments joined end to end. Per-segment work downstream needs to know
how many of each unit's spikes landed in each segment, keyed by
``(unit, segment)``.

**Subsampling matters.** ``random_spikes`` keeps a capped, seeded subset per
unit, so a count meant to weight template averages must count the SELECTED
spikes, not the full trains. :func:`selected_spikes_per_segment` reads the
selection back off the analyzer for exactly that reason; using the full counts
silently miscounts whenever a unit is capped.

Pure library: no argparse, no printing, no ``__main__``.
"""

import logging

logger = logging.getLogger(__name__)


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


__all__ = [
    "selected_spikes_per_segment",
]
