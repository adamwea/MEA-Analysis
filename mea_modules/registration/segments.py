"""Put a backbone sort back onto the segments it was sorted from.

The sort happened on the concatenated recording, whose frames run 0..N across
every segment laid end to end, and whose channels are only the electrodes routed
in *all* of them. Each original segment has its own frame origin and its own,
much larger, electrode set. Registration is the arithmetic that connects the
two: a spike at concatenated frame ``f`` inside segment ``k`` is the same spike
at segment-local frame ``f - start_frame[k]``.

Unit identity is carried across unchanged. Nothing is re-sorted and nothing is
matched — the backbone sort is the authority on which spike belongs to which
unit, and this module only re-indexes time. That is what makes the recovered
per-segment templates comparable: they are the same units, seen through
different electrode subsets.

The payoff is spatial. A segment routing 1000 electrodes yields a template on
all 1000, not on the 353 the sort could see, and the union across segments
covers the array.

Pure library: no argparse, no printing, no ``__main__``.
"""

import logging

logger = logging.getLogger(__name__)


def segment_bounds_from_manifest(manifest):
    """``[(rec, start_frame, end_frame), ...]`` from a concat manifest.

    Reads the ``segments`` list concatenate wrote, which already carries the
    frame span each segment occupies on the concatenated timeline. Falling back
    to recomputing from ``stitch_frames`` would risk disagreeing with the file
    the binary was actually written from.
    """
    entries = manifest.get("segments") or ()
    if not entries:
        raise ValueError(
            "concat manifest carries no `segments`; it was written by an older "
            "concatenate than this registration expects"
        )

    bounds = []
    for entry in entries:
        start = entry.get("start_frame")
        end = entry.get("end_frame")
        if start is None or end is None:
            raise ValueError(
                f"segment {entry.get('rec')!r} has no start/end frame in the concat "
                "manifest; cannot place its spikes"
            )
        bounds.append((str(entry.get("rec")), int(start), int(end)))

    # Contiguity is the assumption every offset here rests on. A gap or an
    # overlap means the manifest does not describe the timeline the sorter saw,
    # and every spike after the discrepancy would land in the wrong segment.
    for (rec_a, _start_a, end_a), (rec_b, start_b, _end_b) in zip(bounds, bounds[1:]):
        if end_a != start_b:
            raise ValueError(
                f"segments {rec_a!r} and {rec_b!r} are not contiguous on the "
                f"concatenated timeline ({end_a} != {start_b}); the manifest does "
                "not describe the recording that was sorted"
            )
    return bounds


def split_spike_train(frames, start_frame, end_frame):
    """The spikes inside ``[start, end)``, re-indexed to segment-local frames."""
    import numpy as np

    frames = np.asarray(frames, dtype=np.int64)
    inside = frames[(frames >= int(start_frame)) & (frames < int(end_frame))]
    return inside - int(start_frame)


def register_sorting_to_segment(sorting, start_frame, end_frame, sampling_frequency=None):
    """The part of `sorting` that falls in one segment, on that segment's clock.

    Returns a SpikeInterface ``NumpySorting`` carrying every unit of the input —
    including units with no spikes in this segment, which stay present as empty
    trains so unit ids remain comparable across segments. A unit that is absent
    from one segment's sorting would otherwise look like a different unit
    downstream.
    """
    from spikeinterface.core import NumpySorting

    fs = float(sampling_frequency or sorting.get_sampling_frequency())
    local = {}
    kept = 0
    for unit_id in sorting.unit_ids:
        train = split_spike_train(
            sorting.get_unit_spike_train(unit_id), start_frame, end_frame
        )
        local[unit_id] = train
        kept += int(train.size)

    logger.debug(
        "segment [%d, %d): %d spike(s) across %d unit(s)",
        int(start_frame), int(end_frame), kept, len(local),
    )
    return NumpySorting.from_unit_dict([local], fs)


def registration_summary(sorting, bounds):
    """Per-segment spike counts plus a check that every spike was placed.

    A spike that falls outside every segment span means the sorter emitted a
    frame the manifest cannot account for. It is reported rather than dropped
    silently, because the usual cause is a manifest describing a different
    concatenation than the one that was sorted.
    """
    import numpy as np

    total = 0
    per_segment = []
    for rec, start, end in bounds:
        count = 0
        for unit_id in sorting.unit_ids:
            count += int(split_spike_train(
                sorting.get_unit_spike_train(unit_id), start, end
            ).size)
        per_segment.append({"rec": rec, "start_frame": int(start), "end_frame": int(end),
                            "n_spikes": count})
        total += count

    emitted = sum(
        int(np.asarray(sorting.get_unit_spike_train(u)).size) for u in sorting.unit_ids
    )
    unplaced = emitted - total
    if unplaced:
        logger.warning(
            "%d of %d spike(s) fall outside every segment span and were not "
            "registered; the concat manifest may not match the sorted recording",
            unplaced, emitted,
        )

    return {
        "n_units": len(sorting.unit_ids),
        "n_spikes_sorted": emitted,
        "n_spikes_registered": total,
        "n_spikes_unplaced": int(unplaced),
        "segments": per_segment,
    }


__all__ = [
    "segment_bounds_from_manifest",
    "split_spike_train",
    "register_sorting_to_segment",
    "registration_summary",
]
