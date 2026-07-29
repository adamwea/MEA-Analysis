"""Join per-segment recordings into one timeline and materialize it for a sorter.

An AxonTracking scan is recorded as many short segments (one per electrode
configuration). Spike sorting wants a single continuous recording, so the
segments are laid end to end — :func:`concatenate_segments` does that lazily, and
:func:`save_concatenated` is the one place in this library that actually reads
traces and writes bytes: Kilosort and friends need a real binary file on disk,
not a lazy graph.

Two details are easy to get wrong and are handled here:

* **Channel sets must match exactly.** Segments are preprocessed on their full
  native channel set so per-segment template extraction can use every channel,
  but concatenation requires one shared channel set. Pass `channel_ids` (the
  precomputed common-electrode subset) and each segment is sliced down to it,
  with the resulting order verified — a silently permuted channel map produces
  plausible-looking garbage downstream.
* **A single segment is passed straight through.** Wrapping one recording in a
  concatenation object buys nothing and only complicates provenance.

Sample offsets of the joins are available from :func:`stitch_frames`, computed
from segment metadata alone so it stays lazy.
"""

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# Defaults carried over from the old build's concat_binary phase.
DEFAULT_CHUNK_DURATION = "1s"


def _num_samples(recording):
    """Total sample count of a recording, however many SI segments it holds."""
    total = getattr(recording, "get_total_samples", None)
    return int(total() if callable(total) else recording.get_num_samples())


def _select_channels(recording, channel_ids, label):
    """Slice a recording to `channel_ids`, refusing a reordered channel map.

    ``select_channels`` is documented to preserve the requested order, but a
    mismatch here would misalign every downstream template, so it is checked
    rather than trusted.
    """
    selected = recording.select_channels(list(channel_ids))
    got = list(selected.get_channel_ids())
    want = list(channel_ids)
    if got != want:
        raise RuntimeError(
            f"channel selection for segment {label!r} did not preserve the requested "
            f"channel set: asked for {len(want)} channels, got {len(got)}; "
            "refusing to concatenate misaligned channels"
        )
    return selected


def concatenate_segments(recordings, channel_ids=None, ignore_times=True):
    """Lay segment recordings end to end and return one lazy recording.

    `recordings` is any iterable of SpikeInterface recordings, in the order they
    should appear on the concatenated timeline. `channel_ids` is an optional
    shared channel set (the common electrodes across segments); when given every
    segment is sliced to exactly that set, in that order.

    Nothing is read: the result is a lazy view over its inputs. With one input
    recording that recording is returned unchanged.

    `ignore_times` matches SpikeInterface's own default — per-segment time
    vectors are dropped and the concatenated recording is indexed by sample.
    """
    recordings = list(recordings)
    if not recordings:
        raise ValueError("no recordings to concatenate")

    wanted = list(channel_ids) if channel_ids is not None else []
    if wanted:
        recordings = [
            _select_channels(rec, wanted, label=index) for index, rec in enumerate(recordings)
        ]

    logger.info(
        "concatenating %d segment(s), %d channel(s)",
        len(recordings),
        recordings[0].get_num_channels(),
    )

    if len(recordings) == 1:
        return recordings[0]

    from spikeinterface.core import concatenate_recordings

    return concatenate_recordings(recordings, ignore_times=ignore_times)


def stitch_frames(recordings):
    """Sample indices where consecutive segments join, for the same input order.

    Returns ``len(recordings) - 1`` cumulative offsets — the boundaries inside
    the concatenated timeline, excluding 0 and the total length. Useful for
    splitting sorted spike trains back into their source segments, and for
    marking discontinuities in trace plots.

    Reads only sample counts from metadata, so this stays lazy.
    """
    recordings = list(recordings)
    frames = []
    offset = 0
    for recording in recordings[:-1]:
        offset += _num_samples(recording)
        frames.append(int(offset))
    return frames


def load_concatenated(recording_dir):
    """Load a previously saved recording folder back as a SpikeInterface object.

    SpikeInterface has renamed its loader more than once; try the spellings in
    order so a folder written by one version still opens under another.
    """
    recording_dir = Path(recording_dir).expanduser()
    if not recording_dir.exists():
        raise FileNotFoundError(f"no such saved recording: {recording_dir}")

    import spikeinterface.core as sc

    errors = []
    for name in ("load", "load_recording", "load_extractor"):
        loader = getattr(sc, name, None)
        if not callable(loader):
            continue
        try:
            return loader(recording_dir)
        except Exception as exc:  # noqa: BLE001 - try the next spelling
            errors.append(f"{name}: {exc}")
    raise RuntimeError(
        f"could not load a saved recording from {recording_dir} ({'; '.join(errors)})"
    )


def save_concatenated(
    recording,
    output_dir,
    overwrite=False,
    n_jobs=1,
    chunk_duration=DEFAULT_CHUNK_DURATION,
    progress_bar=True,
    dtype=None,
    verbose=False,
):
    """Write `recording` to `output_dir` as a binary folder and return it loaded.

    This is the one function in this library that materializes traces to disk —
    sorters need a real binary file, not a lazy graph. Expect it to take real
    time and real disk: a concatenated AxonTracking scan is many GB.

    An existing `output_dir` is reused as-is unless `overwrite` is true, in which
    case it is removed first. Reuse returns the recording already on disk, so a
    rerun is cheap and the caller does not have to branch on it.

    `dtype` defaults to the recording's own dtype (float32 after preprocessing);
    pass e.g. ``"int16"`` to halve the file at the cost of precision.
    `n_jobs`/`chunk_duration` are SpikeInterface job kwargs — chunking is what
    keeps peak memory bounded while writing.

    Returns the loaded binary recording; the bytes live at `output_dir`.
    """
    output_dir = Path(output_dir).expanduser()

    if output_dir.exists():
        if not overwrite:
            logger.info("reusing existing concatenated recording: %s", output_dir)
            return load_concatenated(output_dir)
        # Clear it ourselves: the path may be a stray file rather than a folder,
        # which SpikeInterface's own overwrite would not remove.
        if output_dir.is_dir():
            shutil.rmtree(output_dir)
        else:
            output_dir.unlink()

    output_dir.parent.mkdir(parents=True, exist_ok=True)

    save_kwargs = {
        "folder": output_dir,
        "format": "binary",
        "overwrite": True,
        "verbose": bool(verbose),
        "n_jobs": max(1, int(n_jobs)),
        "chunk_duration": str(chunk_duration),
        "progress_bar": bool(progress_bar),
    }
    # Omitted rather than passed as None, so SpikeInterface keeps the
    # recording's own dtype instead of being handed a null to interpret.
    if dtype is not None:
        save_kwargs["dtype"] = dtype

    logger.info(
        "writing concatenated recording to %s (n_jobs=%d, chunk_duration=%s)",
        output_dir,
        save_kwargs["n_jobs"],
        save_kwargs["chunk_duration"],
    )
    saved = recording.save(**save_kwargs)
    logger.info("wrote concatenated recording: %s", output_dir)
    return saved
