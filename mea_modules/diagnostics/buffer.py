"""Read a lazy recording once, for every diagnostic that wants it.

A SpikeInterface recording is a recipe, not data: every ``get_traces`` re-runs
the whole filter chain over the span asked for and throws the result away.
Six diagnostics reading one segment paid for six filter passes -- measured on a
985-channel segment, 126.4 s of a 127 s detection was the chain and 0.53 s the
detector. Buffering the filtered segment once and serving every diagnostic from
the copy removes that cost without changing a single number the diagnostics
compute from it.

The mode is decided by the caller, never here. How much memory a process may
take depends on how many siblings it runs beside, which is a fact about the
machine and the scheduler, not about the mechanic:

``memory``
    the whole filtered span in RAM, freed when the caller is done.
``disk``
    a binary copy in `scratch_dir`, deleted when the caller is done -- eager at
    the cost of transient disk, never an artifact that accumulates.
``lazy``
    the recording as it came, re-filtered on every read.

Nothing here is written anywhere the caller did not name, and a ``disk`` buffer
is removed even when the work inside the block raises.
"""

import logging
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

SIGNAL_BUFFER_MODES = ("memory", "disk", "lazy")

# Whole chunks per worker, in process, quietly: the buffer is one read of the
# chain, and a progress bar per chunk in a pipeline log is noise.
_JOB_KWARGS = {"n_jobs": 1, "chunk_duration": "1s", "progress_bar": False}


def signal_bytes(recording):
    """Bytes the recording occupies once read, at its own dtype."""
    return int(recording.get_total_memory_size())


def _keep_filtered_flag(source, buffered):
    """Carry ``is_filtered`` across, whatever the copy did with annotations.

    It is load-bearing: the noise and bad-channel estimators high-pass a
    recording that does NOT report itself filtered, so a buffer that lost the
    flag would be filtered a second time and every noise number would move.
    """
    try:
        if source.is_filtered() and not buffered.is_filtered():
            buffered.annotate(is_filtered=True)
    except Exception:  # noqa: BLE001 - an extractor without annotations
        logger.debug("could not carry is_filtered onto the buffer", exc_info=True)
    return buffered


@contextmanager
def buffered_signal(recording, mode, scratch_dir=None, job_kwargs=None):
    """Yield ``(recording, info)`` with the signal buffered as `mode` says.

    `info` records what happened: the mode, the bytes held, and how long the
    read took -- the one-off price every diagnostic after it no longer pays,
    which a per-diagnostic profile would otherwise silently leave out.
    """
    if mode not in SIGNAL_BUFFER_MODES:
        raise ValueError(f"signal buffer must be one of {SIGNAL_BUFFER_MODES}, got {mode!r}")

    info = {"mode": mode, "bytes": signal_bytes(recording), "seconds": 0.0}
    if mode == "lazy":
        yield recording, info
        return

    kwargs = {**_JOB_KWARGS, **(job_kwargs or {})}
    folder = None
    started = time.perf_counter()
    info["started"] = round(time.time(), 3)
    if mode == "memory":
        # A private copy, not shared memory: one process owns it and frees it.
        buffered = recording.save(format="memory", sharedmem=False, **kwargs)
    else:
        if scratch_dir is None:
            raise ValueError("a disk buffer needs a scratch_dir to live in")
        folder = Path(scratch_dir) / f"signal_buffer_{int(time.time() * 1e6)}"
        folder.parent.mkdir(parents=True, exist_ok=True)
        buffered = recording.save(format="binary", folder=folder, **kwargs)
    info["seconds"] = round(time.perf_counter() - started, 3)
    info["ended"] = round(time.time(), 3)
    buffered = _keep_filtered_flag(recording, buffered)
    logger.info(
        "buffered %.2f GB of signal in %s in %.1f s", info["bytes"] / 1e9, mode, info["seconds"]
    )
    try:
        yield buffered, info
    finally:
        del buffered
        if folder is not None:
            shutil.rmtree(folder, ignore_errors=True)
            logger.info("removed the disk buffer at %s", folder)


__all__ = ["SIGNAL_BUFFER_MODES", "buffered_signal", "signal_bytes"]
