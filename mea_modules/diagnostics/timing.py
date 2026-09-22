"""What one entity's diagnostics cost: a readable JSON beside its cache.

The collector already times every diagnostic it runs (:func:`.collect._tolerant`
returns ``seconds``, ``started`` and ``ended`` for each) and the buffer records
its own materialisation. Those numbers live inside the cache record, where a
person reading a finished run does not look. This writes them out as
``timing.json`` beside the cache, together with the steps the capsule timed
around the collector -- resolving the buffer, collecting, writing the cache,
writing the readable JSON -- so a profile can lay one entity on a single
timeline.

It is NOT part of the cache: nothing here enters the cache record, its version
or any fingerprint, and a reader that never opens this file loses nothing. The
capsule that writes it clears the previous one when it opens an entity to
compute, so a launch whose diagnostics were switched off leaves no timing from
the launch before it.
"""

import datetime
import json

from .cache import CACHE_DIRNAME, _jsonable, cache_dir

TIMING_NAME = "timing.json"


def write_timing(capsule_out_dir, *, capsule, entity, diagnostics=None, buffer=None, steps=None):
    """Write ``diagnostics/timing.json`` for one entity and return its path.

    `entity` names it the way the capsule does (``{"well": ..., "rec": ...}``).
    `diagnostics` is the collector's own ``metrics["timings"]`` and `buffer` its
    ``metrics["buffer"]``, both taken verbatim; `steps` is what the capsule timed
    itself. Every entry carries ``seconds``, ``started`` and ``ended``.
    """
    folder = cache_dir(capsule_out_dir, create=True)
    document = {
        "capsule": capsule,
        "entity": _jsonable(entity),
        "written_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "diagnostics": _jsonable(diagnostics or {}),
        "steps": _jsonable(steps or {}),
    }
    if buffer is not None:
        document["buffer"] = _jsonable(buffer)
    path = folder / TIMING_NAME
    path.write_text(json.dumps(document, indent=2, sort_keys=False), encoding="utf-8")
    return path


def clear_timing(capsule_out_dir):
    """Remove an entity's ``timing.json``; True when one was there.

    Never creates the diagnostics folder, and removes it again when it was the
    only thing in it: an empty ``diagnostics/`` is how a run says a cache lives
    here rather than in the run it borrows from.
    """
    folder = cache_dir(capsule_out_dir, create=False)
    path = folder / TIMING_NAME
    existed = path.is_file()
    path.unlink(missing_ok=True)
    if folder.is_dir() and not any(folder.iterdir()):
        folder.rmdir()
    return existed
