"""Writing a diagnostics file whole, or not at all.

A capsule writes its cache while a plot suite may already be reading the entity
beside it, and a segment-scoped capsule fans out: two processes can reach the
same well's readable JSONs at once. A file written in place is readable
half-written, and half a JSON document is not an error a reader notices -- it is
a number nobody computed.

The content goes to a temporary file in the SAME directory, so the replace is a
rename inside one filesystem, and the reader sees the old file or the new one.
The temp name carries the writer's pid so two writers never share one.
"""

import json
import os
from pathlib import Path


def write_with(path, write_body):
    """Replace `path` with whatever `write_body(tmp)` writes; returns the path.

    For a file a library writes by path -- a compressed array bundle -- rather
    than one held as a string. A body that raises leaves neither the target nor
    the temporary behind.
    """
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        write_body(tmp)
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return path


def write_text(path, text):
    """Replace `path` with `text`, whole."""
    return write_with(path, lambda tmp: tmp.write_text(text, encoding="utf-8"))


def write_json(path, payload, *, indent=2):
    """Replace `path` with `payload` as JSON, whole."""
    return write_text(path, json.dumps(payload, indent=indent))
