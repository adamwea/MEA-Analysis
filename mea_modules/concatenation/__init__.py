"""Joining recording segments into one timeline.

Public API::

    from mea_modules.concatenation import concatenate_segments, save_concatenated

`concatenate_segments` is lazy; `save_concatenated` is the one place in this
library that reads traces and writes bytes, because sorters need a real binary
file on disk.
"""

from .concat import (
    DEFAULT_CHUNK_DURATION,
    concatenate_segments,
    load_concatenated,
    save_concatenated,
    stitch_frames,
)

__all__ = [
    "concatenate_segments",
    "stitch_frames",
    "save_concatenated",
    "load_concatenated",
    "DEFAULT_CHUNK_DURATION",
]
