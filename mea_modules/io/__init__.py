"""Reading MEA recordings off disk.

Public API::

    from mea_modules.io import load_segment, list_segments, describe_segment

`load_segment`/`list_segments` are segment-oriented aliases for the Maxwell-named
functions; both spellings resolve to the same objects.
"""

from .hdf5_plugin import find_plugin_dir, is_plugin_dir, plugin_lib_name, set_plugin_path
from .load import (
    count_segments,
    describe_segment,
    iter_segments,
    list_maxwell_streams,
    load_maxwell,
)

# Segment-oriented aliases: the vocabulary the pipeline speaks.
load_segment = load_maxwell
list_segments = list_maxwell_streams

__all__ = [
    # loading
    "load_segment",
    "load_maxwell",
    "list_segments",
    "list_maxwell_streams",
    "iter_segments",
    "count_segments",
    "describe_segment",
    # hdf5 plugin
    "set_plugin_path",
    "find_plugin_dir",
    "is_plugin_dir",
    "plugin_lib_name",
]
