"""Reading MEA recordings off disk.

Public API::

    from mea_modules.io import load_segment, list_segments, describe_segment

`load_segment`/`list_segments` are segment-oriented aliases for the Maxwell-named
functions; both spellings resolve to the same objects.

:mod:`mea_modules.io.gaps` covers the other half of "what is in this file": how
much time it is missing, both inside a segment and between segments. Its output
feeds :mod:`mea_modules.diagnostics.timebase`.
"""

from .gaps import frame_gaps, segment_time_bounds, well_gap_summary
from .hdf5_plugin import find_plugin_dir, is_plugin_dir, plugin_lib_name, set_plugin_path
from .load import (
    count_segments,
    describe_segment,
    iter_segments,
    list_maxwell_streams,
    load_maxwell,
    segment_index,
)
from .metadata import extract_metadata, find_common_electrodes, save_metadata

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
    # segment structure across all three axes (well x group x recording)
    "segment_index",
    # missing time: dropped frames within a segment, wall clock between them
    "frame_gaps",
    "segment_time_bounds",
    "well_gap_summary",
    # metadata
    "extract_metadata",
    "save_metadata",
    "find_common_electrodes",
    # hdf5 plugin
    "set_plugin_path",
    "find_plugin_dir",
    "is_plugin_dir",
    "plugin_lib_name",
]
