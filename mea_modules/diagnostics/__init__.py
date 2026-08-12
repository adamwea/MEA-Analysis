"""Review artifacts: the PNGs a human looks at before trusting a run.

Public API::

    from mea_modules.diagnostics import (
        plot_channel_layout,   # where the routed electrodes are
        plot_traces,           # what the loudest channels look like
        plot_raster_threshold, # whether anything is firing, and when
        plot_unit_locations,   # where the recomputed units sit on the array
    )

Every emitter takes an opened recording (or, for ``plot_unit_locations``, the
arrays the caller already holds) plus an explicit output path, writes one
figure, and returns that path — nothing here invents a directory or decides a
filename. Figures are built on an Agg canvas without pyplot, so they are safe on
a headless node and leave no global state behind when called in a loop over
segments.

Reading traces is unavoidable here, but each emitter is bounded by a time window
and a channel cap by default; pass ``duration_s=None`` only when you know the
recording is short.

The selection helpers are exported too, since knowing *which* channels a plot
was made from is usually worth recording alongside it.

``plot_traces`` and ``plot_raster_threshold`` draw against the contiguous sample
timeline by default. Pass them ``time_gaps`` — the gap structure
:mod:`mea_modules.io.gaps` reads off the file — and the x axis becomes real
elapsed time with the missing stretches marked; :mod:`.timebase` is the
arithmetic behind that, and is exported for callers that need the axis without
the figure.
"""

from .channel_layout import (
    detect_electrode_clusters,
    estimate_electrode_pitch,
    plot_channel_layout,
)
from .raster import (
    detect_threshold_crossings,
    estimate_channel_thresholds,
    plot_raster_threshold,
)
from .timebase import (
    gap_spans,
    real_time_axis,
    resolve_time_gaps,
    sample_times,
)
from .traces import (
    channel_activity_rms,
    plot_traces,
    select_representative_channels,
)
from .unit_locations import (
    UNIT_LOCATIONS_PLOT_FILENAME,
    plot_unit_locations,
)

__all__ = [
    # plot emitters
    "plot_channel_layout",
    "plot_traces",
    "plot_raster_threshold",
    "plot_unit_locations",
    "UNIT_LOCATIONS_PLOT_FILENAME",
    # channel selection
    "select_representative_channels",
    "channel_activity_rms",
    "detect_electrode_clusters",
    "estimate_electrode_pitch",
    # threshold detection
    "estimate_channel_thresholds",
    "detect_threshold_crossings",
    # real elapsed time, for plots that must not pretend the gaps are not there
    "real_time_axis",
    "sample_times",
    "gap_spans",
    "resolve_time_gaps",
]
