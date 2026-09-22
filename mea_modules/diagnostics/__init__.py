"""Review artifacts: the PNGs a human looks at before trusting a run.

Public API::

    from mea_modules.diagnostics import (
        plot_channel_layout,   # where the routed electrodes are
        plot_traces,           # what the loudest channels look like
        plot_raster_threshold, # whether anything is firing, and when
        plot_unit_locations,   # where the recomputed units sit on the array
        plot_noise_activity_map,  # both QC metrics on the probe geometry
        plot_spectra_panels,   # what the filter chain did, raw vs preprocessed
    )

Not everything here renders. Three of the per-recording diagnostics produce
numbers a report writes rather than a picture — :func:`flag_channels` (which
channel ids not to trust, and why), :func:`clipping_census` (which channels sat
on a rail) and :func:`artifact_census` (when the whole array moved at once) —
and the emitters that do produce both keep the arithmetic in its own function,
so a number can become a check without dragging a figure along.

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

from .activity_map import (
    WHOLE_CHIP_ACTIVITY_FILENAME,
    plot_whole_chip_activity,
    template_projected_activity,
)
from .artifacts import artifact_census
from .collect import (
    SEGMENT_DIAGNOSTICS,
    SEGMENT_DIAGNOSTIC_NAMES,
    DiagnosticSpec,
    collect_segment_diagnostics,
)
from .cache import (
    CachedProbe,
    CachedTraces,
    CacheMissing,
    DiagnosticCache,
    cache_ready,
    read_cache,
    write_cache,
)
from .channel_flags import flag_channels, flagged_channel_groups, flagged_channel_note
from .channel_layout import (
    cluster_center_channels,
    detect_electrode_clusters,
    estimate_electrode_pitch,
    plot_channel_layout,
)
from .clipping import clipping_census
from .readable import CONCAT_REPORTS, SEGMENT_REPORTS, write_concat_reports, write_segment_reports
from .figure_style import emptiest_corner, legend_corner, tighten
from .motion import estimate_motion_over_recording, plot_motion_estimate
from .metric_maps import (
    plot_firing_rate_map,
    plot_metric_maps,
    plot_noise_activity_map,
    plot_noise_map,
    plot_rms_mad_ratio_map,
    plot_rms_map,
    plot_rms_vs_mad,
    robust_color_limits,
)
from .buffer import SIGNAL_BUFFER_MODES, buffered_signal, signal_bytes
from .collect_concat import (
    CONCAT_DIAGNOSTIC_NAMES,
    CONCAT_DIAGNOSTICS,
    collect_concat_diagnostics,
    segment_table,
)
from .electrode_coverage import plot_electrode_coverage, plot_segment_electrode_counts
from .raster import plot_raster_threshold
from .spectra import (
    plot_spectra_panels,
    welch_spectra,
)
from .timebase import (
    gap_spans,
    real_time_axis,
    rescale_time_gaps,
    resolve_time_gaps,
    sample_times,
)
from .traces import (
    channel_activity_rms,
    plot_traces,
    plot_traces_with_layout,
    select_representative_channels,
)
from .unit_locations import (
    UNIT_LOCATIONS_PLOT_FILENAME,
    plot_unit_locations,
)

__all__ = [
    # readable JSON beside a segment's or a concatenated well's cache
    "CONCAT_REPORTS",
    "SEGMENT_REPORTS",
    "write_concat_reports",
    "write_segment_reports",
    # plot emitters
    "plot_channel_layout",
    "plot_traces",
    "plot_traces_with_layout",
    "plot_raster_threshold",
    "plot_unit_locations",
    "UNIT_LOCATIONS_PLOT_FILENAME",
    # whole-chip template-projected activity field
    "template_projected_activity",
    "plot_whole_chip_activity",
    "WHOLE_CHIP_ACTIVITY_FILENAME",
    # per-channel metrics painted on the probe geometry
    "plot_metric_maps",
    "plot_noise_activity_map",
    "plot_noise_map",
    "plot_firing_rate_map",
    "plot_rms_map",
    "plot_rms_mad_ratio_map",
    "plot_rms_vs_mad",
    "robust_color_limits",
    # how much of the array one well's recordings share
    "plot_electrode_coverage",
    "plot_segment_electrode_counts",
    # power spectra: the numbers, then the raw-vs-preprocessed panel
    "welch_spectra",
    "plot_spectra_panels",
    # drift over a concatenated well: the estimate, then the trace
    "estimate_motion_over_recording",
    "plot_motion_estimate",
    # numbers without a figure: ids, rails, array-wide events
    "flag_channels",
    "flagged_channel_groups",
    "flagged_channel_note",
    "clipping_census",
    "artifact_census",
    # the compute-once cache a capsule writes and a plot tool reads
    "collect_segment_diagnostics",
    "SEGMENT_DIAGNOSTICS",
    "SEGMENT_DIAGNOSTIC_NAMES",
    "collect_concat_diagnostics",
    "CONCAT_DIAGNOSTICS",
    "CONCAT_DIAGNOSTIC_NAMES",
    "segment_table",
    "SIGNAL_BUFFER_MODES",
    "buffered_signal",
    "signal_bytes",
    "DiagnosticSpec",
    "CachedProbe",
    "CachedTraces",
    "CacheMissing",
    "DiagnosticCache",
    "cache_ready",
    "read_cache",
    "write_cache",
    # channel selection
    "select_representative_channels",
    "channel_activity_rms",
    "detect_electrode_clusters",
    "estimate_electrode_pitch",
    # real elapsed time, for plots that must not pretend the gaps are not there
    "real_time_axis",
    "sample_times",
    "gap_spans",
    "rescale_time_gaps",
    "resolve_time_gaps",
]
