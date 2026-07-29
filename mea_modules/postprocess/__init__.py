"""Post-sort review: what a sorted unit looks like, and where on the chip it is.

Public API::

    from mea_modules.postprocess import (
        build_analyzer,        # the bounded SortingAnalyzer everything reads from
        load_analyzer,         # reopen one written earlier
        extremum_channels,     # {unit_id: loudest channel}
        plot_unit_waveform,    # spikes + template on the extremum channel
        plot_unit_footprint,   # the template across the electrode layout
        plot_footprint_grid,   # several footprints in one figure
    )

`build_analyzer` is the only expensive call here and the only one that reads
traces: it makes one bounded pass over the recording and leaves a
SortingAnalyzer that both plot emitters then read from, in memory or on disk.
Everything after it is arithmetic on arrays that already exist, so a loop over
hundreds of units costs a pass over the sorting, not a pass over the binary.

The two plots answer different questions and are both needed. The waveform says
*is this unit real* — a hundred snippets and the template on the one channel
where the unit is loudest. The footprint says *where does it go* — the same
template drawn at every electrode's true micrometre position, which is the
entire reason an AxonTracking scan exists and is invisible in any channel-ordered
view.

Every emitter takes an analyzer plus an explicit output path, writes one figure,
and returns that path. Figures are built on an Agg canvas without pyplot and
released after writing, so these are safe on a headless node and leave nothing
behind when called in a loop.
"""

from .analyzer import (
    DEFAULT_MAX_SPIKES_PER_UNIT,
    DEFAULT_MS_AFTER,
    DEFAULT_MS_BEFORE,
    DEFAULT_SPARSITY_RADIUS_UM,
    build_analyzer,
    extremum_channels,
    load_analyzer,
    template_nbefore,
    unit_channel_ids,
    unit_extremum_channel,
    unit_random_spike_count,
    unit_template,
)
from .footprints import plot_footprint_grid, plot_unit_footprint
from .unit_locations import plot_unit_locations, unit_location_array
from .unit_raster import plot_unit_raster, unit_firing_rates
from .waveforms import plot_unit_waveform, unit_waveforms

__all__ = [
    # the analyzer, and the things that read structure off it
    "build_analyzer",
    "load_analyzer",
    "extremum_channels",
    "unit_extremum_channel",
    "unit_channel_ids",
    "unit_template",
    "unit_random_spike_count",
    "template_nbefore",
    # plot emitters
    "plot_unit_waveform",
    "plot_unit_locations",
    "unit_location_array",
    "plot_unit_raster",
    "unit_firing_rates",
    "plot_unit_footprint",
    "plot_footprint_grid",
    # numbers behind the waveform plot, for callers that want them without a PNG
    "unit_waveforms",
    # defaults, so a caller can report what an analyzer was built with
    "DEFAULT_MAX_SPIKES_PER_UNIT",
    "DEFAULT_SPARSITY_RADIUS_UM",
    "DEFAULT_MS_BEFORE",
    "DEFAULT_MS_AFTER",
]
