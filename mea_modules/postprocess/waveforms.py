"""The single-channel waveform plot: what one unit's spikes actually look like.

This is the first thing anyone asks of a sorted unit — show me the spikes, on
the channel where the unit is loudest, with the template on top. Everything else
(quality metrics, footprints, curation) is downstream of trusting this picture.

Two choices in here are the whole content of the plot:

* **which channel.** The extremum channel, resolved by the same rule as
  :func:`mea_modules.postprocess.analyzer.extremum_channels` — the channel whose
  template carries the deepest trough. Plotting a unit on an arbitrary channel
  of its sparsity mask makes a perfectly good unit look like noise.
* **which spikes.** A seeded random subset of the snippets the analyzer already
  holds, never a fresh read of the recording. The snippet count is small on
  purpose: a hundred faint traces show the spread, a thousand show a solid band.

The mean line is the stored template, i.e. the average over every spike
``random_spikes`` kept, not over the subset that happens to be drawn — so the
same unit gets the same template regardless of how many traces are shown, and
the title reports both counts so the two are never confused.

Figures are built on an Agg canvas without pyplot and released after writing;
this runs in a loop over hundreds of units.
"""

import logging

from ..diagnostics.channel_layout import _new_figure, _save_and_release

logger = logging.getLogger(__name__)

_WAVEFORM_FIGSIZE = (6.5, 4.5)
_WAVEFORM_DPI = 180

# A hundred traces reads as a cloud with visible outliers; a thousand reads as a
# filled block that hides exactly the spread it is supposed to show.
_DEFAULT_N_SPIKES = 100

# Per-sample quantile the y range is taken from. 2% is enough to reject the
# occasional overlapping-spike snippet without touching the honest spread of a
# clean unit.
_DEFAULT_YLIM_QUANTILE = 0.02

_SPIKE_COLOR = "#4a4a4a"
_TEMPLATE_COLOR = "#c0392b"


def unit_waveforms(analyzer, unit_id, channel_id=None, n_spikes=_DEFAULT_N_SPIKES, seed=0):
    """Snippets for one unit on one channel: ``(snippets, template, channel_id)``.

    `snippets` is ``(n_drawn, n_samples)`` and `template` is ``(n_samples,)``,
    both in microvolts when the analyzer was built with ``return_in_uV=True``
    (which :func:`mea_modules.postprocess.analyzer.build_analyzer` always does).
    With `channel_id` None the unit's extremum channel is used.

    Exposed separately from the plot because the numbers are worth having on
    their own — for a metric, a report table, or a different rendering — and
    because it is the only place that knows how to turn a channel id into a
    column of the sparse waveform buffer.
    """
    import numpy as np

    from .analyzer import unit_channel_ids, unit_extremum_channel, unit_template

    extension = analyzer.get_extension("waveforms")
    if extension is None:
        raise ValueError(
            "analyzer has no 'waveforms' extension; build it with build_analyzer() first"
        )

    if channel_id is None:
        channel_id = unit_extremum_channel(analyzer, unit_id)

    channel_ids = unit_channel_ids(analyzer, unit_id)
    try:
        # Position within the unit's own channel list, which is the layout of
        # the last axis of the sparse waveform buffer — not the global index.
        column = channel_ids.index(channel_id)
    except ValueError:
        raise ValueError(
            f"channel {channel_id} is not in unit {unit_id}'s sparsity mask "
            f"({len(channel_ids)} channels); pass one of those or leave channel_id None"
        ) from None

    # (n_spikes_kept, n_samples, n_unit_channels) -> one channel
    snippets = np.asarray(extension.get_waveforms_one_unit(unit_id))[:, :, column]
    n_available = int(snippets.shape[0])

    if n_spikes is not None and 0 < int(n_spikes) < n_available:
        rng = np.random.default_rng(int(seed))
        chosen = np.sort(rng.choice(n_available, size=int(n_spikes), replace=False))
        snippets = snippets[chosen]

    template = unit_template(analyzer, unit_id)[:, column]
    return np.asarray(snippets, dtype=float), np.asarray(template, dtype=float), channel_id


def _trough_aligned_time_ms(template, fs, nbefore):
    """Milliseconds per sample with zero at the template's own extremum.

    The x axis is supposed to say "relative to trough", so it is anchored on
    where the trough actually is rather than on `nbefore`. The two normally
    differ by a sample or two — the sorter aligns on its own detection index,
    which is not exactly the averaged template's minimum — and at 20 kHz a
    couple of samples is 0.1 ms of visible offset. `nbefore` is the fallback for
    a degenerate (flat) template where there is no trough to find.
    """
    import numpy as np

    template = np.asarray(template, dtype=float)
    n_samples = int(template.size)
    if n_samples == 0:
        return np.asarray([], dtype=float), 0
    zero_index = int(np.argmax(np.abs(template))) if np.any(template) else int(nbefore)
    return (np.arange(n_samples, dtype=float) - zero_index) / float(fs) * 1000.0, zero_index


def _robust_ylim(snippets, template, quantile):
    """(low, high) that frame the template and the bulk of the snippets.

    Min/max limits are the obvious choice and the wrong one on real data: a
    single artifact snippet ten times the template's amplitude — an overlapping
    spike, a stimulation transient — sets the whole y range and flattens the
    template into a horizontal line at zero. That is the difference between a
    plot that shows a unit is marginal and a plot that shows nothing at all.

    So the range is taken from a per-sample quantile envelope across snippets,
    unioned with the template's full range so the template is never clipped.
    Outlier snippets are still drawn; they simply run out of the axes, which
    reads as "this unit has contaminated spikes" rather than hiding them.
    Pass `quantile` 0 to get true min/max back.
    """
    import numpy as np

    values = [np.asarray(template, dtype=float)]
    snippets = np.asarray(snippets, dtype=float)
    if snippets.size:
        q = float(quantile or 0.0)
        if q > 0.0:
            values.append(np.nanquantile(snippets, [q, 1.0 - q], axis=0).ravel())
        else:
            values.append(snippets.ravel())

    finite = np.concatenate(values)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return None
    low, high = float(np.min(finite)), float(np.max(finite))
    pad = 0.05 * (high - low) or 1.0
    return low - pad, high + pad


def plot_unit_waveform(
    analyzer,
    unit_id,
    out_path,
    channel_id=None,
    n_spikes=_DEFAULT_N_SPIKES,
    title=None,
    seed=0,
    ylim_quantile=_DEFAULT_YLIM_QUANTILE,
    figsize=_WAVEFORM_FIGSIZE,
    dpi=_WAVEFORM_DPI,
):
    """Draw one unit's spikes and template on its extremum channel; return `out_path`.

    Individual snippets go down faintly, the template goes over them, x is
    milliseconds relative to the trough and y is microvolts. With `channel_id`
    None the extremum channel is chosen for you, which is almost always what you
    want; pass one explicitly to inspect a unit on a neighbouring electrode.

    `n_spikes` caps how many snippets are drawn (None or <= 0 draws all the
    analyzer kept). The title carries the unit id, the channel id and both spike
    counts, so a PNG dropped into a report is self-describing.

    `ylim_quantile` trims the y range to the bulk of the snippets so one artifact
    spike cannot flatten the template — see :func:`_robust_ylim`, and pass 0 for
    literal min/max. The count of snippets that leave the frame is logged.
    """
    import numpy as np
    from matplotlib.collections import LineCollection

    from .analyzer import template_nbefore, unit_random_spike_count

    snippets, template, channel_id = unit_waveforms(
        analyzer, unit_id, channel_id=channel_id, n_spikes=n_spikes, seed=seed
    )
    n_drawn = int(snippets.shape[0])
    n_kept = unit_random_spike_count(analyzer, unit_id)

    fs = float(analyzer.sampling_frequency)
    time_ms, zero_index = _trough_aligned_time_ms(template, fs, template_nbefore(analyzer))

    fig = _new_figure(figsize, dpi)
    ax = fig.subplots()

    if n_drawn:
        # One LineCollection instead of n_drawn plot() calls: identical output,
        # but it keeps a 800-unit loop from spending its time in artist setup.
        segments = [np.column_stack((time_ms, snippet)) for snippet in snippets]
        ax.add_collection(
            LineCollection(segments, colors=_SPIKE_COLOR, linewidths=0.35, alpha=0.15)
        )

    ax.plot(time_ms, template, color=_TEMPLATE_COLOR, lw=1.8, zorder=3)
    ax.axvline(0.0, color="#999999", lw=0.6, ls=":", zorder=1)
    ax.axhline(0.0, color="#999999", lw=0.6, ls=":", zorder=1)

    # add_collection does not update the data limits the way plot does.
    ax.set_xlim(float(time_ms[0]), float(time_ms[-1]))
    ylim = _robust_ylim(snippets, template, ylim_quantile)
    n_clipped = 0
    if ylim is not None:
        ax.set_ylim(*ylim)
        if n_drawn:
            n_clipped = int(
                np.count_nonzero(
                    (np.nanmin(snippets, axis=1) < ylim[0]) | (np.nanmax(snippets, axis=1) > ylim[1])
                )
            )

    ax.set_xlabel("time (ms, relative to trough)")
    ax.set_ylabel("amplitude (uV)")
    ax.set_title(
        title
        or f"unit {unit_id} - channel {channel_id} - {n_drawn}/{n_kept} spikes"
    )
    fig.tight_layout()

    out_path = _save_and_release(fig, out_path)
    logger.info(
        "wrote unit waveform: %s (unit=%s channel=%s drawn=%d kept=%d trough_sample=%d clipped=%d)",
        out_path,
        unit_id,
        channel_id,
        n_drawn,
        n_kept,
        zero_index,
        n_clipped,
    )
    return out_path
