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


def plot_unit_waveform(
    analyzer,
    unit_id,
    out_path,
    channel_id=None,
    n_spikes=_DEFAULT_N_SPIKES,
    title=None,
    seed=0,
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
    """
    import numpy as np
    from matplotlib.collections import LineCollection

    from .analyzer import template_nbefore

    snippets, template, channel_id = unit_waveforms(
        analyzer, unit_id, channel_id=channel_id, n_spikes=n_spikes, seed=seed
    )
    n_drawn = int(snippets.shape[0])
    n_kept = int(np.asarray(analyzer.get_extension("waveforms").get_waveforms_one_unit(unit_id)).shape[0])

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
    finite = np.concatenate([snippets.ravel(), template]) if n_drawn else template
    finite = finite[np.isfinite(finite)]
    if finite.size:
        span = float(np.nanmax(finite) - np.nanmin(finite)) or 1.0
        ax.set_ylim(float(np.nanmin(finite)) - 0.05 * span, float(np.nanmax(finite)) + 0.05 * span)

    ax.set_xlabel("time (ms, relative to trough)")
    ax.set_ylabel("amplitude (uV)")
    ax.set_title(
        title
        or f"unit {unit_id} - channel {channel_id} - {n_drawn}/{n_kept} spikes"
    )
    fig.tight_layout()

    out_path = _save_and_release(fig, out_path)
    logger.info(
        "wrote unit waveform: %s (unit=%s channel=%s drawn=%d kept=%d trough_sample=%d)",
        out_path,
        unit_id,
        channel_id,
        n_drawn,
        n_kept,
        zero_index,
    )
    return out_path
