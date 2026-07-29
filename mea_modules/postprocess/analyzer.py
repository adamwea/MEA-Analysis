"""The SortingAnalyzer every post-sort plot reads from, and how it stays bounded.

A sorted AxonTracking scan is the worst possible shape for naive waveform code:
hundreds of units, a concatenated binary in the tens of gigabytes, and every unit
potentially touching every routed electrode. Pulling snippets per unit per
channel is what turns a review step into an overnight job, so everything here
goes through one SpikeInterface :class:`SortingAnalyzer` that is bounded on both
axes before a single trace is read:

* **spikes** — ``random_spikes`` keeps at most `max_spikes_per_unit` per unit, so
  a unit that fired 66k times costs the same as one that fired 500 times, and
* **channels** — a sparsity mask keeps each unit to the electrodes it actually
  reaches, so the waveform buffer is ``n_spikes x n_samples x max_channels``
  rather than ``... x n_all_channels``.

The sparsity radius is the one knob worth understanding. SpikeInterface defaults
to 100 um, which on a Maxwell layout is *one electrode clump* — the routed
electrodes come in tight clusters a pitch or two across, with hundreds of
micrometres of dead space between clusters. A 100 um mask therefore produces a
"footprint" that is a single blob, which defeats the point of an AxonTracking
scan: the interesting signal is the small, delayed deflection on a clump a few
hundred micrometres away. :data:`DEFAULT_SPARSITY_RADIUS_UM` is deliberately
wider so a footprint crosses clumps, and it is the first thing to raise if axons
look truncated.

Building the analyzer is the expensive step — it is the only pass over the raw
binary — so it is made resumable: point `build_analyzer` at an `output_dir` that
already holds one and it is loaded instead of recomputed, with only the missing
extensions filled in.

Persistence uses SpikeInterface 0.103.2's ``"binary_folder"`` format by default.
0.103.2 offers ``"memory"``, ``"binary_folder"`` and ``"zarr"``; binary_folder
wins here because the waveforms buffer is several GB of float32 that gets sliced
one unit at a time, and a memory-mapped .npy serves that without a decompression
round trip per read. Pass ``format="zarr"`` when the artifact has to match the
AIND/Varda convention instead.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Waveform window. 1 ms before / 2 ms after is the SpikeInterface default and
# what the working build used; it comfortably contains a Maxwell spike plus the
# repolarisation that makes a template recognisable.
DEFAULT_MS_BEFORE = 1.0
DEFAULT_MS_AFTER = 2.0

# Enough spikes for a stable mean and a visually honest spread, few enough that
# 800 units stay in single-digit GB.
DEFAULT_MAX_SPIKES_PER_UNIT = 500

# See the module docstring: 100 um (SpikeInterface's default) is one Maxwell
# electrode clump. This is wide enough to cross into neighbouring clumps.
DEFAULT_SPARSITY_RADIUS_UM = 300.0

# Fixed so a rebuild picks the same spikes and the plots are reproducible.
DEFAULT_SEED = 0

DEFAULT_FORMAT = "binary_folder"

# Extensions the plotting modules need, in dependency order. `noise_levels` is
# not needed to draw a waveform, but it is what makes an amplitude interpretable
# (and what quality metrics would want next), so it is computed while the
# analyzer is open rather than forcing a second pass later.
_EXTENSION_ORDER = ("random_spikes", "noise_levels", "waveforms", "templates")

# Above this the analyzer stops being a review artifact and starts being a
# storage problem; worth saying out loud before spending the pass.
_WAVEFORM_BUFFER_WARN_BYTES = 32 * 1024**3

_DEFAULT_JOB_KWARGS = {
    "n_jobs": 1,
    "chunk_duration": "1s",
    # A library must not write to stdout/stderr; the caller owns progress
    # reporting, and these run inside pipeline workers with no tty.
    "progress_bar": False,
}


def _resolve_job_kwargs(n_jobs, job_kwargs):
    """Merge caller job kwargs over the defaults, with `n_jobs` winning."""
    resolved = dict(_DEFAULT_JOB_KWARGS)
    resolved.update(dict(job_kwargs or {}))
    resolved["n_jobs"] = int(n_jobs)
    return resolved


def _zarr_path(analyzer_dir):
    """Path SpikeInterface will actually write a zarr analyzer to.

    ``create_sorting_analyzer(format="zarr")`` silently appends ``.zarr`` when
    the folder is not already named that way. Applying the same rule up front is
    what keeps the resume path pointing at the folder that was written, instead
    of rebuilding every time because it looked in the un-suffixed one.
    """
    analyzer_dir = Path(analyzer_dir)
    if analyzer_dir.suffix == ".zarr":
        return analyzer_dir
    return analyzer_dir.parent / (analyzer_dir.name + ".zarr")


def _try_load_analyzer(analyzer_dir):
    """Load an analyzer from `analyzer_dir`, or return None if there isn't one.

    Used only by the resume path, where "no analyzer here" is an ordinary
    outcome and not an error: a half-written folder from a killed job should
    lead to a rebuild, not a traceback.
    """
    analyzer_dir = Path(analyzer_dir).expanduser()
    if not analyzer_dir.exists():
        return None
    try:
        return load_analyzer(analyzer_dir)
    except Exception as exc:  # noqa: BLE001 - any failure means "rebuild it"
        logger.info("no reusable analyzer at %s (%s); rebuilding", analyzer_dir, exc)
        return None


def load_analyzer(analyzer_dir, load_extensions=True):
    """Reload a persisted SortingAnalyzer from `analyzer_dir`.

    The format (binary_folder or zarr) is detected from the folder, so a caller
    does not have to remember which one :func:`build_analyzer` wrote.

    The analyzer carries its own probe and channel ids, so the plots keep
    working even when the recording it was built from is not reachable from this
    machine — only recomputing an extension needs the traces back.

    `load_extensions` False is worth knowing about. SpikeInterface reads
    extension arrays eagerly rather than memory-mapping them (deliberately — see
    its issue #3041), so loading everything pulls the whole waveforms buffer into
    RAM: on an 800-unit scan that is several GB before a single figure is drawn.
    Footprints need only ``templates``, which is two orders of magnitude smaller,
    so a footprint-only pass should load nothing and then ask for what it needs::

        analyzer = load_analyzer(analyzer_dir, load_extensions=False)
        analyzer.load_extension("templates")

    Waveform plots do need the snippets, and therefore the RAM.
    """
    analyzer_dir = Path(analyzer_dir).expanduser()
    if not analyzer_dir.exists():
        # A caller who passed `output_dir=.../analyzer` to build_analyzer with
        # format="zarr" got `.../analyzer.zarr` on disk; find it rather than
        # making them remember the rename.
        zarr_dir = _zarr_path(analyzer_dir)
        if not zarr_dir.exists():
            raise FileNotFoundError(f"no such analyzer folder: {analyzer_dir}")
        analyzer_dir = zarr_dir

    import spikeinterface.core as sc

    analyzer = sc.load_sorting_analyzer(analyzer_dir, load_extensions=bool(load_extensions))
    logger.info(
        "loaded analyzer: %s (%d units, %d channels, sparse=%s, extensions=%s)",
        analyzer_dir,
        analyzer.get_num_units(),
        analyzer.get_num_channels(),
        analyzer.is_sparse(),
        sorted(analyzer.get_loaded_extension_names()),
    )
    return analyzer


def _missing_extensions(analyzer, wanted=_EXTENSION_ORDER):
    """Names from `wanted` the analyzer does not already carry, in order."""
    return [name for name in wanted if not analyzer.has_extension(name)]


def _max_channels_per_unit(analyzer):
    """Widest sparsity mask, i.e. the channel axis of the waveforms buffer."""
    if analyzer.sparsity is None:
        return int(analyzer.get_num_channels())
    return int(analyzer.sparsity.mask.sum(axis=1).max())


def _log_waveform_budget(analyzer, sorting, max_spikes_per_unit, ms_before, ms_after):
    """Log (and warn about) the size of the buffer the waveforms pass will write.

    Worth doing before the pass rather than after: the projection is arithmetic
    on counts already in memory, and finding out that a run needs 40 GB is much
    cheaper here than 20 minutes into a read of an 86 GB binary.
    """
    import numpy as np

    counts = np.asarray(list(sorting.count_num_spikes_per_unit().values()), dtype=np.int64)
    n_spikes = int(np.minimum(counts, int(max_spikes_per_unit)).sum())
    fs = float(analyzer.sampling_frequency)
    n_samples = int(round(ms_before * fs / 1000.0)) + int(round(ms_after * fs / 1000.0))
    max_channels = _max_channels_per_unit(analyzer)
    n_bytes = n_spikes * n_samples * max_channels * 4

    logger.info(
        "waveform budget: %d spikes x %d samples x <=%d channels = %.2f GiB "
        "(%d units, sparse=%s)",
        n_spikes,
        n_samples,
        max_channels,
        n_bytes / 1024**3,
        analyzer.get_num_units(),
        analyzer.is_sparse(),
    )
    if n_bytes > _WAVEFORM_BUFFER_WARN_BYTES:
        logger.warning(
            "waveform buffer projected at %.1f GiB; lower max_spikes_per_unit or "
            "tighten the sparsity radius",
            n_bytes / 1024**3,
        )
    return n_bytes


def build_analyzer(
    recording,
    sorting,
    output_dir=None,
    sparse=True,
    max_spikes_per_unit=DEFAULT_MAX_SPIKES_PER_UNIT,
    n_jobs=1,
    ms_before=DEFAULT_MS_BEFORE,
    ms_after=DEFAULT_MS_AFTER,
    seed=DEFAULT_SEED,
    format=DEFAULT_FORMAT,
    overwrite=False,
    job_kwargs=None,
    **kwargs,
):
    """Build (or reuse) the SortingAnalyzer the waveform and footprint plots read.

    Computes ``random_spikes``, ``noise_levels``, ``waveforms`` and
    ``templates`` — everything :mod:`mea_modules.postprocess.waveforms` and
    :mod:`mea_modules.postprocess.footprints` need, and nothing else.
    ``templates`` is computed after ``waveforms`` on purpose: SpikeInterface then
    reduces the snippets it already has instead of making a second pass over the
    recording. Both ``average`` and ``std`` are kept, since a template without a
    spread is not reviewable.

    With `output_dir` given the analyzer is persisted there and the call is
    resumable — an existing analyzer is loaded and only its missing extensions
    are computed, which matters because this is the one step that reads the raw
    binary end to end. Pass ``overwrite=True`` to force a rebuild (for instance
    after changing `max_spikes_per_unit`, which is *not* detected: a loaded
    analyzer keeps the parameters it was built with). With `output_dir` None the
    analyzer is built in memory and the waveforms buffer is resident RAM — check
    the budget line this function logs before doing that on a full scan.

    `format` selects the persistence backend. SpikeInterface 0.103.2 supports
    ``"memory"``, ``"binary_folder"`` and ``"zarr"``; the default here is
    ``"binary_folder"`` (see the module docstring for why).

    `job_kwargs` overrides the parallelism defaults (``chunk_duration``,
    ``progress_bar``, ``mp_context``, ...); `n_jobs` always wins over whatever it
    contains. Remaining `kwargs` are forwarded to ``create_sorting_analyzer`` and
    thus to ``estimate_sparsity`` — ``method``, ``radius_um``, ``num_channels``,
    ``peak_sign``. When `sparse` is True and none of those are given, the mask is
    a radius one at :data:`DEFAULT_SPARSITY_RADIUS_UM`.

    Returns the SortingAnalyzer.
    """
    import spikeinterface.core as sc

    resolved_jobs = _resolve_job_kwargs(n_jobs, job_kwargs)

    analyzer = None
    if output_dir is not None:
        output_dir = Path(output_dir).expanduser()
        if format == "zarr":
            output_dir = _zarr_path(output_dir)
        if overwrite:
            logger.info("overwrite requested; ignoring any analyzer at %s", output_dir)
        else:
            analyzer = _try_load_analyzer(output_dir)

    if analyzer is None:
        sparsity_kwargs = dict(kwargs)
        if sparse and sparsity_kwargs.get("sparsity") is None:
            # Only fill in a default when the caller has not chosen a method;
            # passing radius_um alongside method="best_channels" would be an error.
            sparsity_kwargs.setdefault("method", "radius")
            if sparsity_kwargs["method"] == "radius":
                sparsity_kwargs.setdefault("radius_um", DEFAULT_SPARSITY_RADIUS_UM)
            # estimate_sparsity makes its own pass over the recording, so it
            # needs the same parallelism as the extensions do.
            for key, value in resolved_jobs.items():
                sparsity_kwargs.setdefault(key, value)

        logger.info(
            "creating analyzer: format=%s folder=%s sparse=%s sparsity=%s",
            format,
            output_dir,
            sparse,
            {k: v for k, v in sparsity_kwargs.items() if k not in _DEFAULT_JOB_KWARGS},
        )
        analyzer = sc.create_sorting_analyzer(
            sorting,
            recording,
            format="memory" if output_dir is None else format,
            folder=None if output_dir is None else output_dir,
            sparse=bool(sparse),
            # uV throughout: an amplitude axis in ADC units is not reviewable,
            # and mixing scaled and unscaled extensions silently corrupts SNR.
            return_in_uV=True,
            overwrite=bool(overwrite),
            **sparsity_kwargs,
        )

    missing = _missing_extensions(analyzer)
    if not missing:
        logger.info("analyzer already carries every needed extension; nothing to compute")
        return analyzer
    logger.info("computing extensions %s", missing)

    _log_waveform_budget(analyzer, sorting, max_spikes_per_unit, ms_before, ms_after)

    params = {
        "random_spikes": {
            "method": "uniform",
            "max_spikes_per_unit": int(max_spikes_per_unit),
            "seed": int(seed),
        },
        "noise_levels": {},
        "waveforms": {"ms_before": float(ms_before), "ms_after": float(ms_after)},
        "templates": {
            "ms_before": float(ms_before),
            "ms_after": float(ms_after),
            "operators": ["average", "std"],
        },
    }
    # One call so SpikeInterface can fuse the extensions that share a node
    # pipeline into a single traversal of the recording.
    analyzer.compute({name: params[name] for name in missing}, **resolved_jobs)

    logger.info(
        "analyzer ready: %d units, %d channels, extensions=%s",
        analyzer.get_num_units(),
        analyzer.get_num_channels(),
        sorted(analyzer.get_loaded_extension_names()),
    )
    return analyzer


def extremum_channels(analyzer, peak_sign="neg", mode="extremum"):
    """Map every unit to the channel where its template reaches its extremum.

    Returns ``{unit_id: channel_id}`` in the analyzer's own id types, ready to
    hand straight back to a plotting call.

    This delegates to :func:`spikeinterface.core.get_template_extremum_channel`
    rather than reimplementing the selection. The reference implementation in
    the lab's ``mea_waveform.py`` is ``argmin(min(template, axis=0))`` — the
    channel carrying the deepest negative trough — and the SpikeInterface
    function with ``peak_sign="neg", mode="extremum"`` computes exactly that
    (per-channel minimum over time, then the largest magnitude), so the library
    call is behaviour-preserving and gets the sparse/dense template handling for
    free. The defaults here are those two values for that reason; ``peak_sign``
    is exposed because a template inverted by a referencing scheme needs
    ``"both"``.

    Channels outside a unit's sparsity mask are exactly zero in the stored
    dense template, so they can never win the extremum — the mask does not have
    to be undone first.
    """
    if not analyzer.has_extension("templates"):
        raise ValueError(
            "analyzer has no 'templates' extension; build it with build_analyzer() first"
        )

    from spikeinterface.core import get_template_extremum_channel

    mapping = get_template_extremum_channel(
        analyzer, peak_sign=str(peak_sign), mode=str(mode), outputs="id"
    )
    logger.info(
        "resolved extremum channels for %d units (peak_sign=%s mode=%s)",
        len(mapping),
        peak_sign,
        mode,
    )
    return mapping


def _extremum_index(template, peak_sign="neg"):
    """Column of a ``(n_samples, n_channels)`` template carrying the extremum.

    The single implementation of the selection rule. :func:`extremum_channels`
    goes through SpikeInterface for the all-units case, but both plotting
    modules need the answer for one unit from a template they already hold, and
    a second hand-rolled ``argmin`` in each of them is exactly how the plot and
    the reported channel end up disagreeing.
    """
    import numpy as np

    if peak_sign == "neg":
        amplitudes = np.min(template, axis=0)
    elif peak_sign == "pos":
        amplitudes = np.max(template, axis=0)
    else:
        amplitudes = np.max(np.abs(template), axis=0)
    return int(np.argmax(np.abs(amplitudes)))


def unit_extremum_channel(analyzer, unit_id, peak_sign="neg"):
    """Extremum channel for one unit, without resolving the other 799.

    Same rule and same answer as :func:`extremum_channels`, evaluated on this
    unit's template alone. The SpikeInterface call resolves every unit on every
    invocation, which is free once and wasteful inside a per-unit plotting loop.
    """
    return unit_channel_ids(analyzer, unit_id)[
        _extremum_index(unit_template(analyzer, unit_id), peak_sign=peak_sign)
    ]


def unit_random_spike_count(analyzer, unit_id):
    """How many spikes ``random_spikes`` kept for this unit.

    Read off the (small) selected-spike vector rather than by measuring the
    waveform buffer, so asking the question does not pull the unit's snippets
    off disk a second time.
    """
    import numpy as np

    extension = analyzer.get_extension("random_spikes")
    if extension is None:
        raise ValueError("analyzer has no 'random_spikes' extension")
    unit_index = list(analyzer.unit_ids).index(unit_id)
    return int(np.count_nonzero(extension.get_random_spikes()["unit_index"] == unit_index))


def unit_channel_ids(analyzer, unit_id):
    """Channel ids a unit was actually extracted on, in waveform-buffer order.

    For a sparse analyzer this is the unit's sparsity mask; for a dense one it is
    every channel. The order matters and is not decorative: it is the order of
    the last axis of ``get_waveforms_one_unit``, so this is what turns a channel
    id into a column index into that array.
    """
    if analyzer.sparsity is None:
        return list(analyzer.channel_ids)
    return list(analyzer.sparsity.unit_id_to_channel_ids[unit_id])


def unit_template(analyzer, unit_id, operator="average"):
    """``(n_samples, n_unit_channels)`` template for one unit, in uV.

    Sliced down to the unit's own channels (see :func:`unit_channel_ids`) rather
    than returned dense, because the dense form is mostly structural zeros and
    plotting those draws hundreds of flat lines across the array.
    """
    import numpy as np

    if not analyzer.has_extension("templates"):
        raise ValueError(
            "analyzer has no 'templates' extension; build it with build_analyzer() first"
        )

    extension = analyzer.get_extension("templates")
    dense = np.asarray(extension.get_data(operator=operator))
    unit_index = list(analyzer.unit_ids).index(unit_id)
    template = dense[unit_index]

    if analyzer.sparsity is not None:
        channel_indices = analyzer.sparsity.unit_id_to_channel_indices[unit_id]
        template = template[:, channel_indices]
    return np.asarray(template, dtype=float)


def template_nbefore(analyzer):
    """Samples before the alignment point in the stored templates/waveforms.

    Read off whichever extension is present rather than recomputed from the ms
    parameters, so it stays correct on an analyzer someone else built.
    """
    for name in ("waveforms", "templates"):
        extension = analyzer.get_extension(name)
        if extension is not None:
            return int(extension.nbefore)
    raise ValueError("analyzer carries neither 'waveforms' nor 'templates'")
