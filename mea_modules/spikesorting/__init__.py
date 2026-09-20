"""Spike sorting: run a sorter, then account for what it produced.

Public API::

    from mea_modules.spikesorting import run_sorter, snapshot_output

`run_sorter` is the local (non-container) SpikeInterface path; `snapshot_output`
reads a finished folder back into a JSON-serializable provenance record. Sorter
packages are imported lazily, so importing this module never requires kilosort —
call `require_sorter` for a preflight check with an install hint.
"""

from .sorter import (
    build_kilosort_params,
    container_problem,
    gpu_passthrough_ready,
    hash_directory,
    install_hint,
    read_sorting,
    require_sorter,
    run_sorter,
    snapshot_output,
    sorter_installed_locally,
    sorter_is_available,
)

__all__ = [
    # sorting
    "run_sorter",
    "read_sorting",
    "build_kilosort_params",
    # availability — one definition of every probe, so two callers asking the
    # same question about the same host cannot get two answers
    "sorter_is_available",
    "sorter_installed_locally",
    "require_sorter",
    "container_problem",
    "gpu_passthrough_ready",
    "install_hint",
    # provenance
    "snapshot_output",
    "hash_directory",
]
