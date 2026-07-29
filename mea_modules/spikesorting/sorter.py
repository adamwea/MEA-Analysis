"""Run a spike sorter over a recording, and fingerprint what it produced.

Two jobs, deliberately kept apart:

* :func:`run_sorter` drives SpikeInterface's sorter machinery, either in this
  interpreter or inside a container (``docker_image=`` / ``singularity_image=``).
  It takes an already-built recording object (lazy or materialized; that choice
  belongs to whoever assembled it) and a folder to write into.
* :func:`snapshot_output` reads a finished sorter folder back and returns a
  plain dict: how many units came out, the parameters the sorter actually ran
  with, and a sha256 per output file. Nothing in it needs the sorter installed,
  so a sort produced on a GPU node can be verified anywhere.

Sorter packages are imported lazily, inside the call. ``kilosort`` is a heavy,
CUDA-shaped dependency that most consumers of this library never touch, so
``from mea_modules.spikesorting import run_sorter`` must not require it —
:func:`require_sorter` turns a missing sorter into an actionable error at call
time instead of an ImportError at import time.

Containers are the primary execution path on machines that have Docker but no
local kilosort, which is the normal state of a workstation. The consequence for
availability checks: "is the sorter importable here" is the wrong question once
a container is requested, because the sorter lives *inside the image*. So
:func:`sorter_is_available` and :func:`require_sorter` take the same
``docker_image`` / ``singularity_image`` arguments as :func:`run_sorter` and
switch to asking whether the container runtime works instead.

Container execution serializes the recording and bind-mounts the files it reads
into the image. A materialized binary folder (what
``mea_modules.concatenation.save_concatenated`` writes) is the dependable input;
a lazy Maxwell recording would additionally need its HDF5 compression plugin
present *inside* the image, which the stock sorter images do not carry.

Folder layout, as SpikeInterface writes it::

    <output_dir>/
        spikeinterface_params.json    sorter_name + the params actually used
        spikeinterface_log.json       version, runtime, error state
        spikeinterface_recording.json the recording that was sorted
        sorter_output/                the sorter's own native output

Callers frequently name ``<output_dir>`` "sorter_output" as well, which yields
``sorter_output/sorter_output``; the readers here accept either level.
"""

import fnmatch
import hashlib
import json
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_PARAMS_FILENAME = "spikeinterface_params.json"
_LOG_FILENAME = "spikeinterface_log.json"
_RAW_OUTPUT_DIRNAME = "sorter_output"

# Container execution is requested through the explicit `docker_image` /
# `singularity_image` parameters. These other spellings are not SpikeInterface
# run_sorter arguments, so left alone they would be forwarded to the sorter as
# if they were sorting parameters and fail somewhere far less legible. Catch
# them here and name the parameter the caller meant.
_CONTAINER_KWARG_ALIASES = {
    "container_image": "docker_image=<image> (or singularity_image=<image>)",
    "use_docker": "docker_image=True",
    "use_singularity": "singularity_image=True",
}

# Where SpikeInterface's container-runtime probes live. 0.103.2 re-exports them
# from runsorter; the others are fallbacks so a version bump degrades to a bad
# availability answer rather than an ImportError.
_CONTAINER_PROBE_MODULES = (
    "spikeinterface.sorters.runsorter",
    "spikeinterface.sorters.utils",
    "spikeinterface.sorters.container_tools",
)

_HASH_CHUNK_BYTES = 1024 * 1024


def _normalize_container_image(value):
    """Turn one container request into the value SpikeInterface expects.

    SpikeInterface reads these arguments as truthy-or-not: falsy means "run in
    this process", True means "the image SpikeInterface maps this sorter to",
    and a string is an explicit image reference. Two spellings need fixing up
    before they reach it. ``None`` is this library's "not requested" default and
    must become False. An empty or whitespace string is *falsy* over there, so a
    caller writing ``docker_image=""`` to mean "you pick the image" — the shape
    an unset CLI/YAML field naturally takes — would silently get a local run;
    read it as True instead.
    """
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    text = str(value).strip()
    return text if text else True


def _container_request(docker_image=None, singularity_image=None):
    """Resolve a docker/singularity pair to ``(mode, image)``.

    `mode` is "docker", "singularity", or None for local execution; `image` is
    what to hand SpikeInterface (True for its default image, or an explicit
    reference). SpikeInterface asserts on being given both, which surfaces as a
    bare AssertionError deep in its dispatch — say what is wrong here instead.
    """
    docker = _normalize_container_image(docker_image)
    singularity = _normalize_container_image(singularity_image)

    if docker and singularity:
        raise ValueError(
            "docker_image and singularity_image both request container execution; pass one. "
            f"(docker_image={docker_image!r}, singularity_image={singularity_image!r})"
        )
    if docker:
        return "docker", docker
    if singularity:
        return "singularity", singularity
    return None, False


def _container_probes():
    """SpikeInterface's runtime probes, as a name -> callable dict."""
    from importlib import import_module

    wanted = ("has_docker", "has_docker_python", "has_singularity", "has_spython")
    for module_name in _CONTAINER_PROBE_MODULES:
        try:
            module = import_module(module_name)
        except Exception:
            continue
        if all(callable(getattr(module, name, None)) for name in wanted):
            return {name: getattr(module, name) for name in wanted}
    raise RuntimeError(
        "this SpikeInterface build does not expose its container probes "
        f"(looked in {', '.join(_CONTAINER_PROBE_MODULES)}); container execution cannot be checked"
    )


def _container_backend_problem(mode):
    """Why `mode` cannot run a sort here, as a sentence, or None when it can.

    Mirrors the two conditions SpikeInterface checks before dispatching — the
    runtime binary and the Python client that drives it. Having them here means
    a preflight fails for the same reason and with the same fix as the real run,
    hours before the recording has been concatenated.
    """
    probes = _container_probes()

    if mode == "docker":
        if not probes["has_docker"]():
            return "`docker --version` failed — Docker is not installed or not on PATH"
        if not probes["has_docker_python"]():
            return "the Python Docker client is missing — `pip install docker`"
        return None

    if mode == "singularity":
        if not probes["has_singularity"]():
            return "neither `singularity` nor `apptainer` is on PATH"
        if not probes["has_spython"]():
            return "the Python Singularity client is missing — `pip install spython`"
        return None

    raise ValueError(f"unknown container mode {mode!r}")


def _default_container_image(sorter_name):
    """The image SpikeInterface would pick for `sorter_name`, or None.

    Only ever used to make log lines and errors concrete — never to choose an
    image, which stays SpikeInterface's job.
    """
    try:
        from spikeinterface.sorters.runsorter import SORTER_DOCKER_MAP

        return SORTER_DOCKER_MAP.get(str(sorter_name))
    except Exception:
        logger.debug("could not read SORTER_DOCKER_MAP", exc_info=True)
        return None


def _sorter_installed_locally(sorter_name):
    """True when the sorter package imports in *this* interpreter."""
    import spikeinterface.sorters as ss

    klass = ss.sorter_dict.get(str(sorter_name))
    if klass is None:
        return False
    try:
        return bool(klass.is_installed())
    except Exception:  # a broken install is an unavailable install
        logger.debug("is_installed() failed for sorter %r", sorter_name, exc_info=True)
        return False


def sorter_is_available(sorter_name, docker_image=None, singularity_image=None):
    """True when `sorter_name` is known to SpikeInterface and runnable here.

    "Runnable here" is a different question depending on where the sort will
    happen, and answering the wrong one is how a perfectly good machine gets
    reported as unusable. Locally it means the sorter package imports. In a
    container it means the container runtime works — the sorter is baked into
    the image, so kilosort being absent from this interpreter says nothing at
    all about whether the sort can run.

    Pass the same `docker_image` / `singularity_image` you intend to pass to
    :func:`run_sorter`; the default (both None) checks the local path.

    Cheap enough to call in a preflight check — the local branch tests for the
    sorter with importlib.find_spec rather than importing it, and the container
    branch runs `docker --version`.
    """
    import spikeinterface.sorters as ss

    sorter_name = str(sorter_name)
    if ss.sorter_dict.get(sorter_name) is None:
        return False

    mode, _image = _container_request(docker_image, singularity_image)
    if mode is None:
        return _sorter_installed_locally(sorter_name)
    try:
        return _container_backend_problem(mode) is None
    except Exception:
        logger.debug("could not probe %s availability", mode, exc_info=True)
        return False


def require_sorter(sorter_name, docker_image=None, singularity_image=None):
    """Return the SpikeInterface sorter class, or raise saying how to get it.

    Kept separate from :func:`run_sorter` so a caller can fail fast — before
    spending an hour concatenating and preprocessing a scan only to discover at
    the end that the GPU node has no kilosort.

    The two execution paths fail for unrelated reasons, so each error names its
    own fix and the state of the *other* path: a machine with no local kilosort
    and working Docker is fine, and should not read like a broken one.
    """
    import spikeinterface.sorters as ss

    sorter_name = str(sorter_name)
    klass = ss.sorter_dict.get(sorter_name)
    if klass is None:
        raise ValueError(
            f"unknown sorter {sorter_name!r}; SpikeInterface knows: {sorted(ss.sorter_dict)}"
        )

    # installation_mesg is a multi-line block with doctest prompts in it; folded
    # to one line so it composes with the rest of the sentence.
    install_hint = " ".join(str(getattr(klass, "installation_mesg", "") or "").split())
    install_hint = install_hint or f"Install the {sorter_name} package and its dependencies."
    # The image itself is SpikeInterface's problem; only the mode matters here.
    mode, _image = _container_request(docker_image, singularity_image)

    if mode is not None:
        problem = _container_backend_problem(mode)
        if problem is None:
            return klass
        if _sorter_installed_locally(sorter_name):
            fallback = (
                f"The sorter *is* installed locally, so dropping the "
                f"{mode}_image argument would run it here instead."
            )
        else:
            fallback = f"It is not installed locally either, so fix {mode} or: {install_hint}"
        raise RuntimeError(
            f"sorter {sorter_name!r} was requested in a {mode} container, "
            f"but {mode} is unusable in this environment: {problem}. {fallback}"
        )

    if _sorter_installed_locally(sorter_name):
        return klass

    default_image = _default_container_image(sorter_name)
    container_hint = (
        f"Or run it in a container: docker_image=True "
        f"(SpikeInterface default image for {sorter_name}: {default_image})."
        if default_image
        else "Or run it in a container: docker_image='<image>'."
    )
    raise RuntimeError(
        f"sorter {sorter_name!r} is not installed in this environment. "
        f"{install_hint} {container_hint}"
    )


def build_kilosort_params(
    recording=None,
    batch_size=None,
    batch_duration_s=None,
    th_universal=None,
    th_learned=None,
    th_single_ch=None,
    cluster_downsampling=None,
    nearest_chans=None,
    max_channel_distance=None,
):
    """Translate MEA-facing sorting knobs into Kilosort4's parameter names.

    Only the values you actually set appear in the result, so Kilosort keeps its
    own defaults for everything else — passing an explicit ``None`` through to
    the sorter is not the same thing as leaving a parameter alone.

    `batch_duration_s` is the reason this exists: batches are natural to reason
    about in seconds, but Kilosort wants samples, and the conversion needs the
    recording's sampling rate. `batch_size` wins if both are given.
    """
    params = {}

    if batch_size is not None:
        params["batch_size"] = int(batch_size)
    elif batch_duration_s is not None:
        if recording is None:
            raise ValueError("batch_duration_s needs `recording` to convert seconds to samples")
        fs = float(recording.get_sampling_frequency())
        params["batch_size"] = int(round(fs * float(batch_duration_s)))

    if th_universal is not None:
        params["Th_universal"] = float(th_universal)
    if th_learned is not None:
        params["Th_learned"] = float(th_learned)
    if th_single_ch is not None:
        params["Th_single_ch"] = float(th_single_ch)
    if cluster_downsampling is not None:
        params["cluster_downsampling"] = int(cluster_downsampling)
    if nearest_chans is not None:
        params["nearest_chans"] = int(nearest_chans)
    if max_channel_distance is not None:
        params["max_channel_distance"] = float(max_channel_distance)

    return params


@contextmanager
def _global_job_kwargs(job_kwargs):
    """Apply SpikeInterface's process-global job kwargs for one call, then undo.

    n_jobs/chunk_duration/progress_bar are global state in SpikeInterface, not
    arguments to run_sorter. A library that sets them must also put them back,
    or it silently reconfigures every later call in the process.
    """
    if not job_kwargs:
        yield
        return

    import spikeinterface as si

    previous = dict(si.get_global_job_kwargs())
    si.set_global_job_kwargs(**dict(job_kwargs))
    logger.debug("global job kwargs set for this sort: %s", job_kwargs)
    try:
        yield
    finally:
        si.set_global_job_kwargs(**previous)


def _execution_label(sorter_name, mode, image):
    """One human-readable phrase describing where this sort will run."""
    if mode is None:
        return "local"
    if image is True:
        default_image = _default_container_image(sorter_name)
        return f"{mode}:{default_image or '<SpikeInterface default>'} (default image)"
    return f"{mode}:{image}"


def run_sorter(
    recording,
    output_dir,
    sorter_name="kilosort4",
    remove_existing=False,
    verbose=False,
    raise_error=True,
    job_kwargs=None,
    docker_image=None,
    singularity_image=None,
    **sorter_params,
):
    """Sort `recording` into `output_dir` and return the resulting Sorting.

    `sorter_params` go straight to the sorter (for Kilosort, build them with
    :func:`build_kilosort_params`). `job_kwargs` is the SpikeInterface job
    config — ``n_jobs``, ``chunk_duration``, ``progress_bar`` — applied globally
    for the duration of the call and restored afterwards. Those stay on the host
    even in container mode: they govern how the recording is written out to be
    handed over, not how the sorter runs once inside.

    Execution location:

    * neither `docker_image` nor `singularity_image` — run in this interpreter,
      which requires the sorter package to be installed here.
    * ``docker_image=True`` (or ``""``) — the image SpikeInterface maps this
      sorter to; for kilosort4 that is ``spikeinterface/kilosort4-base``.
    * ``docker_image="repo/image:tag"`` — that image, exactly.
    * `singularity_image` — same two forms, for HPC where Docker is not offered.

    Passing both is an error. A container run needs the sorter only inside the
    image, so a workstation with Docker and no local kilosort is a supported
    setup, not a degraded one.

    `remove_existing` False means an existing `output_dir` is an error, which is
    what you want when a folder may hold a real prior sort.

    The sorter and, in container mode, the container runtime are checked up
    front, so a missing kilosort or a missing Docker client fails immediately
    with a fix rather than partway through writing the recording out.
    """
    sorter_name = str(sorter_name)

    for key, replacement in _CONTAINER_KWARG_ALIASES.items():
        if key in sorter_params:
            raise ValueError(
                f"{key!r} is not a run_sorter parameter; request container execution with {replacement}."
            )

    mode, image = _container_request(docker_image, singularity_image)
    require_sorter(sorter_name, docker_image=docker_image, singularity_image=singularity_image)

    output_dir = Path(output_dir).expanduser()

    import spikeinterface.sorters as ss

    logger.info(
        "sorting with %s -> %s (execution=%s params=%s job_kwargs=%s)",
        sorter_name,
        output_dir,
        _execution_label(sorter_name, mode, image),
        sorter_params,
        job_kwargs or {},
    )
    with _global_job_kwargs(job_kwargs):
        sorting = ss.run_sorter(
            sorter_name=sorter_name,
            recording=recording,
            folder=output_dir,
            remove_existing_folder=bool(remove_existing),
            verbose=bool(verbose),
            raise_error=bool(raise_error),
            # Exactly one of these can be truthy; _container_request enforced it.
            docker_image=image if mode == "docker" else False,
            singularity_image=image if mode == "singularity" else False,
            **sorter_params,
        )

    # Some sorters/paths hand back None even on success; the folder is the real
    # artifact, so reload from it rather than reporting a failure that isn't one.
    if sorting is None:
        sorting = read_sorting(output_dir, raise_error=raise_error)
    if sorting is None:
        # raise_error False is the caller saying a failed sort is theirs to handle.
        if raise_error:
            raise RuntimeError(
                f"{sorter_name} returned no sorting and none could be read back from {output_dir}"
            )
        logger.warning("%s produced no sorting in %s", sorter_name, output_dir)
        return None

    logger.info("%s produced %d units in %s", sorter_name, len(sorting.unit_ids), output_dir)
    return sorting


def read_sorting(sorter_output_dir, raise_error=True):
    """Reload the Sorting a previous :func:`run_sorter` left on disk.

    Reading needs no sorter installed — the output is plain .npy/.tsv — so a
    sort produced on a GPU node reloads anywhere. Returns None when the run
    recorded a failure and `raise_error` is False.

    The sorted recording is reattached when the folder still describes it. Only
    when: SpikeInterface 0.103.2 raises an UnboundLocalError instead of a useful
    message if asked to reattach a recording whose json/pickle is absent, which
    is the normal state of a sorter folder copied off a compute node.
    """
    run_dir = _resolve_run_dir(sorter_output_dir)
    has_recording = any(
        (run_dir / f"spikeinterface_recording{ext}").is_file() for ext in (".json", ".pickle")
    )

    import spikeinterface.sorters as ss

    return ss.read_sorter_folder(
        run_dir,
        register_recording=has_recording,
        raise_error=bool(raise_error),
    )


def hash_directory(root, exclude_globs=()):
    """Return {posix_relative_path: sha256_hex} for every file under `root`.

    Paths are relative and posix-formatted so the mapping compares equal across
    machines. `exclude_globs` are matched against those relative paths.
    """
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        return {}
    mapping = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if _is_excluded(rel, exclude_globs):
            continue
        mapping[rel] = _hash_file(path)
    return dict(sorted(mapping.items()))


def snapshot_output(sorter_output_dir, hash_files=True, exclude_globs=()):
    """Summarize a finished sorter folder as a JSON-serializable dict.

    Reports the unit count, the parameters the sorter actually ran with (read
    back from the folder, not from what the caller believed it passed), and a
    sha256 per output file plus one combined digest over the whole tree. That
    combination is what makes a sort reproducible-checkable: same params, same
    bytes out.

    Hashing reads every byte the sorter wrote, which includes the binary
    recording copy some sorters leave behind (``sorter_output/recording.dat``,
    routinely tens of GB for a concatenated MEA scan). Pass
    ``exclude_globs=("sorter_output/recording.dat",)`` to skip it, or
    ``hash_files=False`` to collect sizes and counts only.

    Accepts either the run folder or its parent, and never needs the sorter
    package installed.
    """
    run_dir = _resolve_run_dir(sorter_output_dir)
    raw_dir = run_dir / _RAW_OUTPUT_DIRNAME
    if not raw_dir.is_dir():
        raw_dir = run_dir

    params_doc = _read_json(run_dir / _PARAMS_FILENAME) or {}
    log_doc = _read_json(run_dir / _LOG_FILENAME) or {}

    files = sorted(p for p in run_dir.rglob("*") if p.is_file())
    kept = [(p, p.relative_to(run_dir).as_posix()) for p in files]
    kept = [(p, rel) for p, rel in kept if not _is_excluded(rel, exclude_globs)]

    per_file = {}
    combined = hashlib.sha256()
    total_bytes = 0
    for path, rel in kept:
        total_bytes += path.stat().st_size
        if not hash_files:
            continue
        digest = _hash_file(path)
        per_file[rel] = digest
        # Fold the path in too, so a rename is a change even when bytes match.
        combined.update(rel.encode("utf-8"))
        combined.update(b"\x00")
        combined.update(digest.encode("ascii"))
        combined.update(b"\x00")

    return {
        "run_dir": str(run_dir),
        "raw_output_dir": str(raw_dir),
        "sorter_name": params_doc.get("sorter_name") or log_doc.get("sorter_name"),
        "sorter_version": log_doc.get("sorter_version"),
        "params": params_doc.get("sorter_params", {}),
        "n_units": _count_units(run_dir, raw_dir),
        "run_time_s": log_doc.get("run_time"),
        "sorted_at": log_doc.get("datetime"),
        "error": log_doc.get("error"),
        "file_count": len(kept),
        "total_bytes": int(total_bytes),
        "combined_sha256": combined.hexdigest() if hash_files else None,
        "per_file_sha256": per_file,
        "snapshotted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _resolve_run_dir(sorter_output_dir):
    """Find the folder holding SpikeInterface's run metadata.

    Callers pass either the run folder itself or the parent that contains it —
    the doubly nested ``sorter_output/sorter_output`` layout comes from naming a
    run folder after the subfolder SpikeInterface creates inside it.
    """
    root = Path(sorter_output_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"no such sorter output directory: {root}")
    for candidate in (root, root / _RAW_OUTPUT_DIRNAME):
        if (candidate / _LOG_FILENAME).is_file() or (candidate / _PARAMS_FILENAME).is_file():
            return candidate
    return root


def _read_json(path):
    """Parse a metadata file, or None — a half-written run is still worth summarizing."""
    if not Path(path).is_file():
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        logger.debug("could not parse %s", path, exc_info=True)
        return None


def _count_units(run_dir, raw_dir):
    """Number of sorted units, by whichever route the folder supports.

    Preferring the real Sorting keeps curation flags (keep_good_only) honored;
    the raw-file fallbacks exist because a failed or partial run still has a
    spike_clusters.npy worth counting.
    """
    try:
        sorting = read_sorting(run_dir, raise_error=False)
        if sorting is not None:
            return int(len(sorting.unit_ids))
    except Exception:
        logger.debug("could not load sorting from %s", run_dir, exc_info=True)

    clusters_npy = raw_dir / "spike_clusters.npy"
    if clusters_npy.is_file():
        try:
            import numpy as np

            return int(np.unique(np.load(clusters_npy, mmap_mode="r")).size)
        except Exception:
            logger.debug("could not read %s", clusters_npy, exc_info=True)

    for name in ("cluster_group.tsv", "cluster_KSLabel.tsv"):
        tsv = raw_dir / name
        if not tsv.is_file():
            continue
        try:
            lines = [ln for ln in tsv.read_text(encoding="utf-8").splitlines() if ln.strip()]
            return max(len(lines) - 1, 0)  # drop the header row
        except Exception:
            logger.debug("could not read %s", tsv, exc_info=True)

    return None


def _is_excluded(rel_path, exclude_globs):
    return any(fnmatch.fnmatch(rel_path, str(pattern)) for pattern in exclude_globs)


def _hash_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK_BYTES), b""):
            h.update(chunk)
    return h.hexdigest()
