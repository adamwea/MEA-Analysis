"""The diagnostic cache: what a compute capsule leaves behind for a plot tool.

The shape this exists to enforce: **a diagnostic is computed once, coincidently,
inside the capsule that already has the recording open, and lands in that
capsule's own canonical output. A plot tool then draws from that cache and never
touches the recording again.**

Why it matters in numbers. Drawing the per-segment and per-well review figures
by re-deriving them cost 2 h 06 m on one well, of which 94% was threshold
detection and artifact scanning -- work the capsule upstream had every input
for, thrown away as soon as a PNG was written. Changing a legend therefore cost
two hours. Reading this cache instead costs seconds.

What is cached, and why each thing rather than the obvious alternative:

``geometry``
    Channel ids and their x/y. Every map, layout and flag figure needs only
    this, so a :class:`CachedProbe` built from it substitutes for the recording
    in those emitters with no change to their signatures.

``metrics``
    The JSON dicts as the capsule computed them -- noise, activity, flags,
    clipping, artifacts. These are the capsule's data contract, not a figure's
    private working, which is the other half of "landing inside the canonical
    outputs".

``traces``
    The sample windows the trace figures actually draw, for the representative
    channels only, at full rate and keyed by their ORIGINAL frame numbers. A few
    channels over a bounded window is megabytes, and keeping the frames means
    the cached figure decimates and shades its dropped stretches in exactly the
    places the live one did -- the same figure, not an approximation of it.

``events``
    The raster caches DETECTED CROSSINGS, not traces. The raster draws every
    channel, and an all-channel window is ~400 MB per well against a few MB of
    event times -- and the detection is the expensive step anyway, so caching
    its result is what removes the cost rather than moving it.

``spectra``
    Welch output per source, already reduced.

One file per kind: a JSON for the dicts and geometry, an ``.npz`` for the
arrays. Both live in a ``diagnostics/`` folder inside the capsule's own output
directory, so a run that has the capsule has the cache, and a tool that cannot
find it names the capsule to run rather than quietly recomputing.
"""

import json
from pathlib import Path

import numpy as np

CACHE_DIRNAME = "diagnostics"
RECORD_NAME = "diagnostics_cache.json"
ARRAYS_NAME = "diagnostics_cache.npz"

# Bumped when a reader can no longer make sense of an older writer's output. A
# tool that meets a newer version says so and stops rather than half-drawing.
CACHE_VERSION = 1


class CacheMissing(FileNotFoundError):
    """No cache where one was expected. Carries the capsule that writes it."""

    def __init__(self, path, capsule):
        self.path = Path(path)
        self.capsule = capsule
        super().__init__(
            f"no diagnostic cache at {self.path} -- run the `{capsule}` capsule "
            "for this entity first; diagnostics are computed there, not here"
        )


class CacheVersionMismatch(ValueError):
    """A cache written by a version this reader does not understand."""


# --------------------------------------------------------------------------
# the recording stand-in
# --------------------------------------------------------------------------


class CachedProbe:
    """Geometry with the read surface the map and layout emitters use.

    Those emitters take a recording only to ask it for channel ids and channel
    locations (see ``channel_layout._channel_xy``). Handing them this instead
    keeps ONE definition of every figure: the cached path and the live path draw
    through the same function, so they cannot drift.

    It is deliberately not a recording: it has no samples and
    :meth:`get_traces` raises. An emitter that reaches for traces is one that
    needs its own cached array, and should fail loudly here rather than silently
    draw something else.
    """

    def __init__(self, channel_ids, locations, sampling_frequency=None, num_frames=None):
        self._channel_ids = list(channel_ids)
        locations = np.asarray(locations, dtype=float)
        if locations.ndim != 2 or locations.shape[0] != len(self._channel_ids):
            raise ValueError(
                f"locations must be (n_channels, 2); got {locations.shape} for "
                f"{len(self._channel_ids)} channels"
            )
        self._locations = locations
        self._fs = None if sampling_frequency is None else float(sampling_frequency)
        self._num_frames = None if num_frames is None else int(num_frames)

    def get_channel_ids(self):
        return list(self._channel_ids)

    def get_num_channels(self):
        return len(self._channel_ids)

    def get_channel_locations(self):
        return self._locations.copy()

    def get_sampling_frequency(self):
        if self._fs is None:
            raise ValueError("this cached probe carries no sampling frequency")
        return self._fs

    def get_num_frames(self, segment_index=None):
        """The span the capsule analysed, so a raster can set its own x limits.

        A raster draws the window that was ANALYSED, not the extent of its
        events -- a window where firing stops halfway has to show the silence --
        so the frame count travels with the cache rather than being inferred
        from the last event.
        """
        if self._num_frames is None:
            raise ValueError("this cached probe carries no frame count")
        return self._num_frames

    # SpikeInterface spells this both ways and the emitters use `get_num_samples`.
    get_num_samples = get_num_frames

    def get_num_segments(self):
        return 1

    def has_time_vector(self, segment_index=None):
        return False

    def has_scaleable_traces(self):
        return False

    def get_traces(self, *args, **kwargs):
        raise TypeError(
            "CachedProbe carries geometry only -- a figure that needs samples "
            "reads them from the cache's own trace or event arrays"
        )

    @classmethod
    def from_recording(cls, recording):
        """Snapshot a live recording's geometry, for the capsule to cache."""
        channel_ids = list(recording.get_channel_ids())
        try:
            locations = np.asarray(recording.get_channel_locations(), dtype=float)
        except Exception:  # noqa: BLE001 - a recording with no probe is a real case
            locations = np.full((len(channel_ids), 2), np.nan)
        try:
            fs = float(recording.get_sampling_frequency())
        except Exception:  # noqa: BLE001
            fs = None
        try:
            num_frames = int(recording.get_num_frames())
        except Exception:  # noqa: BLE001
            num_frames = None
        return cls(channel_ids, locations, sampling_frequency=fs, num_frames=num_frames)


class CachedTraces:
    """A frame-faithful window of samples, with the recording read surface.

    The trace figures decimate, place a real-elapsed axis and shade dropped
    stretches, all from FRAME NUMBERS. A view that renumbered its samples would
    put the gap shading in the wrong place, so this one keeps the original
    frames: `frame_offset` is where the cached block starts in the recording it
    came from, and :meth:`get_traces` is addressed in those same numbers.

    The consequence worth stating: the cached figure is not an approximation of
    the live one. Same frames, same decimation, same axis -- the same figure,
    drawn by the same emitter, from samples that were read once.
    """

    def __init__(
        self,
        channel_ids,
        traces,
        sampling_frequency,
        frame_offset=0,
        locations=None,
        unit=None,
    ):
        self._channel_ids = list(channel_ids)
        self._traces = np.asarray(traces)
        if self._traces.ndim != 2 or self._traces.shape[1] != len(self._channel_ids):
            raise ValueError(
                f"traces must be (n_samples, n_channels); got {self._traces.shape} "
                f"for {len(self._channel_ids)} channels"
            )
        self._fs = float(sampling_frequency)
        self._offset = int(frame_offset)
        self._locations = None if locations is None else np.asarray(locations, dtype=float)
        self.unit = unit

    def get_channel_ids(self):
        return list(self._channel_ids)

    def get_num_channels(self):
        return len(self._channel_ids)

    def get_num_frames(self, segment_index=None):
        return self._offset + self._traces.shape[0]

    # SpikeInterface spells this both ways and the emitters use `get_num_samples`.
    get_num_samples = get_num_frames

    def get_num_segments(self):
        return 1

    def get_sampling_frequency(self):
        return self._fs

    def get_channel_locations(self):
        if self._locations is None:
            raise ValueError("this cached trace window carries no geometry")
        return self._locations.copy()

    def has_time_vector(self, segment_index=None):
        return False

    def has_scaleable_traces(self):
        return False

    def get_traces(
        self, start_frame=None, end_frame=None, channel_ids=None, return_in_uV=None, **kwargs
    ):
        start = self._offset if start_frame is None else int(start_frame)
        end = self.get_num_frames() if end_frame is None else int(end_frame)
        lo = start - self._offset
        hi = end - self._offset
        if lo < 0 or hi > self._traces.shape[0]:
            raise ValueError(
                f"frames {start}..{end} fall outside the cached window "
                f"{self._offset}..{self._offset + self._traces.shape[0]}"
            )
        block = self._traces[lo:hi, :]
        if channel_ids is None:
            return block
        index = [self._channel_ids.index(cid) for cid in channel_ids]
        return block[:, index]

    @property
    def start_time_s(self):
        return self._offset / self._fs if self._fs else 0.0

    @property
    def duration_s(self):
        return self._traces.shape[0] / self._fs if self._fs else 0.0


# --------------------------------------------------------------------------
# write
# --------------------------------------------------------------------------


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def cache_dir(capsule_out_dir, create=False):
    """Where the cache lives inside a capsule's own output directory."""
    path = Path(capsule_out_dir) / CACHE_DIRNAME
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def write_cache(
    capsule_out_dir,
    *,
    probe,
    metrics=None,
    traces=None,
    events=None,
    spectra=None,
    meta=None,
):
    """Write one entity's diagnostic cache; return the directory it went to.

    `probe` is a :class:`CachedProbe` (or a live recording, snapshotted here).
    `traces` maps a source name to ``{"channel_ids", "traces", "frame_offset",
    "sampling_frequency", "unit"}`` -- frames, not seconds, because that is what
    the trace emitters decimate and shade from; `events` maps a name to
    ``{"times_s", "labels"}``; `spectra` maps a source name to a
    :func:`mea_modules.diagnostics.spectra.welch_spectra` result. `metrics` and
    `meta` are plain JSON-able dicts.

    Arrays and dicts are written to two files, not interleaved: the JSON stays
    readable by a person opening it to check a number, which is most of why the
    metric dicts are worth caching at all.
    """
    if not isinstance(probe, CachedProbe):
        probe = CachedProbe.from_recording(probe)

    out = cache_dir(capsule_out_dir, create=True)
    arrays = {}

    def _store(prefix, name, key, values, dtype=float):
        arrays[f"{prefix}/{name}/{key}"] = np.asarray(values, dtype=dtype)

    trace_sources = {}
    for name, block in (traces or {}).items():
        trace_sources[name] = {
            "channel_ids": _jsonable(list(block["channel_ids"])),
            "unit": block.get("unit"),
            # Frame-faithful: where this block starts in the recording it was
            # read from, so the cached figure decimates and shades exactly where
            # the live one did.
            "frame_offset": int(block.get("frame_offset", 0)),
            "sampling_frequency": float(
                block.get("sampling_frequency", probe._fs or 0.0)
            ),
        }
        _store("traces", name, "traces", block["traces"], dtype=np.float32)

    event_sources = {}
    for name, block in (events or {}).items():
        event_sources[name] = {"n_events": int(np.asarray(block["times_s"]).size)}
        _store("events", name, "times_s", block["times_s"])
        _store("events", name, "labels", block["labels"], dtype=np.int64)

    spectra_sources = {}
    for name, block in (spectra or {}).items():
        spectra_sources[name] = {
            key: _jsonable(value)
            for key, value in block.items()
            if not isinstance(value, np.ndarray) and key not in ("frequencies", "psd")
        }
        _store("spectra", name, "frequencies", block["frequencies"])
        _store("spectra", name, "psd", block["psd"])

    record = {
        "version": CACHE_VERSION,
        "geometry": {
            "channel_ids": _jsonable(probe.get_channel_ids()),
            "locations": probe.get_channel_locations().tolist(),
            "sampling_frequency": probe._fs,
            "num_frames": probe._num_frames,
        },
        "metrics": _jsonable(metrics or {}),
        "traces": trace_sources,
        "events": event_sources,
        "spectra": spectra_sources,
        "meta": _jsonable(meta or {}),
    }

    (out / RECORD_NAME).write_text(json.dumps(record, indent=2))
    np.savez_compressed(out / ARRAYS_NAME, **arrays)
    return out


# --------------------------------------------------------------------------
# read
# --------------------------------------------------------------------------


class DiagnosticCache:
    """One entity's cached diagnostics, as a plot tool consumes them."""

    def __init__(self, record, arrays, path):
        self.path = Path(path)
        self._record = record
        self._arrays = arrays
        geometry = record["geometry"]
        self.probe = CachedProbe(
            geometry["channel_ids"],
            geometry["locations"],
            sampling_frequency=geometry.get("sampling_frequency"),
            num_frames=geometry.get("num_frames"),
        )

    @property
    def metrics(self):
        return self._record.get("metrics", {})

    @property
    def meta(self):
        return self._record.get("meta", {})

    def metric(self, name):
        """One metric dict, or None when the capsule did not compute it."""
        return self._record.get("metrics", {}).get(name)

    def trace_names(self):
        return list(self._record.get("traces", {}))

    def event_names(self):
        return list(self._record.get("events", {}))

    def spectra_names(self):
        return list(self._record.get("spectra", {}))

    def traces(self, name):
        """One cached trace window, as a :class:`CachedTraces` view.

        A view rather than raw arrays, because the trace emitters take a
        recording: handing them this keeps ONE definition of the figure, drawn
        by the same function whether the samples came off disk or out of cache.
        """
        block = self._record["traces"][name]
        return CachedTraces(
            block["channel_ids"],
            self._arrays[f"traces/{name}/traces"],
            sampling_frequency=block["sampling_frequency"],
            frame_offset=block.get("frame_offset", 0),
            locations=self._locations_for(block["channel_ids"]),
            unit=block.get("unit"),
        )

    def _locations_for(self, channel_ids):
        """The cached geometry, restricted to and ordered by `channel_ids`."""
        known = self.probe.get_channel_ids()
        locations = self.probe.get_channel_locations()
        try:
            index = [known.index(cid) for cid in channel_ids]
        except ValueError:
            return None
        return locations[index, :]

    def events(self, name):
        """``(times_s, labels)`` -- the pair detection returns, cached."""
        return (
            self._arrays[f"events/{name}/times_s"],
            self._arrays[f"events/{name}/labels"],
        )

    def spectra(self, name):
        """One Welch result, reassembled in the shape the panel emitter takes."""
        block = dict(self._record["spectra"][name])
        block["frequencies"] = self._arrays[f"spectra/{name}/frequencies"]
        block["psd"] = self._arrays[f"spectra/{name}/psd"]
        return block

    def all_spectra(self):
        """Every cached source, in the order the capsule wrote them."""
        return {name: self.spectra(name) for name in self.spectra_names()}


def read_cache(capsule_out_dir, capsule="the upstream capsule"):
    """Load one entity's cache. Raises :class:`CacheMissing` when absent.

    `capsule` names the capsule that writes it, so a tool run against a run
    folder that predates the cache tells the operator what to run rather than
    falling back to recomputing -- the fallback is what this whole module
    exists to remove.
    """
    directory = cache_dir(capsule_out_dir)
    record_path = directory / RECORD_NAME
    if not record_path.is_file():
        raise CacheMissing(record_path, capsule)

    record = json.loads(record_path.read_text())
    version = int(record.get("version", 0))
    if version > CACHE_VERSION:
        raise CacheVersionMismatch(
            f"{record_path} was written at cache version {version}; this reader "
            f"understands {CACHE_VERSION}. Re-run the `{capsule}` capsule."
        )

    arrays_path = directory / ARRAYS_NAME
    arrays = dict(np.load(arrays_path)) if arrays_path.is_file() else {}
    return DiagnosticCache(record, arrays, directory)


def cache_ready(capsule_out_dir):
    """True when a readable cache is present. For a resume check, not a gate."""
    return (cache_dir(capsule_out_dir) / RECORD_NAME).is_file()
