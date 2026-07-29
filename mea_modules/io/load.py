"""Read Maxwell HDF5 recordings, one segment at a time.

A Maxwell file holds one or more *segments* — a (well, recording) pair. How many
there are is a property of the file, not of the experiment type, so it is always
discovered rather than assumed:

* an AxonTracking scan is multi-segment (one recording per electrode config —
  the P003454 reference scan has 21 in a single well),
* a network scan is typically single-segment,
* the pre-2016 MaxOne format has one implicit well and no recording ids at all.

:func:`list_maxwell_streams` reports that layout, :func:`load_maxwell` opens one
segment, and :func:`iter_segments` walks them all by calling :func:`load_maxwell`
per segment — so there is a single open path, not one for "single" files and
another for "multi" files.

This module is pure library: no argparse, no printing, no ``__main__``. CLIs live
in the consuming pipeline's capsules.
"""

from pathlib import Path

from .hdf5_plugin import set_plugin_path

# Maxwell format version that predates the per-well "wells" group.
_OLD_FORMAT_VERSION = 20160704
_OLD_FORMAT_STREAM = "well000"


def list_maxwell_streams(h5_path):
    """Return {"version": int, "streams": {stream_id: [rec_name, ...]}}.

    Read straight from the HDF5 layout rather than through SpikeInterface: the
    extractor refuses to open a file holding multiple recording ids until you
    have already picked one, which is exactly what this is here to tell you.
    """
    import h5py

    with h5py.File(str(h5_path), mode="r") as h5:
        version = int(h5["version"][0].decode())
        if version <= _OLD_FORMAT_VERSION:
            # Old MaxOne format: a single implicit well, no rec_name.
            return {"version": version, "streams": {_OLD_FORMAT_STREAM: []}}
        streams = {well: sorted(h5["wells"][well].keys()) for well in sorted(h5["wells"].keys())}
    return {"version": version, "streams": streams}


def count_segments(h5_path):
    """Total number of (well, recording) segments in the file."""
    layout = list_maxwell_streams(h5_path)
    return sum(len(recs) or 1 for recs in layout["streams"].values())


def load_maxwell(h5_path, stream_id=None, rec_name=None, hdf5_plugin_path=None, plugin_candidates=()):
    """Open one Maxwell stream/recording and return the SpikeInterface recording.

    stream_id is a well ("well000"); rec_name is a recording within it
    ("rec0000"). Either may be omitted, in which case the first available one is
    used. Only the selected segment is opened — traces stay lazy, so this returns
    in well under a second even on a multi-GB scan.
    """
    h5_path = Path(h5_path).expanduser()
    if not h5_path.exists():
        raise FileNotFoundError(f"no such Maxwell file: {h5_path}")

    plugin_dir = set_plugin_path(hdf5_plugin_path, extra_candidates=plugin_candidates)

    available = list_maxwell_streams(h5_path)["streams"]

    if stream_id is None:
        stream_id = next(iter(available))
    elif stream_id not in available:
        raise ValueError(f"stream {stream_id!r} not in file; available: {sorted(available)}")

    rec_names = available[stream_id]
    if rec_name is None:
        # Old format carries no rec_name; neo derives it itself.
        rec_name = rec_names[0] if rec_names else None
    elif rec_names and rec_name not in rec_names:
        raise ValueError(f"rec_name {rec_name!r} not in stream {stream_id!r}; available: {rec_names}")

    import spikeinterface.extractors as se

    # Having resolved the plugin ourselves, stop SpikeInterface from running its
    # own install check — it re-announces the library on every single open.
    kwargs = {"stream_id": stream_id, "rec_name": rec_name}
    if plugin_dir is not None:
        kwargs["install_maxwell_plugin"] = False

    # read_maxwell is the current entry point; fall back for older SI versions.
    reader = getattr(se, "read_maxwell", None) or se.MaxwellRecordingExtractor
    try:
        recording = reader(file_path=str(h5_path), **kwargs)
    except TypeError:  # pragma: no cover - older SI without install_maxwell_plugin
        kwargs.pop("install_maxwell_plugin", None)
        recording = reader(file_path=str(h5_path), **kwargs)

    return recording


def iter_segments(h5_path, stream_id=None, hdf5_plugin_path=None, plugin_candidates=(), strict=True):
    """Yield ``(stream_id, rec_name, recording)`` for each segment in the file.

    Walks whatever the file actually contains — every well by default, or just
    `stream_id` when given. Each segment is opened through :func:`load_maxwell`,
    so single- and multi-segment files take the same path; a single-segment
    network scan simply yields once.

    Recordings are opened lazily one at a time, so iterating a 21-segment
    AxonTracking scan costs metadata reads, not traces.

    With `strict` False, a segment that fails to open yields its exception in
    place of the recording instead of aborting the walk — useful when surveying
    a file that may hold a bad segment.
    """
    layout = list_maxwell_streams(h5_path)
    streams = layout["streams"]
    if stream_id is not None:
        if stream_id not in streams:
            raise ValueError(f"stream {stream_id!r} not in file; available: {sorted(streams)}")
        streams = {stream_id: streams[stream_id]}

    for well, rec_names in streams.items():
        for rec in rec_names or [None]:
            try:
                recording = load_maxwell(
                    h5_path,
                    stream_id=well,
                    rec_name=rec,
                    hdf5_plugin_path=hdf5_plugin_path,
                    plugin_candidates=plugin_candidates,
                )
            except Exception as exc:
                if strict:
                    raise
                yield well, rec, exc
                continue
            yield well, rec, recording


def describe_segment(recording):
    """Summarize an opened recording as a plain dict of scalars."""
    fs = float(recording.get_sampling_frequency())
    n_samples = int(recording.get_num_samples())
    return {
        "channels": int(recording.get_num_channels()),
        "fs_hz": fs,
        "n_samples": n_samples,
        "duration_s": n_samples / fs if fs else 0.0,
        "dtype": str(recording.get_dtype()),
    }
