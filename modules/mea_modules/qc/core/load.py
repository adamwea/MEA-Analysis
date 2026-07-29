"""Load a single Maxwell recording segment.

Individually runnable, so the read path can be checked on its own::

    python -m mea_modules.qc.core.load --data-h5 <path>              # survey every segment
    python -m mea_modules.qc.core.load --data-h5 <path> --rec-name rec0005

Or through the pipeline front door, which fills the arguments in from the
analysis config::

    axon-recon qc.core.load --config runtime.yml

Nothing here filters, references, or scales the data — it only opens one
stream/recording and hands back the SpikeInterface recording object.

Plugin path handling lives in :mod:`mea_modules.io.hdf5_plugin`; this module
just passes the caller's choice through, and hardcodes no path of its own.
"""

import argparse
import sys

from ...io.hdf5_plugin import set_plugin_path

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


def load_maxwell(h5_path, stream_id=None, rec_name=None, hdf5_plugin_path=None, plugin_candidates=()):
    """Open one Maxwell stream/recording and return the SpikeInterface recording.

    stream_id is a well ("well000"); rec_name is a recording within it
    ("rec0000"). Either may be omitted, in which case the first available one is
    used. Only the selected segment is opened — traces stay lazy, so this returns
    in well under a second even on a multi-GB scan.
    """
    from pathlib import Path

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


def build_parser():
    parser = argparse.ArgumentParser(
        prog="mea_modules.qc.core.load",
        description="Load Maxwell segments and report what is in them.",
    )
    parser.add_argument("--data-h5", required=True, help="path to data.raw.h5")
    parser.add_argument("--stream", default=None, help="well/stream id, e.g. well000")
    parser.add_argument("--rec-name", default=None, help="recording id, e.g. rec0000")
    parser.add_argument(
        "--hdf5-plugin-path",
        default=None,
        help="dir holding the Maxwell HDF5 compression plugin (default: $HDF5_PLUGIN_PATH)",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="optional name for this dataset, used as a heading in the output",
    )
    return parser


def _print_segment_detail(args, layout):
    """Open one selected segment and print it in full, chunk shape included."""
    recording = load_maxwell(
        args.data_h5,
        stream_id=args.stream,
        rec_name=args.rec_name,
        hdf5_plugin_path=args.hdf5_plugin_path,
    )
    info = describe_segment(recording)
    chunk_end = min(int(round(info["fs_hz"])), info["n_samples"])
    chunk = recording.get_traces(start_frame=0, end_frame=chunk_end)

    print(f"stream        : {args.stream or next(iter(layout['streams']))}")
    print(f"rec_name      : {getattr(recording, '_kwargs', {}).get('rec_name')}")
    print(f"channels      : {info['channels']}")
    print(f"sampling rate : {info['fs_hz']:.1f} Hz")
    print(f"duration      : {info['duration_s']:.2f} s ({info['n_samples']} samples)")
    print(f"dtype         : {info['dtype']}")
    print(f"1 s chunk     : {chunk.shape} (samples, channels)")


def _print_segment_table(args, layout):
    """Walk every stream/recording in the file and print one row each."""
    print(f"{'well':<10} {'rec':<10} {'ch':>6} {'fs (Hz)':>10} {'dur (s)':>10}  dtype")
    for stream_id, rec_names in layout["streams"].items():
        for rec_name in rec_names or [None]:
            try:
                recording = load_maxwell(
                    args.data_h5,
                    stream_id=stream_id,
                    rec_name=rec_name,
                    hdf5_plugin_path=args.hdf5_plugin_path,
                )
                info = describe_segment(recording)
            except Exception as exc:  # one bad segment should not hide the rest
                print(f"{stream_id:<10} {str(rec_name):<10} {'ERROR':>6}  {exc}")
                continue
            print(
                f"{stream_id:<10} {str(rec_name):<10} {info['channels']:>6} "
                f"{info['fs_hz']:>10.1f} {info['duration_s']:>10.2f}  {info['dtype']}"
            )


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.label:
        print(f"=== {args.label} ===")

    layout = list_maxwell_streams(args.data_h5)
    n_segments = sum(len(recs) or 1 for recs in layout["streams"].values())
    print(f"file          : {args.data_h5}")
    print(f"format        : {layout['version']}")
    print(f"streams       : {len(layout['streams'])} well(s), {n_segments} segment(s)")
    print()

    # Naming any part of the selection means "show me that one in detail";
    # otherwise survey the whole file.
    if args.stream or args.rec_name:
        _print_segment_detail(args, layout)
    else:
        _print_segment_table(args, layout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
