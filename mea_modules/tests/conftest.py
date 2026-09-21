"""Shared fixtures for the mea_modules tests."""
import numpy as np
import pytest

# 2026-01-01T00:00:00Z in milliseconds: the unit the reference scan stores its
# start/stop timestamps in, inferred by the reader from the magnitude.
_EPOCH_MS = 1_767_225_600_000
_ARRAY_COLUMNS = 220
_PITCH_UM = 17.5


@pytest.fixture
def maxtwo_file(tmp_path):
    """Write a minimal MaxTwo-layout HDF5 file; return its path.

    Called with one dict per recording, in file order::

        {"rec": "rec0000", "n_samples": 1000, "electrodes": [...],
         "breaks": [(sample_index, missing_frames), ...],
         "start_s": 0.0, "stop_s": 10.0}

    Only what the readers under test open is written: the format version, each
    recording's wall-clock bounds, its channel mapping (electrode, channel,
    position; plus one unrouted row with channel -1 and one electrode listed
    twice) and its routed frame counter.
    """

    def write(recordings, well="well000", name="scan.raw.h5"):
        import h5py

        path = tmp_path / name
        with h5py.File(path, "w") as h5:
            h5.create_dataset("version", data=[b"20190530"])
            for entry in recordings:
                base = f"wells/{well}/{entry['rec']}"
                h5.create_dataset(f"{base}/start_time", data=[_EPOCH_MS + int(entry["start_s"] * 1000)])
                h5.create_dataset(f"{base}/stop_time", data=[_EPOCH_MS + int(entry["stop_s"] * 1000)])

                electrodes = np.asarray(entry["electrodes"], dtype=np.int32)
                rows = np.concatenate((electrodes, electrodes[:1], [999_999])).astype(np.int32)
                channels = np.concatenate(
                    (np.arange(electrodes.size), [0], [-1])
                ).astype(np.int32)
                mapping = np.zeros(rows.size, dtype=[
                    ("channel", "<i4"), ("electrode", "<i4"), ("x", "<f8"), ("y", "<f8"),
                ])
                mapping["electrode"] = rows
                mapping["channel"] = channels
                mapping["x"] = (rows % _ARRAY_COLUMNS) * _PITCH_UM
                mapping["y"] = (rows // _ARRAY_COLUMNS) * _PITCH_UM
                h5.create_dataset(f"{base}/settings/mapping", data=mapping)

                frames = np.arange(int(entry["n_samples"]), dtype=np.int64)
                for index, missing in entry.get("breaks", ()):
                    frames[int(index):] += int(missing)
                h5.create_dataset(f"{base}/groups/routed/frame_nos", data=frames)
        return path

    return write
