"""The readable JSON beside a segment's cache: copied out of it, never recomputed.

Fixtures come from the real producer (`collect_segment_diagnostics` into
`write_cache`), so the reports are built from the cache a capsule actually
writes, not from a dict shaped to agree with them.
"""

import json

import numpy as np
import pytest

from mea_modules.diagnostics import (
    SEGMENT_REPORTS,
    collect_segment_diagnostics,
    read_cache,
    write_cache,
    write_segment_reports,
)

FS_HZ = 10_000.0
N_CHANNELS = 8


def _recording():
    from probeinterface import Probe
    from spikeinterface.core import NumpyRecording

    rng = np.random.default_rng(3)
    traces = rng.normal(0.0, 8.0, (40_000, N_CHANNELS)).astype(np.float32)
    traces[400:30_000:311, :] = -180.0
    recording = NumpyRecording([traces], sampling_frequency=FS_HZ)
    probe = Probe(ndim=2)
    probe.set_contacts(positions=[[(i % 4) * 17.5, (i // 4) * 17.5] for i in range(N_CHANNELS)],
                       shapes="square", shape_params={"width": 5})
    probe.set_device_channel_indices(np.arange(N_CHANNELS))
    return recording.set_probe(probe)


def _cache(out_dir, enabled=None):
    recording = _recording()
    payload = collect_segment_diagnostics(
        recording, recording, source="preprocessed", duration_s=1.0, num_chunks=2, seed=5,
        mad_threshold=5.0, dead_noise_ratio=0.1, artifacts_duration_s=1.0, window_s=1.0,
        trace_channels=2, raster_max_channels=8, enabled=enabled,
        meta={"data_h5": "/data/scan.raw.h5"},
    )
    write_cache(out_dir, **payload)


@pytest.fixture
def reported(tmp_path):
    _cache(tmp_path)
    return tmp_path, write_segment_reports(tmp_path, capsule="preprocess_segment",
                                           well="well000", rec="rec0000", label="toy")


def test_every_report_is_written_beside_the_cache(reported):
    out_dir, written = reported
    assert set(written) == set(SEGMENT_REPORTS)
    for name, path in written.items():
        assert path == out_dir / "diagnostics" / name and path.is_file(), name


def test_the_reports_carry_the_cache_s_own_numbers(reported):
    out_dir, _ = reported
    cache = read_cache(out_dir)
    folder = out_dir / "diagnostics"
    qc = json.loads((folder / "qc_report.json").read_text())
    assert qc["computed_by"] == "preprocess_segment" and qc["well"] == "well000"
    assert qc["rec"] == "rec0000" and qc["label"] == "toy" and qc["data_h5"] == "/data/scan.raw.h5"
    assert qc["noise"]["median_noise"] == cache.metric("noise")["median_noise"]
    assert qc["summary"]["median_rate_hz"] == float(np.median(cache.metric("activity")["rate_hz"]))
    assert qc["n_channels"] == N_CHANNELS and qc["incomplete"] == {}
    bad = json.loads((folder / "bad_channels.json").read_text())
    assert bad["thresholds"]["mad_threshold"] == 5.0
    assert bad["thresholds"]["dead_noise_ratio"] == 0.1  # the flag rules' own, kept beside it
    assert bad["flagged_fraction"] == cache.metric("flagged")["flagged_fraction"]
    for name in ("clipping", "artifacts"):
        census = json.loads((folder / f"{name}.json").read_text())
        assert {k: census[k] for k in cache.metric(name)} == json.loads(
            json.dumps(cache.metric(name)))


def test_a_diagnostic_switched_off_leaves_no_old_report_behind(reported):
    out_dir, _ = reported
    _cache(out_dir, enabled={"clipping": False, "artifacts": False, "noise": True,
                             "rms": True, "activity": True, "bad_channels": True,
                             "flags": True, "traces": True, "raster": True, "spectra": True})
    written = write_segment_reports(out_dir, capsule="preprocess_segment",
                                    well="well000", rec="rec0000")
    assert written["clipping.json"] is None and written["artifacts.json"] is None
    assert not (out_dir / "diagnostics" / "clipping.json").exists()
    assert not (out_dir / "diagnostics" / "artifacts.json").exists()
    assert written["qc_report.json"].is_file()


def _concat_cache(out_dir, enabled=None):
    from mea_modules.diagnostics.collect_concat import collect_concat_diagnostics

    recording = _recording()
    rows = [{"rec": "rec0000", "n_samples": 20_000, "start_frame": 0, "end_frame": 20_000,
             "n_electrodes": 10},
            {"rec": "rec0001", "n_samples": 20_000, "start_frame": 20_000, "end_frame": 40_000,
             "n_electrodes": 9}]
    payload = collect_concat_diagnostics(
        recording, segments=rows, stitch_frames=[20_000], seed=5, detect_threshold=5.0,
        duration_s=1.0, num_chunks=2, trace_channels=2, enabled=enabled)
    write_cache(out_dir, **payload)


def test_a_concatenated_well_s_reports_come_from_its_cache(tmp_path):
    from mea_modules.diagnostics import CONCAT_REPORTS, write_concat_reports

    _concat_cache(tmp_path)
    written = write_concat_reports(tmp_path, capsule="concatenate", well="well000")
    assert set(written) == set(CONCAT_REPORTS)
    activity = json.loads(written["segment_activity.json"].read_text())
    cache = read_cache(tmp_path)
    assert activity["computed_by"] == "concatenate" and activity["threshold_factor"] == 5.0
    assert [row["n_electrodes"] for row in activity["segments"]] == [10, 9]
    assert activity["activity"] == json.loads(json.dumps(cache.metric("segment_activity")))
    # no source file was named, so no gap table; motion is off by default
    assert written["gap_table.json"] is None and written["motion.json"] is None


def test_the_median_rate_is_the_detector_s_own_number(tmp_path):
    """The QC report copies the median rate the detector recorded; it never
    recomputes one from the cached array (a viewer must not re-derive numbers)."""
    recording = _recording()
    payload = collect_segment_diagnostics(
        recording, recording, source="preprocessed", duration_s=1.0, num_chunks=2, seed=5,
        mad_threshold=5.0, dead_noise_ratio=0.1, artifacts_duration_s=1.0, window_s=1.0,
        trace_channels=2, raster_max_channels=8, meta={},
    )
    payload["metrics"]["activity"]["median_rate_hz"] = 123.0  # a number only the detector wrote
    write_cache(tmp_path, **payload)
    written = write_segment_reports(tmp_path, capsule="preprocess_segment", well="well000")
    qc = json.loads(written["qc_report.json"].read_text())
    assert qc["summary"]["median_rate_hz"] == 123.0


def test_without_noise_there_is_no_qc_report(tmp_path):
    _cache(tmp_path, enabled={"clipping": True})
    written = write_segment_reports(tmp_path, capsule="preprocess_segment", well="well000")
    assert written["qc_report.json"] is None and written["bad_channels.json"] is None
    assert written["clipping.json"].is_file()
