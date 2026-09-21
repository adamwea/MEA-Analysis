"""The concatenated well's diagnostics, computed once where the binary is open.

The input is made the way the concatenate capsule makes it -- segments joined by
`concatenate_segments`, joins from `stitch_frames`, the result written by
`save_concatenated` -- and the gap table comes from a source file laid out like
the real one. What the collector owes its consumers:

* one detection over the whole timeline, which the per-segment activity summary
  is a view of;
* the per-segment table carrying each segment's own frame count from the SOURCE
  file, the independent number a continuity check compares the binary against;
* the whole-timeline traces as exactly the samples a stride of the binary
  gives, with the joins and the gap table renumbered to match;
* all of it surviving the cache round trip.
"""
import numpy as np
import pytest

from mea_modules.concatenation import concatenate_segments, save_concatenated, stitch_frames
from mea_modules.diagnostics import collect_concat_diagnostics, read_cache, write_cache
from mea_modules.diagnostics.collect_concat import CONCAT_DIAGNOSTIC_NAMES
from mea_modules.diagnostics.timebase import rescale_time_gaps
from mea_modules.io.metadata import electrode_coverage, find_common_electrodes

FS = 10_000.0
TRACE_HZ = 500.0
STEP = 20
SEGMENTS = [("rec0000", 12_000), ("rec0001", 8_000), ("rec0002", 10_000)]
COMMON = list(range(20, 28))
TROUGH_EVERY = 500

SOURCE = [
    {"rec": "rec0000", "n_samples": 12_000, "electrodes": list(range(0, 40)),
     "breaks": [(3_000, 6)], "start_s": 0.0, "stop_s": 1.2},
    {"rec": "rec0001", "n_samples": 8_000, "electrodes": list(range(15, 60)),
     "breaks": [], "start_s": 31.2, "stop_s": 32.0},
    {"rec": "rec0002", "n_samples": 10_000, "electrodes": list(range(20, 28)) + [900],
     "breaks": [(5_001, 3)], "start_s": 62.0, "stop_s": 63.0},
]


def _segment(n_samples, seed):
    from probeinterface import Probe
    from spikeinterface.core import NumpyRecording

    rng = np.random.default_rng(seed)
    traces = rng.normal(0.0, 1.0, (n_samples, len(COMMON))).astype(np.float32)
    width = int(0.001 * FS)
    shape = (-25.0 * np.sin(np.linspace(0.0, np.pi, width))).astype(np.float32)
    for start in range(100, n_samples - width, TROUGH_EVERY):
        traces[start:start + width, :] += shape[:, None]
    recording = NumpyRecording([traces], sampling_frequency=FS,
                               channel_ids=[str(e) for e in COMMON])
    probe = Probe(ndim=2)
    probe.set_contacts(
        positions=np.asarray([[(e % 220) * 17.5, (e // 220) * 17.5] for e in COMMON]),
        shapes="square", shape_params={"width": 5},
    )
    probe.set_device_channel_indices(np.arange(len(COMMON)))
    recording.set_probe(probe, in_place=True)
    return recording


@pytest.fixture
def well(tmp_path, maxtwo_file):
    """``(saved_binary, rows, stitch, source_path)`` as the capsule holds them."""
    source = maxtwo_file(SOURCE)
    assert find_common_electrodes(source, "well000")["common_electrodes"] == COMMON
    coverage = electrode_coverage(source, "well000")
    own = {row["rec"]: row["n_electrodes"] for row in coverage["segments"]}

    segments = [_segment(n, seed) for seed, (_rec, n) in enumerate(SEGMENTS)]
    joins = stitch_frames(segments)
    saved = save_concatenated(concatenate_segments(segments), tmp_path / "recording",
                              progress_bar=False)
    rows, start = [], 0
    for rec, n in SEGMENTS:
        rows.append({"rec": rec, "n_samples": n, "start_frame": start,
                     "end_frame": start + n, "n_electrodes": own[rec]})
        start += n
    return saved, rows, joins, source


def _collect(fixture, **kwargs):
    saved, rows, joins, source = fixture
    base = dict(segments=rows, stitch_frames=joins, seed=5, detect_threshold=5.0,
                duration_s=0.5, num_chunks=3, trace_channels=3, trace_hz=TRACE_HZ,
                data_h5=source, well="well000")
    base.update(kwargs)
    return collect_concat_diagnostics(saved, **base)


def _planted(n_samples):
    return len(range(100, n_samples - 10, TROUGH_EVERY))


def test_everything_the_default_set_computes_comes_back_clean(well):
    payload = _collect(well)
    assert set(payload) == {"probe", "metrics", "traces", "events", "spectra", "time_gaps", "meta"}
    metrics = payload["metrics"]
    assert metrics["errors"] == {}
    # Motion is off by default: unvalidated here, and the most expensive step.
    assert metrics["skipped"] == {"motion": "not requested"}
    assert metrics["motion"] is None
    assert payload["meta"]["diagnostics"] == {
        name: name != "motion" for name in CONCAT_DIAGNOSTIC_NAMES
    }


def test_the_segment_table_carries_the_source_files_own_counts(well):
    rows = _collect(well)["metrics"]["segment_table"]
    assert [row["rec"] for row in rows] == [rec for rec, _ in SEGMENTS]
    assert [row["source_n_samples"] for row in rows] == [n for _, n in SEGMENTS]
    assert [row["n_samples"] for row in rows] == [n for _, n in SEGMENTS]
    assert [row["n_electrodes"] for row in rows] == [40, 45, 9]
    assert rows[0]["gap_before_s"] is None
    assert rows[1]["gap_before_s"] == pytest.approx(30.0)
    assert [row["n_breaks"] for row in rows] == [1, 0, 1]


def test_one_detection_serves_the_raster_and_the_activity_summary(well):
    payload = _collect(well)
    events = payload["metrics"]["events"]
    expected = sum(_planted(n) for _, n in SEGMENTS) * len(COMMON)
    assert events["n_events"] == expected
    assert set(payload["events"]["raster"]["labels"].tolist()) == set(COMMON)

    activity = payload["metrics"]["segment_activity"]
    assert [row["n_events"] for row in activity["segments"]] == [
        _planted(n) * len(COMMON) for _, n in SEGMENTS
    ]
    assert activity["total_events"] == expected
    assert activity["stability"]["n_segments_compared"] == 3
    # Every electrode fires at exactly 20 Hz in every segment (24 / 1.2 s,
    # 16 / 0.8 s, 20 / 1.0 s): nothing moved, and the record says so rather
    # than inventing a statistic.
    assert activity["stability"]["cv_of_segment_means"] == pytest.approx(0.0)
    assert activity["activity_comparison"]["test"] is None
    assert activity["activity_comparison"]["significant"] is None
    assert "no test was run" in activity["activity_comparison"]["note"]


def test_the_traces_are_a_stride_of_the_binary_with_the_joins_renumbered(well):
    saved = well[0]
    payload = _collect(well)
    block = payload["traces"]["preprocessed"]
    assert block["frame_step"] == STEP
    assert block["sampling_frequency"] == pytest.approx(TRACE_HZ)
    rep = payload["metrics"]["representative_channels"]
    full = saved.get_traces(channel_ids=rep, return_in_uV=block["unit"] == "uV")
    assert np.array_equal(block["traces"], full[::STEP].astype(np.float32))
    assert payload["metrics"]["traces"]["stitch_frames"] == [600, 1_000]


def test_both_gap_tables_are_cached_and_agree(well):
    payload = _collect(well)
    native = payload["time_gaps"]["native"]
    assert native["gaps"]["break_sample_indices"] == [3_000, 20_000 + 5_001]
    assert payload["time_gaps"]["traces"] == rescale_time_gaps(native, STEP)


def test_the_payload_survives_the_cache_round_trip(well, tmp_path):
    payload = _collect(well)
    write_cache(tmp_path / "concatenate", **payload)
    cached = read_cache(tmp_path / "concatenate", capsule="concatenate")

    assert cached.metric("segment_table") == payload["metrics"]["segment_table"]
    times, labels = cached.events("raster")
    assert np.array_equal(labels, payload["events"]["raster"]["labels"])
    assert np.allclose(times, payload["events"]["raster"]["times_s"])
    assert cached.trace_block("preprocessed")["frame_step"] == STEP
    assert cached.trace_block("preprocessed")["time_gaps"] == "traces"
    assert list(cached.time_gaps("traces")["gaps"]["break_sample_indices"]) == [150, 1_251]
    view = cached.traces("preprocessed")
    assert view.get_num_samples() == payload["traces"]["preprocessed"]["traces"].shape[0]


def test_no_source_file_skips_the_gap_table_with_the_reason(well):
    payload = _collect(well, data_h5=None, well=None)
    metrics = payload["metrics"]
    assert "gaps" not in metrics["errors"]
    assert "no source file" in metrics["skipped"]["gaps"]
    assert payload["time_gaps"] == {}
    assert all(row["source_n_samples"] is None for row in metrics["segment_table"])


def test_switches_and_dependencies_are_honoured_and_recorded(well):
    payload = _collect(well, enabled={"noise", "events"})
    skipped = payload["metrics"]["skipped"]
    assert skipped["traces"] == "not requested"
    assert skipped["activity_summary"] == "not requested"
    assert skipped["gaps"] == "not requested"
    assert payload["traces"] == {} and payload["metrics"]["segment_activity"] is None

    alone = _collect(well, enabled={"events", "activity_summary"})["metrics"]["skipped"]
    assert alone["events"] == "requires noise"
    assert alone["activity_summary"] == "requires events"


def test_an_unknown_switch_is_refused(well):
    with pytest.raises(ValueError, match="unknown diagnostics"):
        _collect(well, enabled={"noise", "evnets"})
