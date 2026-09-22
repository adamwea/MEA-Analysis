"""timing.json: what one entity's diagnostics cost, beside the cache it describes.

The fixtures come from the real producer -- the collector's own metrics and
write_cache -- so the file records what a capsule would actually have to hand.
"""
import json

import numpy as np
import pytest
from spikeinterface.core import NumpyRecording, generate_recording

from mea_modules.diagnostics.cache import RECORD_NAME, cache_dir, read_cache, write_cache
from mea_modules.diagnostics.collect import collect_segment_diagnostics
from mea_modules.diagnostics.timing import TIMING_NAME, clear_timing, write_timing

FS = 10_000.0


@pytest.fixture
def computed(tmp_path):
    """One segment's diagnostics, collected and cached as a capsule does."""
    rng = np.random.default_rng(0)
    traces = rng.normal(0.0, 5.0, size=(int(FS), 6)).astype(np.float32)
    probe = generate_recording(num_channels=6, sampling_frequency=FS, durations=[1.0])
    rec = NumpyRecording([traces], sampling_frequency=FS, channel_ids=list(range(6)))
    rec.set_probe(probe.get_probe(), in_place=True)
    payload = collect_segment_diagnostics(
        rec, rec, duration_s=0.2, num_chunks=2, seed=0, mad_threshold=5.0,
        dead_noise_ratio=0.1, artifacts_duration_s=0.2, window_s=0.5,
        trace_channels=3, raster_max_channels=6, signal_buffer="memory",
    )
    write_cache(tmp_path, **payload)
    return tmp_path, payload


def test_the_timing_lands_beside_the_cache_and_carries_the_collector_s_own_numbers(computed):
    out_dir, payload = computed
    metrics = payload["metrics"]
    path = write_timing(out_dir, capsule="preprocess_segment",
                        entity={"well": "well000", "rec": "rec0000"},
                        diagnostics=metrics["timings"], buffer=metrics["buffer"],
                        steps={"collect": {"seconds": 1.5, "started": 1.0, "ended": 2.5}})
    assert path == cache_dir(out_dir) / TIMING_NAME
    assert (cache_dir(out_dir) / RECORD_NAME).is_file()  # the cache it describes

    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["capsule"] == "preprocess_segment"
    assert document["entity"] == {"well": "well000", "rec": "rec0000"}
    assert set(document["diagnostics"]) == set(metrics["timings"])
    assert document["buffer"]["mode"] == metrics["buffer"]["mode"]
    for name, entry in {**document["diagnostics"], **document["steps"]}.items():
        assert {"seconds", "started", "ended"} <= set(entry), name
        assert entry["started"] <= entry["ended"], name
    # the cache is untouched by it
    assert read_cache(out_dir, capsule="preprocess_segment")


def test_clearing_is_idempotent_and_never_makes_a_diagnostics_folder(computed, tmp_path):
    out_dir, payload = computed
    write_timing(out_dir, capsule="preprocess_segment", entity={"well": "well000"},
                 diagnostics=payload["metrics"]["timings"])
    assert clear_timing(out_dir) is True
    assert not (cache_dir(out_dir) / TIMING_NAME).exists()
    assert (cache_dir(out_dir) / RECORD_NAME).is_file()  # the cache stands
    assert clear_timing(out_dir) is False

    # an entity whose cache is borrowed from another run: no folder is made, so
    # nothing starts claiming a cache lives here
    borrowed = tmp_path / "borrowed"
    borrowed.mkdir()
    assert clear_timing(borrowed) is False
    assert not cache_dir(borrowed).exists()


def test_a_timing_only_folder_does_not_survive_the_clear(tmp_path):
    """The stale-file rule cannot leave an empty diagnostics/ behind: the run
    layer reads one as 'the cache is local', and a viewer would then look here
    instead of the run this entity borrows from."""
    write_timing(tmp_path, capsule="ingest", entity={"well": "well000"},
                 steps={"common_electrodes": {"seconds": 0.1, "started": 1.0, "ended": 1.1}})
    assert cache_dir(tmp_path).is_dir()
    assert clear_timing(tmp_path) is True
    assert not cache_dir(tmp_path).exists()
