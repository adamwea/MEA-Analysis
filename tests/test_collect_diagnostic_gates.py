"""Per-diagnostic switches, and the record a switched-off diagnostic leaves.

The point of the registry is that the list of diagnostics exists ONCE. These
tests hold that: the names the config can offer are the names this function
answers to, and a name that is neither is an error rather than a silent no-op.
"""
import numpy as np
import pytest
from spikeinterface.core import NumpyRecording, generate_recording

from mea_modules.diagnostics.collect import (
    SEGMENT_DIAGNOSTIC_NAMES,
    SEGMENT_DIAGNOSTICS,
    collect_segment_diagnostics,
)

FS = 10_000.0


@pytest.fixture
def pair():
    """`(raw_view, qc_rec)` -- two objects, as the real caller passes."""
    rng = np.random.default_rng(0)
    n = int(1.0 * FS)
    traces = rng.normal(0.0, 5.0, size=(n, 6)).astype(np.float32)
    ids = list(range(6))
    probe = generate_recording(num_channels=6, sampling_frequency=FS, durations=[1.0])
    rec = NumpyRecording([traces], sampling_frequency=FS, channel_ids=ids)
    rec.set_probe(probe.get_probe(), in_place=True)
    return rec, rec


def _collect(pair, **kwargs):
    raw, qc = pair
    base = dict(
        duration_s=0.2, num_chunks=2, seed=0, mad_threshold=5.0,
        dead_noise_ratio=0.1, artifacts_duration_s=0.2, window_s=0.5,
        trace_channels=3, raster_max_channels=6,
    )
    base.update(kwargs)
    return collect_segment_diagnostics(raw, qc, **base)


def test_the_registry_is_the_one_list_of_diagnostics(pair):
    """Every declared name is a name the collector answers to, and nothing the
    collector computes is missing from the list. A config generated from the
    registry therefore cannot offer a setting that does nothing."""
    assert SEGMENT_DIAGNOSTIC_NAMES == tuple(s.name for s in SEGMENT_DIAGNOSTICS)
    payload = _collect(pair, enabled=set(SEGMENT_DIAGNOSTIC_NAMES))
    assert payload["metrics"]["skipped"] == {}
    # Every name either timed or dependency-skipped; none silently absent.
    timed = set(payload["metrics"]["timings"])
    for name in SEGMENT_DIAGNOSTIC_NAMES:
        assert any(key == name or key.startswith(f"{name}_") for key in timed), name


def test_an_unknown_name_is_refused_rather_than_ignored(pair):
    """A typo that silently switches nothing off is worse than a crash: the run
    looks like it honoured a setting it never saw."""
    with pytest.raises(ValueError, match="unknown diagnostics"):
        _collect(pair, enabled={"raster", "rastre"})


def test_default_is_everything_the_registry_defaults_to(pair):
    both = (_collect(pair, enabled=None), _collect(pair))
    for payload in both:
        assert payload["metrics"]["skipped"] == {}


def test_a_switched_off_diagnostic_records_why_and_costs_nothing(pair):
    payload = _collect(pair, enabled=set(SEGMENT_DIAGNOSTIC_NAMES) - {"raster", "artifacts"})
    metrics = payload["metrics"]

    assert metrics["skipped"]["raster"] == "not requested"
    assert metrics["skipped"]["artifacts"] == "not requested"
    assert "raster" not in payload["events"]
    assert metrics["artifacts"] is None
    # Not an error. The two are different facts about a run.
    assert "raster" not in metrics["errors"]
    assert "artifacts" not in metrics["errors"]
    # And no time was spent on either.
    assert "raster" not in metrics["timings"]
    assert "artifacts" not in metrics["timings"]


def test_the_old_skip_keys_still_answer_for_the_two_the_tool_reads_by_name(pair):
    """`raster_skipped` / `artifacts_skipped` are what the segment tool reads to
    tell a reader why a figure is missing. They are now a view of `skipped`."""
    payload = _collect(pair, enabled=set(SEGMENT_DIAGNOSTIC_NAMES) - {"raster"})
    metrics = payload["metrics"]
    assert metrics["raster_skipped"] == metrics["skipped"]["raster"]


def test_a_dependency_that_was_switched_off_skips_its_dependants_with_the_reason(pair):
    """activity thresholds off the noise estimate. With noise off there is
    nothing to threshold against, and saying so beats reporting a bare None."""
    payload = _collect(pair, enabled=set(SEGMENT_DIAGNOSTIC_NAMES) - {"noise"})
    skipped = payload["metrics"]["skipped"]
    assert skipped["noise"] == "not requested"
    assert "noise" in skipped["activity"]
    assert "noise" in skipped["flags"]
    assert payload["metrics"]["activity"] is None


def test_every_diagnostic_is_timed_including_one_that_fails(pair):
    """How long something took to NOT work is part of what a profile is for;
    a missing entry would read as free."""
    raw, qc = pair

    class Broken:
        """Answers geometry, raises on traces -- the shape a dead reader has."""

        def __getattr__(self, name):
            return getattr(qc, name)

        def get_traces(self, *args, **kwargs):
            raise RuntimeError("reader is gone")

    payload = _collect((Broken(), qc), enabled={"clipping"})
    metrics = payload["metrics"]
    assert "clipping" in metrics["timings"]
    assert "clipping" in metrics["errors"]
    assert metrics["timings"]["clipping"] >= 0.0


def test_the_cache_meta_records_what_was_asked_for(pair):
    """A later reader must be able to tell a cache that is thin by choice from
    one that is thin because something broke."""
    payload = _collect(pair, enabled={"noise", "clipping"})
    asked = payload["meta"]["diagnostics"]
    assert set(asked) == set(SEGMENT_DIAGNOSTIC_NAMES)
    assert asked["noise"] is True and asked["clipping"] is True
    assert asked["raster"] is False and asked["spectra"] is False


def test_the_detected_rate_recorded_is_the_rate_achieved_not_the_one_asked_for(pair):
    payload = _collect(pair, enabled={"raster"}, raster_downsample_hz=3000.0)
    raster = payload["metrics"]["raster"]
    assert raster["decimation_factor"] == 3
    assert raster["detection_hz"] == pytest.approx(FS / 3)
    assert payload["meta"]["raster_downsample_hz"] == 3000.0


def test_no_downsample_records_the_native_rate_and_factor_one(pair):
    payload = _collect(pair, enabled={"raster"})
    raster = payload["metrics"]["raster"]
    assert raster["decimation_factor"] == 1
    assert raster["detection_hz"] == pytest.approx(FS)
    assert payload["meta"]["raster_downsample_hz"] is None
