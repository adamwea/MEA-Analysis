"""The PSD pool is one channel per electrode cluster, not every shared electrode.

`welch_spectra` documents the pool it wants -- one representative per electrode
cluster, drawn from the shared set -- and the collector was handing it the
shared set itself. On an AxonTracking well routed as 30 patches of 12 that is
354 estimates where 30 were meant: twelve neighbours a few micrometres apart do
not carry twelve different spectra, the panel cannot draw 354 curves apart, and
every one of them was paid for.

The guard that matters is the one that runs the collector and looks at the pool
it actually used. Testing `cluster_center_channels` alone would pass just as
well with the reduction unwired, which is the state this fixes.
"""
import numpy as np
import pytest

from mea_modules.diagnostics import collect_segment_diagnostics
from mea_modules.diagnostics.electrodes import cluster_center_channels

FS_HZ = 10_000.0
N_FRAMES = 30_000
PATCHES = 6
PER_PATCH = 4
PITCH_UM = 17.5
PATCH_GAP_UM = 400.0


def _locations():
    """Dense patches far apart, the shape a routed AxonTracking well has."""
    out = []
    for patch in range(PATCHES):
        cx = (patch % 3) * PATCH_GAP_UM
        cy = (patch // 3) * PATCH_GAP_UM
        for member in range(PER_PATCH):
            out.append([cx + (member % 2) * PITCH_UM, cy + (member // 2) * PITCH_UM])
    return np.asarray(out, dtype=float)


def _recording():
    from probeinterface import Probe
    from spikeinterface.core import NumpyRecording

    n_channels = PATCHES * PER_PATCH
    rng = np.random.default_rng(3)
    traces = rng.normal(0.0, 8.0, (N_FRAMES, n_channels)).astype(np.float32)
    traces[500:20_000:311, :] = -180.0

    recording = NumpyRecording([traces], sampling_frequency=FS_HZ)
    probe = Probe(ndim=2)
    probe.set_contacts(positions=_locations(), shapes="square",
                       shape_params={"width": 5})
    probe.set_device_channel_indices(np.arange(n_channels))
    return recording.set_probe(probe)


@pytest.fixture(scope="module")
def recording():
    return _recording()


def _collect(recording, pool):
    return collect_segment_diagnostics(
        recording, recording,
        source="preprocessed",
        duration_s=1.0, num_chunks=2, seed=5,
        mad_threshold=5.0, dead_noise_ratio=0.1,
        artifacts_duration_s=1.0, window_s=1.0,
        trace_channels=4, raster_max_channels=0,
        psd_channel_ids=pool,
    )


def test_the_shared_set_is_reduced_to_one_channel_per_patch(recording):
    """The wiring, not the helper: what pool did the collector actually use?"""
    pool = list(recording.get_channel_ids())
    assert len(pool) == PATCHES * PER_PATCH, "fixture is not dense; nothing to reduce"

    collected = _collect(recording, pool)
    used = collected["metrics"]["psd_channels"]

    assert len(used) == PATCHES, (
        f"handed {len(pool)} shared electrodes in {PATCHES} patches, the "
        f"collector estimated {len(used)} -- the cluster reduction is not wired"
    )
    assert set(map(str, used)) <= set(map(str, pool))
    # one from each patch, not several from one
    order = {str(cid): index for index, cid in enumerate(pool)}
    assert len({order[str(cid)] // PER_PATCH for cid in used}) == PATCHES


def test_the_spectra_carry_one_curve_per_patch(recording):
    """The saving is in the ESTIMATE, not only in what gets drawn."""
    pool = list(recording.get_channel_ids())
    collected = _collect(recording, pool)
    for name, block in collected["spectra"].items():
        assert block["power"].shape[1] == PATCHES, (
            f"{name} holds {block['power'].shape[1]} curves, expected {PATCHES}"
        )
        assert len(block["channel_ids"]) == PATCHES


def test_a_probe_without_locations_keeps_every_channel(recording):
    """Fewer curves is a choice; no curves is a loss."""
    pool = list(recording.get_channel_ids())

    class _NoLocations:
        def __getattr__(self, name):
            return getattr(recording, name)

        def get_channel_locations(self):
            raise AttributeError("no probe")

    assert list(cluster_center_channels(_NoLocations(), pool)) == list(pool)
