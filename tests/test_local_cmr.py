"""Local common-median-reference tests on a small synthetic geometry.

Regression suite for Adam's 2026-08-11 ruling (R-A/R-B): the default
``local_radius`` moved from the older build's zero-width ``(250, 250)`` annulus
— empty on every geometry, so ``common_reference('local')`` ALWAYS raised and
the global-median fallback is what actually ran — to a genuine ``(0, 250)``
local reference. These tests prove, on a geometry small enough to reason about
by hand:

1. the ``(0, 250)`` default yields a NON-EMPTY neighbour set for every channel
   (the geometric fact the zero-width trap violated),
2. ``common_median_reference`` applies the LOCAL reference without falling
   back, and truthfully says so via the honest-provenance annotations
   (``reference_requested`` / ``reference_effective`` / ``reference_fallback``),
3. the local result genuinely differs from a global median reference on a grid
   whose extent exceeds the radius (i.e. the fix changed real output, it did
   not just relabel the fallback),
4. the legacy zero-width radius still reproduces the historical behaviour:
   fallback to global, recorded honestly,
5. :func:`mea_modules.preprocessing.preprocess_segment` carries the provenance
   through to the final (cast) recording, and omits it when
   ``apply_reference=False``.

The grid: 4x4 channels at 100 um pitch. Nearest-neighbour distance 100 um
(inside the 250 um radius, so every channel has neighbours), grid diagonal
~424 um (outside it, so no channel's local set is the whole array — which is
what makes local vs global distinguishable in the output).
"""

import numpy as np
import pytest

sc = pytest.importorskip("spikeinterface.core")
spre = pytest.importorskip("spikeinterface.preprocessing")

from mea_modules.preprocessing import (  # noqa: E402
    common_median_reference,
    preprocess_segment,
    reference_provenance,
)
from mea_modules.preprocessing.filters import (  # noqa: E402
    DEFAULT_LOCAL_RADIUS,
    LEGACY_ZERO_WIDTH_RADIUS,
)

FS = 20000.0
DURATION_S = 0.2
N_SIDE = 4
PITCH_UM = 100.0
N_CHANNELS = N_SIDE * N_SIDE


def _grid_locations():
    rows, cols = np.meshgrid(np.arange(N_SIDE), np.arange(N_SIDE), indexing="ij")
    xs = cols.ravel().astype(float) * PITCH_UM
    ys = rows.ravel().astype(float) * PITCH_UM
    return np.stack([xs, ys], axis=1)


@pytest.fixture()
def grid_recording():
    recording = sc.generate_recording(
        num_channels=N_CHANNELS, sampling_frequency=FS, durations=[DURATION_S], seed=7,
    )
    recording.set_dummy_probe_from_locations(_grid_locations())
    return recording


def test_default_radius_is_zero_to_250():
    """The ruling itself: (0, 250), not the legacy zero-width annulus."""
    assert tuple(DEFAULT_LOCAL_RADIUS) == (0.0, 250.0)
    assert tuple(LEGACY_ZERO_WIDTH_RADIUS) == (250.0, 250.0)


def test_default_radius_yields_nonempty_neighbor_sets():
    """Every channel has >= 1 neighbour in the (0, 250] um annulus — the
    geometric precondition the zero-width default violated for ALL channels."""
    locations = _grid_locations()
    exclude, include = DEFAULT_LOCAL_RADIUS
    distances = np.linalg.norm(locations[:, None, :] - locations[None, :, :], axis=2)
    for channel in range(N_CHANNELS):
        others = np.delete(distances[channel], channel)
        neighbors = np.count_nonzero((others > exclude) & (others <= include))
        assert neighbors >= 1, f"channel {channel} has an empty local reference set"

    # And the legacy annulus is empty everywhere — the documented trap.
    legacy_exclude, legacy_include = LEGACY_ZERO_WIDTH_RADIUS
    for channel in range(N_CHANNELS):
        others = np.delete(distances[channel], channel)
        assert np.count_nonzero((others > legacy_exclude) & (others <= legacy_include)) == 0


def test_local_reference_applies_without_fallback(grid_recording):
    """The default chain genuinely runs LOCAL: no exception, no fallback, and
    the provenance annotations say local/local/False."""
    referenced = common_median_reference(grid_recording)
    provenance = reference_provenance(referenced)
    assert provenance == {
        "reference_requested": "local",
        "reference_effective": "local",
        "reference_fallback": False,
    }

    traces = referenced.get_traces(start_frame=0, end_frame=200)
    assert traces.shape == (200, N_CHANNELS)
    assert np.all(np.isfinite(traces))

    # Local must DIFFER from global on this geometry (diagonal > radius), or
    # the "fix" would just be relabeling the fallback.
    global_traces = spre.common_reference(
        grid_recording, reference="global", operator="median"
    ).get_traces(start_frame=0, end_frame=200)
    assert not np.allclose(traces, global_traces), (
        "local-median output is identical to global-median output — the local "
        "reference cannot actually be applying"
    )


def test_legacy_zero_width_radius_falls_back_and_says_so(grid_recording):
    """The historical (250, 250) behaviour, now recorded honestly: the empty
    annulus fails, the global fallback runs, provenance reports it."""
    referenced = common_median_reference(
        grid_recording, local_radius=LEGACY_ZERO_WIDTH_RADIUS
    )
    provenance = reference_provenance(referenced)
    assert provenance == {
        "reference_requested": "local",
        "reference_effective": "global",
        "reference_fallback": True,
    }


def test_preprocess_segment_carries_provenance_to_final_recording(grid_recording):
    """The full standard chain ends float32 AND still carries the reference
    provenance on the final wrapper (a cast must not drop it)."""
    processed = preprocess_segment(grid_recording, rename_to_electrodes=False)
    assert str(processed.get_dtype()) == "float32"
    provenance = reference_provenance(processed)
    assert provenance == {
        "reference_requested": "local",
        "reference_effective": "local",
        "reference_fallback": False,
    }


def test_preprocess_segment_without_reference_carries_no_provenance(grid_recording):
    processed = preprocess_segment(
        grid_recording, apply_reference=False, rename_to_electrodes=False
    )
    provenance = reference_provenance(processed)
    assert provenance == {
        "reference_requested": None,
        "reference_effective": None,
        "reference_fallback": None,
    }


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
