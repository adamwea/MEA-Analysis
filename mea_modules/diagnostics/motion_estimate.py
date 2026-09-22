"""Drift of the recording over time, estimated with SpikeInterface.

The estimate the concatenated well's collector caches when motion is switched
on; :mod:`.motion` draws it and imports this back.
"""

import logging

from mea_modules.quality import DEFAULT_SEED

logger = logging.getLogger(__name__)


# Ported verbatim from the retired capsule. These ARE the scientific settings;
# changing one changes the result, so they are named here rather than inlined.
_DETECT_METHOD = "locally_exclusive"


_DETECT_THRESHOLD = 5.0


_LOCALIZE_METHOD = "center_of_mass"


_ESTIMATE_METHOD = "decentralized"


_RIGID = True


METHOD_CHAIN = (
    "detect_peaks(locally_exclusive) -> center_of_mass -> "
    "estimate_motion(decentralized, rigid)"
)


TIMELINE_CAVEAT = (
    "Displacement is on the concatenated file timeline, not real elapsed time. "
    "A step at a segment join is an electrode re-routing artifact until proven "
    "otherwise; read the trace within each segment band."
)


def estimate_motion_over_recording(recording, well=None, seed=DEFAULT_SEED):
    """Rigid drift estimate over `recording`. Returns a JSON-serializable dict.

    Keys: ``well``, ``method``, ``n_peaks``, ``max_abs_displacement_um``,
    ``temporal_bins_s``, ``displacement_um``, ``seed``, ``note``.

    `seed` fixes the chunks the noise estimate is drawn from. SpikeInterface
    defaults that picker to ``seed=None``, which draws from OS entropy and does
    NOT follow the process-wide numpy seed -- so without this argument two runs
    over the same recording give different noise levels, hence different peaks,
    hence a different displacement trace. Detection and the estimator itself are
    deterministic once the noise levels are fixed.

    Unit coherence, ported with the code because it is easy to get wrong: noise
    levels are taken in DEVICE COUNTS, because ``detect_peaks`` cuts the
    recording's UNSCALED traces against ``detect_threshold * noise``. Passing
    microvolt noise here would scale the threshold by the gain (~6.3x on this
    hardware) and miss real peaks. The figure's axes are µm against seconds and
    never see the unit either way.
    """
    import numpy as np
    from spikeinterface.core import get_noise_levels
    from spikeinterface.sortingcomponents.motion import estimate_motion
    from spikeinterface.sortingcomponents.peak_detection import detect_peaks
    from spikeinterface.sortingcomponents.peak_localization import localize_peaks

    noise = get_noise_levels(
        recording, return_in_uV=False, random_slices_kwargs={"seed": int(seed)}
    )
    peaks = detect_peaks(
        recording,
        method=_DETECT_METHOD,
        detect_threshold=_DETECT_THRESHOLD,
        noise_levels=noise,
    )
    locations = localize_peaks(recording, peaks, method=_LOCALIZE_METHOD)
    motion = estimate_motion(
        recording, peaks, locations, method=_ESTIMATE_METHOD, rigid=_RIGID
    )

    displacement = np.atleast_1d(np.asarray(motion.displacement[0]).squeeze())
    bins_s = np.asarray(motion.temporal_bins_s[0])
    summary = {
        "well": well,
        "method": METHOD_CHAIN,
        "n_peaks": int(len(peaks)),
        "seed": int(seed),
        "max_abs_displacement_um": (
            float(np.nanmax(np.abs(displacement))) if displacement.size else None
        ),
        "note": TIMELINE_CAVEAT,
        "temporal_bins_s": [float(b) for b in bins_s],
        "displacement_um": [float(d) for d in displacement],
    }
    logger.info(
        "motion estimate for %s: %d peaks, max |displacement| %s um",
        well, summary["n_peaks"], summary["max_abs_displacement_um"],
    )
    return summary
