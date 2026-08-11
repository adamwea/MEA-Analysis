"""Canonical reader-facing wording for figures and generated READMEs.

Two rulings from Adam (2026-08-11) land here, and they land here *once* so the
same sentence cannot drift between a figure, its capsule README, and the
report generator:

1. **Every acronym is expanded at least once on the figure.** The README rule
   ("all acronyms fully defined at least once per README", 2026-08-10) now
   covers plots too. The trigger was capsule 05's ``segment_activity.png``,
   which printed a threshold as a MAD multiple with nothing on the figure
   saying what MAD is.

2. **No insider jargon in reader-facing text.** Adam explicitly did not follow
   "within each band" or "never across a seam" — those are our terms of art.
   Anything a competent neuroscientist who has never read this source cannot
   decode gets replaced with plain wording, or defined inline on first use.

The test for anything added here is Adam's: could a neuroscientist who has
never seen our code read the figure unaided? Keep entries terse — an expansion
plus a clause of meaning, not a paragraph. A figure caption is a caption.
"""

# --- Acronyms -------------------------------------------------------------
#
# Expansion + one clause of what it MEANS. Used on first appearance in a
# figure's legend, axis label or caption. Keys are the bare acronym as it
# appears in figure text.
ACRONYMS = {
    "MAD": (
        "MAD = median absolute deviation, the outlier-robust noise estimate "
        "(÷0.6745 gives the equivalent Gaussian sigma)"
    ),
    "RMS": "RMS = root mean square, the typical signal size over a window",
    "PSD": "PSD = power spectral density, signal power split by frequency",
    "ADC": (
        "ADC = analog-to-digital converter; device counts are the raw integers "
        "the chip reports, before any conversion to voltage"
    ),
    "SNR": "SNR = signal-to-noise ratio, spike size divided by the noise level",
    "ISI": "ISI = inter-spike interval, the time between one spike and the next",
    "CMR": (
        "CMR = common median reference, subtracting the median across channels "
        "to remove noise shared by the whole array"
    ),
    "QC": "QC = quality control",
    "PTP": "PTP = peak-to-peak, the full trough-to-peak height of a waveform",
    "CoM": "CoM = centre of mass, the amplitude-weighted average position",
    "a.u.": "a.u. = arbitrary units, a relative score with no physical unit",
    "LSB": "LSB = least significant bit, one step of the device's digital scale",
}


def acronym_note(*names, joiner="  ·  "):
    """Caption fragment defining `names`, in the order given, skipping unknowns.

    Callers list the acronyms their figure actually prints — passing the whole
    table would crowd the figure with definitions of terms that never appear.
    """
    parts = [ACRONYMS[name] for name in names if name in ACRONYMS]
    return joiner.join(parts)


# --- Plain language for our terms of art ----------------------------------
#
# Adam-approved wording, 2026-08-11. Use these verbatim rather than
# re-paraphrasing: the point is that the same explanation appears everywhere.

SEAM = (
    "Segment join: the join between two segments in the concatenated file. The "
    "file looks continuous there, but real time jumps — minutes may have passed "
    "while the chip re-routed electrodes."
)

SEGMENT_BAND = (
    "Segment band: the stretch of the concatenated timeline belonging to one "
    "segment."
)

PER_SEGMENT_ONLY = (
    "Each segment's number is computed from its own samples and its own recorded "
    "duration; nothing is computed across a join, because the gap between one "
    "segment's last spike and the next segment's first is microseconds in the "
    "file but minutes in real time — any rate, interval, correlation or slope "
    "spanning a join would be inventing structure out of the stitching."
)

REAL_ELAPSED_AXIS = (
    "The x axis is real elapsed time: every sample is shifted right by the time "
    "missing before it, and stretches where nothing was recorded are shaded, so "
    "a shaded band means 'not recorded', not 'silent'."
)

CONTIGUOUS_AXIS = (
    "The x axis is position in the recorded file, not real elapsed time: every "
    "stretch the instrument did not record has been removed, so two points "
    "either side of a break look adjacent when they are not."
)

REPRESENTATIVE_CHANNELS = (
    "Representative channels: the electrodes sit in tight clumps, so one channel "
    "is taken from each clump and those are ranked by how large their signal is, "
    "loudest first."
)

NO_DATA_SHADING = "Shaded: no data was recorded over this stretch."

BACKBONE_CHANNELS = (
    "Backbone electrodes: the electrodes that stayed routed in every segment, so "
    "they are the only ones present throughout the whole recording."
)

DENSE_STITCH = (
    "Stitched footprint: each segment records a different subset of electrodes, "
    "so the per-segment footprints are combined into one array-wide footprint."
)

PROXY_NOT_MODEL = (
    "This is a descriptive summary of the recorded data, not a fitted model — "
    "read it as a review aid, not as validation."
)


__all__ = [
    "ACRONYMS",
    "acronym_note",
    "SEAM",
    "SEGMENT_BAND",
    "PER_SEGMENT_ONLY",
    "REAL_ELAPSED_AXIS",
    "CONTIGUOUS_AXIS",
    "REPRESENTATIVE_CHANNELS",
    "NO_DATA_SHADING",
    "BACKBONE_CHANNELS",
    "DENSE_STITCH",
    "PROXY_NOT_MODEL",
]
