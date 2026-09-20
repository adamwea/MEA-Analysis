"""Canonical reader-facing wording for figures and generated READMEs.

Two reviewer rulings (2026-08-11) land here, and they land here *once* so the
same sentence cannot drift between a figure, its capsule README, and the
report generator:

1. **Every acronym is expanded at least once on the figure.** The README rule
   ("all acronyms fully defined at least once per README", 2026-08-10) now
   covers plots too. The trigger was capsule 05's ``segment_activity.png``,
   which printed a threshold as a MAD multiple with nothing on the figure
   saying what MAD is.

2. **No insider jargon in reader-facing text.** The reviewer did not follow
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
    "SD": "SD = standard deviation, the spread of the measurements themselves",
    "SEM": (
        "SEM = standard error of the mean, how precisely the mean is known "
        "(SD divided by the square root of the count)"
    ),
}


# The SHORT form: the expansion and nothing else. A figure legend is not the
# place for the explanatory clause -- a legend key reads the way it would in a
# published figure, and a three-line gloss inside the box is the "too much
# partial sentence" the review keeps catching (2026-09-20). The long form above
# still carries the meaning, and the README is where every acronym is defined
# in full, so nothing is lost by keeping the box to one line.
ACRONYMS_SHORT = {
    "MAD": "MAD = median absolute deviation",
    "RMS": "RMS = root mean square",
    "PSD": "PSD = power spectral density",
    "ADC": "ADC = analog-to-digital converter",
    "SNR": "SNR = signal-to-noise ratio",
    "ISI": "ISI = inter-spike interval",
    "CMR": "CMR = common median reference",
    "QC": "QC = quality control",
    "PTP": "PTP = peak-to-peak",
    "CoM": "CoM = centre of mass",
    "a.u.": "a.u. = arbitrary units",
    "LSB": "LSB = least significant bit",
    "SD": "SD = standard deviation",
    "SEM": "SEM = standard error of the mean",
}


def acronym_note(*names, joiner="  ·  ", short=False):
    """Caption fragment defining `names`, in the order given, skipping unknowns.

    Callers list the acronyms their figure actually prints — passing the whole
    table would crowd the figure with definitions of terms that never appear.

    `short` takes the expansion alone, without the clause of meaning, which is
    what belongs inside a legend box; the default long form is for a caption or
    a README, where there is room for it.
    """
    table = ACRONYMS_SHORT if short else ACRONYMS
    parts = [table[name] for name in names if name in table]
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


# --- Legend entries and axis labels ---------------------------------------
#
# Publication shorthand (review ruling, 2026-09-19): "use legend descriptions
# as we'd expect to see them in a publication. Descriptive and clear, but as
# short-handed as possible. Not even a partial sentence."
#
# The prose blocks above are NOT the fallback for these. They explain a term to
# someone meeting it for the first time and belong in a README or a report; a
# legend key is two or three words. The trigger was `segment_activity`'s
# "one electrode's own rate" — a sentence fragment doing a legend's job.
#
# The information a longer label used to carry now lives in two other places,
# both of which a reader can reach: the figure's own self-describing filename
# (`..._2seg_realtime.png`), and the diagnostic JSON beside it.

# The two gap kinds. These are genuinely different events — a frame-counter
# break inside one segment lasts microseconds, a between-segment gap lasts as
# long as the chip needed to re-route — so they keep separate keys, but the
# explanation of WHY moved out of the legend.
GAP_WITHIN = "within-segment gap"
GAP_BETWEEN = "between-segment gap"

# A join is one instant on the file timeline and a span on the real-elapsed
# one, so it needs two labels rather than one.
JOIN_INSTANT = "segment join"
JOIN_SPANNING = "segment end / start"

# Axis labels for the two timelines. The distinction the old labels spelled out
# ("not real elapsed time") is carried by the filename token, `filetime` vs
# `realtime`, and by these two labels being visibly different.
FILE_TIME_AXIS = "file time (s)"
REAL_TIME_AXIS = "elapsed time (s)"

# One bar per segment, left to right in the order they were recorded. The
# clause that used to spell that out ("one recording configuration, in the
# order ...") was a full sentence doing an axis label's job; the ordering is
# also the drawing order, so a reader sees it, and the fuller explanation
# belongs in the figure's caption instead.
SEGMENT_AXIS = "segment (recording order)"

# Channel-selection keys.
TRACED_CHANNELS = "traced channels"
REPRESENTATIVE_KEY = "representative channels"
BACKBONE_KEY = "shared electrodes"

# Per-electrode summary keys, for a bar chart carrying scatter and a whisker.
PER_ELECTRODE = "per electrode"
MEAN_SD = "mean ± SD"
MEAN_SEM = "mean ± SEM"


def dispersion_key(kind):
    """Legend key for an error bar: `kind` is "sd" or "sem"."""
    return MEAN_SEM if str(kind).lower() == "sem" else MEAN_SD


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
    "GAP_WITHIN",
    "GAP_BETWEEN",
    "JOIN_INSTANT",
    "JOIN_SPANNING",
    "FILE_TIME_AXIS",
    "REAL_TIME_AXIS",
    "SEGMENT_AXIS",
    "TRACED_CHANNELS",
    "REPRESENTATIVE_KEY",
    "BACKBONE_KEY",
    "PER_ELECTRODE",
    "MEAN_SD",
    "MEAN_SEM",
    "dispersion_key",
]
