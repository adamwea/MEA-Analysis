"""Filter and reference steps, composed into the standard segment recipe.

Every function here returns a *lazy* SpikeInterface recording — nothing reads
traces and nothing touches disk. A caller can therefore build the whole chain on
a 21-segment AxonTracking scan for the cost of metadata reads, and only pay for
samples at the point where something downstream actually pulls them.

The recipe in :func:`preprocess_segment` is a direct port of the older build's
``apply_standard_preprocessing``; the ordering is load-bearing and is spelled out
in that function's docstring. Each step is also exported on its own so callers
who need a different chain are not forced to fork the whole thing.
"""

import logging

logger = logging.getLogger(__name__)

# Maxwell writes unsigned counts. Filtering those without a signed cast wraps
# around at zero, so the cast has to happen before anything subtracts.
_UNSIGNED_PREFIX = "uint"

DEFAULT_FREQ_MIN = 300.0
DEFAULT_FREQ_MAX = 6000.0
DEFAULT_DTYPE = "float32"

# SpikeInterface reads local_radius as (exclude, include): the reference set is
# the annulus `exclude < distance <= include` micrometres. (0, 250) is a genuine
# local reference -- for each channel, the median over ALL neighbours within
# 250 um.
#
# HISTORY (keep documented; do not re-simplify). The older build shipped
# (250, 250) -- a zero-width annulus, empty for every channel on every
# geometry -- so common_reference('local') always raised and the global-median
# fallback below is what actually ran on every segment of every scan that
# build processed. That trap was carried into this port verbatim (as
# DEFAULT_LOCAL_RADIUS) to reproduce those results exactly, until Adam's
# 2026-08-11 ruling: genuine local CMR becomes the default. Every run built
# before 2026-08-11 was therefore EFFECTIVELY GLOBAL-median referenced no
# matter what "local" its config recorded; preprocessed traces from this
# default onward legitimately differ from every prior run.
# LEGACY_ZERO_WIDTH_RADIUS reproduces the old (effectively global) behaviour
# for anyone who needs a byte-faithful replay of a pre-ruling chain.
DEFAULT_LOCAL_RADIUS = (0.0, 250.0)
LOCAL_RADIUS_250UM = (0.0, 250.0)  # pre-ruling opt-in name; now equals the default
LEGACY_ZERO_WIDTH_RADIUS = (250.0, 250.0)

# Annotation keys `common_median_reference` stamps on the recording it returns,
# recording what ACTUALLY ran (honest provenance -- Adam ruling R-B,
# 2026-08-11). Read them back with :func:`reference_provenance`.
REFERENCE_ANNOTATIONS = (
    "reference_requested",
    "reference_effective",
    "reference_fallback",
)

# The old loader centred each segment before filtering, using a chunk one shy of
# 10k samples. Kept as a default so `center` reproduces that behaviour.
DEFAULT_CENTER_CHUNK_SIZE = 10_000
_CENTER_CHUNK_MARGIN = 100
_MIN_CENTER_CHUNK = 100


def _dtype_name(recording):
    """dtype as a plain string, or "" when the extractor cannot report one."""
    try:
        return str(recording.get_dtype())
    except Exception:  # pragma: no cover - defensive; some extractors lack dtype
        return ""


def ensure_signed(recording):
    """Cast an unsigned recording to signed, otherwise pass it through.

    Guard, not a blanket cast: applying ``unsigned_to_signed`` to something that
    is already signed shifts the baseline by half the dynamic range. Only the
    ``uint*`` case is converted.

    ``unsigned_to_signed`` shifts the DATA down by half the dtype's range
    (2**15 counts for uint16) but copies ``offset_to_uV`` verbatim, which
    silently breaks ``return_in_uV`` on the signed view: microvolts built from
    the shifted counts land ~2**15 gains below what the unsigned recording
    reported (~-203 mV on Maxwell data that should read ~0 µV). The cast here
    compensates — the signed view's ``offset_to_uV`` is bumped by
    ``2**(bits-1) * gain`` so ``gain*x + offset`` maps to the SAME microvolts
    before and after the cast. Downstream filters still zero the offset (they
    remove the DC it describes), so the preprocessed chain is unaffected.
    """
    import numpy as np
    import spikeinterface.preprocessing as spre

    dtype_name = _dtype_name(recording)
    if not dtype_name.startswith(_UNSIGNED_PREFIX):
        return recording
    signed = spre.unsigned_to_signed(recording)

    gains = signed.get_property("gain_to_uV")
    if gains is not None:
        # Mirror unsigned_to_signed's own shift: bit_depth=None means it
        # subtracts half the STORAGE dtype's range, whatever the ADC used.
        shift_counts = float(2 ** (np.dtype(dtype_name).itemsize * 8 - 1))
        gains = np.asarray(gains, dtype="float64")
        offsets = signed.get_property("offset_to_uV")
        offsets = (
            np.zeros_like(gains) if offsets is None else np.asarray(offsets, dtype="float64")
        )
        signed.set_property("offset_to_uV", offsets + shift_counts * gains)
    return signed


def highpass(recording, freq_min=DEFAULT_FREQ_MIN, **filter_kwargs):
    """High-pass filter at `freq_min` Hz (300 Hz is the standard spike band edge)."""
    import spikeinterface.preprocessing as spre

    return spre.highpass_filter(recording, freq_min=float(freq_min), **filter_kwargs)


def bandpass(recording, freq_min=DEFAULT_FREQ_MIN, freq_max=DEFAULT_FREQ_MAX, **filter_kwargs):
    """Band-pass filter between `freq_min` and `freq_max` Hz.

    Not part of the standard chain — that one high-passes only, leaving the top
    of the band alone — but offered for callers who want the classic spike band.
    """
    import spikeinterface.preprocessing as spre

    return spre.bandpass_filter(
        recording,
        freq_min=float(freq_min),
        freq_max=float(freq_max),
        **filter_kwargs,
    )


def _annotate_reference(recording, requested, effective, fallback):
    """Stamp the honest-provenance annotations; advisory, never fatal."""
    try:
        recording.annotate(
            reference_requested=requested,
            reference_effective=effective,
            reference_fallback=bool(fallback),
        )
    except Exception:  # pragma: no cover - annotation is advisory only
        logger.debug("Could not annotate reference provenance on %r", type(recording).__name__)


def reference_provenance(recording):
    """What :func:`common_median_reference` actually did, read off `recording`.

    Returns ``{"reference_requested": ..., "reference_effective": ...,
    "reference_fallback": ...}`` with ``None`` for any key the recording does
    not carry (e.g. a chain built with ``apply_reference=False``, or a
    recording that never went through this module). Capsules record these
    fields into their descriptors so what a run REPORTS is what actually ran
    (Adam ruling R-B, 2026-08-11).
    """
    annotations = getattr(recording, "_annotations", None) or {}
    return {key: annotations.get(key) for key in REFERENCE_ANNOTATIONS}


def common_median_reference(
    recording,
    reference="local",
    operator="median",
    local_radius=DEFAULT_LOCAL_RADIUS,
    fallback_to_global=True,
):
    """Common reference, local median by default, with a global-median fallback.

    A *local* reference subtracts the median of the channels in the annulus
    ``local_radius[0] < distance <= local_radius[1]`` micrometres from each
    channel, rather than the median of the whole array. On a dense MEA that
    matters: a global reference subtracts real axonal signal picked up by
    neighbouring electrodes, whereas a local one removes only the noise those
    neighbours share.

    The default ``local_radius`` is a GENUINE local reference since Adam's
    2026-08-11 ruling: ``(0, 250)`` — all neighbours within 250 um. (Before
    that ruling the default was the older build's zero-width ``(250, 250)``
    annulus, which always failed into the global fallback — see
    ``DEFAULT_LOCAL_RADIUS`` / ``LEGACY_ZERO_WIDTH_RADIUS`` above for the full
    history. With a working default, the fallback firing is now a REAL
    anomaly, not the normal path.)

    The fallback exists because a global median reference is far better than no
    referencing at all; the failure is logged at WARNING and the chain continues.
    Set `fallback_to_global` False to surface the failure instead.

    Honest provenance (ruling R-B): the returned recording is annotated with
    ``reference_requested`` / ``reference_effective`` / ``reference_fallback``
    recording what ACTUALLY ran — read them back via
    :func:`reference_provenance`.
    """
    import spikeinterface.preprocessing as spre

    try:
        referenced = spre.common_reference(
            recording,
            reference=reference,
            operator=operator,
            local_radius=local_radius,
        )
    except Exception as exc:
        if not fallback_to_global or reference == "global":
            raise
        logger.warning(
            "Local common_reference FAILED; falling back to global median reference "
            "— with the (0, 250) default radius this is a real anomaly, not the "
            "expected path (%s)",
            exc,
        )
        referenced = spre.common_reference(recording, reference="global", operator=operator)
        _annotate_reference(referenced, requested=reference, effective="global", fallback=True)
        return referenced

    _annotate_reference(referenced, requested=reference, effective=reference, fallback=False)
    return referenced


def center(recording, chunk_size=DEFAULT_CENTER_CHUNK_SIZE):
    """Subtract each channel's offset, estimated from a chunk of `chunk_size`.

    The older build ran this at load time, before the filter chain, so raw
    Maxwell offsets never reached the filters. It is exported separately because
    it belongs to whoever opens the segment, not to the filter recipe — see
    `center_chunk_size` on :func:`preprocess_segment` to fold it back in.

    The chunk is clamped to the recording length less a small margin: asking for
    more samples than exist raises inside SpikeInterface.
    """
    import spikeinterface.preprocessing as spre

    n_samples = int(recording.get_num_samples())
    chunk = min(int(chunk_size), n_samples) - _CENTER_CHUNK_MARGIN
    chunk = max(chunk, _MIN_CENTER_CHUNK)
    return spre.center(recording, chunk_size=chunk)


def to_float32(recording, dtype=DEFAULT_DTYPE):
    """Cast to `dtype` unless already there.

    Re-casting a recording that is already float32 buys nothing but an extra
    lazy wrapper in the provenance chain, so the no-op case is skipped.
    """
    import spikeinterface.preprocessing as spre

    if _dtype_name(recording) == str(dtype):
        return recording
    return spre.astype(recording, str(dtype))


def rename_channels_to_electrodes(recording):
    """Re-key a recording's channels by ELECTRODE id.

    SpikeInterface labels channels with the amplifier channel index, which is an
    artefact of how the chip was routed for that particular recording. The same
    physical electrode therefore appears under different channel ids in different
    AxonTracking configurations, and the electrode id is the only identifier
    stable across them.

    Everything downstream that compares segments — above all the common-electrode
    intersection that concatenation depends on — is expressed in electrode ids,
    so segments must be re-keyed before any of it is meaningful.

    Ported from the older build's loader, including both guards: duplicate
    electrodes make the mapping ambiguous, and a rename that does not land
    exactly is worse than none at all, because it silently misaligns channels.
    """
    import numpy as np

    contact_vector = recording.get_property("contact_vector")
    if contact_vector is None:
        raise ValueError("recording has no contact_vector; cannot map channels to electrodes")

    electrodes = np.asarray(contact_vector["electrode"], dtype=int)
    if np.unique(electrodes).size != electrodes.size:
        raise ValueError(
            "duplicate electrode ids in contact_vector; cannot map channels to "
            "electrodes reliably"
        )

    renamed = recording.rename_channels([int(value) for value in electrodes])

    got = np.asarray(renamed.get_channel_ids())
    if got.shape != electrodes.shape or not np.array_equal(got.astype(int), electrodes):
        raise RuntimeError("failed to rename channel ids to electrode ids")
    return renamed


def preprocess_segment(
    recording,
    freq_min=DEFAULT_FREQ_MIN,
    reference="local",
    operator="median",
    local_radius=DEFAULT_LOCAL_RADIUS,
    fallback_to_global=True,
    dtype=DEFAULT_DTYPE,
    center_chunk_size=None,
    apply_reference=True,
    rename_to_electrodes=True,
):
    """The standard per-segment chain, returned lazily.

    Ordering, ported unchanged from the older build's
    ``apply_standard_preprocessing``:

    1. optional :func:`center` (off unless `center_chunk_size` is given — the old
       build did this in its loader, not in the chain),
    2. :func:`ensure_signed` — only when the dtype is ``uint*``,
    3. :func:`highpass` at `freq_min` (300 Hz),
    4. :func:`common_median_reference` — a GENUINE local median by default
       since Adam's 2026-08-11 ruling ((0, 250) um: all neighbours within
       250 um). Before that ruling the ported default radius was the older
       build's empty (250, 250) annulus, so every pre-ruling run actually got
       the global-median fallback — see ``DEFAULT_LOCAL_RADIUS``'s history
       note; traces from this default onward differ from those runs,
    5. ``annotate(is_filtered=True)``,
    6. :func:`to_float32`.

    The returned recording carries the honest-provenance reference annotations
    (``reference_requested`` / ``reference_effective`` / ``reference_fallback``,
    re-stamped onto the final wrapper so a cast cannot drop them) — read them
    with :func:`reference_provenance`. A chain built with
    ``apply_reference=False`` carries none.

    Steps 2 and 3 are the pairing that must not be reordered: high-passing
    unsigned traces wraps at zero. Referencing comes after the high-pass so the
    median being subtracted is a median of spike-band noise, not of drift.

    Channels are re-keyed to electrode ids first (`rename_to_electrodes`), which
    is what makes segments comparable across electrode configurations.

    Set `apply_reference` False to stop after the high-pass — useful when the
    caller means to reference across concatenated segments instead of per
    segment.
    """
    processed = recording

    if center_chunk_size is not None:
        processed = center(processed, chunk_size=center_chunk_size)

    # Before anything else: channel ids become ELECTRODE ids. Without this the
    # common-electrode intersection concatenation relies on cannot match, since
    # it is expressed in electrode ids while the recording is keyed by channel.
    if rename_to_electrodes:
        processed = rename_channels_to_electrodes(processed)

    processed = ensure_signed(processed)
    processed = highpass(processed, freq_min=freq_min)

    reference_info = None
    if apply_reference:
        processed = common_median_reference(
            processed,
            reference=reference,
            operator=operator,
            local_radius=local_radius,
            fallback_to_global=fallback_to_global,
        )
        # Read the provenance straight off the wrapper that was just
        # annotated, so it can be re-stamped on the FINAL recording below —
        # annotations do not reliably survive later lazy wrappers (same class
        # of problem as is_filtered).
        reference_info = reference_provenance(processed)

    # Sorters warn (or refuse) when handed a recording they cannot tell has been
    # filtered; the filter step does not always propagate the flag through the
    # reference wrapper, so it is set explicitly.
    try:
        processed.annotate(is_filtered=True)
    except Exception:  # pragma: no cover - annotation is advisory only
        logger.debug("Could not annotate is_filtered on %r", type(processed).__name__)

    final = to_float32(processed, dtype=dtype)
    if reference_info is not None and reference_info.get("reference_effective") is not None:
        _annotate_reference(
            final,
            requested=reference_info["reference_requested"],
            effective=reference_info["reference_effective"],
            fallback=reference_info["reference_fallback"],
        )
    return final
