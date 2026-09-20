"""The one robust-scale primitive: MAD, rescaled to the sigma it implies.

Every spike-detection threshold in this pipeline is a multiple of "the noise on
this electrode", and this is the only place that number is defined.

**Why a MAD and not a standard deviation.** The quantity wanted is the size of a
typical *baseline* fluctuation, measured on a recording that is full of the very
events the threshold exists to catch. A standard deviation is inflated by those
events, so an electrode carrying many large spikes gets a large sigma, therefore
a high threshold, therefore fewer detections -- the metric fights the
measurement it is there to enable. A median absolute deviation ignores the
sparse tails, so a busy electrode and a quiet one with the same baseline get the
same threshold.

**Why it is centred.** ``median(|x|)`` is a noise estimate only when the signal
already sits on zero; otherwise it returns the offset. Four copies of this
calculation existed, two of them uncentred, and they were correct only because
every caller happened to pass a high-passed recording -- an invariant no
signature stated and no test pinned. On this project's raw view, which sits at
about -32255 ADC counts, the uncentred form returned 47821 where the true noise
was 4.45: a factor of 10754, and a threshold built on it falls outside the
converter's range so that *nothing is ever detected* and the raster comes out
blank. Centring costs one extra median per window and removes the invariant, so
this function always centres.

**Why the cast comes first.** Integer traces must be widened before either the
subtraction or the absolute value, and neither is safe in place:

* ``numpy.abs`` of ``int16`` at the negative rail is *negative* -- there is no
  +32768 in int16, so ``abs(-32768) == -32768`` and a corrupted sample silently
  drags the median down.
* an int16 sample minus an int16 median can leave the int16 range: 32000 minus
  -32000 wraps to -1536 rather than 64000.

Both are reachable from raw Maxwell data, which uses the full 16-bit span.
"""

import numpy as np

# scipy.stats.norm.ppf(0.75): for Gaussian noise the MAD is this fraction of
# sigma, so dividing by it (equivalently, multiplying by the reciprocal below)
# turns a MAD into the sigma it implies. Defined once here because the same name
# previously held this value in two modules and its reciprocal in a third --
# each used in its own matching direction, so the results agreed, but that is
# exactly how a factor of 2.2 gets into a threshold unnoticed.
MAD_OVER_SIGMA = 0.6744897501960817
MAD_TO_SIGMA = 1.0 / MAD_OVER_SIGMA


def _working_dtype(values):
    """Float type wide enough for `values`, without needlessly doubling memory.

    float64 only when the input already is: a noise window is easily 20000
    frames by 1000 channels, and the intermediate ``|x - median|`` doubles it
    again, so promoting int16 to float64 would cost 320 MB where 160 MB does the
    same job exactly. int16 and int32 are both represented exactly in float32's
    24-bit mantissa, and after centring the values are small, so nothing is lost.
    """
    dtype = np.asarray(values).dtype
    return np.float64 if dtype == np.float64 else np.float32


def mad_sigma(values, axis=0, nan_safe=False):
    """MAD of `values` along `axis`, rescaled to a Gaussian-equivalent sigma.

    Traces are ``(samples, channels)`` and want the default ``axis=0``; a
    waveform table measuring along its last axis passes ``axis=-1``.

    `nan_safe` skips NaNs instead of propagating them. Traces do not want it --
    a NaN in a recording is a real problem and should surface rather than be
    averaged away -- but a template or footprint table legitimately carries NaN
    for channels a unit never reached, and there the NaNs are absence, not
    corruption. A slice that is entirely NaN yields NaN either way; callers
    substitute their own fallback, because what an unmeasurable channel should
    become depends on what they are about to do with it.

    Returns float64 so a caller can compare or divide without a second cast.
    An empty input gives an empty result rather than raising, and a flat channel
    gives exactly 0.0 -- callers that turn this into a threshold clip it away
    themselves, because what a zero should become is the caller's decision.
    """
    values = np.asarray(values)
    if values.size == 0:
        shape = list(values.shape)
        if values.ndim:
            shape.pop(axis)
        return np.zeros(shape, dtype=np.float64)

    # Widen BEFORE the median, the subtraction and the absolute value -- see the
    # module docstring for the two integer traps this avoids.
    values = values.astype(_working_dtype(values), copy=False)
    middle = np.nanmedian if nan_safe else np.median
    with np.errstate(invalid="ignore"):
        median = middle(values, axis=axis, keepdims=True)
        scale = middle(np.abs(values - median), axis=axis)
    return (scale * MAD_TO_SIGMA).astype(np.float64, copy=False)


__all__ = ["MAD_OVER_SIGMA", "MAD_TO_SIGMA", "mad_sigma"]
