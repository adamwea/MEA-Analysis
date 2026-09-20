"""The shared MAD primitive, and the traps that justified consolidating it."""

import pathlib
import tokenize

import numpy as np
import pytest

from mea_modules.quality.robust import MAD_OVER_SIGMA, MAD_TO_SIGMA, mad_sigma


def test_scaling_recovers_a_known_gaussian_sigma():
    """For Gaussian noise the rescaled MAD is an estimate of sigma itself."""
    rng = np.random.default_rng(0)
    values = rng.normal(0.0, 7.0, (200_000, 2))

    assert mad_sigma(values) == pytest.approx([7.0, 7.0], rel=0.02)


def test_the_constant_and_its_reciprocal_agree():
    """The bug this guards is a same-named constant holding reciprocal values.

    Two modules defined `_MAD_TO_SIGMA` as 0.6745 and a third as its reciprocal,
    each used in its own matching direction. The results agreed, but nothing
    said they had to, and that is how a factor of 2.2 reaches a threshold.
    """
    assert MAD_TO_SIGMA == pytest.approx(1.0 / MAD_OVER_SIGMA)
    assert MAD_OVER_SIGMA == pytest.approx(0.6744897501960817)


def test_a_dc_offset_does_not_become_the_noise_estimate():
    """The reason this function centres.

    The raw view of a Maxwell scan sits near -32255 ADC counts. An uncentred
    `median(|x|)` reports that offset rather than the noise -- measured 47821
    against a true 4.45 on this project's own data. A threshold built on it
    lands outside the converter's range, so nothing is ever detected and the
    raster comes out blank, which is indistinguishable from a quiet well.
    """
    rng = np.random.default_rng(1)
    noise = rng.normal(0.0, 4.0, (20_000, 3))
    offset = (noise - 32_255).astype(np.int16)

    uncentred = np.median(np.abs(offset.astype(float)), axis=0) * MAD_TO_SIGMA
    assert uncentred.min() > 40_000  # the trap, reproduced

    assert mad_sigma(offset) == pytest.approx([4.0, 4.0, 4.0], abs=0.6)


def test_absolute_value_at_the_int16_negative_rail():
    """`numpy.abs` of int16 -32768 is -32768: there is no +32768 in int16.

    Taking the absolute value before widening therefore lets a corrupted,
    still-negative sample drag the median down. Maxwell data reaches the rail.
    """
    assert np.abs(np.int16(-32768)) == -32768  # the trap still exists in numpy

    rail = np.full((101, 1), -32768, dtype=np.int16)
    rail[50, 0] = -32700
    # A flat-but-for-one-sample channel has a MAD of 0; the point is that it is
    # not negative and not enormous, which is what the unwidened path produced.
    assert mad_sigma(rail)[0] == pytest.approx(0.0)


def test_centring_an_int16_trace_cannot_wrap():
    """32000 - (-32000) leaves the int16 range and wraps to -1536."""
    a = np.array([32000, -32000], dtype=np.int16)
    assert (a - np.int16(-32000))[0] == -1536  # the trap, reproduced

    values = np.array([[32000], [-32000], [32000], [-32000]], dtype=np.int16)
    # The true MAD about the median (0) is 32000, so the sigma is that rescaled.
    assert mad_sigma(values)[0] == pytest.approx(32000 * MAD_TO_SIGMA)


@pytest.mark.parametrize("dtype", [np.int16, np.int32, np.float32, np.float64])
def test_integer_and_float_inputs_agree(dtype):
    """Widening must not change the answer, only make it computable."""
    base = np.array([[-32768, -32000, 100], [-32760, -32010, 90], [-32764, -31990, 110]])
    assert mad_sigma(base.astype(dtype)) == pytest.approx(mad_sigma(base.astype(np.float64)), rel=1e-6)


def test_empty_input_returns_an_empty_result_rather_than_raising():
    assert mad_sigma(np.zeros((0, 4), dtype=np.int16)).shape == (4,)


def test_a_flat_channel_is_exactly_zero():
    """Zero, not epsilon: what a flat channel should become is the caller's
    decision, and every caller that builds a threshold clips it itself."""
    assert mad_sigma(np.full((10, 2), 7, dtype=np.int16)) == pytest.approx([0.0, 0.0])


def test_axis_follows_the_caller():
    values = np.array([[1.0, 1.0, 9.0, 1.0], [2.0, 2.0, 2.0, 2.0]])
    assert mad_sigma(values, axis=-1).shape == (2,)
    assert mad_sigma(values, axis=0).shape == (4,)


def test_nan_propagates_by_default_and_is_skipped_on_request():
    """A NaN in a recording is a defect and should surface. A NaN in a template
    is a channel the unit never reached, and skipping it is the right answer."""
    values = np.array([[1.0, 1.0], [2.0, np.nan], [9.0, 2.0], [1.0, 9.0], [2.0, 1.0]])

    assert np.isnan(mad_sigma(values, axis=0)[1])
    assert np.isfinite(mad_sigma(values, axis=0, nan_safe=True)[1])
    # the clean column is unaffected either way
    assert mad_sigma(values, axis=0)[0] == pytest.approx(
        mad_sigma(values, axis=0, nan_safe=True)[0]
    )


def test_no_module_outside_robust_reimplements_the_scaling():
    """Structural: the consolidation has to stay consolidated.

    Six copies of this calculation existed across quality, diagnostics,
    preprocessing and curation. A seventh is easy to add by accident, and the
    cost of that is two thresholds that disagree without anything saying so.
    Prose may name the constant; code may not re-derive it.
    """
    root = pathlib.Path(__file__).resolve().parents[1] / "mea_modules"
    wanted = {"0.6744897501960817", "0.6745", "1.4826"}
    offenders = []
    for path in root.rglob("*.py"):
        if path.name == "robust.py":
            continue
        with tokenize.open(path) as handle:
            # Token-level, not a text search: a constant named in a docstring is
            # a STRING token and is prose, while the same characters as a NUMBER
            # token are a second implementation. A regex over lines cannot tell
            # those apart, and the docstrings here legitimately name the value.
            for token in tokenize.generate_tokens(handle.readline):
                if token.type == tokenize.NUMBER and token.string in wanted:
                    offenders.append(
                        "%s:%d: %s" % (path.relative_to(root), token.start[0], token.line.strip())
                    )
    joined = "\n".join(offenders)
    assert not offenders, "the MAD scaling was re-derived outside robust.py:\n" + joined
