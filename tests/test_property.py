"""Property-based tests (Hypothesis) for metric invariants.

These complement the example-based tests in ``test_metrics.py`` by asserting
laws that must hold for *any* input, not just a handful of hand-picked cases:

- Joint row permutation never changes a metric (correlations are functions of
  the paired multiset, not row order).
- A constant column makes every metric unavailable (``value=None``).
- Bounded metrics stay inside their mathematical range.
- Symmetric metrics are unchanged when X and Y swap roles; Chatterjee's ξ
  (asymmetric) is consistent under the forward/reverse split.

Only dependency-free metrics (Pearson, Spearman, Kendall, Chatterjee's ξ) are
exercised so the suite runs on the lite install; distance correlation is added
under an ``importorskip`` guard.
"""

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as npst

from corrsleuth.metrics import (
    compute_chatterjee_xi,
    compute_chatterjee_xi_reverse,
    compute_kendall,
    compute_pearson,
    compute_spearman,
)
from corrsleuth.validation.input import validate_pair

# Symmetric, dependency-free metrics: f(x, y) == f(y, x).
SYMMETRIC_METRICS = (compute_pearson, compute_spearman, compute_kendall)
# Metrics that are a pure function of the (x, y) multiset, hence invariant to
# row order. Chatterjee's ξ is deliberately excluded: it breaks sort-key ties
# with a seeded random permutation, so under ties (which Hypothesis readily
# generates) its value depends on the input row order. Its calibration and
# determinism are covered by dedicated tests in test_metrics.py.
ROW_INVARIANT_METRICS = SYMMETRIC_METRICS
# Every dependency-free metric, including the asymmetric ξ in both directions.
ALL_METRICS = (*SYMMETRIC_METRICS, compute_chatterjee_xi, compute_chatterjee_xi_reverse)

# Bounded-magnitude metrics whose value must lie in [-1, 1].
UNIT_INTERVAL_METRICS = (compute_pearson, compute_spearman, compute_kendall)

_TOL = 1e-9
# Generous settings: a few of these build 200-row frames and run ξ, which can
# brush the default per-example deadline on a cold CI runner.
_SETTINGS = settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)

_FINITE_FLOATS = st.floats(
    allow_nan=False,
    allow_infinity=False,
    min_value=-1e6,
    max_value=1e6,
    width=64,
)


@st.composite
def paired_xy(draw, *, min_size=30, max_size=120):
    """Draw two equal-length float arrays representing a paired (x, y) sample."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    x = draw(npst.arrays(np.float64, n, elements=_FINITE_FLOATS))
    y = draw(npst.arrays(np.float64, n, elements=_FINITE_FLOATS))
    return x, y


def _pair(x, y):
    import pandas as pd

    return validate_pair(pd.DataFrame({"x": x, "y": y}), "x", "y")


def _approx_equal(a, b):
    """Compare two metric values, treating ``None`` as a value and allowing
    floating-point slack (row permutation changes summation order)."""
    if a is None or b is None:
        return a is None and b is None
    return a == pytest.approx(b, rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("metric", ROW_INVARIANT_METRICS, ids=lambda m: m.__name__)
@_SETTINGS
@given(data=paired_xy())
def test_metric_is_invariant_to_joint_row_permutation(metric, data):
    """Permuting the rows of (x, y) together must not change a multiset metric."""
    x, y = data
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(x))

    original = metric(_pair(x, y)).value
    permuted = metric(_pair(x[perm], y[perm])).value

    assert _approx_equal(original, permuted)


# Shape diagnostics were historically excluded from permutation checks because
# the old position-based binning split tied X across bins by input order. With
# tie-safe binning they are now invariant, so this closes that gap.
_SHAPE_DIAG_KEYS = (
    ("bin_lof", "bin_lof_r2_gain"),
    ("bin_lof", "bin_reversal_count"),
    ("bin_lof", "bin_lof_r2_gain_robust"),
    ("seg", "segment_gain"),
    ("seg", "segment_stepness"),
)


@_SETTINGS
@given(data=paired_xy(min_size=60, max_size=160))
def test_shape_diagnostics_invariant_to_joint_row_permutation(data):
    """The bin-lof and segmentation shape diagnostics must not change when the
    rows of (x, y) are permuted together — including when Hypothesis generates
    ties in X (which the previous position-based binning made order-dependent).
    Withheld (None) results must stay withheld under permutation too."""
    from corrsleuth.metrics.shape import compute_bin_lof, compute_segmentation

    x, y = data
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(x))

    def diagnostics(pair):
        return {"bin_lof": compute_bin_lof(pair), "seg": compute_segmentation(pair)}

    base = diagnostics(_pair(x, y))
    permuted = diagnostics(_pair(x[perm], y[perm]))
    for group, key in _SHAPE_DIAG_KEYS:
        assert _approx_equal(base[group][key].value, permuted[group][key].value)


@pytest.mark.parametrize("metric", ALL_METRICS, ids=lambda m: m.__name__)
@_SETTINGS
@given(
    const=_FINITE_FLOATS,
    y=npst.arrays(np.float64, st.integers(30, 120), elements=_FINITE_FLOATS),
)
def test_constant_input_is_never_computable(metric, const, y):
    """A constant column yields ``value=None`` (but ``available=True``) for
    every metric — the check ran, the data just cannot support a number."""
    x = np.full(len(y), const)
    result = metric(_pair(x, y))

    assert result.value is None
    assert result.available is True


@pytest.mark.parametrize("metric", UNIT_INTERVAL_METRICS, ids=lambda m: m.__name__)
@_SETTINGS
@given(data=paired_xy())
def test_correlation_magnitude_stays_in_unit_interval(metric, data):
    """Pearson/Spearman/Kendall must stay within [-1, 1] when computed."""
    x, y = data
    value = metric(_pair(x, y)).value
    if value is not None:
        assert -1.0 - _TOL <= value <= 1.0 + _TOL


@_SETTINGS
@given(data=paired_xy())
def test_chatterjee_xi_upper_bounded_by_one(data):
    """The tie-corrected ξ is bounded above by 1 (it approaches 1 for a perfect
    functional dependence); allow a tiny epsilon for float error."""
    x, y = data
    for metric in (compute_chatterjee_xi, compute_chatterjee_xi_reverse):
        value = metric(_pair(x, y)).value
        if value is not None:
            assert value <= 1.0 + _TOL


@pytest.mark.parametrize("metric", SYMMETRIC_METRICS, ids=lambda m: m.__name__)
@_SETTINGS
@given(data=paired_xy())
def test_symmetric_metrics_are_order_independent_in_arguments(metric, data):
    """Pearson, Spearman, and Kendall are symmetric: f(x, y) == f(y, x)."""
    x, y = data
    assert _approx_equal(metric(_pair(x, y)).value, metric(_pair(y, x)).value)


@_SETTINGS
@given(data=paired_xy())
def test_chatterjee_xi_forward_reverse_consistency(data):
    """ξ is asymmetric, but the two directions are mirror images: the forward
    ξ(X→Y) on (x, y) must equal the reverse ξ(Y→X) on the column-swapped pair."""
    x, y = data
    forward = compute_chatterjee_xi(_pair(x, y)).value
    reverse_on_swapped = compute_chatterjee_xi_reverse(_pair(y, x)).value
    assert _approx_equal(forward, reverse_on_swapped)


@st.composite
def _paired_moderate(draw, *, min_size=30, max_size=120):
    """Like ``paired_xy`` but bounded to a moderate magnitude, so squaring stays
    well-conditioned and the invariance below is not obscured by float error."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    elems = st.floats(
        allow_nan=False, allow_infinity=False, min_value=-1e3, max_value=1e3, width=64
    )
    x = draw(npst.arrays(np.float64, n, elements=elems))
    y = draw(npst.arrays(np.float64, n, elements=elems))
    return x, y


_OFFSET = st.floats(
    allow_nan=False, allow_infinity=False, min_value=-1e3, max_value=1e3, width=64
)


@_SETTINGS
@given(data=_paired_moderate(), dx=_OFFSET, dy=_OFFSET)
def test_squared_correlation_is_translation_invariant(data, dx, dy):
    """sq_corr must depend only on the shape, not where it sits in the plane.

    Squaring is not translation-invariant, so ``compute_squared_correlation``
    centers X and Y first; shifting both by any finite offset must leave the
    value unchanged (a circle at (5, 5) reads the same as one at the origin).
    Before that fix, ``corr(X², Y²)`` on raw values collapsed toward
    ``corr(X, Y)`` for off-origin data."""
    from corrsleuth.metrics import compute_squared_correlation

    x, y = data
    # Well-conditioned inputs only. A value so tiny that a finite shift absorbs
    # it (e.g. 1e-65 + 1.0 == 1.0, collapsing the column to a constant) leaves the
    # diagnostic undefined on one side — a floating-point artifact, not a
    # translation-variance — so require real spread in both variables.
    assume((x.max() - x.min()) > 1e-3 and (y.max() - y.min()) > 1e-3)
    base = compute_squared_correlation(_pair(x, y)).value
    shifted = compute_squared_correlation(_pair(x + dx, y + dy)).value
    assume(base is not None and shifted is not None)
    assert shifted == pytest.approx(base, rel=1e-7, abs=1e-7)


def test_distance_correlation_properties():
    """Distance correlation (optional dep) is symmetric and in [0, 1].

    A single representative case under an importorskip guard, since the property
    sweep above is restricted to dependency-free metrics."""
    pytest.importorskip("dcor")
    from corrsleuth.metrics import compute_distance_correlation

    rng = np.random.default_rng(0)
    x = rng.normal(size=100)
    y = x**2 + rng.normal(scale=0.1, size=100)

    def dcor_value(a, b):
        return compute_distance_correlation(
            _pair(a, b), mode="standard", max_n_for_dcor=None, random_state=0
        ).value

    forward = dcor_value(x, y)
    swapped = dcor_value(y, x)
    assert forward is not None
    assert 0.0 - _TOL <= forward <= 1.0 + _TOL
    assert forward == pytest.approx(swapped, rel=1e-9, abs=1e-9)
