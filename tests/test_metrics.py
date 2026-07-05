import sys

import numpy as np
import pandas as pd
import pytest

from corrsleuth.api import profile_pair
from corrsleuth.datasets import make_relationship
from corrsleuth.exceptions import MetricComputationError, OptionalDependencyError
from corrsleuth.metrics import (
    compute_bin_lof_r2_gain,
    compute_biweight_midcorrelation,
    compute_chatterjee_xi,
    compute_chatterjee_xi_reverse,
    compute_distance_correlation,
    compute_heteroscedasticity,
    compute_influence,
    compute_kendall,
    compute_median_clipped_pearson,
    compute_mutual_information,
    compute_pearson,
    compute_segmentation,
    compute_spearman,
    compute_squared_correlation,
    compute_trimmed_pearson,
    compute_winsorized_pearson,
)
from corrsleuth.validation.input import validate_pair


def test_core_metrics():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 5, 4, 5]})
    pair = validate_pair(df, "x", "y")

    p = compute_pearson(pair)
    assert p.name == "pearson"
    assert p.value is not None

    s = compute_spearman(pair)
    assert s.name == "spearman"
    assert s.value is not None

    k = compute_kendall(pair)
    assert k.name == "kendall_tau_b"
    assert k.value is not None


def test_optional_metrics():
    pytest.importorskip("dcor")
    pytest.importorskip("sklearn")
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 5, 4, 5]})
    pair = validate_pair(df, "x", "y")

    dc = compute_distance_correlation(pair, mode="standard")
    assert dc.available is True
    assert dc.value is not None

    mi = compute_mutual_information(pair, mode="standard")
    assert mi.available is True
    assert mi.value is not None


def test_optional_metrics_downsampling_override():
    pytest.importorskip("dcor")
    df = pd.DataFrame({"x": range(100), "y": range(100)})
    pair = validate_pair(df, "x", "y")

    # Cap at 50
    compute_distance_correlation(pair, mode="standard", max_n_for_dcor=50)
    assert any("n_used > 50" in w for w in pair.warnings)

    # Disable cap
    pair2 = validate_pair(df, "x", "y")
    compute_distance_correlation(pair2, mode="standard", max_n_for_dcor=None)
    assert not any("n_used >" in w for w in pair2.warnings)


def _hide_module(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Make ``import name`` (and any submodule import) raise ImportError."""
    for mod in list(sys.modules):
        if mod == name or mod.startswith(name + "."):
            monkeypatch.setitem(sys.modules, mod, None)
    monkeypatch.setitem(sys.modules, name, None)


def test_distance_correlation_missing_dependency_raises_in_standard_mode(monkeypatch):
    _hide_module(monkeypatch, "dcor")
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 5, 4, 5]})
    pair = validate_pair(df, "x", "y")

    with pytest.raises(OptionalDependencyError, match="corrsleuth\\[standard\\]"):
        compute_distance_correlation(pair, mode="standard")


def test_distance_correlation_missing_dependency_returns_unavailable_in_lite_mode(
    monkeypatch,
):
    _hide_module(monkeypatch, "dcor")
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 5, 4, 5]})
    pair = validate_pair(df, "x", "y")

    result = compute_distance_correlation(pair, mode="lite")
    assert result.available is False
    assert result.value is None


def test_mutual_information_missing_dependency_raises_in_standard_mode(monkeypatch):
    _hide_module(monkeypatch, "sklearn")
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 5, 4, 5]})
    pair = validate_pair(df, "x", "y")

    with pytest.raises(OptionalDependencyError, match="corrsleuth\\[standard\\]"):
        compute_mutual_information(pair, mode="standard")


def test_mutual_information_missing_dependency_returns_unavailable_in_lite_mode(
    monkeypatch,
):
    _hide_module(monkeypatch, "sklearn")
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 5, 4, 5]})
    pair = validate_pair(df, "x", "y")

    result = compute_mutual_information(pair, mode="lite")
    assert result.available is False
    assert result.value is None


def test_standard_mode_small_sample_returns_low_power_result():
    pytest.importorskip("dcor")
    pytest.importorskip("sklearn")
    df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 3, 2]})

    result = profile_pair(df, "x", "y", mode="standard")

    assert result.pattern == "low_power_or_uncertain"
    mi = result.metrics.loc[
        result.metrics["metric"] == "mutual_information", "value"
    ].iloc[0]
    assert pd.isna(mi)
    assert any("Mutual information is not computed" in w for w in result.warnings)


def test_compute_pearson_wraps_unexpected_failures_as_metric_error(monkeypatch):
    """Unexpected scipy failures must surface as MetricComputationError with the metric name."""
    import scipy.stats as stats

    def boom(*args, **kwargs):
        raise ValueError("simulated scipy failure")

    monkeypatch.setattr(stats, "pearsonr", boom)
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 5, 4, 5]})
    pair = validate_pair(df, "x", "y")

    with pytest.raises(MetricComputationError, match="Failed to compute pearson"):
        compute_pearson(pair)


def test_deep_mode_adds_robust_metrics_without_standard_dependencies():
    df = pd.DataFrame({"x": range(80), "y": range(80)})

    lite = profile_pair(df, "x", "y", mode="lite")
    deep = profile_pair(df, "x", "y", mode="deep")

    lite_metrics = set(lite.metrics["metric"])
    deep_metrics = set(deep.metrics["metric"])
    robust_metrics = {
        "pearson_trimmed_1pct",
        "pearson_winsorized_1pct",
        "biweight_midcorrelation",
        "pearson_median_clipped_20pct",
    }
    assert robust_metrics.isdisjoint(lite_metrics)
    assert robust_metrics <= deep_metrics
    assert "distance_correlation" not in deep_metrics
    assert "mutual_information" not in deep_metrics


def test_deep_mode_emits_one_small_sample_robust_warning():
    df = pd.DataFrame({"x": range(40), "y": range(40)})

    result = profile_pair(df, "x", "y", mode="deep")

    robust_warnings = [
        w for w in result.warnings if "deep-mode robust correlation diagnostics" in w
    ]
    assert robust_warnings == [
        "n_used < 50; deep-mode robust correlation diagnostics are not computed."
    ]


def test_robust_metrics_are_near_pearson_for_clean_linear_data():
    df = pd.DataFrame({"x": range(100), "y": range(100)})
    pair = validate_pair(df, "x", "y")

    results = [
        compute_trimmed_pearson(pair),
        compute_winsorized_pearson(pair),
        compute_biweight_midcorrelation(pair),
        compute_median_clipped_pearson(pair),
    ]

    assert all(r.value == pytest.approx(1.0) for r in results)


def test_outlier_driven_data_shows_meaningful_pearson_robust_gap():
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(0, 0.1, size=n)
    y = rng.normal(0, 0.1, size=n)
    x[-2:] = rng.uniform(8, 10, size=2)
    y[-2:] = rng.uniform(8, 10, size=2)
    df = pd.DataFrame({"x": x, "y": y})

    result = profile_pair(df, "x", "y", mode="deep")
    metrics = {row["metric"]: row["value"] for _, row in result.metrics.iterrows()}

    assert metrics["pearson"] > 0.90
    for metric_name in (
        "pearson_trimmed_1pct",
        "pearson_winsorized_1pct",
        "biweight_midcorrelation",
        "pearson_median_clipped_20pct",
    ):
        assert metrics[metric_name] < 0.50
        assert metrics["pearson"] - metrics[metric_name] > 0.40


def test_robust_metrics_return_none_when_mad_or_bend_scale_is_zero():
    x = [0.0] * 51 + [1.0] * 9
    y = list(range(60))
    pair = validate_pair(pd.DataFrame({"x": x, "y": y}), "x", "y")

    biweight = compute_biweight_midcorrelation(pair)
    median_clipped = compute_median_clipped_pearson(pair)

    assert biweight.value is None
    assert median_clipped.value is None


# --- Chatterjee's xi (deep-mode nonlinear dependence) ---


def test_chatterjee_xi_high_for_clean_linear_data():
    rng = np.random.default_rng(0)
    n = 300
    x = rng.uniform(-3, 3, size=n)
    y = x + rng.normal(0, 0.05, size=n)
    pair = validate_pair(pd.DataFrame({"x": x, "y": y}), "x", "y")

    result = compute_chatterjee_xi(pair)
    assert result.name == "chatterjee_xi"
    assert result.available is True
    assert result.value is not None
    assert result.value > 0.85


def test_chatterjee_xi_detects_u_shape_that_pearson_misses():
    """For Y = X^2, Pearson and Spearman are near zero but xi(X->Y) is high."""
    rng = np.random.default_rng(0)
    n = 500
    x = rng.uniform(-3, 3, size=n)
    y = x**2 + rng.normal(0, 0.05, size=n)

    deep = profile_pair(pd.DataFrame({"x": x, "y": y}), "x", "y", mode="deep")
    metrics = {row["metric"]: row["value"] for _, row in deep.metrics.iterrows()}

    assert abs(metrics["pearson"]) < 0.20
    assert abs(metrics["spearman"]) < 0.30
    # xi should still detect the dependence
    assert metrics["chatterjee_xi"] > 0.80


def test_chatterjee_xi_is_asymmetric_for_many_to_one_relationship():
    """Y = X^2 maps two X values to the same Y, so X is not a function of Y.

    xi(X -> Y) should be much higher than xi(Y -> X)."""
    rng = np.random.default_rng(0)
    n = 500
    x = rng.uniform(-3, 3, size=n)
    y = x**2 + rng.normal(0, 0.05, size=n)
    df = pd.DataFrame({"x": x, "y": y})

    fwd = profile_pair(df, "x", "y", mode="deep")
    rev = profile_pair(df, "y", "x", mode="deep")
    fwd_xi = next(
        r["value"] for _, r in fwd.metrics.iterrows() if r["metric"] == "chatterjee_xi"
    )
    rev_xi = next(
        r["value"] for _, r in rev.metrics.iterrows() if r["metric"] == "chatterjee_xi"
    )

    assert fwd_xi - rev_xi > 0.40


def test_chatterjee_xi_near_zero_for_independent_variables():
    rng = np.random.default_rng(0)
    n = 500
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    pair = validate_pair(pd.DataFrame({"x": x, "y": y}), "x", "y")

    result = compute_chatterjee_xi(pair)
    assert abs(result.value) < 0.10


def _reference_tie_corrected_xi(x, y):
    """Brute-force Chatterjee (2020) tie-corrected ξ, independent of the
    library implementation, used as an oracle in the tests below."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    order = np.lexsort((y, x))
    ys = y[order]
    r = np.array([np.sum(ys <= v) for v in ys], dtype=float)
    l_counts = np.array([np.sum(ys >= v) for v in ys], dtype=float)
    numerator = n * np.sum(np.abs(np.diff(r)))
    denominator = 2.0 * np.sum(l_counts * (n - l_counts))
    return 1.0 - numerator / denominator


def test_chatterjee_xi_matches_tie_corrected_reference_on_discrete_y():
    """With a tied (low-cardinality) response, ξ must equal the tie-corrected
    formula, not the no-ties simplification."""
    rng = np.random.default_rng(0)
    n = 500
    x = rng.uniform(0, 1, size=n)
    y = np.round(rng.uniform(0, 1, size=n) * 3)  # only 4 distinct Y values
    pair = validate_pair(pd.DataFrame({"x": x, "y": y}), "x", "y")

    result = compute_chatterjee_xi(pair)
    expected = _reference_tie_corrected_xi(x, y)
    assert result.value == pytest.approx(expected, abs=1e-12)


def test_chatterjee_xi_near_zero_for_independent_discrete_y():
    """Independent X with a discrete Y: the tie-corrected ξ stays near zero,
    whereas the uncorrected n^2-1 formula overstates dependence."""
    rng = np.random.default_rng(0)
    n = 1000
    x = rng.uniform(0, 1, size=n)
    y = np.round(rng.uniform(0, 1, size=n) * 3)
    pair = validate_pair(pd.DataFrame({"x": x, "y": y}), "x", "y")

    result = compute_chatterjee_xi(pair)
    assert abs(result.value) < 0.05


def test_chatterjee_xi_returns_none_for_constant_input():
    df = pd.DataFrame({"x": [1.0] * 50, "y": list(range(50))})
    pair = validate_pair(df, "x", "y")

    result = compute_chatterjee_xi(pair)
    assert result.value is None
    assert result.available is True


def test_chatterjee_xi_returns_none_with_warning_below_min_n():
    df = pd.DataFrame({"x": list(range(15)), "y": list(range(15))})
    pair = validate_pair(df, "x", "y")

    result = compute_chatterjee_xi(pair)

    assert result.value is None
    assert result.available is True
    assert any("chatterjee_xi" in w and "converges slowly" in w for w in pair.warnings)


def test_chatterjee_xi_only_appears_in_deep_mode():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.uniform(size=80), "y": rng.uniform(size=80)})

    lite_metrics = set(profile_pair(df, "x", "y", mode="lite").metrics["metric"])
    deep_metrics = set(profile_pair(df, "x", "y", mode="deep").metrics["metric"])

    assert "chatterjee_xi" not in lite_metrics
    assert "chatterjee_xi_reverse" not in lite_metrics
    assert "chatterjee_xi" in deep_metrics
    assert "chatterjee_xi_reverse" in deep_metrics


def test_chatterjee_xi_is_calibrated_when_sort_variable_has_heavy_ties():
    """ξ must stay near 0 under independence even when the sort variable is
    heavily tied. Breaking ties by Y (the old behavior) leaked the response and
    drove ξ toward 1 for, e.g., independent binary X — a false dependence
    signal. The seeded random tie-break keeps it calibrated."""
    rng = np.random.default_rng(0)
    n = 1000
    x = rng.integers(0, 2, size=n).astype(float)  # binary -> ~50% ties
    y = rng.normal(size=n)  # independent of x
    pair = validate_pair(pd.DataFrame({"x": x, "y": y}), "x", "y")

    forward = compute_chatterjee_xi(pair).value  # sorts by the tied x
    reverse = compute_chatterjee_xi_reverse(
        validate_pair(pd.DataFrame({"x": y, "y": x}), "x", "y")
    ).value  # sorts by the tied x in the reverse role

    assert abs(forward) < 0.15, forward
    assert abs(reverse) < 0.15, reverse


def test_chatterjee_xi_still_detects_dependence_on_a_discrete_predictor():
    """Calibration must not cost sensitivity: when Y is a clean function of a
    discrete X, ξ should be clearly positive (well above the independence band
    and the dependence-warning threshold)."""
    rng = np.random.default_rng(1)
    n = 1000
    x = rng.integers(0, 2, size=n).astype(float)
    y = 3.0 * x + rng.normal(0, 0.01, size=n)  # Y is essentially a function of X
    pair = validate_pair(pd.DataFrame({"x": x, "y": y}), "x", "y")

    assert compute_chatterjee_xi(pair).value > 0.35


def test_chatterjee_xi_is_reproducible_for_a_fixed_seed():
    """The seeded tie-break makes ξ reproducible for a given input + seed."""
    rng = np.random.default_rng(2)
    n = 200
    x = rng.integers(0, 5, size=n).astype(float)
    y = rng.normal(size=n)
    pair = validate_pair(pd.DataFrame({"x": x, "y": y}), "x", "y")

    assert (
        compute_chatterjee_xi(pair, random_state=7).value
        == compute_chatterjee_xi(pair, random_state=7).value
    )


def test_chatterjee_xi_exposes_both_directions_in_a_single_deep_call():
    """For Y = X^2 the forward direction should be high (Y is a function of X)
    and the reverse direction should be lower (X is not a function of Y)."""
    rng = np.random.default_rng(0)
    n = 500
    x = rng.uniform(-3, 3, size=n)
    y = x**2 + rng.normal(0, 0.05, size=n)

    deep = profile_pair(pd.DataFrame({"x": x, "y": y}), "x", "y", mode="deep")
    metrics = {row["metric"]: row["value"] for _, row in deep.metrics.iterrows()}

    assert metrics["chatterjee_xi"] > 0.80
    assert metrics["chatterjee_xi_reverse"] < 0.50


# --- Shape diagnostics (bin_lof_r2_gain, sq_corr) ---


def _reference_bin_lof_r2_gain(x, y, target_per_bin=10, min_bins=5, max_bins=20):
    """Independent reimplementation of the equal-frequency-bin lack-of-fit
    test, used as an oracle below. Uses an explicit per-bin loop for the bin
    R^2 (rather than the vectorized ``array_split`` + broadcast approach in
    ``metrics/shape.py``) and the R^2-equals-squared-Pearson-r identity for
    the linear R^2 (rather than an explicit polyfit residual sum), so the two
    implementations share only the bin-split rule, not the arithmetic path."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ys = y[order]
    n_bins = int(np.clip(n // target_per_bin, min_bins, max_bins))
    bins = np.array_split(np.arange(n), n_bins)
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    ss_res_bins = sum(float(np.sum((ys[b] - ys[b].mean()) ** 2)) for b in bins)
    r2_bins = 1.0 - ss_res_bins / ss_tot
    from scipy.stats import pearsonr

    r_lin, _ = pearsonr(xs, ys)
    r2_linear = r_lin**2
    return r2_bins - r2_linear


def test_bin_lof_r2_gain_matches_reference_on_curved_data():
    rng = np.random.default_rng(0)
    n = 307  # deliberately not evenly divisible by any bin count in range
    x = rng.uniform(0, 3, size=n)
    y = np.exp(x) + rng.normal(0, 0.1, size=n) * np.exp(x).std()
    pair = validate_pair(pd.DataFrame({"x": x, "y": y}), "x", "y")

    result = compute_bin_lof_r2_gain(pair)
    expected = _reference_bin_lof_r2_gain(x, y)

    assert result.value == pytest.approx(expected, abs=1e-9)


def test_bin_lof_r2_gain_positive_for_curved_data_near_zero_for_linear():
    rng = np.random.default_rng(0)
    n = 300

    x_lin = rng.uniform(-3, 3, size=n)
    y_lin = x_lin + rng.normal(0, 0.1, size=n)
    linear_pair = validate_pair(pd.DataFrame({"x": x_lin, "y": y_lin}), "x", "y")

    x_curved = rng.uniform(-3, 3, size=n)
    y_curved = x_curved**2 + rng.normal(0, 0.1, size=n)
    curved_pair = validate_pair(pd.DataFrame({"x": x_curved, "y": y_curved}), "x", "y")

    linear_gain = compute_bin_lof_r2_gain(linear_pair).value
    curved_gain = compute_bin_lof_r2_gain(curved_pair).value

    assert linear_gain < 0.05
    assert curved_gain > 0.5


def test_bin_lof_r2_gain_returns_none_for_constant_input():
    df = pd.DataFrame({"x": [1.0] * 60, "y": list(range(60))})
    pair = validate_pair(df, "x", "y")

    result = compute_bin_lof_r2_gain(pair)
    assert result.value is None
    assert result.available is True


def test_bin_lof_r2_gain_returns_none_below_min_n():
    rng = np.random.default_rng(0)
    n = 40  # below _MIN_N_FOR_BIN_LOF (50)
    df = pd.DataFrame({"x": rng.uniform(size=n), "y": rng.uniform(size=n)})
    pair = validate_pair(df, "x", "y")

    result = compute_bin_lof_r2_gain(pair)
    assert result.value is None
    assert result.available is True


def test_bin_lof_r2_gain_handles_heavy_ties_without_raising():
    """Heavy ties in X don't shrink any bin below 2 points here — binning is
    by sorted *position*, not by distinct X value, and n >= _MIN_N_FOR_BIN_LOF
    with at most 20 bins guarantees every bin has >= 2 points — but a heavily
    tied sort key is exactly the shape that could regress this if the binning
    logic ever changed to group by value instead of position."""
    n = 200
    # All but a handful of rows share the same X value.
    x = np.concatenate([np.zeros(n - 3), [1.0, 2.0, 3.0]])
    y = np.arange(n, dtype=float)
    pair = validate_pair(pd.DataFrame({"x": x, "y": y}), "x", "y")

    result = compute_bin_lof_r2_gain(pair)
    assert result.available is True
    assert result.value is not None


def test_squared_correlation_strongly_negative_for_circular_data():
    rng = np.random.default_rng(0)
    n = 500
    theta = rng.uniform(0, 2 * np.pi, size=n)
    radius = 5.0 * (1 + rng.normal(0, 0.05, size=n))
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    pair = validate_pair(pd.DataFrame({"x": x, "y": y}), "x", "y")

    result = compute_squared_correlation(pair)
    assert result.name == "sq_corr"
    assert result.value < -0.7


def test_squared_correlation_near_zero_for_independent_variables():
    rng = np.random.default_rng(0)
    n = 500
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    pair = validate_pair(pd.DataFrame({"x": x, "y": y}), "x", "y")

    result = compute_squared_correlation(pair)
    assert abs(result.value) < 0.15


def test_squared_correlation_returns_none_for_constant_input():
    df = pd.DataFrame({"x": [1.0] * 50, "y": list(range(50))})
    pair = validate_pair(df, "x", "y")

    result = compute_squared_correlation(pair)
    assert result.value is None
    assert result.available is True


def test_squared_correlation_returns_none_when_squared_x_is_constant():
    """X alternating between +c and -c makes X^2 constant even though X itself
    is not."""
    n = 50
    x = np.array([2.0, -2.0] * (n // 2))
    y = np.arange(n, dtype=float)
    pair = validate_pair(pd.DataFrame({"x": x, "y": y}), "x", "y")

    result = compute_squared_correlation(pair)
    assert result.value is None
    assert result.available is True


# --- Heteroscedasticity (variance-shape diagnostics) ---


def _reference_breusch_pagan_pvalue(x, y):
    """Independent Koenker Breusch-Pagan p-value via the single-regressor
    identity R²(e² ~ x) = corr(e², x)², a different arithmetic path than the
    polyfit-residual R² used in metrics/variance.py."""
    from scipy.stats import chi2, pearsonr

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    slope, intercept = np.polyfit(x, y, 1)
    e2 = (y - (slope * x + intercept)) ** 2
    r = pearsonr(x, e2)[0]
    lm = n * r**2
    return float(chi2.sf(lm, 1))


def test_heteroscedasticity_bp_matches_reference_on_funnel_data():
    rng = np.random.default_rng(0)
    n = 400
    x = rng.uniform(0, 4, size=n)
    y = x + rng.normal(0, 1, size=n) * (0.5 + x)  # spread grows with x
    pair = validate_pair(pd.DataFrame({"x": x, "y": y}), "x", "y")

    result = compute_heteroscedasticity(pair)
    expected = _reference_breusch_pagan_pvalue(x, y)

    assert result["bp_pvalue"].value == pytest.approx(expected, abs=1e-9)


def test_heteroscedasticity_detects_increasing_and_constant_spread():
    rng = np.random.default_rng(0)
    n = 400
    x = rng.uniform(0, 4, size=n)

    funnel_y = x + rng.normal(0, 1, size=n) * (0.5 + x)
    funnel = validate_pair(pd.DataFrame({"x": x, "y": funnel_y}), "x", "y")
    het = compute_heteroscedasticity(funnel)
    assert het["bp_pvalue"].value < 0.01  # clearly heteroscedastic
    assert het["gq_ratio"].value > 1.5  # more spread on the high-x side

    homo_y = x + rng.normal(0, 0.3, size=n)  # constant spread
    homo = validate_pair(pd.DataFrame({"x": x, "y": homo_y}), "x", "y")
    het_homo = compute_heteroscedasticity(homo)
    # Constant spread: Goldfeld-Quandt ratio near 1 in either direction.
    assert 1.0 / 1.5 < het_homo["gq_ratio"].value < 1.5


def test_heteroscedasticity_gq_ratio_below_one_for_decreasing_spread():
    rng = np.random.default_rng(1)
    n = 400
    x = rng.uniform(0, 4, size=n)
    y = x + rng.normal(0, 1, size=n) * (0.5 + (4 - x))  # spread shrinks with x
    pair = validate_pair(pd.DataFrame({"x": x, "y": y}), "x", "y")

    result = compute_heteroscedasticity(pair)
    assert result["gq_ratio"].value < 1.0 / 1.5


def test_heteroscedasticity_returns_none_for_constant_input():
    df = pd.DataFrame({"x": [1.0] * 80, "y": list(range(80))})
    pair = validate_pair(df, "x", "y")

    result = compute_heteroscedasticity(pair)
    assert result["bp_pvalue"].value is None
    assert result["gq_ratio"].value is None
    assert result["bp_pvalue"].available is True


def test_heteroscedasticity_returns_none_below_min_n():
    rng = np.random.default_rng(0)
    n = 40  # below _MIN_N_FOR_HETEROSCEDASTICITY (50)
    df = pd.DataFrame({"x": rng.uniform(0, 4, n), "y": rng.uniform(0, 4, n)})
    pair = validate_pair(df, "x", "y")

    result = compute_heteroscedasticity(pair)
    assert result["bp_pvalue"].value is None
    assert result["gq_ratio"].value is None


def test_heteroscedasticity_returns_none_for_perfect_linear_fit():
    """A perfect linear fit leaves no residual variance to test; the shared fit
    would otherwise divide by zero."""
    x = np.arange(80, dtype=float)
    y = 2.0 * x + 1.0  # exactly linear, zero residuals
    pair = validate_pair(pd.DataFrame({"x": x, "y": y}), "x", "y")

    result = compute_heteroscedasticity(pair)
    assert result["bp_pvalue"].value is None
    assert result["gq_ratio"].value is None


# --- Segmentation (single-breakpoint, mean-shape refinement) ---


def _reference_segmentation(x, y, min_frac=0.1):
    """Independent brute-force single-breakpoint search using per-segment
    np.polyfit / np.var, a different path than the prefix-sum identities in
    metrics/shape.py. Returns (segment_gain, stepness, breakpoint_x)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    order = np.argsort(x, kind="mergesort")
    xs, ys = x[order], y[order]

    def line_ssr(xx, yy):
        slope, intercept = np.polyfit(xx, yy, 1)
        return float(np.sum((yy - (slope * xx + intercept)) ** 2))

    def mean_ssr(a, b):
        return float(np.sum((a - a.mean()) ** 2) + np.sum((b - b.mean()) ** 2))

    sst = float(np.sum((ys - ys.mean()) ** 2))
    r2_line = 1.0 - line_ssr(xs, ys) / sst
    min_seg = max(5, int(min_frac * n))
    best_line = best_mean = None
    for k in range(min_seg, n - min_seg + 1):
        tl = line_ssr(xs[:k], ys[:k]) + line_ssr(xs[k:], ys[k:])
        tm = mean_ssr(ys[:k], ys[k:])
        if best_line is None or tl < best_line[0]:
            best_line = (tl, k)
        if best_mean is None or tm < best_mean[0]:
            best_mean = (tm, k)
    seg_gain = (1.0 - best_line[0] / sst) - r2_line
    step_gain = (1.0 - best_mean[0] / sst) - r2_line
    stepness = step_gain / seg_gain if seg_gain > 1e-6 else 0.0
    bp_x = 0.5 * (xs[best_mean[1] - 1] + xs[best_mean[1]])
    return seg_gain, stepness, bp_x


def test_segmentation_matches_reference_on_step_and_smooth():
    rng = np.random.default_rng(0)
    n = 407  # not a round number, to exercise the split arithmetic

    x_step = rng.uniform(-3, 3, size=n)
    y_step = np.where(x_step > 0, 1.0, -1.0) + rng.normal(0, 0.1, size=n)
    step_pair = validate_pair(pd.DataFrame({"x": x_step, "y": y_step}), "x", "y")

    x_smooth = rng.uniform(0, 3, size=n)
    y_smooth = np.exp(x_smooth) + rng.normal(0, 0.1, size=n)
    smooth_pair = validate_pair(pd.DataFrame({"x": x_smooth, "y": y_smooth}), "x", "y")

    for pair, (x, y) in (
        (step_pair, (x_step, y_step)),
        (smooth_pair, (x_smooth, y_smooth)),
    ):
        result = compute_segmentation(pair)
        exp_gain, exp_step, exp_bp = _reference_segmentation(x, y)
        assert result["segment_gain"].value == pytest.approx(exp_gain, abs=1e-9)
        assert result["segment_stepness"].value == pytest.approx(exp_step, abs=1e-9)
        assert result["breakpoint_x"].value == pytest.approx(exp_bp, abs=1e-9)


def test_segmentation_stepness_separates_step_from_smooth():
    rng = np.random.default_rng(0)
    n = 400

    x = rng.uniform(-3, 3, size=n)
    y_step = np.where(x > 0, 1.0, -1.0) + rng.normal(0, 0.1, size=n)
    step = compute_segmentation(
        validate_pair(pd.DataFrame({"x": x, "y": y_step}), "x", "y")
    )
    # Flat segments: the two-level model captures ~all the two-line improvement.
    assert step["segment_stepness"].value > 0.8
    # The breakpoint sits at the true cut.
    assert abs(step["breakpoint_x"].value) < 0.3

    xs = rng.uniform(0, 3, size=n)
    y_smooth = np.exp(xs) + rng.normal(0, 0.1, size=n)
    smooth = compute_segmentation(
        validate_pair(pd.DataFrame({"x": xs, "y": y_smooth}), "x", "y")
    )
    # Sloped segments: flattening them is strictly worse, so stepness <= 0.
    assert smooth["segment_stepness"].value < 0.5


def test_segmentation_returns_none_below_min_n():
    rng = np.random.default_rng(0)
    n = 40  # below _MIN_N_FOR_SEGMENTATION (50)
    df = pd.DataFrame({"x": rng.uniform(0, 3, n), "y": rng.uniform(0, 3, n)})
    pair = validate_pair(df, "x", "y")

    result = compute_segmentation(pair)
    assert result["segment_gain"].value is None
    assert result["breakpoint_x"].value is None
    assert result["segment_stepness"].value is None


def test_segmentation_returns_none_for_constant_input():
    df = pd.DataFrame({"x": [1.0] * 80, "y": list(range(80))})
    pair = validate_pair(df, "x", "y")

    result = compute_segmentation(pair)
    assert result["segment_gain"].value is None
    assert result["breakpoint_x"].available is True


# --- Regression influence (Cook's distance) ---


def _reference_cooks_distance(x, y):
    """Independent Cook's distance via an explicit hat matrix on the [1, x]
    design, a different path than the elementary one-predictor identities in
    metrics/influence.py."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    design = np.column_stack([np.ones(n), x])
    hat = design @ np.linalg.inv(design.T @ design) @ design.T
    h = np.diag(hat)
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    s2 = np.sum(resid**2) / (n - 2)
    return resid**2 * h / (2 * s2 * (1 - h) ** 2)


def test_influence_max_cook_matches_hat_matrix_reference():
    rng = np.random.default_rng(0)
    n = 200
    x = rng.uniform(-3, 3, size=n)
    y = x + rng.normal(0, 0.3, size=n)
    x[-1], y[-1] = 20.0, -20.0  # one strongly influential row
    pair = validate_pair(pd.DataFrame({"x": x, "y": y}), "x", "y")

    result = compute_influence(pair)
    expected = _reference_cooks_distance(x, y)

    assert result["max_cook_distance"].value == pytest.approx(
        float(expected.max()), rel=1e-9
    )


def test_influence_single_dominant_point_counts_one():
    rng = np.random.default_rng(0)
    n = 300
    x = rng.uniform(-3, 3, size=n)
    y = x + rng.normal(0, 0.3, size=n)
    x[-1], y[-1] = 20.0, -20.0  # high leverage, large residual
    pair = validate_pair(pd.DataFrame({"x": x, "y": y}), "x", "y")

    result = compute_influence(pair)
    assert result["max_cook_distance"].value > 1.0
    assert result["n_influential_points"].value == 1.0


def test_influence_detects_masked_leverage_cluster():
    # The outlier_driven scenario is a ~2% cluster of high-leverage points. They
    # mask each other, deflating each point's Cook's distance below the classical
    # D > 1 cutoff -- the softer 0.5 cutoff still counts them as a cluster.
    df = make_relationship("outlier_driven", n=500, noise=0.1, random_state=0)
    pair = validate_pair(df, "x", "y")

    result = compute_influence(pair)
    assert result["max_cook_distance"].value < 1.0  # masked below the strict cutoff
    assert result["n_influential_points"].value >= 2.0


def test_influence_clean_data_has_no_influential_points():
    rng = np.random.default_rng(0)
    n = 500
    x = rng.uniform(-3, 3, size=n)
    y = x + rng.normal(0, 0.3, size=n)
    pair = validate_pair(pd.DataFrame({"x": x, "y": y}), "x", "y")

    result = compute_influence(pair)
    assert result["n_influential_points"].value == 0.0
    assert result["max_cook_distance"].value < 0.5


def test_influence_returns_none_below_min_n():
    rng = np.random.default_rng(0)
    n = 40  # below _MIN_N_FOR_INFLUENCE (50)
    df = pd.DataFrame({"x": rng.uniform(-3, 3, n), "y": rng.uniform(-3, 3, n)})
    pair = validate_pair(df, "x", "y")

    result = compute_influence(pair)
    assert result["max_cook_distance"].value is None
    assert result["n_influential_points"].value is None


def test_influence_perfect_fit_reports_no_influence():
    x = np.arange(80, dtype=float)
    y = 2.0 * x + 1.0  # exactly linear, no residual structure
    pair = validate_pair(pd.DataFrame({"x": x, "y": y}), "x", "y")

    result = compute_influence(pair)
    assert result["max_cook_distance"].value == 0.0
    assert result["n_influential_points"].value == 0.0
