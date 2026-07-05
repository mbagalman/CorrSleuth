import sys

import numpy as np
import pandas as pd
import pytest

from corrsleuth.api import profile_pair
from corrsleuth.exceptions import MetricComputationError, OptionalDependencyError
from corrsleuth.metrics import (
    compute_bin_lof_r2_gain,
    compute_biweight_midcorrelation,
    compute_chatterjee_xi,
    compute_chatterjee_xi_reverse,
    compute_distance_correlation,
    compute_kendall,
    compute_median_clipped_pearson,
    compute_mutual_information,
    compute_pearson,
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
