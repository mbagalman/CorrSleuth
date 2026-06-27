import sys

import numpy as np
import pandas as pd
import pytest

from corrsleuth.validation.input import validate_pair
from corrsleuth.api import profile_pair
from corrsleuth.exceptions import MetricComputationError, OptionalDependencyError
from corrsleuth.metrics import (
    compute_pearson, compute_spearman, compute_kendall,
    compute_distance_correlation, compute_mutual_information,
    compute_trimmed_pearson, compute_winsorized_pearson,
    compute_biweight_midcorrelation, compute_median_clipped_pearson,
    compute_chatterjee_xi, compute_chatterjee_xi_reverse,
)

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
    dc1 = compute_distance_correlation(pair, mode="standard", max_n_for_dcor=50)
    assert any("n_used > 50" in w for w in pair.warnings)

    # Disable cap
    pair2 = validate_pair(df, "x", "y")
    dc2 = compute_distance_correlation(pair2, mode="standard", max_n_for_dcor=None)
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


def test_distance_correlation_missing_dependency_returns_unavailable_in_lite_mode(monkeypatch):
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


def test_mutual_information_missing_dependency_returns_unavailable_in_lite_mode(monkeypatch):
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
    mi = result.metrics.loc[result.metrics["metric"] == "mutual_information", "value"].iloc[0]
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
    metrics = {
        row["metric"]: row["value"]
        for _, row in result.metrics.iterrows()
    }

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
    fwd_xi = next(r["value"] for _, r in fwd.metrics.iterrows() if r["metric"] == "chatterjee_xi")
    rev_xi = next(r["value"] for _, r in rev.metrics.iterrows() if r["metric"] == "chatterjee_xi")

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
    l = np.array([np.sum(ys >= v) for v in ys], dtype=float)
    numerator = n * np.sum(np.abs(np.diff(r)))
    denominator = 2.0 * np.sum(l * (n - l))
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


def test_chatterjee_xi_is_invariant_to_row_order_with_x_ties():
    """Shuffling the input rows must not change ξ — the metric should be a
    function of the (x, y) multiset, not the row order."""
    rng = np.random.default_rng(0)
    n = 200
    # Heavy ties on x via discretization; y stays continuous.
    x = rng.integers(0, 5, size=n).astype(float)
    y = rng.normal(size=n)
    df = pd.DataFrame({"x": x, "y": y})

    forward_xi = compute_chatterjee_xi(validate_pair(df, "x", "y"))
    reverse_xi = compute_chatterjee_xi_reverse(validate_pair(df, "x", "y"))

    df_shuffled = df.sample(frac=1, random_state=7).reset_index(drop=True)
    fwd_shuf = compute_chatterjee_xi(validate_pair(df_shuffled, "x", "y"))
    rev_shuf = compute_chatterjee_xi_reverse(validate_pair(df_shuffled, "x", "y"))

    assert forward_xi.value == fwd_shuf.value
    assert reverse_xi.value == rev_shuf.value


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
    assert metrics["chatterjee_xi"] - metrics["chatterjee_xi_reverse"] > 0.40
