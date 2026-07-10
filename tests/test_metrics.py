import sys

import numpy as np
import pandas as pd
import pytest

from corrsleuth.api import profile_pair
from corrsleuth.datasets import make_relationship
from corrsleuth.exceptions import MetricComputationError, OptionalDependencyError
from corrsleuth.heuristics.classifier import (
    OSCILLATION_BIN_LOF_FLOOR,
    SQ_CORR_ROBUST_FLOOR,
    SQ_CORR_THRESHOLD,
)
from corrsleuth.metrics import (
    compute_bin_lof,
    compute_biweight_midcorrelation,
    compute_chatterjee_xi,
    compute_chatterjee_xi_reverse,
    compute_cluster_split,
    compute_distance_correlation,
    compute_heteroscedasticity,
    compute_heteroscedasticity_excluding,
    compute_influence,
    compute_influential_mask,
    compute_kendall,
    compute_median_clipped_pearson,
    compute_mutual_information,
    compute_pearson,
    compute_segmentation,
    compute_spearman,
    compute_squared_correlation,
    compute_squared_correlation_robust,
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
    rng = np.random.default_rng(0)
    x = rng.normal(size=60)
    df = pd.DataFrame({"x": x, "y": x + rng.normal(0, 0.5, 60)})
    pair = validate_pair(df, "x", "y")

    dc = compute_distance_correlation(pair, mode="standard")
    assert dc.available is True
    assert dc.value is not None

    mi = compute_mutual_information(pair, mode="standard")
    assert mi.available is True
    assert mi.value is not None


def test_mutual_information_discreteness_policy():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(1)

    # Discrete (low-cardinality integer) FEATURE with a continuous target:
    # declared discrete to scikit-learn, so MI is still computed (correctly).
    xd = rng.integers(0, 8, 400).astype(float)
    cont = xd + rng.normal(0, 1.0, 400)
    disc_feat = validate_pair(pd.DataFrame({"x": xd, "y": cont}), "x", "y")
    res_feat = compute_mutual_information(disc_feat, mode="standard")
    assert res_feat.value is not None and res_feat.value > 0

    # Discrete TARGET: mutual_info_regression assumes a continuous target, so MI
    # is withheld with a warning rather than reported wrong.
    pair_disc_y = validate_pair(pd.DataFrame({"x": cont, "y": xd}), "x", "y")
    res_tgt = compute_mutual_information(pair_disc_y, mode="standard")
    assert res_tgt.value is None
    assert any("target (y) is discrete" in w for w in pair_disc_y.warnings)

    # Continuous data is unaffected (treated as continuous features).
    xc = rng.normal(size=400)
    cont_pair = validate_pair(
        pd.DataFrame({"x": xc, "y": xc + rng.normal(0, 0.5, 400)}), "x", "y"
    )
    assert compute_mutual_information(cont_pair, mode="standard").value is not None


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


def test_lite_profile_with_standard_bootstrap_metrics_names_bootstrap_metrics(
    monkeypatch,
):
    """A lite-mode profile that opts into standard bootstrap metrics without the
    extras must fail with a message naming *bootstrap_metrics*, not "standard
    mode" — the profile mode is lite (C4 #5)."""
    _hide_module(monkeypatch, "dcor")
    df = make_relationship("linear_positive", n=80, random_state=42)

    with pytest.raises(OptionalDependencyError, match="bootstrap_metrics"):
        profile_pair(
            df, "x", "y", mode="lite", bootstrap=5, bootstrap_metrics="standard"
        )


def test_assess_outlier_sensitivity_treats_non_finite_baseline_as_unavailable():
    """A NaN/inf baseline Pearson must yield status 'unavailable', not 'stable':
    'stable' is an affirmative all-clear that would block the leverage label on no
    real evidence (C3 #3)."""
    from corrsleuth.metrics.robust import assess_outlier_sensitivity

    df = make_relationship("linear_positive", n=80, random_state=42)
    pair = validate_pair(df, "x", "y")

    assert assess_outlier_sensitivity(pair, float("nan")).status == "unavailable"
    assert assess_outlier_sensitivity(pair, float("inf")).status == "unavailable"
    # A finite baseline still resolves normally.
    assert assess_outlier_sensitivity(pair, 0.9).status in ("stable", "sensitive")


def test_deep_mode_requires_standard_extras_and_names_deep(monkeypatch):
    """deep is a superset of standard, so it also requires the [standard] extras
    and raises OptionalDependencyError — with a message naming *deep* mode, not
    standard — when they are missing."""
    _hide_module(monkeypatch, "dcor")
    df = pd.DataFrame({"x": range(60), "y": range(60)})
    with pytest.raises(OptionalDependencyError, match="deep mode"):
        profile_pair(df, "x", "y", mode="deep")


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


def test_deep_mode_is_a_superset_of_standard_plus_robust_and_xi():
    pytest.importorskip("dcor")
    pytest.importorskip("sklearn")
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
    # deep is a strict superset of standard: the standard metrics (distance
    # correlation, mutual information) plus the robust family and Chatterjee's xi.
    assert robust_metrics.isdisjoint(lite_metrics)
    assert robust_metrics <= deep_metrics
    assert {"chatterjee_xi", "chatterjee_xi_reverse"} <= deep_metrics
    assert "distance_correlation" in deep_metrics
    assert "mutual_information" in deep_metrics


def test_deep_mode_emits_one_small_sample_robust_warning():
    pytest.importorskip("dcor")
    pytest.importorskip("sklearn")
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
    pytest.importorskip("dcor")
    pytest.importorskip("sklearn")
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


def _canonical_bicor(x, y):
    """Textbook biweight midcorrelation (Langfelder & Horvath 2012): the Tukey
    rejection indicator is applied *per variable* — an x-outlier zeroes only its
    x-weight, keeping every point in all three sums. A different code path than
    metrics/robust.py."""
    from scipy import stats

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mx, my = np.median(x), np.median(y)
    madx = stats.median_abs_deviation(x, scale=1.0)
    mady = stats.median_abs_deviation(y, scale=1.0)
    ux = (x - mx) / (9.0 * madx)
    uy = (y - my) / (9.0 * mady)
    wx = np.where(np.abs(ux) < 1, (x - mx) * (1 - ux**2) ** 2, 0.0)
    wy = np.where(np.abs(uy) < 1, (y - my) * (1 - uy**2) ** 2, 0.0)
    return float(np.sum(wx * wy) / np.sqrt(np.sum(wx**2) * np.sum(wy**2)))


def _joint_mask_bicor(x, y):
    """The former (incorrect) implementation: a single joint mask drops the whole
    row whenever *either* variable rejects it."""
    from scipy import stats

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mx, my = np.median(x), np.median(y)
    madx = stats.median_abs_deviation(x, scale=1.0)
    mady = stats.median_abs_deviation(y, scale=1.0)
    ux = (x - mx) / (9.0 * madx)
    uy = (y - my) / (9.0 * mady)
    mask = (np.abs(ux) < 1) & (np.abs(uy) < 1)
    xc = (x[mask] - mx) * (1 - ux[mask] ** 2) ** 2
    yc = (y[mask] - my) * (1 - uy[mask] ** 2) ** 2
    return float(np.sum(xc * yc) / np.sqrt(np.sum(xc**2) * np.sum(yc**2)))


def test_biweight_midcorrelation_matches_canonical_per_variable_oracle():
    rng = np.random.default_rng(7)
    n = 120
    x = rng.normal(0.0, 1.0, size=n)
    y = 0.8 * x + rng.normal(0.0, 0.4, size=n)
    # One-sided contamination: points extreme in x but near the y-median, and
    # vice versa. A joint mask discards them entirely; the canonical estimator
    # keeps each in the non-outlying variable's scale.
    x[0], y[0] = 40.0, 0.0
    x[1], y[1] = -40.0, 0.1
    x[2], y[2] = 0.0, 40.0
    x[3], y[3] = 0.05, -40.0
    pair = validate_pair(pd.DataFrame({"x": x, "y": y}), "x", "y")

    result = compute_biweight_midcorrelation(pair)

    assert result.value == pytest.approx(_canonical_bicor(x, y), abs=1e-12)
    # And the fix genuinely changed behavior: the old joint-mask value differs.
    assert abs(result.value - _joint_mask_bicor(x, y)) > 1e-4


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
    pytest.importorskip("dcor")
    pytest.importorskip("sklearn")
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
    pytest.importorskip("dcor")
    pytest.importorskip("sklearn")
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
    # Breaks x-ties by y — the exact leak the production code's seeded random
    # tie-break avoids. Fine here ONLY because every caller passes a continuous,
    # tie-free x; do NOT reuse this oracle with a tied sort variable.
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
    pytest.importorskip("dcor")
    pytest.importorskip("sklearn")
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
    pytest.importorskip("dcor")
    pytest.importorskip("sklearn")
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
    """Independent reimplementation of the df-adjusted equal-frequency-bin
    lack-of-fit test, used as an oracle below. Uses an explicit per-bin loop for
    the bin residual SS (rather than the vectorized ``array_split`` + broadcast
    approach in ``metrics/shape.py``) and the R^2-equals-squared-Pearson-r
    identity to derive the linear residual SS (rather than an explicit polyfit
    residual sum), so the two implementations share only the bin-split rule and
    the adjusted-R^2 formula, not the arithmetic path."""
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
    from scipy.stats import pearsonr

    r_lin, _ = pearsonr(xs, ys)
    ss_res_linear = ss_tot * (1.0 - r_lin**2)
    # Adjusted R^2 per model: 1 - (SS_res / (n - p)) / (SS_tot / (n - 1)),
    # p = n_bins for the bin model, 2 for the line.
    adj_r2_bins = 1.0 - (ss_res_bins / (n - n_bins)) / (ss_tot / (n - 1))
    adj_r2_linear = 1.0 - (ss_res_linear / (n - 2)) / (ss_tot / (n - 1))
    return adj_r2_bins - adj_r2_linear


def test_bin_lof_r2_gain_matches_reference_on_curved_data():
    rng = np.random.default_rng(0)
    n = 307  # deliberately not evenly divisible by any bin count in range
    x = rng.uniform(0, 3, size=n)
    y = np.exp(x) + rng.normal(0, 0.1, size=n) * np.exp(x).std()
    pair = validate_pair(pd.DataFrame({"x": x, "y": y}), "x", "y")

    result = compute_bin_lof(pair)["bin_lof_r2_gain"]
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

    linear_gain = compute_bin_lof(linear_pair)["bin_lof_r2_gain"].value
    curved_gain = compute_bin_lof(curved_pair)["bin_lof_r2_gain"].value

    assert linear_gain < 0.05
    assert curved_gain > 0.5


def test_bin_lof_returns_none_for_constant_input():
    df = pd.DataFrame({"x": [1.0] * 60, "y": list(range(60))})
    pair = validate_pair(df, "x", "y")

    result = compute_bin_lof(pair)
    assert result["bin_lof_r2_gain"].value is None
    assert result["bin_reversal_count"].value is None
    assert result["bin_lof_r2_gain"].available is True


def test_bin_lof_returns_none_below_min_n():
    rng = np.random.default_rng(0)
    n = 40  # below _MIN_N_FOR_BIN_LOF (50)
    df = pd.DataFrame({"x": rng.uniform(size=n), "y": rng.uniform(size=n)})
    pair = validate_pair(df, "x", "y")

    result = compute_bin_lof(pair)
    assert result["bin_lof_r2_gain"].value is None
    assert result["bin_reversal_count"].value is None
    assert result["bin_lof_r2_gain"].available is True


def test_bin_lof_withheld_for_too_few_distinct_x():
    """When X has too few distinct values to form the minimum bin count without
    splitting a tied run, the bin-lof diagnostics are *withheld* (no_value)
    rather than computed on order-dependent bins. Binning is tie-safe — a
    boundary never splits a run of equal X — so with only a handful of distinct
    values (here 4, one dominating) fewer than `_MIN_BINS` bins survive and the
    diagnostic is not reported. It must not raise."""
    n = 200
    x = np.concatenate([np.zeros(n - 3), [1.0, 2.0, 3.0]])
    y = np.arange(n, dtype=float)
    pair = validate_pair(pd.DataFrame({"x": x, "y": y}), "x", "y")

    result = compute_bin_lof(pair)
    assert result["bin_lof_r2_gain"].available is True
    assert result["bin_lof_r2_gain"].value is None
    assert result["bin_reversal_count"].value is None


def test_bin_lof_and_segmentation_invariant_under_row_permutation_with_ties():
    """Reproducibility: an identical dataset with tied X must yield identical
    shape diagnostics regardless of row order. The previous position-based
    binning split tied runs by input order, so a reorder could flip
    `bin_lof_r2_gain` / `bin_reversal_count` (and the label). Tie-safe binning
    partitions only at value changes, so the diagnostics are invariant to
    machine precision."""
    rng = np.random.default_rng(7)
    n = 300
    x = np.round(rng.uniform(-3, 3, n), 1)  # many ties (~40-60 distinct values)
    y = x**2 + rng.normal(0, 1.0, n)
    perm = rng.permutation(n)

    def diagnostics(xx, yy):
        pair = validate_pair(pd.DataFrame({"x": xx, "y": yy}), "x", "y")
        bl = compute_bin_lof(pair)
        seg = compute_segmentation(pair)
        return np.array(
            [
                bl["bin_lof_r2_gain"].value,
                bl["bin_reversal_count"].value,
                bl["bin_lof_r2_gain_robust"].value,
                seg["segment_gain"].value,
                seg["segment_stepness"].value,
            ]
        )

    base = diagnostics(x, y)
    permuted = diagnostics(x[perm], y[perm])
    assert np.allclose(base, permuted, rtol=0, atol=1e-9)
    assert base[1] == permuted[1]  # reversal count identical exactly


def test_bin_reversal_count_separates_oscillation_from_single_bend():
    rng = np.random.default_rng(0)
    n = 500

    # A ~2.5-cycle sinusoid: several genuine direction reversals.
    x_sin = rng.uniform(0, 5 * np.pi, size=n)
    y_sin = np.sin(x_sin) + rng.normal(0, 0.1, size=n)
    sine = validate_pair(pd.DataFrame({"x": x_sin, "y": y_sin}), "x", "y")
    sine_result = compute_bin_lof(sine)
    assert sine_result["bin_reversal_count"].value >= 2
    assert sine_result["bin_lof_r2_gain"].value > 0.3

    # A U-shape: exactly one bend, so exactly one reversal.
    x_u = rng.uniform(-3, 3, size=n)
    y_u = x_u**2 + rng.normal(0, 0.1, size=n)
    u_shape = validate_pair(pd.DataFrame({"x": x_u, "y": y_u}), "x", "y")
    assert compute_bin_lof(u_shape)["bin_reversal_count"].value == 1

    # A monotone trend: no reversals.
    x_lin = rng.uniform(-3, 3, size=n)
    y_lin = x_lin + rng.normal(0, 0.1, size=n)
    linear = validate_pair(pd.DataFrame({"x": x_lin, "y": y_lin}), "x", "y")
    assert compute_bin_lof(linear)["bin_reversal_count"].value == 0


def test_bin_lof_robust_gain_collapses_for_heavy_tailed_y_artifact():
    """The leave-one-bin-out gain (``bin_lof_r2_gain_robust``) tracks the raw
    gain for a genuine oscillation but collapses when a lone extreme Y in one
    bin manufactures the gain — the heavy-tailed-Y artifact FU-U guards against."""
    # Genuine sinusoid: gain is spread across bins, so dropping any one barely
    # dents it -- the robust gain stays close to the raw gain and well above the
    # oscillation floor.
    rng = np.random.default_rng(0)
    x_sin = rng.uniform(-np.pi, np.pi, size=200)
    y_sin = np.sin(1.5 * x_sin) + rng.normal(0, 0.3, size=200)
    sine = compute_bin_lof(
        validate_pair(pd.DataFrame({"x": x_sin, "y": y_sin}), "x", "y")
    )
    raw = sine["bin_lof_r2_gain"].value
    robust = sine["bin_lof_r2_gain_robust"].value
    assert robust > OSCILLATION_BIN_LOF_FLOOR
    assert robust > raw - 0.1  # barely moves

    # Heavy-tailed target vs an independent predictor, on a seed whose unlucky
    # draw pushes the raw gain over the 0.15 oscillation floor. The robust gain
    # collapses far below it: the "structure" was one outlier-dominated bin.
    rng = np.random.default_rng(387)
    x_ht = rng.normal(size=100)
    y_ht = np.exp(rng.uniform(0.1, 10, size=100))
    ht = compute_bin_lof(validate_pair(pd.DataFrame({"x": x_ht, "y": y_ht}), "x", "y"))
    assert ht["bin_lof_r2_gain"].value > OSCILLATION_BIN_LOF_FLOOR
    assert ht["bin_lof_r2_gain_robust"].value < OSCILLATION_BIN_LOF_FLOOR


def test_bin_reversal_count_known_zigzag_is_exact():
    """Known-answer check: a noiseless triangle wave whose bin means form an
    exact up-down-up-down zigzag must count exactly 3 reversals (4 legs)."""
    n = 400  # -> 20 bins of 20 rows
    x = np.linspace(0, 4, n, endpoint=False)
    y = np.abs((x % 2) - 1)  # triangle wave: down, up, down, up over [0, 4)
    pair = validate_pair(pd.DataFrame({"x": x, "y": y}), "x", "y")

    result = compute_bin_lof(pair)
    assert result["bin_reversal_count"].value == 3
    assert result["bin_lof_r2_gain"].value > 0.3


@pytest.mark.parametrize("n", [100, 500])
def test_bin_reversal_count_noise_has_many_reversals_but_tiny_gain(n):
    """The reversal count alone must never be trusted: pure noise reverses
    direction constantly, but its df-adjusted bin-fit gain stays near zero and
    well below the oscillation floor — the joint gate (OSCILLATION_BIN_LOF_FLOOR)
    is what separates it from a sinusoid. Checked at n=100 too, where the
    unadjusted statistic's null bias was largest."""
    rng = np.random.default_rng(0)
    x = rng.uniform(-3, 3, size=n)
    y = rng.normal(0, 1, size=n)
    pair = validate_pair(pd.DataFrame({"x": x, "y": y}), "x", "y")

    result = compute_bin_lof(pair)
    assert result["bin_reversal_count"].value >= 2  # noise wiggles a lot
    assert (
        result["bin_lof_r2_gain"].value < OSCILLATION_BIN_LOF_FLOOR
    )  # explains nothing


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


def test_squared_correlation_robust_collapses_for_heavy_tailed_artifact():
    """The robust sq_corr (drop the few most extreme points, take the min |corr|)
    tracks the raw value for a genuine magnitude link but collapses when a
    heavy-tailed variable manufactures the raw sq_corr with a handful of extreme
    squared values — the FU-V artifact."""
    # Genuine U-shape: the magnitude link is spread over many points, so dropping
    # the extreme few barely dents it — robust stays above the classifier floor.
    rng = np.random.default_rng(0)
    x = rng.uniform(-3, 3, size=400)
    df = pd.DataFrame({"x": x, "y": x**2 + rng.normal(0, 0.3, size=400)})
    pair = validate_pair(df, "x", "y")
    raw = abs(compute_squared_correlation(pair).value)
    robust = compute_squared_correlation_robust(pair).value
    assert raw > SQ_CORR_THRESHOLD
    assert robust > SQ_CORR_ROBUST_FLOOR
    assert robust > raw - 0.15  # barely moves

    # Heavy-tailed target vs an independent predictor, seed 574: the raw sq_corr
    # clears the 0.35 threshold but is carried by a few extreme values, so the
    # robust value collapses below the floor.
    rng = np.random.default_rng(574)
    x_ht = rng.normal(size=100)
    y_ht = np.exp(rng.uniform(0.1, 10, size=100))
    ht = validate_pair(pd.DataFrame({"x": x_ht, "y": y_ht}), "x", "y")
    assert abs(compute_squared_correlation(ht).value) > SQ_CORR_THRESHOLD
    assert compute_squared_correlation_robust(ht).value < SQ_CORR_ROBUST_FLOOR


def test_squared_correlation_robust_returns_none_for_constant_input():
    df = pd.DataFrame({"x": [1.0] * 50, "y": list(range(50))})
    pair = validate_pair(df, "x", "y")
    result = compute_squared_correlation_robust(pair)
    assert result.name == "sq_corr_robust"
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
    assert result["bowtie_ratio"].value is None
    assert result["bp_pvalue"].available is True


def test_heteroscedasticity_returns_none_below_min_n():
    rng = np.random.default_rng(0)
    n = 40  # below _MIN_N_FOR_HETEROSCEDASTICITY (50)
    df = pd.DataFrame({"x": rng.uniform(0, 4, n), "y": rng.uniform(0, 4, n)})
    pair = validate_pair(df, "x", "y")

    result = compute_heteroscedasticity(pair)
    assert result["bp_pvalue"].value is None
    assert result["gq_ratio"].value is None
    assert result["bowtie_ratio"].value is None


def test_heteroscedasticity_returns_none_for_perfect_linear_fit():
    """A perfect linear fit leaves no residual variance to test; the shared fit
    would otherwise divide by zero."""
    x = np.arange(80, dtype=float)
    y = 2.0 * x + 1.0  # exactly linear, zero residuals
    pair = validate_pair(pd.DataFrame({"x": x, "y": y}), "x", "y")

    result = compute_heteroscedasticity(pair)
    assert result["bp_pvalue"].value is None
    assert result["gq_ratio"].value is None
    assert result["bowtie_ratio"].value is None


def test_heteroscedasticity_handles_binary_x_without_raising():
    """A low-cardinality/binary x (e.g. a 0/1 flag) can leave an entire
    x-sorted Goldfeld-Quandt group with zero x-variance -- np.polyfit's design
    matrix is singular there. This must fall back gracefully, not raise."""
    rng = np.random.default_rng(0)
    n = 1000
    x = (rng.uniform(size=n) < 0.95).astype(float)  # heavily skewed 0/1 flag
    y = rng.normal(size=n)
    pair = validate_pair(pd.DataFrame({"x": x, "y": y}), "x", "y")

    result = compute_heteroscedasticity(pair)
    assert result["bp_pvalue"].value is not None
    assert result["gq_ratio"].value is not None
    assert result["bowtie_ratio"].value is not None


def _reference_bowtie_ratio(x, y):
    """Independent edge-vs-middle variance ratio: fit the line with np.polyfit,
    split x-sorted residuals into thirds via plain array slicing (rather than
    np.array_split) as a different arithmetic path than metrics/variance.py."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    order = np.argsort(x, kind="mergesort")
    resid_sorted = resid[order]
    third = n // 3
    low, mid, high = (
        resid_sorted[:third],
        resid_sorted[third : n - third],
        resid_sorted[n - third :],
    )
    mid_ms = np.mean(mid**2)
    edge_ms = np.mean(np.concatenate([low, high]) ** 2)
    return float(edge_ms / mid_ms)


def test_bowtie_ratio_matches_reference_on_symmetric_variance_data():
    rng = np.random.default_rng(0)
    n = 900
    x = rng.uniform(-4, 4, size=n)
    y = x + rng.normal(0, 1, size=n) * (0.5 + np.abs(x))  # spread high at both ends
    pair = validate_pair(pd.DataFrame({"x": x, "y": y}), "x", "y")

    result = compute_heteroscedasticity(pair)
    # np.array_split's remainder distribution differs slightly from plain
    # slicing at n not divisible by 3, so allow a small relative tolerance
    # rather than requiring exact equality.
    expected = _reference_bowtie_ratio(x, y)
    assert result["bowtie_ratio"].value == pytest.approx(expected, rel=0.05)


def test_bowtie_ratio_detects_symmetric_variance_and_ignores_funnel():
    rng = np.random.default_rng(0)
    n = 900
    x = rng.uniform(-4, 4, size=n)

    bowtie_y = x + rng.normal(0, 1, size=n) * (0.5 + np.abs(x))
    bowtie = validate_pair(pd.DataFrame({"x": x, "y": bowtie_y}), "x", "y")
    het_bowtie = compute_heteroscedasticity(bowtie)
    assert het_bowtie["bowtie_ratio"].value > 2.5  # clearly edge-high

    # A one-directional funnel has similar low-x/high-x variance to a bowtie's
    # low+high combined, but its middle third is not calm -- bowtie_ratio
    # should stay near 1, confirming the two checks measure different shapes.
    x_pos = rng.uniform(0, 4, size=n)
    funnel_y = x_pos + rng.normal(0, 1, size=n) * (0.5 + x_pos)
    funnel = validate_pair(pd.DataFrame({"x": x_pos, "y": funnel_y}), "x", "y")
    het_funnel = compute_heteroscedasticity(funnel)
    assert 1.0 / 1.5 < het_funnel["bowtie_ratio"].value < 1.5

    homo_y = x + rng.normal(0, 0.3, size=n)
    homo = validate_pair(pd.DataFrame({"x": x, "y": homo_y}), "x", "y")
    het_homo = compute_heteroscedasticity(homo)
    assert 1.0 / 1.5 < het_homo["bowtie_ratio"].value < 1.5


def test_bowtie_ratio_below_one_for_center_high_spread():
    rng = np.random.default_rng(2)
    n = 900
    x = rng.uniform(-4, 4, size=n)
    y = x + rng.normal(0, 1, size=n) * (2.5 - np.abs(x))  # spread high in the middle
    pair = validate_pair(pd.DataFrame({"x": x, "y": y}), "x", "y")

    result = compute_heteroscedasticity(pair)
    assert result["bowtie_ratio"].value < 1.0 / 2.5


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


def test_segmentation_is_numerically_stable_under_large_offset():
    rng = np.random.default_rng(1)
    n = 300
    x = rng.uniform(-3, 3, size=n)
    y = np.where(x > 0, 1.0, -1.0) + rng.normal(0, 0.1, size=n)

    base = compute_segmentation(validate_pair(pd.DataFrame({"x": x, "y": y}), "x", "y"))

    # A large common offset makes the raw sums of squares (~offset**2) dwarf the
    # residuals (~1). Without mean-centering the prefix-sum identities lose the
    # signal to catastrophic cancellation; with it, the shape statistics are
    # invariant and the breakpoint simply tracks the shift.
    offset = 1e8
    shifted = compute_segmentation(
        validate_pair(pd.DataFrame({"x": x + offset, "y": y + offset}), "x", "y")
    )

    assert shifted["segment_gain"].value == pytest.approx(
        base["segment_gain"].value, rel=1e-6, abs=1e-9
    )
    assert shifted["segment_stepness"].value == pytest.approx(
        base["segment_stepness"].value, rel=1e-6, abs=1e-9
    )
    assert shifted["breakpoint_x"].value == pytest.approx(
        base["breakpoint_x"].value + offset, rel=1e-12
    )
    # Sanity: the offset case still reads as a clean step, not numerical mush.
    assert shifted["segment_stepness"].value > 0.8


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


# --- compute_influential_mask / compute_heteroscedasticity_excluding (ticket 1.5) ---


def test_influential_mask_matches_n_influential_points_count():
    rng = np.random.default_rng(0)
    n = 300
    x = rng.uniform(-3, 3, size=n)
    y = x + rng.normal(0, 0.3, size=n)
    x[-1], y[-1] = 20.0, -20.0  # one strongly influential row
    pair = validate_pair(pd.DataFrame({"x": x, "y": y}), "x", "y")

    mask = compute_influential_mask(pair)
    result = compute_influence(pair)
    assert mask is not None
    assert mask.dtype == bool
    assert mask.shape == (n,)
    assert int(mask.sum()) == int(result["n_influential_points"].value)
    assert mask[-1]  # the injected row is the one flagged


def test_influential_mask_none_below_min_n():
    rng = np.random.default_rng(0)
    n = 40  # below _MIN_N_FOR_INFLUENCE (50)
    df = pd.DataFrame({"x": rng.uniform(-3, 3, n), "y": rng.uniform(-3, 3, n)})
    pair = validate_pair(df, "x", "y")

    assert compute_influential_mask(pair) is None


def test_influential_mask_none_for_constant_input():
    df = pd.DataFrame({"x": [1.0] * 80, "y": list(range(80))})
    pair = validate_pair(df, "x", "y")

    assert compute_influential_mask(pair) is None


def test_heteroscedasticity_excluding_removes_leverage_artifact_signal():
    # A single manufactured outlier creates a spurious Goldfeld-Quandt/bowtie
    # signal on the full sample; excluding it (matching the row
    # compute_influential_mask flags) must make the signal disappear, since
    # the remainder is pure noise. This is the X13 pattern from ticket 1.5.
    rng = np.random.default_rng(0)
    n = 500
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    x[-1], y[-1] = 20.0, 20.0
    pair = validate_pair(pd.DataFrame({"x": x, "y": y}), "x", "y")

    full = compute_heteroscedasticity(pair)
    mask = compute_influential_mask(pair)
    assert mask is not None and mask.sum() == 1

    excluded = compute_heteroscedasticity_excluding(pair, mask)
    # Full-sample signal is real (BP rejects, some effect-size ratio clears a
    # floor); after excluding the flagged row it must no longer clear either
    # the funnel or the bowtie floor.
    assert full["bp_pvalue"].value < 0.05
    assert excluded["bp_pvalue"].value > 0.05


def test_heteroscedasticity_excluding_none_when_subset_drops_below_min_n():
    rng = np.random.default_rng(0)
    n = 55  # just above _MIN_N_FOR_HETEROSCEDASTICITY (50)
    x = rng.uniform(0, 4, size=n)
    y = x + rng.normal(0, 1, size=n) * (0.5 + x)
    pair = validate_pair(pd.DataFrame({"x": x, "y": y}), "x", "y")

    # Exclude enough rows to push the remainder below the floor.
    mask = np.zeros(n, dtype=bool)
    mask[:10] = True

    result = compute_heteroscedasticity_excluding(pair, mask)
    assert result["bp_pvalue"].value is None
    assert result["gq_ratio"].value is None
    assert result["bowtie_ratio"].value is None


# --- Two-group / mixture split diagnostics (cluster_split_*) ---


def _two_blob_pair(n=500, sep=5.0, frac=0.5, within=0.0, seed=0):
    """Two diagonal Gaussian blobs `sep` within-stds apart; `within` adds a
    within-group linear trend."""
    rng = np.random.default_rng(seed)
    n1 = int(n * frac)
    x = np.concatenate([rng.normal(0, 1, n1), rng.normal(sep, 1, n - n1)])
    y = within * x + np.concatenate([rng.normal(0, 1, n1), rng.normal(sep, 1, n - n1)])
    return validate_pair(pd.DataFrame({"x": x, "y": y}), "x", "y")


def test_cluster_split_detects_two_separated_blobs():
    result = compute_cluster_split(_two_blob_pair())

    assert result["cluster_split_r2"].value > 0.85
    # "Almost no points bridging the gap": the boundary band is near-empty.
    assert result["cluster_valley_share"].value < 0.03
    assert result["cluster_min_share"].value == pytest.approx(0.5, abs=0.05)
    # The association collapses inside the groups (it is pure between-group shift).
    assert result["pearson_within_cluster"].value < 0.15


def test_cluster_split_unimodal_correlated_pair_stays_below_gates():
    """A plain bivariate normal is the null: its best two-group split explains
    at most ~0.64 of the projection variance (2/pi asymptotically) and its
    density peaks exactly where 2-means splits it, so the valley band is full."""
    rng = np.random.default_rng(1)
    n = 500
    z = rng.normal(size=n)
    y = 0.7 * z + np.sqrt(1 - 0.49) * rng.normal(size=n)
    pair = validate_pair(pd.DataFrame({"x": z, "y": y}), "x", "y")

    result = compute_cluster_split(pair)
    assert result["cluster_split_r2"].value < 0.70
    assert result["cluster_valley_share"].value > 0.05


def test_cluster_split_unavailable_below_min_n_and_for_constant():
    rng = np.random.default_rng(2)
    small = validate_pair(
        pd.DataFrame({"x": rng.normal(size=60), "y": rng.normal(size=60)}), "x", "y"
    )
    for res in compute_cluster_split(small).values():
        assert res.value is None

    const = validate_pair(
        pd.DataFrame({"x": [3.0] * 200, "y": list(range(200))}), "x", "y"
    )
    for res in compute_cluster_split(const).values():
        assert res.value is None


def test_cluster_split_unavailable_for_coarse_discrete_data():
    """Ordinal/Likert data quantizes the projection onto a lattice whose empty
    inter-level spacing fakes a perfect valley, so the diagnostics are withheld
    (the tie-rate validation warnings already cover such columns)."""
    rng = np.random.default_rng(3)
    n = 400
    x = rng.integers(1, 6, n).astype(float)
    y = np.clip(x + rng.integers(-1, 2, n), 1, 5).astype(float)
    pair = validate_pair(pd.DataFrame({"x": x, "y": y}), "x", "y")

    for res in compute_cluster_split(pair).values():
        assert res.value is None


def test_cluster_split_within_pearson_reports_surviving_trend():
    """Blobs with a genuine within-group slope keep a high within-group Pearson
    -- the value that lets the classifier tell a mixture-driven correlation from
    clusters riding on a real trend."""
    result = compute_cluster_split(_two_blob_pair(within=0.8, seed=4))

    assert result["cluster_split_r2"].value > 0.70  # the split is still there
    assert result["pearson_within_cluster"].value > 0.5  # ...but the trend survives


# --- Discontinuity diagnostics (segment_jump_ratio) ---


def _jump_pair(n=500, js=6.0, sigma=0.3, seed=0, slope_change=False):
    """A linear trend with a `js`-sigma level shift at x=0."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(-3, 3, size=n)
    if slope_change:
        y = np.where(u > 0, 0.5 * u + js * sigma, u) + rng.normal(0, sigma, n)
    else:
        y = u + js * sigma * (u > 0) + rng.normal(0, sigma, n)
    return validate_pair(pd.DataFrame({"x": u, "y": y}), "x", "y")


def test_segment_jump_ratio_measures_discontinuity_in_sigmas():
    """A 6-sigma level shift reads ~6 on the jump-ratio scale, with or without
    a slope change -- while segment_gain (R-squared scale) stays tiny because
    the dominant trend soaks up the variance."""
    for slope_change in (False, True):
        seg = compute_segmentation(_jump_pair(slope_change=slope_change))
        assert 4.0 < seg["segment_jump_ratio"].value < 9.0
        assert seg["segment_gain"].value < 0.25


def test_segment_jump_ratio_near_zero_for_continuous_shapes():
    """Continuous relationships -- a line, a piecewise-linear kink (however
    sharp), a smooth curve -- are fitted by two lines that nearly meet at the
    boundary, so the ratio sits far below the 3.0 gate."""
    rng = np.random.default_rng(1)
    n = 500
    u = rng.uniform(-3, 3, size=n)
    shapes = {
        "linear": 0.8 * u + rng.normal(0, 0.3, n),
        "kink": np.where(u > 0, 2.0 * u, 0.2 * u) + rng.normal(0, 0.3, n),
        "exp": np.exp(0.8 * u) + rng.normal(0, 0.2, n),
    }
    for name, y in shapes.items():
        pair = validate_pair(pd.DataFrame({"x": u, "y": y}), "x", "y")
        ratio = compute_segmentation(pair)["segment_jump_ratio"].value
        assert ratio < 2.0, f"{name}: {ratio}"


def test_segment_jump_ratio_localization_collapses_smooth_sigmoid():
    """A moderate sigmoid fakes a boundary gap under the global two-line fit
    (its tails displace the chords vertically), but the localized refit tracks
    the curve through the boundary, so the reported min collapses."""
    rng = np.random.default_rng(2)
    n = 500
    u = rng.uniform(-3, 3, size=n)
    y = 1 / (1 + np.exp(-2 * u)) + rng.normal(0, 0.05, n)
    pair = validate_pair(pd.DataFrame({"x": u, "y": y}), "x", "y")
    assert compute_segmentation(pair)["segment_jump_ratio"].value < 2.0


def test_segment_jump_ratio_unavailable_below_dedicated_floor():
    """Below n=150 a moderate smooth curve is not reliably separable from a
    jump, so the ratio is withheld while the other segmentation outputs (which
    keep the n>=50 floor) still report."""
    seg = compute_segmentation(_jump_pair(n=120))
    assert seg["segment_jump_ratio"].value is None
    assert seg["segment_gain"].value is not None
    assert seg["segment_stepness"].value is not None
