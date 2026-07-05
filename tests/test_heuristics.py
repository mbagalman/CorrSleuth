import pytest

from corrsleuth.api import profile_pair
from corrsleuth.datasets import make_relationship
from corrsleuth.heuristics.classifier import apply_heuristics, detect_metric_warnings
from corrsleuth.result import MetricResult


def test_canonical_shapes_lite():
    # Linear Positive
    df = make_relationship("linear_positive", n=500, noise=0.1, random_state=42)
    res = profile_pair(df, "x", "y", mode="lite")
    assert res.pattern == "near_linear"


def test_canonical_shapes_standard():
    pytest.importorskip("dcor")
    pytest.importorskip("sklearn")
    # U Shape -> nonmonotonic_dependence
    df = make_relationship("u_shape", n=500, noise=0.1, random_state=42)
    res = profile_pair(df, "x", "y", mode="standard")
    assert res.pattern == "nonmonotonic_dependence"

    # Monotonic Log -> monotonic_nonlinear
    df = make_relationship("monotonic_log", n=500, noise=0.1, random_state=42)
    res = profile_pair(df, "x", "y", mode="standard")
    assert res.pattern == "monotonic_nonlinear"

    # Independent -> weak_or_no_relationship
    df = make_relationship("independent", n=500, noise=0.1, random_state=42)
    res = profile_pair(df, "x", "y", mode="standard")
    assert res.pattern == "weak_or_no_relationship"


def test_exponential_monotonic_resolves_to_monotonic_nonlinear():
    # A smooth monotonic curve over an ordinary (non-rigged) X range: Pearson
    # and Spearman stay close together (gap well under
    # RANK_LINEAR_GAP_THRESHOLD), so only bin_lof_r2_gain catches the real
    # curvature. Lite mode: no optional dependency needed.
    df = make_relationship("exponential_monotonic", n=500, noise=0.1, random_state=42)
    res = profile_pair(df, "x", "y", mode="lite")
    assert res.pattern == "monotonic_nonlinear"


def test_logarithmic_monotonic_resolves_to_monotonic_nonlinear():
    df = make_relationship("logarithmic_monotonic", n=500, noise=0.1, random_state=42)
    res = profile_pair(df, "x", "y", mode="lite")
    assert res.pattern == "monotonic_nonlinear"


def test_threshold_step_resolves_to_monotonic_nonlinear():
    # A two-level step function sits in near_linear's regime by the
    # rank-linear gap alone (Pearson and Spearman both moderately strong, gap
    # small), but bin_lof_r2_gain reveals the two flat groups a line doesn't
    # capture.
    df = make_relationship("threshold_step", n=500, noise=0.1, random_state=42)
    res = profile_pair(df, "x", "y", mode="lite")
    assert res.pattern == "monotonic_nonlinear"


def test_circular_resolves_to_nonmonotonic_dependence():
    # Points scattered around a ring: Pearson, Spearman, and distance
    # correlation on the raw values are all near zero (distance correlation is
    # structurally capped around ~0.2 for a true circular relationship), but
    # sq_corr (corr(X^2, Y^2)) is strongly negative. Lite mode: no optional
    # dependency needed.
    df = make_relationship("circular", n=500, noise=0.1, random_state=42)
    res = profile_pair(df, "x", "y", mode="lite")
    assert res.pattern == "nonmonotonic_dependence"


@pytest.mark.parametrize("shape", ["linear_positive", "linear_negative"])
@pytest.mark.parametrize("seed", range(10))
def test_linear_shapes_stay_near_linear_across_seeds(shape, seed):
    # BIN_LOF_R2_GAIN_THRESHOLD (0.05) has a thinner margin than most cascade
    # thresholds (genuinely linear data measures ~-0.01 on any single seed), so
    # this regression check runs across many seeds rather than relying on one.
    df = make_relationship(shape, n=500, noise=0.1, random_state=seed)
    res = profile_pair(df, "x", "y", mode="lite")
    assert res.pattern == "near_linear"


def test_conflicting_signs_warning():
    import numpy as np
    import pandas as pd

    # Create an artificial dataset where Pearson is strongly positive, but Spearman is strongly negative
    # E.g. mostly negative trend but one huge positive outlier
    x = np.arange(50, dtype=float)
    y = -np.arange(50, dtype=float)
    x[-1] = 10000
    y[-1] = 10000

    df = pd.DataFrame({"x": x, "y": y})
    res = profile_pair(df, "x", "y")

    # Check that Pearson is > 0.3 and Spearman is < -0.3
    p = float(res.metrics[res.metrics["metric"] == "pearson"]["value"].iloc[0])
    s = float(res.metrics[res.metrics["metric"] == "spearman"]["value"].iloc[0])

    assert p > 0.3
    assert s < -0.3
    assert any("conflicting directions" in w for w in res.warnings)


def test_sign_conflict_is_never_labeled_near_linear_or_monotonic():
    """Opposite-sign Pearson/Spearman must not read as agreement. With nearly
    equal magnitudes the abs gap is ~0, which previously mislabeled the pair
    near_linear; the cascade now keys off the signed conflict."""

    def metrics(p, s, k):
        return {
            "pearson": MetricResult("pearson", p, True),
            "spearman": MetricResult("spearman", s, True),
            "kendall_tau_b": MetricResult("kendall_tau_b", k, True),
        }

    conflict = metrics(0.8, -0.8, -0.6)
    # With independent leverage evidence, the conflict is a leverage signature.
    assert (
        apply_heuristics(conflict, ["pearson_trim_sensitive"], 200).label
        == "possible_outlier_or_leverage"
    )
    # Without it, it must not be near_linear/monotonic — it is ambiguous.
    assert (
        apply_heuristics(conflict, ["pearson_trim_stable"], 200).label
        == "mixed_or_ambiguous"
    )
    # Same magnitudes but agreeing signs is still near_linear (unchanged).
    assert (
        apply_heuristics(metrics(0.8, 0.8, 0.6), ["pearson_trim_stable"], 200).label
        == "near_linear"
    )


def test_disagreement_score_reflects_sign_conflict():
    """A sign conflict (e.g. +0.96 vs -0.94) must not score as agreement: the
    disagreement uses the signed Pearson-Spearman difference, so the score is
    large rather than ~0."""
    import numpy as np
    import pandas as pd

    n = 200
    x = np.linspace(0, 10, n)
    y = -x.copy()
    x[-2:] = [200, 210]
    y[-2:] = [200, 210]  # leverage points flip Pearson positive
    res = profile_pair(pd.DataFrame({"x": x, "y": y}), "x", "y", mode="deep")
    metrics = {row["metric"]: row["value"] for _, row in res.metrics.iterrows()}

    assert metrics["pearson"] > 0.5 and metrics["spearman"] < -0.5
    assert res.pattern == "possible_outlier_or_leverage"
    assert res.disagreement_score > 1.0


def test_outlier_driven_uses_trim_sensitivity():
    df = make_relationship("outlier_driven", n=500, noise=0.1, random_state=42)
    res = profile_pair(df, "x", "y")

    assert res.pattern == "possible_outlier_or_leverage"
    assert res.diagnostics.pearson_trimmed is not None
    assert res.diagnostics.pearson_trim_delta is not None
    assert res.diagnostics.pearson_trim_delta > 0.20
    assert any("trimming extreme" in warning for warning in res.warnings)


def test_trim_sensitivity_detects_pearson_sign_flip():
    """A few high-leverage points can flip full-sample Pearson negative while
    the trimmed bulk stays positive. The magnitudes are nearly equal, so the
    old abs-of-abs delta would score ~0 and mislabel this 'stable'; the signed
    delta must catch it as leverage-sensitive."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 500
    x = rng.uniform(-1, 1, n)
    y = x + rng.normal(0, 0.05, n)  # tight positive bulk
    # High-leverage outliers forming a strong negative direction.
    ox = np.array([40, 42, 44, 46, 48], dtype=float)
    df = pd.DataFrame({"x": np.concatenate([x, ox]), "y": np.concatenate([y, -ox])})

    res = profile_pair(df, "x", "y")

    p = float(res.metrics[res.metrics["metric"] == "pearson"]["value"].iloc[0])
    assert p < -0.5  # full-sample Pearson is negative...
    assert res.diagnostics.pearson_trimmed > 0.5  # ...but the trimmed bulk is positive
    # Magnitudes are nearly equal, so abs(|p| - |trimmed|) would be < 0.20.
    assert abs(abs(p) - abs(res.diagnostics.pearson_trimmed)) < 0.20
    # The signed delta still flags it.
    assert res.diagnostics.pearson_trim_delta > 0.20
    assert any("trimming extreme" in warning for warning in res.warnings)


def test_detect_metric_warnings_flags_conflicting_signs():
    metrics = {
        "pearson": MetricResult("pearson", 0.6, True),
        "spearman": MetricResult("spearman", -0.5, True),
    }
    warnings = detect_metric_warnings(metrics)
    assert any("conflicting directions" in w for w in warnings)


def test_detect_metric_warnings_silent_when_below_threshold():
    metrics = {
        "pearson": MetricResult("pearson", 0.2, True),
        "spearman": MetricResult("spearman", -0.25, True),
    }
    assert detect_metric_warnings(metrics) == []


def test_detect_metric_warnings_silent_on_aligned_signs():
    metrics = {
        "pearson": MetricResult("pearson", 0.6, True),
        "spearman": MetricResult("spearman", 0.5, True),
    }
    assert detect_metric_warnings(metrics) == []


def _weak_metrics_with_xi(xi=None, xi_reverse=None):
    metrics = {
        "pearson": MetricResult("pearson", 0.05, True),
        "spearman": MetricResult("spearman", 0.04, True),
        "kendall_tau_b": MetricResult("kendall_tau_b", 0.03, True),
    }
    if xi is not None:
        metrics["chatterjee_xi"] = MetricResult("chatterjee_xi", xi, True)
    if xi_reverse is not None:
        metrics["chatterjee_xi_reverse"] = MetricResult(
            "chatterjee_xi_reverse", xi_reverse, True
        )
    return metrics


def test_detect_metric_warnings_flags_high_xi_with_weak_label():
    metrics = _weak_metrics_with_xi(xi=0.9, xi_reverse=0.3)
    warnings = detect_metric_warnings(metrics, label="weak_or_no_relationship")
    assert any("chatterjee_xi (0.900)" in w for w in warnings)
    assert any("weak_or_no_relationship" in w for w in warnings)


def test_detect_metric_warnings_xi_reports_stronger_direction():
    metrics = _weak_metrics_with_xi(xi=0.3, xi_reverse=0.8)
    warnings = detect_metric_warnings(metrics, label="mixed_or_ambiguous")
    assert any("chatterjee_xi_reverse (0.800)" in w for w in warnings)


def test_detect_metric_warnings_xi_silent_without_label():
    metrics = _weak_metrics_with_xi(xi=0.9)
    assert detect_metric_warnings(metrics) == []


def test_detect_metric_warnings_xi_silent_on_other_labels():
    metrics = _weak_metrics_with_xi(xi=0.9)
    assert detect_metric_warnings(metrics, label="near_linear") == []


def test_detect_metric_warnings_xi_silent_below_threshold():
    metrics = _weak_metrics_with_xi(xi=0.30)
    assert detect_metric_warnings(metrics, label="weak_or_no_relationship") == []


def test_detect_metric_warnings_xi_silent_when_not_computed():
    metrics = _weak_metrics_with_xi(xi=None)
    metrics["chatterjee_xi"] = MetricResult("chatterjee_xi", None, True)
    assert detect_metric_warnings(metrics, label="weak_or_no_relationship") == []


def test_classifier_treats_nan_metric_as_not_computable():
    """A NaN metric value must be treated as 'no value' (not_computable), not
    slip past the None guard and fall through to mixed_or_ambiguous."""
    metrics = {
        "pearson": MetricResult("pearson", float("nan"), True),
        "spearman": MetricResult("spearman", 0.5, True),
        "kendall_tau_b": MetricResult("kendall_tau_b", 0.4, True),
    }
    res = apply_heuristics(metrics, flags=[], n_used=100)
    assert res.label == "not_computable"


def test_detect_metric_warnings_ignores_nan_xi():
    """A NaN xi must not trigger (or crash) the dependence warning; max() over
    candidates would otherwise be unreliable with NaN present."""
    metrics = _weak_metrics_with_xi(xi=float("nan"), xi_reverse=0.30)
    warnings = detect_metric_warnings(metrics, label="weak_or_no_relationship")
    assert not any("chatterjee_xi" in w for w in warnings)


def test_deep_mode_u_shape_resolves_via_sq_corr_without_xi_warning():
    # The sq_corr shape diagnostic (corr(X^2, Y^2)) is lite-computable, so a
    # classic U-shape now resolves to nonmonotonic_dependence even without
    # distance correlation (deep mode) — it no longer falls back to
    # weak_or_no_relationship, and no longer needs the chatterjee_xi warning
    # as a safety net for this case.
    df = make_relationship("u_shape", n=500, noise=0.1, random_state=42)
    res = profile_pair(df, "x", "y", mode="deep")

    assert res.pattern == "nonmonotonic_dependence"
    assert not any("understate" in w for w in res.warnings)


def test_lite_mode_u_shape_resolves_via_sq_corr():
    # sq_corr needs no optional dependency, so lite mode resolves the same way.
    df = make_relationship("u_shape", n=500, noise=0.1, random_state=42)
    res = profile_pair(df, "x", "y", mode="lite")

    assert res.pattern == "nonmonotonic_dependence"
    assert not any("chatterjee_xi" in w for w in res.warnings)


def test_stable_trim_sensitivity_avoids_outlier_label():
    metrics = {
        "pearson": MetricResult("pearson", 0.80, True),
        "spearman": MetricResult("spearman", 0.52, True),
        "kendall_tau_b": MetricResult("kendall_tau_b", 0.45, True),
    }

    result = apply_heuristics(metrics, ["pearson_trim_stable"], n_used=100)

    assert result.label != "possible_outlier_or_leverage"
