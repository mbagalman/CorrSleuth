import pytest
from corrsleuth.datasets import make_relationship
from corrsleuth.api import profile_pair
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

def test_conflicting_signs_warning():
    import pandas as pd
    import numpy as np
    
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


def test_outlier_driven_uses_trim_sensitivity():
    df = make_relationship("outlier_driven", n=500, noise=0.1, random_state=42)
    res = profile_pair(df, "x", "y")

    assert res.pattern == "possible_outlier_or_leverage"
    assert res.diagnostics.pearson_trimmed is not None
    assert res.diagnostics.pearson_trim_delta is not None
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


def test_deep_mode_u_shape_warns_about_high_xi():
    # The cascade cannot assign nonmonotonic_dependence without distance
    # correlation, so a deep-mode U-shape keeps the weak_or_no_relationship
    # label — but the high chatterjee_xi must surface as a warning instead of
    # being silently contradicted by the label.
    df = make_relationship("u_shape", n=500, noise=0.1, random_state=42)
    res = profile_pair(df, "x", "y", mode="deep")

    assert res.pattern == "weak_or_no_relationship"
    assert any("chatterjee_xi" in w and "understate" in w for w in res.warnings)


def test_lite_mode_u_shape_has_no_xi_warning():
    # Lite mode never computes xi, so the warning must not fire.
    df = make_relationship("u_shape", n=500, noise=0.1, random_state=42)
    res = profile_pair(df, "x", "y", mode="lite")

    assert not any("chatterjee_xi" in w for w in res.warnings)


def test_stable_trim_sensitivity_avoids_outlier_label():
    metrics = {
        "pearson": MetricResult("pearson", 0.80, True),
        "spearman": MetricResult("spearman", 0.52, True),
        "kendall_tau_b": MetricResult("kendall_tau_b", 0.45, True),
    }

    result = apply_heuristics(metrics, ["pearson_trim_stable"], n_used=100)

    assert result.label != "possible_outlier_or_leverage"
