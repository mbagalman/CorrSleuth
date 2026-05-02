import pytest
from corrsleuth.datasets import make_relationship
from corrsleuth.api import profile_pair
from corrsleuth.heuristics.classifier import apply_heuristics
from corrsleuth.result import MetricResult

def test_canonical_shapes_lite():
    # Linear Positive
    df = make_relationship("linear_positive", n=500, noise=0.1, random_state=42)
    res = profile_pair(df, "x", "y", mode="lite")
    assert res.pattern == "near_linear"

def test_canonical_shapes_standard():
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


def test_stable_trim_sensitivity_avoids_outlier_label():
    metrics = {
        "pearson": MetricResult("pearson", 0.80, True),
        "spearman": MetricResult("spearman", 0.52, True),
        "kendall_tau_b": MetricResult("kendall_tau_b", 0.45, True),
    }

    result = apply_heuristics(metrics, ["pearson_trim_stable"], n_used=100)

    assert result.label != "possible_outlier_or_leverage"
