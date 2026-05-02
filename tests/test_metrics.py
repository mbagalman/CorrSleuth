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
    compute_biweight_midcorrelation, compute_percentage_bend_correlation,
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
        "percentage_bend_correlation",
    }
    assert robust_metrics.isdisjoint(lite_metrics)
    assert robust_metrics <= deep_metrics
    assert "distance_correlation" not in deep_metrics
    assert "mutual_information" not in deep_metrics


def test_robust_metrics_return_none_for_small_samples_with_warning():
    df = pd.DataFrame({"x": range(40), "y": range(40)})
    pair = validate_pair(df, "x", "y")

    result = compute_trimmed_pearson(pair)

    assert result.name == "pearson_trimmed_1pct"
    assert result.value is None
    assert result.available is True
    assert any("robust correlation diagnostics need more observations" in w for w in pair.warnings)


def test_robust_metrics_are_near_pearson_for_clean_linear_data():
    df = pd.DataFrame({"x": range(100), "y": range(100)})
    pair = validate_pair(df, "x", "y")

    results = [
        compute_trimmed_pearson(pair),
        compute_winsorized_pearson(pair),
        compute_biweight_midcorrelation(pair),
        compute_percentage_bend_correlation(pair),
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
    assert metrics["pearson_trimmed_1pct"] < 0.50
    assert metrics["pearson"] - metrics["pearson_trimmed_1pct"] > 0.40
