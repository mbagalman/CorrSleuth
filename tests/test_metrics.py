import sys

import pandas as pd
import pytest

from corrsleuth.validation.input import validate_pair
from corrsleuth.exceptions import MetricComputationError, OptionalDependencyError
from corrsleuth.metrics import (
    compute_pearson, compute_spearman, compute_kendall,
    compute_distance_correlation, compute_mutual_information
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
