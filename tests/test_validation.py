import pytest
import numpy as np
import pandas as pd
from corrsleuth.validation.input import compute_tie_rate, validate_pair
from corrsleuth.exceptions import InputError

def test_validation_missing_data_pairwise():
    df = pd.DataFrame({"x": [1, 2, np.nan, 4], "y": [np.nan, 2, 3, 4]})
    pair = validate_pair(df, "x", "y", missing="pairwise")
    assert pair.n_original == 4
    assert pair.n_used == 2
    assert pair.missing_ratio == 0.5
    assert len(pair.x) == 2

def test_validation_missing_data_listwise():
    df = pd.DataFrame({"x": [1, 2, np.nan, 4], "y": [np.nan, 2, 3, 4]})
    pair = validate_pair(df, "x", "y", missing="listwise")
    assert pair.n_used == 2
    assert pair.missing_ratio == 0.5

def test_validation_missing_data_raise():
    df = pd.DataFrame({"x": [1, 2, np.nan], "y": [1, 2, 3]})
    with pytest.raises(InputError, match="Missing values found"):
        validate_pair(df, "x", "y", missing="raise")

def test_validation_missing_data_invalid():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
    with pytest.raises(InputError, match="Unsupported missing mode"):
        validate_pair(df, "x", "y", missing="invalid")

def test_validation_requires_at_least_two_observations():
    df = pd.DataFrame({"x": [1.0, np.nan, np.nan], "y": [1.0, np.nan, np.nan]})
    with pytest.raises(InputError, match="At least 2 valid observations"):
        validate_pair(df, "x", "y")


def test_validation_rejects_all_nan_columns():
    df = pd.DataFrame({"x": [np.nan] * 5, "y": [1.0, 2.0, 3.0, 4.0, 5.0]})
    with pytest.raises(InputError, match="At least 2 valid observations"):
        validate_pair(df, "x", "y")


def test_validation_non_numeric():
    df = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
    with pytest.raises(InputError, match="is not numeric"):
        validate_pair(df, "x", "y")

def test_validation_flags():
    # Test low_n
    df = pd.DataFrame({"x": range(10), "y": range(10)})
    pair = validate_pair(df, "x", "y")
    assert "low_n" in pair.flags

    # Test high_missingness
    df = pd.DataFrame({"x": [1, 2] + [np.nan]*8, "y": [1, 2] + [np.nan]*8})
    pair = validate_pair(df, "x", "y")
    assert "high_missingness" in pair.flags

    # Test low_unique_ratio
    df = pd.DataFrame({"x": [1]*40 + [2]*10, "y": range(50)})
    pair = validate_pair(df, "x", "y")
    assert "low_unique_ratio" in pair.flags


def test_compute_tie_rate_all_unique():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert compute_tie_rate(series) == 0.0


def test_compute_tie_rate_mixed():
    # 1 appears twice, others once -> 2 of 5 observations are tied
    series = pd.Series([1, 1, 2, 3, 4])
    assert compute_tie_rate(series) == pytest.approx(0.4)


def test_compute_tie_rate_all_same():
    series = pd.Series([7.0] * 10)
    assert compute_tie_rate(series) == 1.0


def test_compute_tie_rate_empty():
    assert compute_tie_rate(pd.Series([], dtype=float)) == 0.0


def test_validation_records_tie_rates_on_clean_pair():
    df = pd.DataFrame({
        "x": list(range(50)),                # all unique
        "y": [1] * 25 + [2] * 25,            # heavy ties
    })
    pair = validate_pair(df, "x", "y")
    assert pair.x_tie_rate == 0.0
    assert pair.y_tie_rate == 1.0


def test_high_tie_rate_warning_names_variable():
    df = pd.DataFrame({
        "discrete_var": [0, 1] * 25 + [2, 3] * 25,
        "continuous_var": np.linspace(0, 1, 100),
    })
    pair = validate_pair(df, "discrete_var", "continuous_var")

    assert "high_tie_rate" in pair.flags
    matching = [w for w in pair.warnings if "discrete_var" in w and "tie rate" in w]
    assert matching, f"expected tie-rate warning naming 'discrete_var', got {pair.warnings}"
    # The continuous variable should not trigger its own tie-rate warning
    assert not any("continuous_var" in w and "tie rate" in w for w in pair.warnings)


def test_high_tie_rate_warns_per_variable_when_both_tied():
    df = pd.DataFrame({
        "x": [0, 1] * 50,
        "y": [10, 20, 30] * 33 + [10],
    })
    pair = validate_pair(df, "x", "y")

    x_warnings = [w for w in pair.warnings if "'x'" in w and "tie rate" in w]
    y_warnings = [w for w in pair.warnings if "'y'" in w and "tie rate" in w]
    assert x_warnings, "expected tie-rate warning for x"
    assert y_warnings, "expected tie-rate warning for y"
    assert pair.flags.count("high_tie_rate") == 1, "high_tie_rate flag should not duplicate"


def test_low_tie_rate_does_not_warn():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "x": rng.normal(size=200),
        "y": rng.normal(size=200),
    })
    pair = validate_pair(df, "x", "y")

    assert "high_tie_rate" not in pair.flags
    assert not any("tie rate" in w for w in pair.warnings)
