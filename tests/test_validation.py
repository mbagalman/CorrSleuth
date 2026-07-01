import warnings

import numpy as np
import pandas as pd
import pytest

from corrsleuth.exceptions import InputError
from corrsleuth.validation.input import compute_tie_rate, validate_pair


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


def test_listwise_drops_rows_missing_in_unrelated_columns():
    """Complete-case deletion: a NaN in a column other than x/y still drops the
    row under listwise, but not under pairwise."""
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [2.0, 4.0, 6.0, 8.0, 10.0],
            "z": [1.0, np.nan, 3.0, 4.0, 5.0],  # NaN in row index 1 only
        }
    )

    pairwise = validate_pair(df, "x", "y", missing="pairwise")
    assert pairwise.n_used == 5  # x and y are fully present

    listwise = validate_pair(df, "x", "y", missing="listwise")
    assert listwise.n_used == 4  # row with NaN in z is dropped
    assert 2.0 not in set(listwise.x)  # the (x=2, y=4) row is gone


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


def test_validation_rejects_complex_dtype():
    # Complex columns pass ``is_numeric_dtype`` but casting to float silently
    # discards the imaginary part; they must be rejected, not projected.
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1 + 2j, 3 + 4j, 5 + 6j]})
    assert df["y"].dtype == np.complex128
    # Rejection happens at the numeric gate, before any float cast, so no
    # ComplexWarning is emitted; turn warnings into errors to prove it.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(InputError, match="complex dtype"):
            validate_pair(df, "x", "y")
    # Rejection also applies when the complex column is x.
    df_x = pd.DataFrame({"x": [1 + 2j, 3 + 4j, 5 + 6j], "y": [1.0, 2.0, 3.0]})
    with pytest.raises(InputError, match="complex dtype"):
        validate_pair(df_x, "x", "y")


def test_validation_rejects_infinite_values_in_used_rows():
    df = pd.DataFrame({"x": [1.0, 2.0, np.inf], "y": [1.0, 2.0, 3.0]})
    with pytest.raises(InputError, match="infinite values"):
        validate_pair(df, "x", "y")


def test_validation_allows_inf_in_rows_dropped_by_missing_policy():
    # The inf sits in a row whose other value is NaN, so pairwise handling
    # drops it before any metric sees it.
    df = pd.DataFrame(
        {"x": [np.inf, 1.0, 2.0, 3.0, 4.0], "y": [np.nan, 2.0, 4.0, 6.0, 8.0]}
    )
    pair = validate_pair(df, "x", "y", missing="pairwise")
    assert pair.n_used == 4
    assert not np.isinf(pair.x).any()


def test_validation_rejects_same_column_for_x_and_y():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    with pytest.raises(InputError, match="must be different columns"):
        validate_pair(df, "x", "x")


def test_validation_rejects_duplicate_column_names():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    df = pd.concat([df, df["x"]], axis=1)  # two 'x' columns
    with pytest.raises(InputError, match="matches multiple columns"):
        validate_pair(df, "x", "y")
    with pytest.raises(InputError, match="matches multiple columns"):
        validate_pair(df, "y", "x")


def test_validation_flags():
    # Test low_n
    df = pd.DataFrame({"x": range(10), "y": range(10)})
    pair = validate_pair(df, "x", "y")
    assert "low_n" in pair.flags

    # Test high_missingness
    df = pd.DataFrame({"x": [1, 2] + [np.nan] * 8, "y": [1, 2] + [np.nan] * 8})
    pair = validate_pair(df, "x", "y")
    assert "high_missingness" in pair.flags

    # Test low_unique_ratio
    df = pd.DataFrame({"x": [1] * 40 + [2] * 10, "y": range(50)})
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
    df = pd.DataFrame(
        {
            "x": list(range(50)),  # all unique
            "y": [1] * 25 + [2] * 25,  # heavy ties
        }
    )
    pair = validate_pair(df, "x", "y")
    assert pair.x_tie_rate == 0.0
    assert pair.y_tie_rate == 1.0


def test_high_tie_rate_warning_names_variable():
    df = pd.DataFrame(
        {
            "discrete_var": [0, 1] * 25 + [2, 3] * 25,
            "continuous_var": np.linspace(0, 1, 100),
        }
    )
    pair = validate_pair(df, "discrete_var", "continuous_var")

    assert "high_tie_rate" in pair.flags
    matching = [w for w in pair.warnings if "discrete_var" in w and "tie rate" in w]
    assert matching, (
        f"expected tie-rate warning naming 'discrete_var', got {pair.warnings}"
    )
    # The continuous variable should not trigger its own tie-rate warning
    assert not any("continuous_var" in w and "tie rate" in w for w in pair.warnings)


def test_high_tie_rate_warns_per_variable_when_both_tied():
    df = pd.DataFrame(
        {
            "x": [0, 1] * 50,
            "y": [10, 20, 30] * 33 + [10],
        }
    )
    pair = validate_pair(df, "x", "y")

    x_warnings = [w for w in pair.warnings if "'x'" in w and "tie rate" in w]
    y_warnings = [w for w in pair.warnings if "'y'" in w and "tie rate" in w]
    assert x_warnings, "expected tie-rate warning for x"
    assert y_warnings, "expected tie-rate warning for y"
    assert pair.flags.count("high_tie_rate") == 1, (
        "high_tie_rate flag should not duplicate"
    )


def test_low_tie_rate_does_not_warn():
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "x": rng.normal(size=200),
            "y": rng.normal(size=200),
        }
    )
    pair = validate_pair(df, "x", "y")

    assert "high_tie_rate" not in pair.flags
    assert not any("tie rate" in w for w in pair.warnings)
