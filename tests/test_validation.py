import pytest
import numpy as np
import pandas as pd
from corrsleuth.validation.input import validate_pair
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
