import pytest
import numpy as np
import pandas as pd
from corrsleuth.datasets import make_relationship

def test_make_relationship_shapes():
    shapes = [
        "linear_positive",
        "linear_negative",
        "monotonic_log",
        "u_shape",
        "outlier_driven",
        "independent"
    ]
    
    for shape in shapes:
        df = make_relationship(shape, n=100, noise=0.1, random_state=42)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["x", "y"]
        assert len(df) == 100

def test_make_relationship_determinism():
    df1 = make_relationship("u_shape", n=50, noise=0.1, random_state=42)
    df2 = make_relationship("u_shape", n=50, noise=0.1, random_state=42)
    pd.testing.assert_frame_equal(df1, df2)

def test_make_relationship_invalid_shape():
    with pytest.raises(ValueError, match="Unknown shape_type"):
        make_relationship("not_a_real_shape")
