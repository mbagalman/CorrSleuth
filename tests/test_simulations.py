import numpy as np
import pandas as pd
import pytest

from corrsleuth.datasets import make_relationship
from corrsleuth.exceptions import InputError


def test_make_relationship_shapes():
    shapes = [
        "linear_positive",
        "linear_negative",
        "monotonic_log",
        "exponential_monotonic",
        "logarithmic_monotonic",
        "threshold_step",
        "u_shape",
        "circular",
        "heteroscedastic",
        "outlier_driven",
        "independent",
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
    with pytest.raises(InputError, match="Unknown shape_type"):
        make_relationship("not_a_real_shape")


@pytest.mark.parametrize("bad_n", [-5, 0, 1, 2.5, True, "10"])
def test_make_relationship_rejects_invalid_n(bad_n):
    with pytest.raises(InputError, match="n must be an integer >= 2"):
        make_relationship("linear_positive", n=bad_n)


@pytest.mark.parametrize(
    "bad_noise",
    [-1, -0.001, float("nan"), float("inf"), -float("inf"), True, "0.1"],
)
def test_make_relationship_rejects_invalid_noise(bad_noise):
    # Non-finite noise must be rejected too: noise=inf previously passed the
    # negative/NaN checks and returned y values of +/-inf.
    with pytest.raises(InputError, match="noise must be a finite non-negative number"):
        make_relationship("linear_positive", n=50, noise=bad_noise)


def test_make_relationship_accepts_zero_noise():
    df = make_relationship("linear_positive", n=50, noise=0, random_state=0)
    # Zero noise is valid: y == x exactly for the linear shape.
    assert np.allclose(df["x"], df["y"])
