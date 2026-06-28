import numpy as np
import pandas as pd

from corrsleuth.exceptions import InputError


def make_relationship(
    shape_type: str,
    n: int = 500,
    noise: float = 0.1,
    random_state: int | np.random.Generator | None = None,
) -> pd.DataFrame:
    """
    Generate a DataFrame with a specific relationship between 'x' and 'y'.

    Supported ``shape_type`` values:
    - linear_positive
    - linear_negative
    - monotonic_log
    - u_shape
    - outlier_driven
    - independent

    Parameters:
    - shape_type (str): The type of relationship to generate.
    - n (int): Number of observations. Must be an integer >= 2. Default is 500.
    - noise (float): Amount of random noise to add. Must be a non-negative
      number. Default is 0.1.
    - random_state (int, numpy.random.Generator, or None): Seed or generator
      for reproducibility. Default is None (nondeterministic).

    Returns:
    - pd.DataFrame: DataFrame with columns 'x' and 'y'.

    Raises:
    - InputError: if ``shape_type`` is unknown, ``n`` is not an integer >= 2,
      or ``noise`` is negative.
    """
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 2:
        raise InputError("n must be an integer >= 2.")
    if (
        isinstance(noise, bool)
        or not isinstance(noise, (int, float, np.floating, np.integer))
        or pd.isna(noise)
        or noise < 0
    ):
        raise InputError("noise must be a non-negative number.")

    rng = np.random.default_rng(random_state)

    x = rng.uniform(-3, 3, size=n)
    y = np.zeros(n)

    if shape_type == "linear_positive":
        y = x + rng.normal(0, noise, size=n)
    elif shape_type == "linear_negative":
        y = -x + rng.normal(0, noise, size=n)
    elif shape_type == "monotonic_log":
        # Create a heavily skewed X so that the log relationship is strongly nonlinear (s - p > 0.20)
        x = np.exp(rng.uniform(0.1, 10, size=n))
        y = np.log(x) + rng.normal(0, noise, size=n)
    elif shape_type == "u_shape":
        y = x**2 + rng.normal(0, noise, size=n)
    elif shape_type == "outlier_driven":
        y = rng.normal(0, noise, size=n)
        # Add a few extreme outliers that drive correlation
        num_outliers = max(1, int(n * 0.02))
        x[-num_outliers:] = rng.uniform(8, 10, size=num_outliers)
        y[-num_outliers:] = rng.uniform(8, 10, size=num_outliers)
    elif shape_type == "independent":
        y = rng.normal(0, 1 + noise, size=n)
    else:
        raise InputError(f"Unknown shape_type: {shape_type}")

    return pd.DataFrame({"x": x, "y": y})
