import numpy as np
import pandas as pd
from typing import Optional, Union

def make_relationship(
    shape_type: str,
    n: int = 500,
    noise: float = 0.1,
    random_state: Optional[Union[int, np.random.Generator]] = None
) -> pd.DataFrame:
    """
    Generate a DataFrame with a specific relationship between 'x' and 'y'.
    
    Required for v0.1 tests:
    - linear_positive
    - linear_negative
    - monotonic_log
    - u_shape
    - outlier_driven
    - independent
    
    Parameters:
    - shape_type (str): The type of relationship to generate.
    - n (int): Number of observations. Default is 500.
    - noise (float): Amount of random noise to add. Default is 0.1.
    - random_state: Random seed for reproducibility.
    
    Returns:
    - pd.DataFrame: DataFrame with columns 'x' and 'y'.
    """
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
        raise ValueError(f"Unknown shape_type: {shape_type}")
        
    return pd.DataFrame({"x": x, "y": y})
