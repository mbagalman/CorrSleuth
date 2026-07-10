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
    - exponential_monotonic
    - logarithmic_monotonic
    - threshold_step
    - u_shape
    - sinusoidal
    - circular
    - heteroscedastic
    - bowtie_variance
    - outlier_driven
    - independent

    Parameters:
    - shape_type (str): The type of relationship to generate.
    - n (int): Number of observations. Must be an integer >= 2. Default is 500.
    - noise (float): Amount of random noise to add. Must be a non-negative
      number. Default is 0.1. **Its scale is shape-specific** — absolute sd for
      the linear/u_shape/sinusoidal/threshold_step/monotonic_log shapes,
      signal-relative sd for the exponential/logarithmic shapes, radius-relative
      jitter for `circular`, a spread-slope scale for the heteroscedastic/bowtie
      shapes, and an additive sd bump for `independent` (sd = 1 + noise). So the
      same `noise` value is not comparable across shapes; see the per-branch
      source comments.
    - random_state (int, numpy.random.Generator, or None): Seed or generator
      for reproducibility. Default is None (nondeterministic).

    Returns:
    - pd.DataFrame: DataFrame with columns 'x' and 'y'.

    Raises:
    - InputError: if ``shape_type`` is unknown, ``n`` is not an integer >= 2,
      or ``noise`` is negative or non-finite.
    """
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 2:
        raise InputError("n must be an integer >= 2.")
    if (
        isinstance(noise, bool)
        or not isinstance(noise, (int, float, np.floating, np.integer))
        or pd.isna(noise)
        or not np.isfinite(noise)
        or noise < 0
    ):
        # np.isfinite: infinite noise would silently produce +/-inf y values
        # that pass this function but fail (or distort) everything downstream.
        raise InputError("noise must be a finite non-negative number.")

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
    elif shape_type == "exponential_monotonic":
        # A smooth monotonic curve over an ordinary (non-rigged) X range: real
        # curvature (see docs/shape-diagnostics-design.md), but mild enough that
        # Pearson stays close to Spearman (s - p well under 0.20) — this is the
        # regime the rank-linear gap alone misses and bin_lof_r2_gain catches.
        x = rng.uniform(0, 3, size=n)
        signal = np.exp(x)
        y = signal + rng.normal(0, noise, size=n) * signal.std()
    elif shape_type == "logarithmic_monotonic":
        # Same idea as exponential_monotonic, in the other direction, over an
        # ordinary X range (unlike the rigged monotonic_log above).
        x = rng.uniform(0.1, 20, size=n)
        signal = np.log(x)
        y = signal + rng.normal(0, noise, size=n) * signal.std()
    elif shape_type == "threshold_step":
        # A two-level step function: Pearson and Spearman both read moderately
        # strong (dominated by the between-group separation), with a small gap
        # between them — near_linear's regime by the rank-linear gap alone, but
        # bin_lof_r2_gain reveals the two flat groups a line doesn't capture.
        y = np.where(x > 0, 1.0, -1.0) + rng.normal(0, noise, size=n)
    elif shape_type == "u_shape":
        y = x**2 + rng.normal(0, noise, size=n)
    elif shape_type == "sinusoidal":
        # ~2.5 full cycles: Pearson, Spearman, and (only marginally above its
        # floor) distance correlation all read weak, yet Y is a strong
        # deterministic function of X. Detected via the bin-mean reversal
        # count jointly with the bin lack-of-fit gain (metrics/shape.py) — the
        # lite-computable oscillation route into nonmonotonic_dependence. A
        # half-integer cycle count is used because over an integer number of
        # cycles the sine's net linear component is larger, pushing
        # Pearson/Spearman toward the rule's monotone ceiling.
        x = rng.uniform(0, 5 * np.pi, size=n)
        y = np.sin(x) + rng.normal(0, noise, size=n)
    elif shape_type == "circular":
        # Points scattered around a ring: X and Y are dependent (X^2 + Y^2 is
        # approximately constant) but Pearson, Spearman, and distance
        # correlation on the raw values are all near zero — only sq_corr
        # (the correlation of the mean-centered squares) reveals it. See
        # docs/shape-diagnostics-design.md.
        theta = rng.uniform(0, 2 * np.pi, size=n)
        radius = 5.0 * (1 + rng.normal(0, noise, size=n))
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
    elif shape_type == "heteroscedastic":
        # A linear *mean* (so the pair stays near_linear) with residual spread
        # that grows with x — the classic funnel. The mean model is fine, but
        # homoscedastic inference on top of it is not: this is what the
        # variance_shape="increasing_spread" axis and its warning exist to
        # catch. x is drawn positive so the spread scaling is monotone.
        x = rng.uniform(0, 4, size=n)
        spread = noise * (0.5 + x)
        y = x + rng.normal(0, 1, size=n) * spread
    elif shape_type == "bowtie_variance":
        # A linear *mean* with residual spread that is high at both extremes
        # of x and low in the middle -- a symmetric ("bowtie") pattern the
        # one-directional Goldfeld-Quandt funnel check is blind to (both edges
        # have similarly high spread, so its low-vs-high ratio reads ~1). This
        # is what variance_shape="edge_high_spread" and its bowtie_ratio
        # diagnostic exist to catch. x is centered at 0 so the spread scales
        # symmetrically with distance from the center.
        x = rng.uniform(-4, 4, size=n)
        spread = noise * (0.5 + np.abs(x))
        y = x + rng.normal(0, 1, size=n) * spread
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
