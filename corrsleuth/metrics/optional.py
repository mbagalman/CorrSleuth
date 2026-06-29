import numpy as np

from corrsleuth.exceptions import MetricComputationError, OptionalDependencyError
from corrsleuth.result import MetricResult
from corrsleuth.validation.input import CleanPair


def compute_distance_correlation(
    pair: CleanPair,
    mode: str = "lite",
    max_n_for_dcor: int | None = 20000,
    random_state: int = 42,
) -> MetricResult:
    """Compute distance correlation, downsampling deterministically for large pairs.

    In ``mode='standard'``, raises :class:`OptionalDependencyError` if ``dcor`` is
    not installed. In ``mode='lite'``, returns an unavailable :class:`MetricResult`
    instead. When ``pair.n_used > max_n_for_dcor`` the input is downsampled with a
    NumPy generator seeded by ``random_state``; pass ``max_n_for_dcor=None`` to
    disable the cap. Returns ``value=None`` when either column is constant.
    """
    try:
        import dcor
    except ImportError:
        if mode == "standard":
            raise OptionalDependencyError(
                "dcor is required for standard mode. Run `pip install corrsleuth[standard]`"
            ) from None
        return MetricResult(name="distance_correlation", value=None, available=False)

    if pair.x_is_constant or pair.y_is_constant:
        return MetricResult.no_value("distance_correlation")

    x = pair.x.to_numpy()
    y = pair.y.to_numpy()

    if max_n_for_dcor is not None and pair.n_used > max_n_for_dcor:
        pair.warnings.append(
            f"n_used > {max_n_for_dcor}. Automatically downsampling to {max_n_for_dcor} "
            f"for distance correlation (random_state={random_state})."
        )
        rng = np.random.default_rng(random_state)
        idx = rng.choice(pair.n_used, max_n_for_dcor, replace=False)
        x = x[idx]
        y = y[idx]

    try:
        dc = dcor.distance_correlation(x, y)
    except (ValueError, RuntimeError, FloatingPointError) as e:
        raise MetricComputationError(
            f"Failed to compute distance_correlation: {type(e).__name__}: {e}"
        ) from e
    return MetricResult(name="distance_correlation", value=float(dc), available=True)


def compute_mutual_information(
    pair: CleanPair,
    mode: str = "lite",
    random_state: int = 42,
) -> MetricResult:
    """Compute mutual information using scikit-learn's KSG estimator.

    The returned value is **raw, unnormalized** mutual information in nats: it is
    ``>= 0`` and unbounded above, **not** scaled to ``[0, 1]``. Do not read its
    magnitude like a correlation coefficient or compare it directly against
    Pearson/Spearman/distance-correlation values — interpret it relatively (a
    larger MI means more shared information) or alongside the other metrics, not
    on the same 0-1 scale.

    In ``mode='standard'``, raises :class:`OptionalDependencyError` if
    ``scikit-learn`` is not installed. In ``mode='lite'`` returns an unavailable
    :class:`MetricResult` instead. ``random_state`` is forwarded to
    :func:`sklearn.feature_selection.mutual_info_regression` for reproducibility.
    Returns ``value=None`` when either column is constant or when
    ``pair.n_used <= 3`` (too few observations for the estimator).
    """
    try:
        from sklearn.feature_selection import mutual_info_regression
    except ImportError:
        if mode == "standard":
            raise OptionalDependencyError(
                "scikit-learn is required for standard mode. Run `pip install corrsleuth[standard]`"
            ) from None
        return MetricResult(name="mutual_information", value=None, available=False)

    if pair.x_is_constant or pair.y_is_constant:
        return MetricResult.no_value("mutual_information")

    if pair.n_used <= 3:
        pair.warnings.append(
            "n_used <= 3. Mutual information is not computed because the estimator "
            "requires more observations."
        )
        return MetricResult.no_value("mutual_information")

    x = pair.x.to_numpy().reshape(-1, 1)
    y = pair.y.to_numpy()

    try:
        mi = mutual_info_regression(x, y, random_state=random_state)[0]
    except (ValueError, RuntimeError, FloatingPointError) as e:
        raise MetricComputationError(
            f"Failed to compute mutual_information: {type(e).__name__}: {e}"
        ) from e
    return MetricResult(name="mutual_information", value=float(mi), available=True)
