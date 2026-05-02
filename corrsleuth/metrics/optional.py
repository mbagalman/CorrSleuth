from typing import Optional

import numpy as np

from corrsleuth.result import MetricResult
from corrsleuth.validation.input import CleanPair
from corrsleuth.exceptions import OptionalDependencyError


def compute_distance_correlation(
    pair: CleanPair,
    mode: str = "lite",
    max_n_for_dcor: Optional[int] = 20000,
    random_state: int = 42,
) -> MetricResult:
    """Compute distance correlation, downsampling deterministically for large pairs.

    In ``mode='standard'``, raises :class:`OptionalDependencyError` if ``dcor`` is
    not installed. In ``mode='lite'``, returns an unavailable :class:`MetricResult`
    instead. When ``pair.n_used > max_n_for_dcor`` the input is downsampled with a
    NumPy generator seeded by ``random_state``; pass ``max_n_for_dcor=None`` to
    disable the cap.
    """
    try:
        import dcor
    except ImportError:
        if mode == "standard":
            raise OptionalDependencyError(
                "dcor is required for standard mode. Run `pip install corrsleuth[standard]`"
            )
        return MetricResult(name="distance_correlation", value=None, available=False)

    if pair.x_is_constant or pair.y_is_constant:
        return MetricResult(name="distance_correlation", value=None, available=True)

    x = pair.x.values
    y = pair.y.values

    if max_n_for_dcor is not None and pair.n_used > max_n_for_dcor:
        pair.warnings.append(
            f"n_used > {max_n_for_dcor}. Automatically downsampling to {max_n_for_dcor} "
            f"for distance correlation (random_state={random_state})."
        )
        rng = np.random.default_rng(random_state)
        idx = rng.choice(pair.n_used, max_n_for_dcor, replace=False)
        x = x[idx]
        y = y[idx]

    dc = dcor.distance_correlation(x, y)
    return MetricResult(name="distance_correlation", value=float(dc), available=True)


def compute_mutual_information(
    pair: CleanPair,
    mode: str = "lite",
    random_state: int = 42,
) -> MetricResult:
    """Compute mutual information using scikit-learn's KSG estimator.

    In ``mode='standard'``, raises :class:`OptionalDependencyError` if
    ``scikit-learn`` is not installed. In ``mode='lite'`` returns an unavailable
    :class:`MetricResult` instead. ``random_state`` is forwarded to
    :func:`sklearn.feature_selection.mutual_info_regression` for reproducibility.
    """
    try:
        from sklearn.feature_selection import mutual_info_regression
    except ImportError:
        if mode == "standard":
            raise OptionalDependencyError(
                "scikit-learn is required for standard mode. Run `pip install corrsleuth[standard]`"
            )
        return MetricResult(name="mutual_information", value=None, available=False)

    if pair.x_is_constant or pair.y_is_constant:
        return MetricResult(name="mutual_information", value=None, available=True)

    x = pair.x.values.reshape(-1, 1)
    y = pair.y.values

    mi = mutual_info_regression(x, y, random_state=random_state)[0]
    return MetricResult(name="mutual_information", value=float(mi), available=True)
