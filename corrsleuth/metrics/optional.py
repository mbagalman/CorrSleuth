import numpy as np

from corrsleuth.exceptions import MetricComputationError, OptionalDependencyError
from corrsleuth.result import MetricResult
from corrsleuth.validation.input import CleanPair

#: Maximum distinct values for a variable to count as **discrete** for the
#: mutual-information estimator. scikit-learn's continuous k-NN (KSG) estimators
#: misestimate a low-cardinality column unless it is declared discrete, so
#: :func:`compute_mutual_information` dispatches on this detection (see its
#: discreteness policy). Detection is by *cardinality and repetition only* —
#: never by the values themselves — because MI is invariant under any bijective
#: relabeling: the same 20 categories encoded ``0…19`` or ``0.0…1.9`` must
#: produce the same estimate, so a whole-number test (which the ``0.0…1.9``
#: encoding fails) would make the result encoding-dependent. Chosen to cover
#: binary/ordinal/small-count columns while leaving genuinely continuous data
#: (even integer-valued, like ages) that has many levels alone.
_MI_DISCRETE_MAX_CARDINALITY = 20


def _looks_discrete(values: np.ndarray) -> bool:
    """True when ``values`` has at most :data:`_MI_DISCRETE_MAX_CARDINALITY`
    distinct levels *and* each level repeats at least twice on average
    (``n_distinct * 2 <= n``) — the low-cardinality case the continuous MI
    estimator misestimates. The repetition requirement keeps a small continuous
    sample (whose draws are almost surely all distinct) from being misread as
    categorical; heuristic detection cannot recover true measurement type, so a
    borderline column falls back to the continuous estimator (the pre-policy
    behavior) rather than guessing discrete."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return False
    n_distinct = np.unique(finite).size
    return n_distinct <= _MI_DISCRETE_MAX_CARDINALITY and n_distinct * 2 <= finite.size


def compute_distance_correlation(
    pair: CleanPair,
    mode: str = "lite",
    max_n_for_dcor: int | None = 20000,
    random_state: int = 42,
) -> MetricResult:
    """Compute distance correlation, downsampling deterministically for large pairs.

    In ``mode='standard'`` or ``'deep'`` (deep is a superset of standard, so both
    need the ``[standard]`` extras), raises :class:`OptionalDependencyError` if
    ``dcor`` is not installed. In ``mode='lite'``, returns an unavailable
    :class:`MetricResult`
    instead. When ``pair.n_used > max_n_for_dcor`` the input is downsampled with a
    NumPy generator seeded by ``random_state``; pass ``max_n_for_dcor=None`` to
    disable the cap. Returns ``value=None`` when either column is constant.
    """
    try:
        import dcor
    except ImportError:
        # Both "standard" and "deep" require the [standard] extras (deep is a
        # superset of standard); "lite" degrades to an unavailable result.
        if mode in ("standard", "deep"):
            raise OptionalDependencyError(
                f"dcor is required for {mode} mode. Run `pip install corrsleuth[standard]`"
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

    **Discreteness policy.** The continuous k-NN (KSG) estimator misestimates
    low-cardinality data unless told otherwise, so each column is classified by
    :func:`_looks_discrete` (cardinality + repetition; deliberately *not*
    value-based, so any bijective re-encoding of the same categories gives the
    same estimate) and the computation dispatches to the matching estimator:
    ``mutual_info_regression`` when Y is continuous (with a discrete X declared
    via ``discrete_features``), ``mutual_info_classif`` on the integer-coded
    levels when Y is discrete. Both mixed cases run scikit-learn's same
    discrete–continuous (Ross 2014) estimator, so — like MI itself — the result
    is symmetric in X and Y. Discreteness detection is heuristic: a column with
    few repeating levels that is *conceptually* continuous is still estimated
    on its level structure, which is the better-calibrated choice for such data.

    In ``mode='standard'`` or ``'deep'`` (both need the ``[standard]`` extras),
    raises :class:`OptionalDependencyError` if ``scikit-learn`` is not installed.
    In ``mode='lite'`` returns an unavailable
    :class:`MetricResult` instead. ``random_state`` is forwarded to the
    scikit-learn estimator for reproducibility.
    Returns ``value=None`` when either column is constant or when
    ``pair.n_used <= 3`` (too few observations for the estimator).
    """
    try:
        from sklearn.feature_selection import (
            mutual_info_classif,
            mutual_info_regression,
        )
    except ImportError:
        # Both "standard" and "deep" require the [standard] extras (deep is a
        # superset of standard); "lite" degrades to an unavailable result.
        if mode in ("standard", "deep"):
            raise OptionalDependencyError(
                f"scikit-learn is required for {mode} mode. Run `pip install corrsleuth[standard]`"
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

    x_arr = pair.x.to_numpy()
    y_arr = pair.y.to_numpy()

    # Dispatch by detected discreteness (see the docstring's discreteness
    # policy). mutual_info_regression assumes a continuous target, so a discrete
    # Y goes through mutual_info_classif instead, on integer level codes — MI is
    # invariant under bijective relabeling, so coding the levels changes nothing
    # while satisfying the classifier's label requirements. Either mixed case
    # runs the same discrete-continuous estimator internally, keeping the
    # reported MI symmetric in X and Y.
    x_discrete = _looks_discrete(x_arr)
    y_discrete = _looks_discrete(y_arr)

    try:
        if y_discrete:
            y_codes = np.unique(y_arr, return_inverse=True)[1]
            feature = (
                np.unique(x_arr, return_inverse=True)[1].astype(float)
                if x_discrete
                else x_arr
            )
            mi = mutual_info_classif(
                feature.reshape(-1, 1),
                y_codes,
                discrete_features=[x_discrete],
                random_state=random_state,
            )[0]
        else:
            mi = mutual_info_regression(
                x_arr.reshape(-1, 1),
                y_arr,
                discrete_features=[x_discrete],
                random_state=random_state,
            )[0]
    except (ValueError, RuntimeError, FloatingPointError) as e:
        raise MetricComputationError(
            f"Failed to compute mutual_information: {type(e).__name__}: {e}"
        ) from e
    return MetricResult(name="mutual_information", value=float(mi), available=True)
