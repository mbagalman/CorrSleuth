import numpy as np

from corrsleuth.exceptions import MetricComputationError, OptionalDependencyError
from corrsleuth.result import MetricResult
from corrsleuth.validation.input import CleanPair

#: Maximum distinct values for a variable to count as **discrete** for the
#: mutual-information estimator. scikit-learn's ``mutual_info_regression`` runs a
#: continuous k-NN (KSG) estimator by default, which misestimates a
#: low-cardinality integer/categorical column unless it is declared discrete via
#: ``discrete_features`` (and assumes a *continuous* target regardless). A
#: variable whose values are all whole numbers and number at most this many
#: distinct levels is treated as discrete: passed as a discrete feature when it
#: is X, and (when it is the target Y) a signal to withhold MI, since
#: ``mutual_info_regression`` has no correct estimate for a discrete target.
#: Chosen to cover binary/ordinal/small-count columns while leaving genuinely
#: continuous data (even integer-valued, like ages) that has many levels alone.
_MI_DISCRETE_MAX_CARDINALITY = 20


def _looks_discrete(values: np.ndarray) -> bool:
    """True when ``values`` are all whole numbers with at most
    :data:`_MI_DISCRETE_MAX_CARDINALITY` distinct levels — the low-cardinality
    integer/categorical case the continuous MI estimator misestimates."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return False
    if not np.all(finite == np.round(finite)):
        return False
    return np.unique(finite).size <= _MI_DISCRETE_MAX_CARDINALITY


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

    **Discreteness policy.** ``mutual_info_regression`` runs a continuous k-NN
    (KSG) estimator that misestimates low-cardinality integer/categorical data
    unless told otherwise. A discrete feature (X — all whole numbers, at most
    :data:`_MI_DISCRETE_MAX_CARDINALITY` distinct levels) is declared via
    ``discrete_features`` so the correct discrete-continuous estimator runs. A
    discrete **target** (Y) has no correct estimate from this function — it
    assumes a continuous target — so MI is *withheld* (``value=None``) with a
    warning rather than reported wrong; distance correlation covers dependence in
    that case.

    In ``mode='standard'`` or ``'deep'`` (both need the ``[standard]`` extras),
    raises :class:`OptionalDependencyError` if ``scikit-learn`` is not installed.
    In ``mode='lite'`` returns an unavailable
    :class:`MetricResult` instead. ``random_state`` is forwarded to
    :func:`sklearn.feature_selection.mutual_info_regression` for reproducibility.
    Returns ``value=None`` when either column is constant, when
    ``pair.n_used <= 3`` (too few observations for the estimator), or when the
    target is discrete (see the discreteness policy above).
    """
    try:
        from sklearn.feature_selection import mutual_info_regression
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

    # scikit-learn's mutual_info_regression assumes a *continuous* target and, by
    # default, continuous features. A discrete/low-cardinality target has no
    # correct estimate here (it needs a classification estimator), so withhold
    # rather than report a misestimated value; a discrete *feature* is handled
    # correctly by declaring it via discrete_features. See _looks_discrete.
    if _looks_discrete(y_arr):
        pair.warnings.append(
            "Mutual information is not computed: the target (y) is discrete / "
            "low-cardinality, and scikit-learn's mutual_info_regression assumes a "
            "continuous target. Rely on distance correlation for dependence here."
        )
        return MetricResult.no_value("mutual_information")

    x = x_arr.reshape(-1, 1)
    y = y_arr
    discrete_features = [_looks_discrete(x_arr)]

    try:
        mi = mutual_info_regression(
            x, y, discrete_features=discrete_features, random_state=random_state
        )[0]
    except (ValueError, RuntimeError, FloatingPointError) as e:
        raise MetricComputationError(
            f"Failed to compute mutual_information: {type(e).__name__}: {e}"
        ) from e
    return MetricResult(name="mutual_information", value=float(mi), available=True)
