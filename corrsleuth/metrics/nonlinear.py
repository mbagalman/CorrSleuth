"""Nonlinear dependence diagnostics for deep mode.

Currently exposes Chatterjee's xi — an asymmetric coefficient of association
that detects whether one variable is a (noisy) function of another. Both
directions are available: ``compute_chatterjee_xi`` returns
``xi(pair.x -> pair.y)`` and ``compute_chatterjee_xi_reverse`` returns
``xi(pair.y -> pair.x)``. See ``docs/phase4-nonlinear-metrics-design-note.md``
for the candidate evaluation that motivated picking xi over HSIC, Hoeffding's
D, MGC, and MIC.
"""
from __future__ import annotations

import numpy as np
import scipy.stats as stats

from corrsleuth.exceptions import MetricComputationError
from corrsleuth.result import MetricResult
from corrsleuth.validation.input import CleanPair


_MIN_N_FOR_CHATTERJEE_XI = 20


def _compute_xi_directional(
    x_sort_key: np.ndarray, y_values: np.ndarray, name: str
) -> MetricResult:
    """Core ξ computation: sort by ``x_sort_key`` (with ``y_values`` as tie-break)
    and accumulate Y-rank differences.

    The lexicographic tie-break makes the result a pure function of the
    multiset of ``(x, y)`` pairs — shuffling rows of the input DataFrame does
    not change the value.
    """
    n = x_sort_key.shape[0]
    try:
        order = np.lexsort((y_values, x_sort_key))
        y_sorted = y_values[order]
        y_ranks = stats.rankdata(y_sorted, method="ordinal")
        diffs = np.abs(np.diff(y_ranks))
        xi = 1.0 - (3.0 * float(np.sum(diffs))) / (n * n - 1.0)
    except (ValueError, RuntimeError, FloatingPointError) as e:
        raise MetricComputationError(
            f"Failed to compute {name}: {type(e).__name__}: {e}"
        ) from e
    return MetricResult(name=name, value=float(xi), available=True)


def compute_chatterjee_xi(pair: CleanPair) -> MetricResult:
    """Compute Chatterjee's coefficient of correlation ``xi(X -> Y)``.

    From Chatterjee (2020), "A new coefficient of correlation" (JASA). The
    statistic is asymmetric: ``xi(X -> Y)`` measures whether ``Y`` is a (noisy)
    function of ``X``. It converges to 0 when ``X`` and ``Y`` are independent
    and approaches 1 when ``Y`` is a measurable function of ``X``. For finite
    ``n`` the empirical value lies roughly in ``[-0.5, 1]``.

    Conventions used here
    ---------------------
    - ``X`` is ``pair.x`` and ``Y`` is ``pair.y`` (matches ``profile_pair``'s
      argument order). For the reverse direction call
      :func:`compute_chatterjee_xi_reverse`.
    - Ties in ``X`` are broken lexicographically by ``Y``, and ties in ``Y``
      are broken via ordinal ranking. This makes the value invariant to the
      row order of the underlying DataFrame; the existing ``high_tie_rate``
      warning already flags datasets where the canonical tie-break may matter.

    Returns ``None`` when either column is constant or when ``n_used`` is too
    small for a reliable estimate.
    """
    name = "chatterjee_xi"

    if pair.x_is_constant or pair.y_is_constant:
        return MetricResult(name=name, value=None, available=True)

    if pair.n_used < _MIN_N_FOR_CHATTERJEE_XI:
        pair.warnings.append(
            f"n_used < {_MIN_N_FOR_CHATTERJEE_XI}. chatterjee_xi is not "
            "computed because it converges slowly on small samples."
        )
        return MetricResult(name=name, value=None, available=True)

    return _compute_xi_directional(pair.x.to_numpy(), pair.y.to_numpy(), name)


def compute_chatterjee_xi_reverse(pair: CleanPair) -> MetricResult:
    """Compute Chatterjee's coefficient in the reverse direction, ``xi(Y -> X)``.

    Mirrors :func:`compute_chatterjee_xi` with the roles of ``X`` and ``Y``
    swapped, so this measures whether ``X`` is a (noisy) function of ``Y``.
    For target scans (where ``profile_pair`` is invoked as
    ``profile_pair(data, target, candidate)``) this is the candidate→target
    direction — the one feature-engineering users typically want when
    prioritizing predictors against a target.

    Returns ``None`` for constant inputs or when ``n_used`` is below the
    small-sample guard. The small-sample warning is emitted by
    :func:`compute_chatterjee_xi` to avoid duplication when both directions
    are computed in the same call (as deep mode does).
    """
    name = "chatterjee_xi_reverse"

    if pair.x_is_constant or pair.y_is_constant:
        return MetricResult(name=name, value=None, available=True)

    if pair.n_used < _MIN_N_FOR_CHATTERJEE_XI:
        return MetricResult(name=name, value=None, available=True)

    return _compute_xi_directional(pair.y.to_numpy(), pair.x.to_numpy(), name)
