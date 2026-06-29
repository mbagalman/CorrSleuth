"""Nonlinear dependence diagnostics for deep mode.

Currently exposes Chatterjee's xi — an asymmetric coefficient of association
that detects whether one variable is a (noisy) function of another. Both
directions are available: ``compute_chatterjee_xi`` returns
``xi(pair.x -> pair.y)`` and ``compute_chatterjee_xi_reverse`` returns
``xi(pair.y -> pair.x)``. xi was chosen over alternatives such as HSIC,
Hoeffding's D, MGC, and MIC for its near-linear ``O(n log n)`` cost, its
interpretable scale (0 under independence, approaching 1 for a functional
relationship), and its directionality.
"""

from __future__ import annotations

import numpy as np
import scipy.stats as stats

from corrsleuth.exceptions import MetricComputationError
from corrsleuth.result import MetricResult
from corrsleuth.validation.input import CleanPair

#: Minimum rows before Chatterjee's xi is computed. xi is consistent but
#: converges slowly, and on tiny samples its finite-sample bias is large enough
#: to be misleading; 20 is a conservative floor (below the n = 30 used to gate
#: the heuristic labels, since xi is only ever reported as a supplementary
#: diagnostic, never as the basis for a label).
_MIN_N_FOR_CHATTERJEE_XI = 20


def _compute_xi_directional(
    x_sort_key: np.ndarray, y_values: np.ndarray, name: str, random_state: int
) -> MetricResult:
    """Core ξ computation: sort by ``x_sort_key`` (ties broken randomly) and
    accumulate Y-rank differences.

    Uses the tie-corrected estimator from Chatterjee (2020), eq. (2):

        ξ = 1 − (n · Σ |r_{i+1} − r_i|) / (2 · Σ l_i (n − l_i))

    where, in ``x_sort_key`` order, ``r_i = #{j : Y_j ≤ Y_i}`` and
    ``l_i = #{j : Y_j ≥ Y_i}``. When ``Y`` has no ties this reduces exactly to
    the simplified ``1 − 3·Σ|Δr| / (n²−1)`` form, but with ties in ``Y`` — e.g.
    a discrete or low-cardinality response — the simplified form overstates
    dependence, so the correction matters most for the feature-engineering
    (reverse) direction against a low-cardinality target.

    Ties in the *sort key* are broken with a seeded random permutation, not by
    ``Y``. Ordering tied ``x_sort_key`` values by ``Y`` would leak the response
    into the ordering and manufacture dependence — under independence with a
    discrete sort key it drives ξ toward 1. Random tie-breaking is Chatterjee's
    prescription and keeps ξ calibrated (→ 0 under independence). The value is
    reproducible for a given input and ``random_state``; when the sort key has
    ties it depends on the random tie-break, so — unlike the tie-free case — it
    is not invariant to the input row order.
    """
    n = x_sort_key.shape[0]
    try:
        # Seeded random permutation as the tie-break on the primary (x_sort_key)
        # sort; the Y-tie correction lives in the denominator below.
        tie_breaker = np.random.default_rng(random_state).permutation(n)
        order = np.lexsort((tie_breaker, x_sort_key))
        y_sorted = y_values[order]
        # r_i = #{j : Y_j <= Y_i} (max rank), l_i = #{j : Y_j >= Y_i}.
        r = stats.rankdata(y_sorted, method="max")
        l_counts = stats.rankdata(-y_sorted, method="max")
        numerator = n * float(np.sum(np.abs(np.diff(r))))
        denominator = 2.0 * float(np.sum(l_counts * (n - l_counts)))
        if denominator == 0.0:
            # All l_i == n, i.e. a constant Y. Constant inputs are guarded
            # upstream, but guard here too rather than divide by zero.
            return MetricResult.no_value(name)
        xi = 1.0 - numerator / denominator
    except (ValueError, RuntimeError, FloatingPointError) as e:
        raise MetricComputationError(
            f"Failed to compute {name}: {type(e).__name__}: {e}"
        ) from e
    return MetricResult(name=name, value=float(xi), available=True)


def compute_chatterjee_xi(pair: CleanPair, random_state: int = 42) -> MetricResult:
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
    - Ties in ``X`` (the sort key) are broken with a seeded random permutation
      — never by ``Y``, which would leak the response and inflate ξ — while ties
      in ``Y`` are handled by Chatterjee's tie-corrected denominator, so the
      value stays well-calibrated for discrete or low-cardinality responses.
    - ``random_state`` seeds that tie-break, so the value is reproducible for a
      given input. When ``X`` has ties the value depends on the random
      tie-break and is therefore not invariant to the input row order; a heavily
      tied ``X`` also carries additional sampling variability in ξ, since a
      different tie-break (a different ``random_state``) would shift the value,
      so treat ξ as noisier for low-cardinality sort variables.

    Returns ``None`` when either column is constant or when ``n_used`` is too
    small for a reliable estimate.
    """
    name = "chatterjee_xi"

    if pair.x_is_constant or pair.y_is_constant:
        return MetricResult.no_value(name)

    if pair.n_used < _MIN_N_FOR_CHATTERJEE_XI:
        pair.warnings.append(
            f"n_used < {_MIN_N_FOR_CHATTERJEE_XI}. chatterjee_xi is not "
            "computed because it converges slowly on small samples."
        )
        return MetricResult.no_value(name)

    return _compute_xi_directional(
        pair.x.to_numpy(), pair.y.to_numpy(), name, random_state
    )


def compute_chatterjee_xi_reverse(
    pair: CleanPair, random_state: int = 42
) -> MetricResult:
    """Compute Chatterjee's coefficient in the reverse direction, ``xi(Y -> X)``.

    Mirrors :func:`compute_chatterjee_xi` with the roles of ``X`` and ``Y``
    swapped, so this measures whether ``X`` is a (noisy) function of ``Y``.
    For target scans (where ``profile_pair`` is invoked as
    ``profile_pair(data, target, candidate)``) this is the candidate→target
    direction — the one feature-engineering users typically want when
    prioritizing predictors against a target.

    ``Y`` is the sort key here, so its ties are broken with the seeded random
    permutation (see :func:`compute_chatterjee_xi`); this is what keeps
    ``xi(Y -> X)`` calibrated when the target is discrete or low-cardinality.

    Returns ``None`` for constant inputs or when ``n_used`` is below the
    small-sample guard. The small-sample warning is emitted by
    :func:`compute_chatterjee_xi` to avoid duplication when both directions
    are computed in the same call (as deep mode does).
    """
    name = "chatterjee_xi_reverse"

    if pair.x_is_constant or pair.y_is_constant:
        return MetricResult.no_value(name)

    if pair.n_used < _MIN_N_FOR_CHATTERJEE_XI:
        return MetricResult.no_value(name)

    return _compute_xi_directional(
        pair.y.to_numpy(), pair.x.to_numpy(), name, random_state
    )
