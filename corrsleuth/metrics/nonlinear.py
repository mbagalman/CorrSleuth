"""Nonlinear dependence diagnostics for deep mode.

Currently exposes Chatterjee's xi — an asymmetric coefficient of association
that detects whether one variable is a (noisy) function of another. See
``docs/phase4-nonlinear-metrics-design-note.md`` for the candidate evaluation
that motivated picking xi over HSIC, Hoeffding's D, MGC, and MIC.
"""
from __future__ import annotations

import numpy as np
import scipy.stats as stats

from corrsleuth.exceptions import MetricComputationError
from corrsleuth.result import MetricResult
from corrsleuth.validation.input import CleanPair


_MIN_N_FOR_CHATTERJEE_XI = 20


def compute_chatterjee_xi(pair: CleanPair) -> MetricResult:
    """Compute Chatterjee's coefficient of correlation ``xi_n(X -> Y)``.

    From Chatterjee (2020), "A new coefficient of correlation" (JASA). The
    statistic is asymmetric: ``xi(X -> Y)`` measures whether ``Y`` is a (noisy)
    function of ``X``. It converges to 0 when ``X`` and ``Y`` are independent
    and approaches 1 when ``Y`` is a measurable function of ``X``. For finite
    ``n`` the empirical value lies roughly in ``[-0.5, 1]``.

    Conventions used here
    ---------------------
    - ``X`` is ``pair.x`` and ``Y`` is ``pair.y`` (matches ``profile_pair``'s
      argument order; for target scans this is target → candidate).
    - Ties in ``X`` are broken via a stable sort on the input order, and ties
      in ``Y`` are broken via ordinal ranking. This keeps the metric
      deterministic across runs at the cost of a small bias under heavy ties;
      the existing ``high_tie_rate`` warning already flags those datasets.

    Returns ``None`` when either column is constant or when ``n_used`` is too
    small for a reliable estimate.
    """
    name = "chatterjee_xi"

    if pair.x_is_constant or pair.y_is_constant:
        return MetricResult(name=name, value=None, available=True)

    if pair.n_used < _MIN_N_FOR_CHATTERJEE_XI:
        pair.warnings.append(
            f"n_used < {_MIN_N_FOR_CHATTERJEE_XI}. {name} is not computed "
            "because it converges slowly on small samples."
        )
        return MetricResult(name=name, value=None, available=True)

    x = pair.x.to_numpy()
    y = pair.y.to_numpy()
    n = x.shape[0]

    try:
        order = np.argsort(x, kind="stable")
        y_sorted = y[order]
        y_ranks = stats.rankdata(y_sorted, method="ordinal")
        diffs = np.abs(np.diff(y_ranks))
        xi = 1.0 - (3.0 * float(np.sum(diffs))) / (n * n - 1.0)
    except (ValueError, RuntimeError, FloatingPointError) as e:
        raise MetricComputationError(
            f"Failed to compute {name}: {type(e).__name__}: {e}"
        ) from e

    return MetricResult(name=name, value=float(xi), available=True)
