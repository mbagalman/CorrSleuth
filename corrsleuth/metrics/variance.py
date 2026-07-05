"""Variance-shape (heteroscedasticity) diagnostics.

Where ``metrics/shape.py`` asks whether the *conditional mean* E[Y|X] departs
from a straight line, this module asks whether the *conditional variance*
Var[Y|X] is constant — the other way a "linear-looking" relationship can
mislead. A pair can have a perfectly linear mean trend (so Pearson is a fine
point estimate) while the residual spread fans out with X, which quietly breaks
any homoscedastic inference built on top of it (standard errors, prediction
intervals, ordinary least squares optimality).

Like the shape diagnostics, this is pure numpy/scipy (no ``statsmodels``
dependency — those tests are cheap to implement directly), runs in every mode,
and is consumed only by the ``variance_shape`` secondary axis and
``MetricDiagnostics``; it is never surfaced in the public metrics table.

Two classical tests, sharing one ordinary-least-squares fit of ``y ~ x``:

- **Breusch-Pagan (Koenker's studentized variant)** — regress the squared
  residuals on ``x``; the statistic ``n · R²`` of that auxiliary regression is
  ``χ²(1)`` under homoscedasticity. Koenker's studentized form is used because
  it is robust to non-normal errors, unlike the original Breusch-Pagan (1979)
  scaling. This is the primary "is there heteroscedasticity?" signal.
- **Goldfeld-Quandt** — sort by ``x``, drop the middle fraction, fit separate
  lines to the low-``x`` and high-``x`` groups, and take the ratio of their
  residual variances. This gives both an effect size (how many times larger the
  spread is on one side) and a direction, which the p-value alone cannot.

The p-value says *whether* the spread changes; the ratio says *how much* and
*which way*. Both matter: at large ``n`` the Breusch-Pagan test rejects for
trivially small heteroscedasticity, so the ``variance_shape`` axis
(``heuristics/classifier.py``) pairs the p-value with a ratio effect-size floor
before calling a pair heteroscedastic.
"""

from __future__ import annotations

import numpy as np
import scipy.stats as stats

from corrsleuth.exceptions import MetricComputationError
from corrsleuth.result import MetricResult
from corrsleuth.validation.input import CleanPair

#: Minimum rows before the heteroscedasticity tests are computed. Mirrors
#: ``metrics/shape.py``'s bin lack-of-fit floor: the Goldfeld-Quandt split needs
#: enough points per group to estimate a residual variance, and the
#: Breusch-Pagan auxiliary regression needs enough residuals to be stable.
_MIN_N_FOR_HETEROSCEDASTICITY = 50

#: Central fraction of the (x-sorted) rows dropped before the Goldfeld-Quandt
#: split, so the low and high groups are well separated in ``x``. The classical
#: choice is ~1/5 to 1/3; 0.2 keeps most rows while giving a clean gap.
_GQ_DROP_FRACTION = 0.2

_BP_NAME = "bp_pvalue"
_GQ_NAME = "gq_ratio"


def _no_value_result() -> dict[str, MetricResult]:
    return {
        _BP_NAME: MetricResult.no_value(_BP_NAME),
        _GQ_NAME: MetricResult.no_value(_GQ_NAME),
    }


def compute_heteroscedasticity(pair: CleanPair) -> dict[str, MetricResult]:
    """Test whether Var[Y|X] is constant, via Breusch-Pagan and Goldfeld-Quandt.

    Returns a dict with two :class:`MetricResult` entries: ``bp_pvalue`` (the
    Koenker studentized Breusch-Pagan p-value — small means heteroscedastic) and
    ``gq_ratio`` (the Goldfeld-Quandt ratio of high-``x`` to low-``x`` residual
    variance — ``> 1`` means spread grows with ``x``, ``< 1`` means it shrinks).

    Both are ``None`` (``MetricResult.no_value``) for constant inputs, ``n_used``
    below :data:`_MIN_N_FOR_HETEROSCEDASTICITY`, or a degenerate fit (an
    essentially perfect linear fit leaves no residual variance to analyze).
    """
    if pair.x_is_constant or pair.y_is_constant:
        return _no_value_result()
    if pair.n_used < _MIN_N_FOR_HETEROSCEDASTICITY:
        return _no_value_result()

    x = pair.x.to_numpy()
    y = pair.y.to_numpy()
    n = x.shape[0]

    try:
        slope, intercept = np.polyfit(x, y, 1)
        residuals = y - (slope * x + intercept)
        squared = residuals**2

        # A degenerate (near-)perfect linear fit leaves essentially no residual
        # spread to analyze; testing the floating-point residual noise would
        # produce meaningless statistics (and risk dividing by zero). Guard
        # relative to the spread of y so it fires for exactly-linear inputs,
        # where polyfit still leaves ~1e-13 residuals rather than exact zeros.
        ss_res = float(np.sum(squared))
        ss_tot_y = float(np.sum((y - y.mean()) ** 2))
        if ss_tot_y <= 0.0 or ss_res / ss_tot_y < 1e-12:
            return _no_value_result()

        # Breusch-Pagan (Koenker studentized): LM = n * R^2 of squared residuals
        # regressed on x, distributed chi^2(1) under homoscedasticity.
        aux_slope, aux_intercept = np.polyfit(x, squared, 1)
        aux_pred = aux_slope * x + aux_intercept
        aux_ss_tot = float(np.sum((squared - squared.mean()) ** 2))
        aux_r2 = (
            1.0 - float(np.sum((squared - aux_pred) ** 2)) / aux_ss_tot
            if aux_ss_tot > 0.0
            else 0.0
        )
        bp_lm = n * max(0.0, aux_r2)
        bp_pvalue = float(stats.chi2.sf(bp_lm, 1))

        # Goldfeld-Quandt: separate line fits on the low- and high-x groups
        # (middle fraction dropped), ratio of their residual variances.
        order = np.argsort(x, kind="mergesort")
        xs, ys = x[order], y[order]
        n_group = (n - int(_GQ_DROP_FRACTION * n)) // 2
        if n_group < 3:  # need >= 1 residual d.o.f. per group (n - 2 slope/intercept)
            return _no_value_result()
        gq_ratio = _group_variance_ratio(
            xs[:n_group], ys[:n_group], xs[n - n_group :], ys[n - n_group :]
        )
        if gq_ratio is None:
            return _no_value_result()
    except (ValueError, RuntimeError, FloatingPointError) as e:
        raise MetricComputationError(
            f"Failed to compute heteroscedasticity: {type(e).__name__}: {e}"
        ) from e

    return {
        _BP_NAME: MetricResult(name=_BP_NAME, value=bp_pvalue, available=True),
        _GQ_NAME: MetricResult(name=_GQ_NAME, value=gq_ratio, available=True),
    }


def _group_variance_ratio(
    x_low: np.ndarray, y_low: np.ndarray, x_high: np.ndarray, y_high: np.ndarray
) -> float | None:
    """Ratio of the high group's residual variance to the low group's, each from
    its own line fit. ``None`` when the low group has no residual spread."""

    def residual_mean_square(xg: np.ndarray, yg: np.ndarray) -> float:
        # A low-cardinality/binary x (e.g. a 0/1 flag) can leave an entire
        # x-sorted group with zero x-variance -- np.polyfit's design matrix is
        # then singular (SVD failure). Mirror metrics/shape.py's segmentation
        # fallback: fit an intercept-only (mean) model instead of a line, since
        # there is no slope to estimate without x variation in the group.
        ss_xx = float(np.sum((xg - xg.mean()) ** 2))
        if ss_xx > 0.0:
            slope, intercept = np.polyfit(xg, yg, 1)
            resid = yg - (slope * xg + intercept)
            dof = len(xg) - 2
        else:
            resid = yg - yg.mean()
            dof = len(xg) - 1
        return float(np.sum(resid**2)) / dof

    low_ms = residual_mean_square(x_low, y_low)
    high_ms = residual_mean_square(x_high, y_high)
    if low_ms <= 0.0:
        return None
    return high_ms / low_ms
