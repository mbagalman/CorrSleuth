"""Regression-influence diagnostics — which rows drive the linear fit.

CorrSleuth already flags leverage in the aggregate: the `possible_outlier_or_
leverage` label is gated on a 1%-trim sensitivity check, and deep mode adds the
robust-Pearson family. Those answer *"is Pearson sensitive to extremes?"* but
not *"which rows, and how many?"* This module answers the row-level question
with the classical regression-influence measure, Cook's distance, so the
`outlier_sensitivity` axis can distinguish a single dominant point from a
leverage cluster.

Everything here is closed-form linear algebra on the simple ``y ~ x`` fit — no
matrix inversion, no new dependency, cheap enough to run in every mode. For a
one-predictor regression the leverage (hat) values and Cook's distances have
elementary forms:

    h_i = 1/n + (x_i − x̄)² / Σ(x_j − x̄)²                 (leverage)
    D_i = e_i² · h_i / (2 · s² · (1 − h_i)²)               (Cook's distance)

where ``e_i`` is the residual, ``s²`` the residual mean square ``SSE/(n − 2)``,
and the ``2`` is the parameter count (slope + intercept).

A note on the threshold: Cook's distance suffers *masking* — a tight cluster of
identical outliers deflates each point's individual ``D_i`` because removing any
one leaves the others holding the fit. The classical ``D > 1`` cutoff therefore
misses such clusters. The softer Cook & Weisberg (1982) ``D > 0.5``
("worth investigating") is used instead: it catches the masked cluster while
sitting far above what clean data produces (clean linear data measured a max
``D`` around 0.03–0.4 depending on ``n``, versus ≥ 0.5 for genuinely
leverage-influenced pairs).
"""

from __future__ import annotations

import numpy as np

from corrsleuth.exceptions import MetricComputationError
from corrsleuth.result import MetricResult
from corrsleuth.validation.input import CleanPair

#: Minimum rows before Cook's distances are computed. Mirrors the other
#: diagnostic floors; also keeps the max Cook's distance of clean data safely
#: below the influence threshold (at n = 50 clean data measured a max around
#: 0.40, at n >= 80 around 0.18, versus the 0.5 cutoff below).
_MIN_N_FOR_INFLUENCE = 50

#: Cook's distance above which a row is counted as influential. The softer
#: Cook & Weisberg (1982) "worth investigating" cutoff rather than the classical
#: ``D > 1``, because a masked cluster of outliers deflates each point's Cook's
#: distance below 1 (see the module docstring). Sits in the wide empty gap
#: between clean data (max ~0.03–0.4) and leverage-influenced pairs (>= 0.5).
COOK_INFLUENTIAL_THRESHOLD = 0.5

_INFLUENCE_NAMES = ("max_cook_distance", "n_influential_points")


def _influence_no_value() -> dict[str, MetricResult]:
    return {name: MetricResult.no_value(name) for name in _INFLUENCE_NAMES}


def _cooks_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    """Cook's distance for each row of the elementary ``y ~ x`` fit.

    ``None`` if ``x`` has no variance (constant input is guarded upstream by
    every caller, but this stays defensive). An all-zero array for a
    (near-)perfect linear fit, which leaves no residual structure for any row
    to be influential.
    """
    n = x.shape[0]
    x_centered = x - x.mean()
    ss_xx = float(np.sum(x_centered**2))
    if ss_xx <= 0.0:
        return None

    slope, intercept = np.polyfit(x, y, 1)
    residuals = y - (slope * x + intercept)
    sse = float(np.sum(residuals**2))
    ss_tot_y = float(np.sum((y - y.mean()) ** 2))
    if ss_tot_y <= 0.0 or sse / ss_tot_y < 1e-12:
        return np.zeros(n)

    s_squared = sse / (n - 2)
    leverage = 1.0 / n + x_centered**2 / ss_xx
    one_minus_h = np.maximum((1.0 - leverage) ** 2, 1e-12)
    return residuals**2 * leverage / (2.0 * s_squared * one_minus_h)


def compute_influence(pair: CleanPair) -> dict[str, MetricResult]:
    """Row-level influence of the ``y ~ x`` fit, via Cook's distance.

    Returns a dict with two :class:`MetricResult` entries: ``max_cook_distance``
    (the largest Cook's distance — how much the most influential single row moves
    the fit) and ``n_influential_points`` (how many rows exceed
    :data:`COOK_INFLUENTIAL_THRESHOLD`, which separates a single dominant point
    from a leverage cluster).

    Both are ``None`` (``MetricResult.no_value``) for constant inputs or
    ``n_used`` below :data:`_MIN_N_FOR_INFLUENCE`. A degenerate (near-perfect)
    linear fit has no residual structure to be influential, so it reports a max
    of 0 and a count of 0.
    """
    if pair.x_is_constant or pair.y_is_constant:
        return _influence_no_value()
    if pair.n_used < _MIN_N_FOR_INFLUENCE:
        return _influence_no_value()

    x = pair.x.to_numpy().astype(float)
    y = pair.y.to_numpy().astype(float)

    try:
        cooks = _cooks_distances(x, y)
        if cooks is None:
            return _influence_no_value()
        max_cook = float(np.max(cooks))
        n_influential = float(int(np.sum(cooks > COOK_INFLUENTIAL_THRESHOLD)))
    except (ValueError, RuntimeError, FloatingPointError) as e:
        raise MetricComputationError(
            f"Failed to compute influence: {type(e).__name__}: {e}"
        ) from e

    return {
        "max_cook_distance": MetricResult(
            name="max_cook_distance", value=max_cook, available=True
        ),
        "n_influential_points": MetricResult(
            name="n_influential_points", value=n_influential, available=True
        ),
    }


def compute_influential_mask(pair: CleanPair) -> np.ndarray | None:
    """Boolean mask (aligned to ``pair.x``/``pair.y``) of rows exceeding
    :data:`COOK_INFLUENTIAL_THRESHOLD`.

    Reuses the same Cook's distances as :func:`compute_influence`, so the rows
    flagged here are exactly what ``n_influential_points`` counts. Exposed for
    callers that need to *exclude* those rows and re-test something else on the
    remainder (e.g. re-testing heteroscedasticity to check whether an apparent
    variance-shape signal is really just this same leverage artifact — see
    ``heuristics/classifier.py``'s ``detect_metric_warnings``).

    ``None`` under the same guards as :func:`compute_influence` (constant
    input, ``n_used`` below :data:`_MIN_N_FOR_INFLUENCE`, or a degenerate fit
    with no residual structure to flag).
    """
    if pair.x_is_constant or pair.y_is_constant:
        return None
    if pair.n_used < _MIN_N_FOR_INFLUENCE:
        return None

    x = pair.x.to_numpy().astype(float)
    y = pair.y.to_numpy().astype(float)

    try:
        cooks = _cooks_distances(x, y)
    except (ValueError, RuntimeError, FloatingPointError) as e:
        raise MetricComputationError(
            f"Failed to compute influence: {type(e).__name__}: {e}"
        ) from e
    if cooks is None:
        return None
    return cooks > COOK_INFLUENTIAL_THRESHOLD
