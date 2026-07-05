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
    n = x.shape[0]

    try:
        x_centered = x - x.mean()
        ss_xx = float(np.sum(x_centered**2))
        if ss_xx <= 0.0:  # constant x (guarded upstream, but be defensive)
            return _influence_no_value()

        slope, intercept = np.polyfit(x, y, 1)
        residuals = y - (slope * x + intercept)
        sse = float(np.sum(residuals**2))
        ss_tot_y = float(np.sum((y - y.mean()) ** 2))

        # A (near-)perfect linear fit leaves no residual structure — removing any
        # single row keeps the line, so no row is influential. Report zeros
        # rather than dividing by an ~0 residual mean square.
        if ss_tot_y <= 0.0 or sse / ss_tot_y < 1e-12:
            return {
                "max_cook_distance": MetricResult(
                    name="max_cook_distance", value=0.0, available=True
                ),
                "n_influential_points": MetricResult(
                    name="n_influential_points", value=0.0, available=True
                ),
            }

        s_squared = sse / (n - 2)
        leverage = 1.0 / n + x_centered**2 / ss_xx
        one_minus_h = np.maximum((1.0 - leverage) ** 2, 1e-12)
        cooks = residuals**2 * leverage / (2.0 * s_squared * one_minus_h)

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
