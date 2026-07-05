"""Shape diagnostics that supplement the linear/rank/distance metrics.

Both diagnostics here are pure numpy/scipy — no optional dependency, cheap
enough to run in every mode — and are consumed only by the heuristic cascade
and :class:`~corrsleuth.result.MetricDiagnostics`, never surfaced in the
public metrics table the way Pearson/distance correlation/mutual information
are. They exist to catch two blind spots the existing rank/linear/distance
metrics leave open (see docs/shape-diagnostics-design.md):

- :func:`compute_bin_lof_r2_gain` — a lack-of-fit test comparing an
  equal-frequency-bin model of Y|X to a straight-line fit. A classical
  technique (Neter, Kutner, Nachtsheim & Wasserman, *Applied Linear
  Statistical Models*, lack-of-fit F-test using grouped X as a stand-in for
  replicates), used here to catch smooth monotonic curvature (exponential,
  logarithmic) and step/threshold functions that keep Pearson and Spearman
  close together despite real nonlinearity.
- :func:`compute_squared_correlation` — the correlation between X² and Y²,
  used to catch dependence that shows up in magnitude but not in the raw
  signed values (e.g. points scattered around a circle, where X and Y are
  strongly dependent but Pearson/Spearman/distance correlation on the raw
  values are all near zero).
"""

from __future__ import annotations

import numpy as np
import scipy.stats as stats

from corrsleuth.exceptions import MetricComputationError
from corrsleuth.result import MetricResult
from corrsleuth.validation.input import CleanPair

#: Rows per bin the equal-frequency binning aims for. Chosen so bin means are
#: themselves reasonably stable (n=10 gives a bin mean of ~10 points) without
#: needing an excessive sample size.
_TARGET_POINTS_PER_BIN = 10

#: Bin-count bounds. The floor (5) keeps at least a few bins so the test can
#: still distinguish curvature from a straight line; the ceiling (20) keeps
#: bins from becoming so numerous that each is dominated by noise.
_MIN_BINS = 5
_MAX_BINS = 20

#: Minimum rows before the lack-of-fit test is computed: enough for
#: ``_MIN_BINS`` bins of at least ``_TARGET_POINTS_PER_BIN`` rows each.
_MIN_N_FOR_BIN_LOF = _MIN_BINS * _TARGET_POINTS_PER_BIN


def compute_bin_lof_r2_gain(pair: CleanPair) -> MetricResult:
    """Bin-mean R² minus linear-fit R² — a lack-of-fit test for curvature.

    Sorts by X, splits into equal-frequency bins, and compares how much
    variance in Y a piecewise-constant "bin mean" model explains versus a
    single straight line. A positive gain means the data has structure (a
    curve, a step, a bend) a straight line does not capture — the bin model
    can only do better than or as well as the line, so this is bounded below
    by (slightly negative, from finite-sample noise) and unbounded above
    up to ``1 - r2_linear``.

    Returns ``None`` (``MetricResult.no_value``) for constant inputs,
    ``n_used`` below :data:`_MIN_N_FOR_BIN_LOF`, or a degenerate bin split
    (fewer than 2 points in some bin, which can happen with heavy ties in X).
    """
    name = "bin_lof_r2_gain"

    if pair.x_is_constant or pair.y_is_constant:
        return MetricResult.no_value(name)

    if pair.n_used < _MIN_N_FOR_BIN_LOF:
        return MetricResult.no_value(name)

    x = pair.x.to_numpy()
    y = pair.y.to_numpy()
    n = x.shape[0]

    try:
        order = np.argsort(x, kind="mergesort")
        xs = x[order]
        ys = y[order]

        n_bins = int(np.clip(n // _TARGET_POINTS_PER_BIN, _MIN_BINS, _MAX_BINS))
        bin_indices = np.array_split(np.arange(n), n_bins)
        if any(len(idx) < 2 for idx in bin_indices):
            return MetricResult.no_value(name)

        ss_tot = float(np.sum((ys - ys.mean()) ** 2))
        if ss_tot == 0.0:
            return MetricResult.no_value(name)

        y_bin_pred = np.empty_like(ys)
        for idx in bin_indices:
            y_bin_pred[idx] = ys[idx].mean()
        r2_bins = 1.0 - float(np.sum((ys - y_bin_pred) ** 2)) / ss_tot

        slope, intercept = np.polyfit(xs, ys, 1)
        y_linear_pred = slope * xs + intercept
        r2_linear = 1.0 - float(np.sum((ys - y_linear_pred) ** 2)) / ss_tot
    except (ValueError, RuntimeError, FloatingPointError) as e:
        raise MetricComputationError(
            f"Failed to compute {name}: {type(e).__name__}: {e}"
        ) from e

    return MetricResult(name=name, value=float(r2_bins - r2_linear), available=True)


def compute_squared_correlation(pair: CleanPair) -> MetricResult:
    """Pearson correlation between X² and Y².

    Catches dependence carried in magnitude rather than sign or rank — e.g.
    points scattered around a circle (X² + Y² ≈ const), where knowing X
    constrains |Y| but not sign(Y), so Pearson/Spearman/distance correlation
    on the raw values are all near zero while ``corr(X², Y²)`` is strongly
    (typically negatively) correlated.

    Returns ``None`` (``MetricResult.no_value``) for constant inputs, or when
    X² or Y² is itself constant (e.g. X is symmetric two-valued data).
    """
    name = "sq_corr"

    if pair.x_is_constant or pair.y_is_constant:
        return MetricResult.no_value(name)

    x2 = pair.x.to_numpy() ** 2
    y2 = pair.y.to_numpy() ** 2

    if np.std(x2) == 0.0 or np.std(y2) == 0.0:
        return MetricResult.no_value(name)

    try:
        r, _ = stats.pearsonr(x2, y2)
    except (ValueError, RuntimeError, FloatingPointError) as e:
        raise MetricComputationError(
            f"Failed to compute {name}: {type(e).__name__}: {e}"
        ) from e

    return MetricResult(name=name, value=float(r), available=True)
