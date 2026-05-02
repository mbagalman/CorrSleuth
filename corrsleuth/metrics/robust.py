"""Robust correlation diagnostics for deep mode.

These metrics are lightweight sensitivity checks for Pearson-style association.
They are intended to help identify leverage-sensitive relationships, not to
replace visual inspection or model validation.
"""
from __future__ import annotations

import numpy as np
import scipy.stats as stats

from corrsleuth.exceptions import MetricComputationError
from corrsleuth.result import MetricResult
from corrsleuth.validation.input import CleanPair


_TAIL_FRACTION = 0.01
_MIN_N_FOR_ROBUST = 50
_MIN_N_AFTER_TRIM = 30
ROBUST_METRIC_MIN_N = _MIN_N_FOR_ROBUST


def _pearson_from_arrays(name: str, x: np.ndarray, y: np.ndarray) -> MetricResult:
    if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return MetricResult(name=name, value=None, available=True)
    try:
        r, _ = stats.pearsonr(x, y)
    except (ValueError, RuntimeError, FloatingPointError) as e:
        raise MetricComputationError(
            f"Failed to compute {name}: {type(e).__name__}: {e}"
        ) from e
    return MetricResult(name=name, value=float(r), available=True)


def _insufficient_pair(pair: CleanPair, name: str) -> MetricResult | None:
    if pair.x_is_constant or pair.y_is_constant:
        return MetricResult(name=name, value=None, available=True)
    if pair.n_used < _MIN_N_FOR_ROBUST:
        return MetricResult(name=name, value=None, available=True)
    return None


def compute_trimmed_pearson(pair: CleanPair) -> MetricResult:
    """Pearson correlation after dropping the outer 1% of either variable."""
    name = "pearson_trimmed_1pct"
    insufficient = _insufficient_pair(pair, name)
    if insufficient is not None:
        return insufficient

    x_low = pair.x.quantile(_TAIL_FRACTION)
    x_high = pair.x.quantile(1 - _TAIL_FRACTION)
    y_low = pair.y.quantile(_TAIL_FRACTION)
    y_high = pair.y.quantile(1 - _TAIL_FRACTION)
    mask = pair.x.between(x_low, x_high) & pair.y.between(y_low, y_high)
    x = pair.x[mask].to_numpy()
    y = pair.y[mask].to_numpy()

    if len(x) < _MIN_N_AFTER_TRIM:
        pair.warnings.append(
            f"Fewer than {_MIN_N_AFTER_TRIM} rows remain after trimming extremes; "
            f"{name} is not computed."
        )
        return MetricResult(name=name, value=None, available=True)
    return _pearson_from_arrays(name, x, y)


def compute_winsorized_pearson(pair: CleanPair) -> MetricResult:
    """Pearson correlation after clipping both variables at the 1st/99th percentiles."""
    name = "pearson_winsorized_1pct"
    insufficient = _insufficient_pair(pair, name)
    if insufficient is not None:
        return insufficient

    x_low = pair.x.quantile(_TAIL_FRACTION)
    x_high = pair.x.quantile(1 - _TAIL_FRACTION)
    y_low = pair.y.quantile(_TAIL_FRACTION)
    y_high = pair.y.quantile(1 - _TAIL_FRACTION)
    x = pair.x.clip(lower=x_low, upper=x_high).to_numpy()
    y = pair.y.clip(lower=y_low, upper=y_high).to_numpy()
    return _pearson_from_arrays(name, x, y)


def compute_biweight_midcorrelation(pair: CleanPair) -> MetricResult:
    """Biweight midcorrelation using median and MAD-based Tukey weights."""
    name = "biweight_midcorrelation"
    insufficient = _insufficient_pair(pair, name)
    if insufficient is not None:
        return insufficient

    x = pair.x.to_numpy(dtype=float)
    y = pair.y.to_numpy(dtype=float)
    x_median = np.median(x)
    y_median = np.median(y)
    x_mad = stats.median_abs_deviation(x, scale="normal")
    y_mad = stats.median_abs_deviation(y, scale="normal")
    if x_mad == 0 or y_mad == 0:
        return MetricResult(name=name, value=None, available=True)

    x_u = (x - x_median) / (9.0 * x_mad)
    y_u = (y - y_median) / (9.0 * y_mad)
    mask = (np.abs(x_u) < 1) & (np.abs(y_u) < 1)
    if mask.sum() < 2:
        return MetricResult(name=name, value=None, available=True)

    x_centered = x[mask] - x_median
    y_centered = y[mask] - y_median
    x_weighted = x_centered * (1 - x_u[mask] ** 2) ** 2
    y_weighted = y_centered * (1 - y_u[mask] ** 2) ** 2
    denominator = np.sqrt(np.sum(x_weighted**2) * np.sum(y_weighted**2))
    if denominator == 0:
        return MetricResult(name=name, value=None, available=True)
    return MetricResult(
        name=name,
        value=float(np.sum(x_weighted * y_weighted) / denominator),
        available=True,
    )


def compute_median_clipped_pearson(pair: CleanPair) -> MetricResult:
    """Pearson after clipping deviations around each median at the 80th percentile."""
    name = "pearson_median_clipped_20pct"
    insufficient = _insufficient_pair(pair, name)
    if insufficient is not None:
        return insufficient

    x = _bend(pair.x.to_numpy(dtype=float))
    y = _bend(pair.y.to_numpy(dtype=float))
    return _pearson_from_arrays(name, x, y)


def _bend(values: np.ndarray, beta: float = 0.20) -> np.ndarray:
    median = np.median(values)
    centered = values - median
    omega = np.quantile(np.abs(centered), 1 - beta)
    if omega == 0:
        return np.zeros_like(centered)
    return np.clip(centered, -omega, omega)
