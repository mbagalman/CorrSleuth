"""Robust correlation diagnostics for deep mode.

These metrics are lightweight sensitivity checks for Pearson-style association.
They are intended to help identify leverage-sensitive relationships, not to
replace visual inspection or model validation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.stats as stats

from corrsleuth.exceptions import MetricComputationError
from corrsleuth.result import MetricResult
from corrsleuth.validation.input import CleanPair

#: Fraction trimmed/winsorized from each tail of each variable (1% per side, so
#: 2% per variable). One percent is a deliberately gentle trim: it is enough to
#: neutralize a handful of extreme leverage points without materially reshaping
#: a clean distribution, so a large baseline-vs-robust gap points to a few
#: influential rows rather than to broad trimming. See the 1%-trim limitation
#: note in docs/interpretation-guide.md for when this misses mid-range leverage.
_TAIL_FRACTION = 0.01
#: Minimum rows before robust deep-mode metrics are computed. Below this a 1%
#: trim removes too few rows to be meaningful and the estimates are noisy; 50
#: keeps at least one row in each trimmed tail.
_MIN_N_FOR_ROBUST = 50
#: Minimum rows that must survive trimming for the robust estimate to be
#: reported. Mirrors :data:`LOW_N_THRESHOLD` so a trimmed correlation is never
#: computed on a sample CorrSleuth would otherwise flag as low-power.
_MIN_N_AFTER_TRIM = 30
ROBUST_METRIC_MIN_N = _MIN_N_FOR_ROBUST
#: Magnitude of the change in Pearson after the 1% trim above which the
#: relationship is flagged leverage-sensitive. Computed from the signed
#: difference (``abs(baseline - trimmed)``) so a sign flip counts in full.
_OUTLIER_SENSITIVE_DELTA = 0.20


def _pearson_from_arrays(name: str, x: np.ndarray, y: np.ndarray) -> MetricResult:
    if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return MetricResult.no_value(name)
    try:
        r, _ = stats.pearsonr(x, y)
    except (ValueError, RuntimeError, FloatingPointError) as e:
        raise MetricComputationError(
            f"Failed to compute {name}: {type(e).__name__}: {e}"
        ) from e
    return MetricResult(name=name, value=float(r), available=True)


def _insufficient_pair(pair: CleanPair, name: str) -> MetricResult | None:
    if pair.x_is_constant or pair.y_is_constant:
        return MetricResult.no_value(name)
    if pair.n_used < _MIN_N_FOR_ROBUST:
        return MetricResult.no_value(name)
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
        return MetricResult.no_value(name)
    return _pearson_from_arrays(name, x, y)


@dataclass
class OutlierSensitivity:
    """Outcome of the leverage-sensitivity check.

    ``status`` is ``"sensitive"``, ``"stable"``, or ``"unavailable"``.
    ``trimmed`` is the 1%-trimmed Pearson :class:`MetricResult` (reused as the
    deep-mode ``pearson_trimmed_1pct`` metric, so the value behind the leverage
    flag and the reported metric are one and the same). ``delta`` is
    ``abs(baseline - trimmed)`` when both are available, else ``None``.
    """

    status: str
    trimmed: MetricResult
    delta: float | None


def assess_outlier_sensitivity(
    pair: CleanPair, baseline_pearson: float | None
) -> OutlierSensitivity:
    """Flag Pearson as leverage-sensitive when the 1% trim moves it materially.

    Delegates the trimming to :func:`compute_trimmed_pearson` so the trimmed
    value is computed exactly once and the same way wherever it is consumed.
    The status is ``"unavailable"`` whenever a baseline or trimmed Pearson
    cannot be computed (constant input, ``n_used`` below the robust minimum, or
    too few rows surviving the trim).
    """
    trimmed = compute_trimmed_pearson(pair)
    if baseline_pearson is None or trimmed.value is None:
        return OutlierSensitivity(status="unavailable", trimmed=trimmed, delta=None)
    # Signed comparison: a sign flip after trimming (e.g. +0.55 -> -0.55) is the
    # most leverage-sensitive case there is; an abs-of-abs delta would score it
    # 0.0 and mislabel it "stable".
    delta = abs(baseline_pearson - trimmed.value)
    status = "sensitive" if delta > _OUTLIER_SENSITIVE_DELTA else "stable"
    return OutlierSensitivity(status=status, trimmed=trimmed, delta=delta)


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
    # Raw MAD (scale=1.0), not the normal-consistent MAD: the canonical biweight
    # midcorrelation (Wilcox; Langfelder & Horvath 2012) defines the Tukey
    # weights as u = (x - median) / (9 * MAD) with the *unscaled* MAD, so the
    # rejection cutoff is 9 raw MADs. Using scale="normal" would multiply MAD by
    # ~1.4826 and push the cutoff to ~13 MADs, downweighting outliers far less
    # aggressively than the estimator this metric is named for.
    x_mad = stats.median_abs_deviation(x, scale=1.0)
    y_mad = stats.median_abs_deviation(y, scale=1.0)
    if x_mad == 0 or y_mad == 0:
        return MetricResult.no_value(name)

    x_u = (x - x_median) / (9.0 * x_mad)
    y_u = (y - y_median) / (9.0 * y_mad)
    mask = (np.abs(x_u) < 1) & (np.abs(y_u) < 1)
    if mask.sum() < 2:
        return MetricResult.no_value(name)

    x_centered = x[mask] - x_median
    y_centered = y[mask] - y_median
    x_weighted = x_centered * (1 - x_u[mask] ** 2) ** 2
    y_weighted = y_centered * (1 - y_u[mask] ** 2) ** 2
    denominator = np.sqrt(np.sum(x_weighted**2) * np.sum(y_weighted**2))
    if denominator == 0:
        return MetricResult.no_value(name)
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
    # beta is the bending constant of the percentage-bend correlation (Wilcox),
    # NOT the biweight midcorrelation above: absolute deviations from the median
    # beyond the (1 - beta) quantile are clipped to that quantile. 0.20 is
    # Wilcox's standard default, trading a small amount of efficiency at the
    # Gaussian for resistance to ~20% contamination.
    median = np.median(values)
    centered = values - median
    omega = np.quantile(np.abs(centered), 1 - beta)
    if omega == 0:
        return np.zeros_like(centered)
    return np.clip(centered, -omega, omega)
