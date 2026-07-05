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
- :func:`compute_segmentation` — a single-breakpoint search that refines a
  *curved monotone* mean into a smooth bend versus a step/threshold jump, and
  reports roughly where a step is located. Distinguishes the two by whether a
  two-*level* (flat-segment) model fits as well as a two-*line* model: a step's
  segments are flat, a smooth curve's are sloped.
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


#: Minimum rows before the single-breakpoint search runs. Mirrors the bin
#: lack-of-fit floor: each of the two segments needs enough points to fit a
#: stable line and mean.
_MIN_N_FOR_SEGMENTATION = 50

#: Smallest fraction of the rows a segment may hold, so a candidate split never
#: leaves a segment too short to estimate a slope. The search only considers
#: splits leaving at least this fraction on each side.
_SEGMENT_MIN_FRACTION = 0.1

_SEGMENT_NAMES = ("segment_gain", "breakpoint_x", "segment_stepness")


def _segmentation_no_value() -> dict[str, MetricResult]:
    return {name: MetricResult.no_value(name) for name in _SEGMENT_NAMES}


def _segment_residual_ss(
    m: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
    sxx: np.ndarray,
    sxy: np.ndarray,
    syy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Residual sum of squares after fitting a mean, and after fitting a line,
    to a segment described by its running sums — all vectorized over candidate
    splits. Uses the closed-form simple-regression identities so every split is
    O(1), giving an O(n) scan overall (no per-split ``polyfit``)."""
    ss_mean = syy - sy**2 / m
    ss_xx = sxx - sx**2 / m
    ss_xy = sxy - sx * sy / m
    # Slope only reduces the residual where the segment's x has spread; a
    # constant-x segment (ss_xx == 0) keeps the mean-only residual.
    safe_ss_xx = np.where(ss_xx > 0.0, ss_xx, 1.0)
    slope_reduction = np.where(ss_xx > 0.0, ss_xy**2 / safe_ss_xx, 0.0)
    ss_line = ss_mean - slope_reduction
    return np.maximum(ss_mean, 0.0), np.maximum(ss_line, 0.0)


def compute_segmentation(pair: CleanPair) -> dict[str, MetricResult]:
    """Single-breakpoint search that characterizes a curved monotone mean.

    Sorts by X and, over every candidate split, fits (a) two independent lines
    and (b) two independent levels (means), returning three
    :class:`MetricResult` entries:

    - ``segment_gain`` — R² of the best two-*line* fit minus the single-line R²
      (how much a single breakpoint improves on a straight line).
    - ``segment_stepness`` — the fraction of that improvement a two-*level*
      (flat-segment) model already captures: ``≈ 1`` when the segments are flat
      (a step/threshold jump), ``≤ 0`` when sloping the segments is essential (a
      smooth or piecewise bend). This is what separates a step from a smooth
      curve (see ``heuristics/classifier.py``).
    - ``breakpoint_x`` — the x-location of the best two-*level* split (where the
      jump sits). The classifier only reports it when the pair reads as a
      step/threshold, since for a smooth curve the split is an artifact of
      forcing a break onto a gradual bend.

    Returns all three as ``None`` (``MetricResult.no_value``) for constant
    inputs, ``n_used`` below :data:`_MIN_N_FOR_SEGMENTATION`, or a degenerate
    (zero-variance) response.
    """
    if pair.x_is_constant or pair.y_is_constant:
        return _segmentation_no_value()
    if pair.n_used < _MIN_N_FOR_SEGMENTATION:
        return _segmentation_no_value()

    x = pair.x.to_numpy()
    y = pair.y.to_numpy()
    n = x.shape[0]

    try:
        order = np.argsort(x, kind="mergesort")
        xs = x[order].astype(float)
        ys = y[order].astype(float)

        # Running sums (index i holds the sum over the first i rows).
        cx = np.concatenate([[0.0], np.cumsum(xs)])
        cy = np.concatenate([[0.0], np.cumsum(ys)])
        cxx = np.concatenate([[0.0], np.cumsum(xs * xs)])
        cxy = np.concatenate([[0.0], np.cumsum(xs * ys)])
        cyy = np.concatenate([[0.0], np.cumsum(ys * ys)])

        ss_tot = float(cyy[n] - cy[n] ** 2 / n)
        if ss_tot <= 0.0:
            return _segmentation_no_value()

        # Single-line residual over the whole range, from the same identities.
        _, ss_line_full = _segment_residual_ss(
            np.array([float(n)]),
            np.array([cx[n]]),
            np.array([cy[n]]),
            np.array([cxx[n]]),
            np.array([cxy[n]]),
            np.array([cyy[n]]),
        )
        r2_line = 1.0 - float(ss_line_full[0]) / ss_tot

        min_seg = max(5, int(_SEGMENT_MIN_FRACTION * n))
        ks = np.arange(min_seg, n - min_seg + 1)
        if ks.size == 0:
            return _segmentation_no_value()

        m_lo = ks.astype(float)
        m_hi = (n - ks).astype(float)
        mean_lo, line_lo = _segment_residual_ss(
            m_lo, cx[ks], cy[ks], cxx[ks], cxy[ks], cyy[ks]
        )
        mean_hi, line_hi = _segment_residual_ss(
            m_hi,
            cx[n] - cx[ks],
            cy[n] - cy[ks],
            cxx[n] - cxx[ks],
            cxy[n] - cxy[ks],
            cyy[n] - cyy[ks],
        )
        two_line = line_lo + line_hi
        two_mean = mean_lo + mean_hi

        r2_two_line = 1.0 - float(np.min(two_line)) / ss_tot
        best_mean_k = int(ks[int(np.argmin(two_mean))])
        r2_two_mean = 1.0 - float(np.min(two_mean)) / ss_tot

        segment_gain = r2_two_line - r2_line
        step_gain = r2_two_mean - r2_line
        stepness = step_gain / segment_gain if segment_gain > 1e-6 else 0.0
        breakpoint_x = 0.5 * (xs[best_mean_k - 1] + xs[best_mean_k])
    except (ValueError, RuntimeError, FloatingPointError) as e:
        raise MetricComputationError(
            f"Failed to compute segmentation: {type(e).__name__}: {e}"
        ) from e

    return {
        "segment_gain": MetricResult(
            name="segment_gain", value=float(segment_gain), available=True
        ),
        "breakpoint_x": MetricResult(
            name="breakpoint_x", value=float(breakpoint_x), available=True
        ),
        "segment_stepness": MetricResult(
            name="segment_stepness", value=float(stepness), available=True
        ),
    }
