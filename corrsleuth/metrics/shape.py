"""Shape diagnostics that supplement the linear/rank/distance metrics.

The diagnostics here are all pure numpy/scipy — no optional dependency, cheap
enough to run in every mode — and are consumed only by the heuristic cascade
and :class:`~corrsleuth.result.MetricDiagnostics`, never surfaced in the
public metrics table the way Pearson/distance correlation/mutual information
are. They exist to catch blind spots the existing rank/linear/distance
metrics leave open (see docs/shape-diagnostics-design.md):

- :func:`compute_bin_lof` — a lack-of-fit test comparing an
  equal-frequency-bin model of Y|X to a straight-line fit (``bin_lof_r2_gain``).
  A classical technique (Neter, Kutner, Nachtsheim & Wasserman, *Applied Linear
  Statistical Models*, lack-of-fit F-test using grouped X as a stand-in for
  replicates), used here to catch smooth monotonic curvature (exponential,
  logarithmic) and step/threshold functions that keep Pearson and Spearman
  close together despite real nonlinearity. From the same bins it also counts
  direction reversals in the sequence of bin means (``bin_reversal_count``),
  which separates *oscillating* dependence (a sinusoid: several reversals)
  from a single bend (a U-shape: exactly one) — the classifier only trusts the
  count when ``bin_lof_r2_gain`` also shows substantial bin structure, since
  pure noise produces many spurious reversals with near-zero gain. A
  leave-one-bin-out companion (``bin_lof_r2_gain_robust``) reports how much of
  the gain survives dropping any single bin, so a lone extreme-Y bin cannot
  fake curvature/oscillation on a structureless predictor.
- :func:`compute_squared_correlation` — the correlation between the squared
  mean-centered X and Y (``corr((X−x̄)², (Y−ȳ)²)``),
  used to catch dependence that shows up in magnitude but not in the raw
  signed values (e.g. points scattered around a circle, where X and Y are
  strongly dependent but Pearson/Spearman/distance correlation on the raw
  values are all near zero). :func:`compute_squared_correlation_robust` is its
  leave-the-top-out companion: the smallest ``|sq_corr|`` after dropping the
  few most extreme squared points, so a heavy-tailed variable's spurious
  magnitude signal collapses while a genuine one survives.
- :func:`compute_segmentation` — a single-breakpoint search that refines a
  *curved monotone* mean into a smooth bend versus a step/threshold jump, and
  reports roughly where a step is located. Distinguishes the two by whether a
  two-*level* (flat-segment) model fits as well as a two-*line* model: a step's
  segments are flat, a smooth curve's are sloped. It also reports
  ``segment_jump_ratio``, the fitted discontinuity at the best two-line split
  in units of residual noise, which catches a level shift embedded in a strong
  trend that the R²-scale gain washes out.
"""

from __future__ import annotations

import warnings

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

#: Hysteresis for the bin-mean reversal count, as a fraction of the bin-mean
#: range: a direction change only counts as a reversal once the sequence has
#: moved at least this far back from its last confirmed extreme. Per-difference
#: de-noising (ignore small bin-to-bin steps) was evaluated and rejected: at
#: high noise a single-bend shape's bin means wiggle enough that noise flips
#: pass a relative-to-max-step filter (~0.4% false "oscillating" reads on
#: U/V-shaped data), while range-scaled hysteresis produced zero false
#: positives over the same 2,080-run sweep with identical sinusoid detection.
#: 0.15 keeps a 1.5-cycle sinusoid's swings (each ~the full range) countable
#: while noise wiggle (a small fraction of the range whenever real structure
#: exists) never confirms a turn.
_BIN_REVERSAL_HYSTERESIS_FRACTION = 0.15

_BIN_LOF_NAMES = ("bin_lof_r2_gain", "bin_reversal_count", "bin_lof_r2_gain_robust")


def _bin_lof_no_value() -> dict[str, MetricResult]:
    return {name: MetricResult.no_value(name) for name in _BIN_LOF_NAMES}


def _tie_safe_bin_indices(xs: np.ndarray, n_bins: int) -> list[np.ndarray]:
    """Equal-frequency-ish bins over pre-sorted ``xs`` that **never split a run
    of equal X values** across a boundary.

    Starts from the plain equal-position split (:func:`numpy.array_split`) and
    moves only those interior boundaries that would fall *inside* a tied run to
    the nearer edge of that run. With no ties every boundary is already between
    distinct values, so the result is byte-identical to ``array_split`` — the
    binning of continuous data (and therefore the calibration of the bin-lof
    gain on it) is unchanged. With ties the boundaries snap to value changes, so
    the bins depend only on the *sorted values*, not on the arbitrary order of
    tied rows — which is what makes the diagnostics reproducible under row
    permutation (the previous position split assigned tied rows to bins by their
    input order). Snapping can merge boundaries, so the returned list may hold
    fewer than ``n_bins`` bins; the caller withholds the diagnostic when too few
    survive.
    """
    n = xs.shape[0]
    sizes = [len(b) for b in np.array_split(np.arange(n), n_bins)]
    raw_bounds = np.cumsum(sizes)[:-1]  # interior boundary positions
    edges = [0]
    for b in raw_bounds:
        b = int(b)
        if xs[b - 1] == xs[b]:
            # b splits a tied run; find the run's [lo, hi) extent and snap to the
            # nearer edge (a real value change), keeping the split as balanced as
            # the ties allow.
            lo = b
            while lo > 0 and xs[lo - 1] == xs[b]:
                lo -= 1
            hi = b
            while hi < n and xs[hi] == xs[b]:
                hi += 1
            b = lo if (b - lo) <= (hi - b) else hi
        if edges[-1] < b < n:
            edges.append(b)
    edges.append(n)
    edges = sorted(set(edges))
    return [np.arange(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def _adjusted_bin_gain(
    xs: np.ndarray, ys: np.ndarray, bin_indices: list[np.ndarray]
) -> tuple[float, np.ndarray] | None:
    """Df-adjusted bin-lack-of-fit gain for pre-sorted ``(xs, ys)`` split into
    the given ``bin_indices``, plus the per-bin means.

    ``adjusted R^2 = 1 - (SS_res / (n - p)) / (SS_tot / (n - 1))`` with ``p`` the
    number of bins; the gain is the k-bin mean model's adjusted R^2 minus the
    straight-line fit's, so the extra parameters a line lacks earn no free credit
    (see ``compute_bin_lof``). Returns ``None`` for a degenerate split (a bin
    under 2 points, fewer than two bins, or zero total variance). Shared by the
    primary gain and the leave-one-bin-out jackknife."""
    n = xs.shape[0]
    n_bins = len(bin_indices)
    if n_bins < 2 or any(len(idx) < 2 for idx in bin_indices):
        return None
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    if ss_tot == 0.0:
        return None
    bin_means = np.array([ys[idx].mean() for idx in bin_indices])
    y_bin_pred = np.empty_like(ys)
    for mean, idx in zip(bin_means, bin_indices, strict=True):
        y_bin_pred[idx] = mean
    ss_res_bins = float(np.sum((ys - y_bin_pred) ** 2))
    try:
        slope, intercept = np.polyfit(xs, ys, 1)
    except np.linalg.LinAlgError:
        # A degenerate (e.g. constant-x) subset has no line to fit — can happen
        # to a leave-one-bin-out subset when nearly all X ties sit in one bin.
        # Report it as unevaluable so the jackknife simply skips this drop.
        return None
    ss_res_linear = float(np.sum((ys - (slope * xs + intercept)) ** 2))
    adj_r2_bins = 1.0 - (ss_res_bins / (n - n_bins)) / (ss_tot / (n - 1))
    adj_r2_linear = 1.0 - (ss_res_linear / (n - 2)) / (ss_tot / (n - 1))
    return adj_r2_bins - adj_r2_linear, bin_means


def _turning_point_count(means: np.ndarray, hysteresis: float) -> int:
    """Count direction reversals in ``means``, confirming a turn only after the
    sequence moves at least ``hysteresis`` away from its last extreme (the
    classic zigzag / turning-point filter, robust to noise wiggle in a way
    per-step thresholds are not)."""
    reversals = 0
    direction = 0  # 0 unknown, +1 rising, -1 falling
    anchor = means[0]  # start point, then the last confirmed extreme
    extreme = means[0]
    for value in means[1:]:
        if direction == 0:
            if abs(value - anchor) >= hysteresis:
                direction = 1 if value > anchor else -1
                extreme = value
        elif direction == 1:
            if value > extreme:
                extreme = value
            elif extreme - value >= hysteresis:
                reversals += 1
                direction = -1
                extreme = value
        else:
            if value < extreme:
                extreme = value
            elif value - extreme >= hysteresis:
                reversals += 1
                direction = 1
                extreme = value
    return reversals


def compute_bin_lof(pair: CleanPair) -> dict[str, MetricResult]:
    """Equal-frequency-bin lack-of-fit diagnostics for the shape of E[Y|X].

    Sorts by X, splits into equal-frequency bins, and returns three
    :class:`MetricResult` entries computed from the same bins:

    - ``bin_lof_r2_gain`` — the **degrees-of-freedom-adjusted** bin-mean-model R²
      minus the linear-fit adjusted R², a lack-of-fit test for curvature. Each
      model's residual is penalized by its own parameter count (``k`` bin means
      vs. 2 line coefficients), so the extra bins a straight line lacks earn no
      free credit: under no curvature the gain sits at ~0, and only genuine
      curvature (a curve, a step, a bend the line cannot capture) pushes it
      clearly positive. A plain (unadjusted) R² difference instead carries a
      positive null bias of ~``(k-2)/(n-1)`` that mislabels ordinary noisy-linear
      data as curved — see the calibration sweep in
      ``validation/bin_lof_sweep.py``.
    - ``bin_reversal_count`` — how many times the sequence of bin means changes
      direction, counted with range-scaled hysteresis
      (:data:`_BIN_REVERSAL_HYSTERESIS_FRACTION`) so noise wiggle is not
      counted as a turn. A monotone trend or step measures 0, a single bend
      (U-shape) exactly 1, an oscillating/periodic relationship 2 or more.
      Only meaningful alongside a substantial ``bin_lof_r2_gain`` — pure noise
      produces many "reversals" with near-zero gain, so the classifier gates
      the count on the gain (see ``OSCILLATION_BIN_LOF_FLOOR`` in
      ``heuristics/classifier.py``).
    - ``bin_lof_r2_gain_robust`` — the leave-one-bin-out **minimum** of the gain:
      the smallest gain obtained by dropping any single bin's rows and refitting.
      It measures how much of the bin structure survives removing its most
      load-bearing bin. Equal to ``bin_lof_r2_gain`` when the structure is spread
      across bins (a genuine oscillation), but far lower when a lone extreme Y in
      one bin manufactures the gain (a heavy-tailed-Y artifact on an otherwise
      structureless predictor). The oscillation and no-trend-curvature gates read
      this value, so a single dominating bin cannot trip them.

    Binning is **tie-safe** (:func:`_tie_safe_bin_indices`): a bin boundary never
    splits a run of equal X, so the bins — and every diagnostic here — depend only
    on the sorted X *values*, not on the arbitrary order of tied rows. (The
    previous position split assigned tied rows to bins by their input order, so a
    mere row permutation of identical data could change the gain, the reversal
    count, and hence the label.) On strictly continuous X the tie-safe bins are
    identical to the plain equal-position split, so the calibration is unchanged.

    Returns all three as ``None`` (``MetricResult.no_value``) for constant inputs,
    ``n_used`` below :data:`_MIN_N_FOR_BIN_LOF`, or when X has **too few distinct
    values** to form :data:`_MIN_BINS` tie-safe bins — there the shape diagnostics
    are neither meaningful nor stably defined (see the low-unique-ratio validation
    warning, which flags such columns).
    """
    if pair.x_is_constant or pair.y_is_constant:
        return _bin_lof_no_value()

    if pair.n_used < _MIN_N_FOR_BIN_LOF:
        return _bin_lof_no_value()

    x = pair.x.to_numpy()
    y = pair.y.to_numpy()
    n = x.shape[0]

    try:
        order = np.argsort(x, kind="mergesort")
        xs = x[order]
        ys = y[order]

        n_bins = int(np.clip(n // _TARGET_POINTS_PER_BIN, _MIN_BINS, _MAX_BINS))
        # Tie-safe bins: boundaries never split a run of equal X, so the bins —
        # and every diagnostic below — depend only on the sorted values, not on
        # the input order of tied rows. When X has too few distinct values to
        # fill even the minimum bin count, the shape diagnostics are not
        # meaningful (and the tie split would otherwise make them order-
        # dependent), so withhold rather than report an unstable value; the
        # low-unique-ratio validation warning already flags such columns.
        bin_indices = _tie_safe_bin_indices(xs, n_bins)
        if len(bin_indices) < _MIN_BINS:
            return _bin_lof_no_value()

        # The df-adjusted gain penalizes each model's residual by its own
        # parameter count, so the k-bin mean model earns no free credit for the
        # degrees of freedom a straight line lacks (the unadjusted gain carries a
        # ~(k-2)/(n-1) positive null bias that mislabels noisy-linear data as
        # curved). See docs/shape-diagnostics-design.md and the calibration sweep
        # in validation/bin_lof_sweep.py.
        primary = _adjusted_bin_gain(xs, ys, bin_indices)
        if primary is None:
            return _bin_lof_no_value()
        bin_lof_gain, bin_means = primary

        # Leave-one-bin-out robustness. A single extreme Y pulls its bin's mean
        # far out, inflating BOTH the gain and the reversal count on a predictor
        # that has no real structure (most visible when Y is heavy-tailed -- in a
        # scan, Y is the target). Recompute the gain with each single bin's rows
        # removed and keep the minimum: how little of the bin structure survives
        # dropping its most load-bearing bin. A genuine oscillation is spread
        # across many bins and barely moves; a one-bin artifact collapses. The
        # oscillation/no-trend-curvature gates (OSCILLATION_BIN_LOF_FLOOR in
        # heuristics/classifier.py) test this robust gain, not the raw one, so
        # they are not fooled by a lone dominating bin. The raw gain still drives
        # the curvature route (BIN_LOF_R2_GAIN_THRESHOLD), which is gated on a
        # strong rank trend and where curvature legitimately concentrates in the
        # extreme bins -- so its calibration is untouched.
        robust_gain = bin_lof_gain
        n_actual_bins = len(bin_indices)
        for j in range(n_actual_bins):
            keep = np.concatenate(
                [bin_indices[i] for i in range(n_actual_bins) if i != j]
            )
            dropped = _adjusted_bin_gain(
                xs[keep], ys[keep], _tie_safe_bin_indices(xs[keep], n_actual_bins - 1)
            )
            # Skip unevaluable (degenerate) or NaN drops so a single bad subset
            # cannot poison the min (and a NaN primary gain stays NaN).
            if dropped is not None and dropped[0] == dropped[0]:
                robust_gain = min(robust_gain, dropped[0])

        means_range = float(bin_means.max() - bin_means.min())
        if means_range <= 0.0:
            reversals = 0
        else:
            reversals = _turning_point_count(
                bin_means, _BIN_REVERSAL_HYSTERESIS_FRACTION * means_range
            )
    except (ValueError, RuntimeError, FloatingPointError) as e:
        raise MetricComputationError(
            f"Failed to compute bin lack-of-fit diagnostics: {type(e).__name__}: {e}"
        ) from e

    return {
        "bin_lof_r2_gain": MetricResult(
            name="bin_lof_r2_gain", value=float(bin_lof_gain), available=True
        ),
        "bin_reversal_count": MetricResult(
            name="bin_reversal_count", value=float(reversals), available=True
        ),
        "bin_lof_r2_gain_robust": MetricResult(
            name="bin_lof_r2_gain_robust", value=float(robust_gain), available=True
        ),
    }


def _squared_correlation(x2: np.ndarray, y2: np.ndarray) -> float | None:
    """Signed Pearson correlation of two (already squared-centered) arrays, or
    ``None`` when either is (near-)constant. Shared by the raw and robust sq_corr.

    The robust variant recomputes this on subsets with the most extreme points
    removed, which can leave a (near-)constant column scipy's own guard flags with
    a warning and a NaN — treated here as "no value", not surfaced to the user."""
    if np.std(x2) == 0.0 or np.std(y2) == 0.0:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r, _ = stats.pearsonr(x2, y2)
    if not np.isfinite(r):
        return None
    return float(r)


def compute_squared_correlation(pair: CleanPair) -> MetricResult:
    """Pearson correlation between the squared **mean-centered** X and Y.

    Catches dependence carried in magnitude rather than sign or rank — e.g.
    points scattered around a circle ((X−x̄)² + (Y−ȳ)² ≈ const), where knowing X
    constrains |Y−ȳ| but not sign(Y−ȳ), so Pearson/Spearman/distance correlation
    on the raw values are all near zero while ``corr((X−x̄)², (Y−ȳ)²)`` is
    strongly (typically negatively) correlated.

    The centering is essential: squaring is not translation-invariant, so
    ``corr(X², Y²)`` on raw (uncentered) values collapses toward ``corr(X, Y)``
    for data far from the origin — a circle centered at (5, 5) would read ~0 and
    be mislabeled "no relationship". Centering first makes the diagnostic depend
    only on the *shape*, not on where it sits in the plane; for origin-centered
    data it is identical to the uncentered form, so the calibration is unchanged.

    Returns ``None`` (``MetricResult.no_value``) for constant inputs, or when the
    centered square of either variable is itself constant (e.g. X is symmetric
    two-valued data).
    """
    name = "sq_corr"

    if pair.x_is_constant or pair.y_is_constant:
        return MetricResult.no_value(name)

    x = pair.x.to_numpy()
    y = pair.y.to_numpy()
    x2 = (x - x.mean()) ** 2
    y2 = (y - y.mean()) ** 2

    try:
        r = _squared_correlation(x2, y2)
    except (ValueError, RuntimeError, FloatingPointError) as e:
        raise MetricComputationError(
            f"Failed to compute {name}: {type(e).__name__}: {e}"
        ) from e

    if r is None:
        return MetricResult.no_value(name)
    return MetricResult(name=name, value=r, available=True)


#: Points removed (the most extreme in either squared variable) when computing
#: the robust sq_corr. A heavy-tailed variable manufactures a spurious sq_corr
#: with a handful of extreme squared values; removing the top few collapses it,
#: while a genuine magnitude link is spread over many points and barely moves.
#: Three balances the two (see validation/sq_corr_sweep.py).
_SQ_CORR_ROBUST_DROP = 3


def compute_squared_correlation_robust(pair: CleanPair) -> MetricResult:
    """Leave-the-top-out robustness companion to :func:`compute_squared_correlation`.

    Reports the smallest ``|corr((X−x̄)², (Y−ȳ)²)|`` obtained across the full data
    and after removing up to :data:`_SQ_CORR_ROBUST_DROP` of the points most
    extreme in *either* squared variable — with the means **re-estimated on the
    retained points** at each drop, so it is a genuine deletion estimate of the
    statistic on the subsample (the dropped extremes no longer bias the center
    the survivors are squared around). A genuine magnitude link (a circle, a
    one-sided U-shape) is carried by many points, so it barely moves; a
    heavy-tailed variable's spurious sq_corr is carried by a few extreme values
    and collapses. The classifier requires this to clear ``SQ_CORR_ROBUST_FLOOR``
    before trusting sq_corr, so a handful of dominating points cannot manufacture
    a magnitude-linked label (see heuristics/classifier.py).

    ``None`` (``MetricResult.no_value``) whenever the raw sq_corr is (constant
    inputs / constant centered squares).
    """
    name = "sq_corr_robust"

    if pair.x_is_constant or pair.y_is_constant:
        return MetricResult.no_value(name)

    x = pair.x.to_numpy()
    y = pair.y.to_numpy()
    x2 = (x - x.mean()) ** 2
    y2 = (y - y.mean()) ** 2

    base = _squared_correlation(x2, y2)
    if base is None:
        return MetricResult.no_value(name)

    try:
        # Rank each point by how extreme it is in *either* squared variable, drop
        # the top-j (j = 1..K), and recompute on the *retained* points. The
        # minimum |corr| — including the full-data value — is how much of the
        # magnitude signal survives removing the most extreme few points.
        #
        # The squared deviations are re-centered on the retained points each
        # iteration (``(x[keep] - x[keep].mean())**2``), not reused from the
        # full-sample centering: dropping the extreme points changes the mean, so
        # a deletion estimate of ``corr((X−x̄)², (Y−ȳ)²)`` must recompute x̄/ȳ on
        # the retained sample — otherwise the dropped extremes still bias the
        # center the survivors are squared around. This also serves the gate's
        # purpose: for a heavy-tailed artifact the dominating points inflate both
        # the correlation and the mean, so re-centering after removing them
        # collapses the value further, while a genuine link (spread over many
        # points) barely moves.
        extremity = np.maximum(stats.rankdata(x2), stats.rankdata(y2))
        order = np.argsort(extremity)[::-1]
        n = x.shape[0]
        worst = abs(base)
        for j in range(1, _SQ_CORR_ROBUST_DROP + 1):
            if n - j < 2:
                break
            keep = np.ones(n, dtype=bool)
            keep[order[:j]] = False
            xk = x[keep]
            yk = y[keep]
            r = _squared_correlation((xk - xk.mean()) ** 2, (yk - yk.mean()) ** 2)
            if r is not None:
                worst = min(worst, abs(r))
    except (ValueError, RuntimeError, FloatingPointError) as e:
        raise MetricComputationError(
            f"Failed to compute {name}: {type(e).__name__}: {e}"
        ) from e

    return MetricResult(name=name, value=float(worst), available=True)


#: Minimum rows before the single-breakpoint search runs. Mirrors the bin
#: lack-of-fit floor: each of the two segments needs enough points to fit a
#: stable line and mean.
_MIN_N_FOR_SEGMENTATION = 50

#: Smallest fraction of the rows a segment may hold, so a candidate split never
#: leaves a segment too short to estimate a slope. The search only considers
#: splits leaving at least this fraction on each side.
_SEGMENT_MIN_FRACTION = 0.1

_SEGMENT_NAMES = (
    "segment_gain",
    "breakpoint_x",
    "segment_stepness",
    "segment_jump_ratio",
)

#: Cap on ``segment_jump_ratio`` when the two-line fit is (near-)noiseless, so a
#: perfect piecewise fit reports a large finite ratio instead of infinity
#: (which would not survive JSON serialization). Far above any gate.
_SEGMENT_JUMP_RATIO_CAP = 100.0

#: Minimum rows before ``segment_jump_ratio`` is reported (the other
#: segmentation outputs keep the lower :data:`_MIN_N_FOR_SEGMENTATION` floor).
#: Below ~150 rows a moderate smooth curve is not reliably separable from a
#: genuine discontinuity: the localization windows are too short to be both
#: stable and local, and on the stress sweep a moderate sigmoid's ratio tail
#: crossed the classifier floor at n = 100 (max 3.58 across 30 seeds) while
#: staying comfortably under it from n = 150 up (max 2.75 across 50 seeds x 6
#: smooth families at n in {150, 200}).
_MIN_N_FOR_JUMP_RATIO = 150

#: Second window size (points per side) for the localized jump refit (see
#: ``compute_segmentation``). The localization is evaluated at both the 10%
#: segment fraction and this fixed count, taking the minimum ratio: at small n
#: the 10% window (10 points at n=100) is noisy and can fail to collapse a
#: smooth curve's spurious global gap, while the fixed window is stabler; at
#: small n the fixed window covers too large a fraction to be local, while the
#: 10% window is. A genuine jump keeps its full gap at every window, so the
#: minimum only removes smooth-curve artifacts, never real discontinuities.
_LOCAL_JUMP_WINDOW_POINTS = 25


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


def _capped_jump_ratio(jump: float, sigma: float) -> float:
    """``jump / sigma`` capped at :data:`_SEGMENT_JUMP_RATIO_CAP`; a noiseless
    fit (sigma == 0) reports the cap for any nonzero gap (an unambiguous jump)
    and 0 otherwise."""
    if sigma > 0.0:
        return min(jump / sigma, _SEGMENT_JUMP_RATIO_CAP)
    return _SEGMENT_JUMP_RATIO_CAP if jump > 0.0 else 0.0


def _segment_line_at(
    m: float, sx: float, sy: float, sxx: float, sxy: float, x0: float
) -> float:
    """Value at ``x0`` of the least-squares line fitted to a segment described
    by its running sums (same closed-form identities as
    :func:`_segment_residual_ss`). A zero-spread (constant-x) segment falls back
    to its mean level — the same convention the residual scan uses."""
    mean_x = sx / m
    mean_y = sy / m
    ss_xx = sxx - sx**2 / m
    if ss_xx <= 0.0:
        return mean_y
    slope = (sxy - sx * sy / m) / ss_xx
    return mean_y + slope * (x0 - mean_x)


def compute_segmentation(pair: CleanPair) -> dict[str, MetricResult]:
    """Single-breakpoint search that characterizes a curved monotone mean.

    Sorts by X and, over every candidate split, fits (a) two independent lines
    and (b) two independent levels (means), returning four
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
      step/threshold or a discontinuous jump, since for a smooth curve the
      split is an artifact of forcing a break onto a gradual bend.
    - ``segment_jump_ratio`` — the fitted **discontinuity** at the best
      two-line split: the gap between the two fitted lines, evaluated at the
      boundary x, divided by the **noisier side's** residual sigma (a genuine
      level shift is the same process with a shifted level, so both regimes
      carry comparable noise; a heavy tail fitted as its own "segment" has far
      larger scatter than the bulk, and measuring against it keeps the tail's
      separation from the body — a leverage artifact — from reading as a
      jump). The two-line
      fit is *unconstrained* (each side gets its own intercept and slope), so
      it is exactly the "discontinuous two-line" model; a continuous
      relationship — a straight line, a smooth curve, or a piecewise-linear
      *kink* — is fitted by two lines that nearly meet at the boundary, so the
      ratio sits near 0 no matter how sharp the bend, while a genuine level
      shift measures the jump in units of noise (a 3-sigma jump reads ~3).
      This is what catches a discontinuity embedded in an otherwise strong
      trend, where the jump is huge relative to noise but tiny relative to the
      trend's variance: on the R-squared scale ``segment_gain`` reads ~0.05 for
      a 7-sigma jump riding a strong linear trend, because the trend soaks up
      the variance — the ratio scale does not wash out. Capped at
      :data:`_SEGMENT_JUMP_RATIO_CAP` for a noiseless piecewise fit.

    The breakpoint search considers only splits at a **value change** in X, so a
    break never falls inside a run of equal X — the segment fits (and hence
    ``segment_gain`` / ``segment_stepness`` / ``breakpoint_x``) depend on the
    sorted X values, not on the order of tied rows. On strictly continuous X this
    removes no candidate.

    Returns all four as ``None`` (``MetricResult.no_value``) for constant
    inputs, ``n_used`` below :data:`_MIN_N_FOR_SEGMENTATION`, a degenerate
    (zero-variance) response, or when no value-change split exists in the
    searchable range (X effectively single-valued there).
    ``segment_jump_ratio`` is additionally ``None`` below its own
    :data:`_MIN_N_FOR_JUMP_RATIO` floor, where a moderate smooth curve is not
    reliably separable from a genuine jump.
    """
    if pair.x_is_constant or pair.y_is_constant:
        return _segmentation_no_value()
    if pair.n_used < _MIN_N_FOR_SEGMENTATION:
        return _segmentation_no_value()

    x = pair.x.to_numpy()
    y = pair.y.to_numpy()
    n = x.shape[0]

    # Mean-center before forming the cumulative sums of squares/products. The
    # closed-form residual identities (``syy - sy**2 / m`` and friends) subtract
    # two large near-equal quantities when x/y sit far from zero, so a large
    # offset swamps the residual in floating point (catastrophic cancellation).
    # Centering is a pure shift: it leaves every residual — and therefore
    # ``segment_gain``/``segment_stepness`` — unchanged, while ``breakpoint_x``
    # is an x-location, so the mean is added back at the end.
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))

    try:
        order = np.argsort(x, kind="mergesort")
        xs = x[order].astype(float) - x_mean
        ys = y[order].astype(float) - y_mean

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
        # A breakpoint at position k splits between rows k-1 and k; restrict the
        # candidates to positions where X actually changes value, so a break
        # never falls inside a run of equal X. Otherwise the chosen split — and
        # the resulting segment_gain / stepness / breakpoint_x — would depend on
        # the arbitrary order of tied rows (the same reproducibility issue the
        # bin-lof binning has). With no ties this removes nothing.
        if ks.size:
            ks = ks[xs[ks] != xs[ks - 1]]
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
        breakpoint_x = 0.5 * (xs[best_mean_k - 1] + xs[best_mean_k]) + x_mean

        # Fitted discontinuity at the best two-LINE split: evaluate each side's
        # fitted line at the shared boundary x and take the gap, in units of
        # residual sigma. The sigma is the **noisier side's** (max of the two
        # per-segment sigmas), not the pooled one: a genuine level shift is the
        # same process with a shifted level, so both regimes carry comparable
        # noise and the max changes nothing — but a heavy tail fitted as its
        # own "segment" has residual scatter far larger than the bulk's, and a
        # pooled sigma would let the bulk's tightness manufacture a large
        # ratio for what is really the tail's separation from the body (a
        # leverage artifact, not a discontinuity). The jump and the boundary
        # use centered coordinates, which is immaterial: the gap is a
        # difference of two values at the same x.
        best_line_idx = int(np.argmin(two_line))
        best_line_k = int(ks[best_line_idx])
        boundary = 0.5 * (xs[best_line_k - 1] + xs[best_line_k])
        lo_at_boundary = _segment_line_at(
            float(best_line_k),
            float(cx[best_line_k]),
            float(cy[best_line_k]),
            float(cxx[best_line_k]),
            float(cxy[best_line_k]),
            boundary,
        )
        hi_at_boundary = _segment_line_at(
            float(n - best_line_k),
            float(cx[n] - cx[best_line_k]),
            float(cy[n] - cy[best_line_k]),
            float(cxx[n] - cxx[best_line_k]),
            float(cxy[n] - cxy[best_line_k]),
            boundary,
        )
        jump = abs(hi_at_boundary - lo_at_boundary)
        sigma_lo = float(
            np.sqrt(float(line_lo[best_line_idx]) / max(best_line_k - 2, 1))
        )
        sigma_hi = float(
            np.sqrt(float(line_hi[best_line_idx]) / max(n - best_line_k - 2, 1))
        )
        global_ratio = _capped_jump_ratio(jump, max(sigma_lo, sigma_hi))

        # Localization check: a smooth curve (a sigmoid, a sine) can fake a
        # boundary gap under the *global* two-line fit — its tails tilt the
        # chords, displacing them vertically at the center even though the
        # relationship is continuous. A genuine discontinuity survives
        # localization: refit each side's line on only the window of points
        # nearest the boundary and re-measure the gap there. Locally, a smooth
        # curve is approximately linear (the local two-line fit tracks it
        # through the boundary, gap -> 0) while a real jump keeps its full gap
        # at any window size. Evaluated at two window sizes — the 10% segment
        # fraction and a fixed point count (see _LOCAL_JUMP_WINDOW_POINTS for
        # why both) — and reported as the min of the global and all local
        # measurements, so every fit must agree the gap is real.
        def _local_ratio(w: int) -> float:
            lo_idx, hi_idx = best_line_k - w, best_line_k + w
            lo_val = _segment_line_at(
                float(w),
                float(cx[best_line_k] - cx[lo_idx]),
                float(cy[best_line_k] - cy[lo_idx]),
                float(cxx[best_line_k] - cxx[lo_idx]),
                float(cxy[best_line_k] - cxy[lo_idx]),
                boundary,
            )
            hi_val = _segment_line_at(
                float(w),
                float(cx[hi_idx] - cx[best_line_k]),
                float(cy[hi_idx] - cy[best_line_k]),
                float(cxx[hi_idx] - cxx[best_line_k]),
                float(cxy[hi_idx] - cxy[best_line_k]),
                boundary,
            )
            _, win_line_lo = _segment_residual_ss(
                np.array([float(w)]),
                np.array([cx[best_line_k] - cx[lo_idx]]),
                np.array([cy[best_line_k] - cy[lo_idx]]),
                np.array([cxx[best_line_k] - cxx[lo_idx]]),
                np.array([cxy[best_line_k] - cxy[lo_idx]]),
                np.array([cyy[best_line_k] - cyy[lo_idx]]),
            )
            _, win_line_hi = _segment_residual_ss(
                np.array([float(w)]),
                np.array([cx[hi_idx] - cx[best_line_k]]),
                np.array([cy[hi_idx] - cy[best_line_k]]),
                np.array([cxx[hi_idx] - cxx[best_line_k]]),
                np.array([cxy[hi_idx] - cxy[best_line_k]]),
                np.array([cyy[hi_idx] - cyy[best_line_k]]),
            )
            # Same noisier-side convention as the global ratio (see above).
            win_sigma_lo = float(np.sqrt(float(win_line_lo[0]) / max(w - 2, 1)))
            win_sigma_hi = float(np.sqrt(float(win_line_hi[0]) / max(w - 2, 1)))
            return _capped_jump_ratio(
                abs(hi_val - lo_val), max(win_sigma_lo, win_sigma_hi)
            )

        max_w = min(best_line_k, n - best_line_k)
        windows = {
            min(min_seg, max_w),
            min(max(min_seg, _LOCAL_JUMP_WINDOW_POINTS), max_w),
        }
        jump_ratio: float | None
        if n >= _MIN_N_FOR_JUMP_RATIO:
            jump_ratio = min([global_ratio] + [_local_ratio(w) for w in windows])
        else:
            # Below the dedicated floor the ratio cannot reliably separate a
            # moderate smooth curve from a real jump (see _MIN_N_FOR_JUMP_RATIO).
            jump_ratio = None
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
        "segment_jump_ratio": (
            MetricResult(
                name="segment_jump_ratio", value=float(jump_ratio), available=True
            )
            if jump_ratio is not None
            else MetricResult.no_value("segment_jump_ratio")
        ),
    }
