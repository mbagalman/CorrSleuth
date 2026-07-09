"""Two-group / mixture diagnostics — is the correlation carried by a split?

A pooled Pearson can be manufactured almost entirely by a *between-group mean
shift*: two well-separated clouds of rows ("low with low, high with high") with
little or no association inside either cloud. The pooled coefficient is then a
statement about the group separation, not about any continuous x-y trend — the
classic mixture / lurking-grouping-variable situation (and the aggregation trap
behind Simpson-style reversals). None of the existing metrics distinguish this:
Pearson/Spearman read the shift as a strong monotone signal, and the shape
diagnostics read the conditional mean as a step.

This module measures the ingredients of that story on the **association axis**
— the first principal component of the standardized pair, which for 2-D
z-scored data is simply ``t = (zx ± zy) / sqrt(2)`` with the sign of the
correlation (the direction along which a correlation-driving separation must
lie). All quantities are closed-form numpy on sorted arrays — no clustering
library, no new dependency, cheap enough to run in every mode:

- ``cluster_split_r2`` — the variance share explained by the best two-group
  split of ``t`` (exact 1-D 2-means via an O(n) prefix-sum scan over the
  sorted values, the same pattern as the segmentation search; the sort makes
  the overall cost O(n log n)). Unimodal data cannot exceed ~0.75 no
  matter how strong the correlation (a normal tops out at 2/pi ~ 0.64, a
  uniform at 0.75, asymptotically); two genuinely separated groups push it
  well above.
- ``cluster_valley_share`` — the fraction of rows within ``+/- 0.15 * std(t)``
  of the split boundary. Unimodal data has its *maximum* density there (~0.09
  to 0.12 for normal/uniform shapes); two separated clouds leave the band
  nearly empty. This is the "almost no points bridging the gap" evidence, and
  is robust where a single min-spacing gap statistic is noise.
- ``cluster_min_share`` — the smaller group's fraction of rows. Distinguishes
  a genuine subpopulation from a handful of leverage outliers (which the
  leverage diagnostics already own).
- ``pearson_within_cluster`` — the size-weighted mean of ``|Pearson|`` inside
  each group: how much association *survives* the split. Near zero when the
  pooled correlation is pure between-group shift; stays high for a step or
  saturating trend that keeps a within-segment slope.

The classifier combines these (see ``_is_two_group_shift`` in
``heuristics/classifier.py``) into the ``dependence_type = "two_group_shift"``
axis value and a warning. Thresholds were calibrated in
``validation/cluster_split_sweep.py``.

A statistical honesty note: from a single (x, y) pair, a *mixture of two
subpopulations* and a *threshold effect* (a flat step of a continuous
variable) are the same joint distribution — two separated groups with a mean
shift and no within-group trend. The diagnostics report the structure; the
warning text presents both readings rather than overclaiming "clusters".
"""

from __future__ import annotations

import numpy as np

from corrsleuth.exceptions import MetricComputationError
from corrsleuth.result import MetricResult
from corrsleuth.validation.input import CleanPair

#: Minimum rows before the split diagnostics are computed. The valley share
#: needs enough points for "an empty band" to be evidence rather than sampling
#: noise (at n = 100 a unimodal band holds ~9-12 points in expectation), and
#: the smaller group must still leave enough rows for a within-group Pearson.
_MIN_N_FOR_CLUSTER_SPLIT = 100

#: Minimum distinct values in *each* variable. Coarse ordinal/Likert/binary
#: data quantizes ``t`` onto a lattice whose empty inter-level spacing fakes a
#: perfect valley (every boundary between lattice points is "empty"), so the
#: split diagnostics are meaningless there — and such columns are already
#: flagged by the tie-rate / low-unique-ratio validation warnings.
_MIN_UNIQUE_FOR_CLUSTER_SPLIT = 10

#: Half-width of the boundary band for ``cluster_valley_share``, as a fraction
#: of ``std(t)``. Wide enough that a unimodal shape puts ~9-12% of its rows in
#: the band (its density peaks where 2-means splits it), narrow enough that
#: moderately separated groups (~3 within-stds apart) leave it near-empty.
_VALLEY_BAND_FRACTION = 0.15

#: Minimum rows per group before a within-group Pearson is reported. Below
#: this the estimate is too noisy to support (or honestly refute) the
#: "association survives inside the groups" question.
_MIN_GROUP_N_FOR_WITHIN_PEARSON = 10

_CLUSTER_SPLIT_NAMES = (
    "cluster_split_r2",
    "cluster_valley_share",
    "cluster_min_share",
    "pearson_within_cluster",
)


def _cluster_split_no_value() -> dict[str, MetricResult]:
    return {name: MetricResult.no_value(name) for name in _CLUSTER_SPLIT_NAMES}


def _best_two_means_split(t: np.ndarray) -> tuple[float, float, np.ndarray] | None:
    """Exact 1-D 2-means of ``t`` via a sorted prefix-sum scan (O(n log n)).

    Returns ``(r2, boundary, upper_mask)`` — the between-group variance share
    at the optimal split, the midpoint between the two adjacent values the
    split separates, and the boolean mask of the upper group — or ``None``
    when ``t`` has zero variance. Maximizing between-group variance over all
    sorted split points is exactly minimizing within-group SS, so this is the
    global 2-means optimum, not a local one.
    """
    n = t.shape[0]
    order = np.argsort(t, kind="mergesort")
    ts = t[order]
    mean = ts.mean()
    ss_tot = float(np.sum((ts - mean) ** 2))
    if ss_tot <= 0.0:
        return None
    csum = np.cumsum(ts)
    n1 = np.arange(1, n)
    m1 = csum[:-1] / n1
    m2 = (csum[-1] - csum[:-1]) / (n - n1)
    between = n1 * (m1 - mean) ** 2 + (n - n1) * (m2 - mean) ** 2
    i = int(np.argmax(between))
    r2 = float(between[i]) / ss_tot
    boundary = 0.5 * (float(ts[i]) + float(ts[i + 1]))
    upper = np.zeros(n, dtype=bool)
    upper[order[i + 1 :]] = True
    return r2, boundary, upper


def _within_group_pearson(
    x: np.ndarray, y: np.ndarray, upper: np.ndarray
) -> float | None:
    """Size-weighted mean of ``|Pearson|`` inside each group, or ``None`` when
    either group is too small or constant in either variable (e.g. the split
    isolated a handful of rows, or a variable is flat within a group)."""
    parts: list[tuple[int, float]] = []
    for mask in (upper, ~upper):
        n_group = int(mask.sum())
        if n_group < _MIN_GROUP_N_FOR_WITHIN_PEARSON:
            return None
        xg = x[mask]
        yg = y[mask]
        if np.std(xg) == 0.0 or np.std(yg) == 0.0:
            return None
        r = float(np.corrcoef(xg, yg)[0, 1])
        if not np.isfinite(r):
            return None
        parts.append((n_group, abs(r)))
    total = sum(n for n, _ in parts)
    return sum(n * r for n, r in parts) / total


def compute_cluster_split(pair: CleanPair) -> dict[str, MetricResult]:
    """Two-group split diagnostics along the pair's association axis.

    Returns four :class:`MetricResult` entries (see the module docstring):
    ``cluster_split_r2``, ``cluster_valley_share``, ``cluster_min_share``, and
    ``pearson_within_cluster``. All four are ``None``
    (``MetricResult.no_value``) for constant inputs, ``n_used`` below
    :data:`_MIN_N_FOR_CLUSTER_SPLIT`, or when either variable has fewer than
    :data:`_MIN_UNIQUE_FOR_CLUSTER_SPLIT` distinct values (coarse
    ordinal/binary data, where lattice spacing fakes an empty valley).
    ``pearson_within_cluster`` can be ``None`` on its own when a group is too
    small or constant — the classifier treats that as "cannot verify the
    within-group collapse" and does not fire.
    """
    if pair.x_is_constant or pair.y_is_constant:
        return _cluster_split_no_value()
    if pair.n_used < _MIN_N_FOR_CLUSTER_SPLIT:
        return _cluster_split_no_value()

    x = pair.x.to_numpy()
    y = pair.y.to_numpy()

    try:
        if (
            np.unique(x).shape[0] < _MIN_UNIQUE_FOR_CLUSTER_SPLIT
            or np.unique(y).shape[0] < _MIN_UNIQUE_FOR_CLUSTER_SPLIT
        ):
            return _cluster_split_no_value()

        zx = (x - x.mean()) / x.std()
        zy = (y - y.mean()) / y.std()
        # Association axis: the first principal component of the standardized
        # pair. For 2-D z-scores the eigenvectors are always the diagonals, so
        # the projection is (zx + zy)/sqrt(2) for a positive correlation and
        # (zx - zy)/sqrt(2) for a negative one; Var(t) = 1 + |corr| >= 1, so t
        # is never degenerate for non-constant inputs.
        sign = 1.0 if float(np.mean(zx * zy)) >= 0.0 else -1.0
        t = (zx + sign * zy) / np.sqrt(2.0)

        split = _best_two_means_split(t)
        if split is None:
            return _cluster_split_no_value()
        r2, boundary, upper = split

        band = _VALLEY_BAND_FRACTION * float(np.std(t))
        valley_share = float(np.mean(np.abs(t - boundary) <= band))
        upper_share = float(upper.mean())
        min_share = min(upper_share, 1.0 - upper_share)
        within = _within_group_pearson(x, y, upper)
    except (ValueError, RuntimeError, FloatingPointError) as e:
        raise MetricComputationError(
            f"Failed to compute cluster split diagnostics: {type(e).__name__}: {e}"
        ) from e

    results = {
        "cluster_split_r2": MetricResult(
            name="cluster_split_r2", value=r2, available=True
        ),
        "cluster_valley_share": MetricResult(
            name="cluster_valley_share", value=valley_share, available=True
        ),
        "cluster_min_share": MetricResult(
            name="cluster_min_share", value=min_share, available=True
        ),
        "pearson_within_cluster": (
            MetricResult(name="pearson_within_cluster", value=within, available=True)
            if within is not None
            else MetricResult.no_value("pearson_within_cluster")
        ),
    }
    return results
