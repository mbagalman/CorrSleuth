import math

from corrsleuth.result import HeuristicResult, MetricResult

from .explanations import generate_recommendations

# ---------------------------------------------------------------------------
# Heuristic cascade thresholds
#
# These cut points convert continuous correlation coefficients into discrete
# labels. They are *conventions*, not parameters fit to data. Two things shape
# the chosen values:
#
#   1. Effect-size convention. The magnitude bands follow the widely used
#      guidance for interpreting a correlation coefficient (Cohen, 1988):
#      |r| ~ 0.5 is a "large" effect, |r| ~ 0.3 "medium", |r| ~ 0.1 "small".
#      ``STRONG_MAGNITUDE_THRESHOLD`` sits on the large boundary;
#      ``WEAK_MAGNITUDE_THRESHOLD`` sits below the medium boundary.
#   2. Separation on the bundled scenarios. The gap/closeness thresholds were
#      chosen so the synthetic relationships in
#      ``corrsleuth/datasets/simulations.py`` land on their intended labels
#      with margin to spare, rather than by formal optimization.
#
# The cascade is deliberately conservative: a borderline pair falls through to
# ``mixed_or_ambiguous`` rather than overclaiming a pattern. These are
# module-level constants that advanced users can read or override on this module
# (e.g. ``import corrsleuth.heuristics.classifier as clf; clf.WEAK_MAGNITUDE_THRESHOLD = ...``);
# they are intentionally not re-exported from the package root. See
# docs/thresholds-and-rationale.md for the full rationale, the documented
# override recipe, and the trade-offs of moving any of them.
# ---------------------------------------------------------------------------

#: Magnitude (|pearson| or |spearman|) above which an association counts as
#: "strong" for labeling. Gates the leverage, monotonic_nonlinear, and
#: near_linear rules. Corresponds to Cohen's "large effect" boundary for a
#: correlation.
STRONG_MAGNITUDE_THRESHOLD = 0.50

#: Magnitude below which an association counts as "very weak / negligible".
#: Gates weak_or_no_relationship — Pearson, Spearman, and (when available)
#: distance correlation must all fall under it. Sits below Cohen's
#: "medium effect" boundary so genuinely moderate relationships are not labeled
#: weak.
WEAK_MAGNITUDE_THRESHOLD = 0.20

#: Minimum gap between a rank coefficient and Pearson before the difference is
#: read as a real nonlinearity/leverage signal rather than sampling noise.
#: Shared by the leverage rule (|pearson| - |spearman|) and the
#: monotonic_nonlinear rule (|spearman| - |pearson|). At the n >= 30 sample
#: sizes CorrSleuth is willing to label, the bootstrap spread of these
#: coefficients is comfortably below 0.20.
RANK_LINEAR_GAP_THRESHOLD = 0.20

#: Pearson-vs-Kendall counterpart of :data:`RANK_LINEAR_GAP_THRESHOLD`, used by
#: the leverage rule. Kendall's tau-b is numerically smaller than Spearman's
#: rho for the same monotonic signal (roughly tau ~ (2/pi)*arcsin(rho)), so a
#: wider gap is required to carry the same evidence.
PEARSON_KENDALL_GAP_THRESHOLD = 0.25

#: Ceiling on |pearson| and |spearman| for the nonmonotonic_dependence rule.
#: Both monotone measures must be weak before a high distance correlation is
#: read as nonmonotonic (e.g. U-shaped) dependence rather than a monotone trend
#: the rank metrics would already have captured.
NONMONOTONIC_MONOTONE_CEILING = 0.25

#: Distance-correlation floor for the nonmonotonic_dependence rule. Distance
#: correlation must clear this while the monotone measures stay under
#: :data:`NONMONOTONIC_MONOTONE_CEILING`. Set equal to
#: :data:`XI_DEPENDENCE_WARN_THRESHOLD` so the cascade and the deep-mode xi
#: warning share a single "this is real dependence" cut point.
NONMONOTONIC_DC_THRESHOLD = 0.35

#: Maximum |pearson - spearman| gap for the near_linear rule. Both coefficients
#: must already be strong (above :data:`STRONG_MAGNITUDE_THRESHOLD`); this
#: closeness test keeps monotone-but-curved relationships out of the
#: "approximately linear" bucket.
NEAR_LINEAR_GAP_THRESHOLD = 0.15

#: Distance-correlation ceiling for weak_or_no_relationship. When distance
#: correlation is available it must also fall under this, so a hidden
#: nonmonotonic signal is not mislabeled "no relationship".
WEAK_DC_THRESHOLD = 0.20

#: Bin-mean-model R² minus linear-fit R² (see metrics/shape.py) above which
#: real curvature exists that the Spearman-vs-Pearson gap misses — smooth
#: monotonic curves (exponential, logarithmic) and step/threshold functions
#: whose Pearson stays nearly as high as Spearman. An alternate trigger for
#: monotonic_nonlinear, alongside RANK_LINEAR_GAP_THRESHOLD. Genuinely linear
#: data measured ~-0.01 on the bundled test scenarios; real curvature measured
#: >=0.06. The margin here (0.05) is thinner than most cascade gaps, so this
#: threshold leans on the simulations.py regression coverage more than most.
BIN_LOF_R2_GAIN_THRESHOLD = 0.05

#: |corr(X^2, Y^2)| above which weak Pearson/Spearman (both under
#: NONMONOTONIC_MONOTONE_CEILING) is read as magnitude/radial dependence — the
#: signature of points scattered around a circle or similar X^2+Y^2-linked
#: structure — rather than no relationship. An alternate trigger for
#: nonmonotonic_dependence, alongside NONMONOTONIC_DC_THRESHOLD, for cases
#: where distance correlation itself is structurally capped (a true circular
#: relationship measures dCor ~0.19-0.20 even noiseless). Set equal to
#: NONMONOTONIC_DC_THRESHOLD for consistency; null pairs on the bundled test
#: scenarios measured <=0.11, real magnitude-linked dependence measured >=0.30.
SQ_CORR_THRESHOLD = 0.35

#: Breusch-Pagan p-value below which the residual variance is treated as
#: non-constant, for the ``variance_shape`` secondary axis (not the primary
#: cascade). Unlike the effect-size-band thresholds, this is a real hypothesis
#: test, so at large n it rejects for trivially small heteroscedasticity — which
#: is why it is paired with :data:`HETEROSCEDASTICITY_RATIO_FLOOR` below rather
#: than used alone. See metrics/variance.py.
HETEROSCEDASTICITY_PVALUE_THRESHOLD = 0.05

#: Goldfeld-Quandt residual-variance ratio (high-x group vs low-x group) that
#: must be cleared — above it, or below its reciprocal — before a
#: Breusch-Pagan rejection is reported as heteroscedastic. A ratio of 1.5 means
#: the spread on one side is at least half again the other's; this effect-size
#: floor keeps the large-n Breusch-Pagan test from flagging negligible variance
#: change (clean linear data measured a ratio ~0.8-1.2 across seeds).
HETEROSCEDASTICITY_RATIO_FLOOR = 1.5

#: Segment "stepness" (see metrics/shape.py) above which a curved monotone mean
#: is read as a step/threshold jump rather than a smooth bend — the fraction of
#: a single breakpoint's fit improvement that a flat-segment (two-level) model
#: already captures. A clean step measures ~1.0 (its segments are flat, so
#: slopes add nothing); smooth and piecewise bends measure <= 0 (sloping the
#: segments is essential). The 0.5 cut sits in the wide empty gap between them.
SEGMENT_STEPNESS_THRESHOLD = 0.5

#: Magnitude above which Pearson and Spearman having *opposite signs* is worth
#: a directionality warning. Below this both coefficients are near zero and a
#: sign disagreement is just noise, so the warning would be spurious.
CONFLICTING_SIGN_THRESHOLD = 0.3

#: Chatterjee's xi value — or, on the same scale via the Gaussian-equivalent-
#: correlation transform, mutual information — above which an otherwise
#: weak/ambiguous label gets a dependence warning. Matches
#: :data:`NONMONOTONIC_DC_THRESHOLD`, the distance-correlation threshold used
#: by the nonmonotonic_dependence rule in the cascade.
XI_DEPENDENCE_WARN_THRESHOLD = 0.35

#: Labels that understate the relationship when Chatterjee's xi or mutual
#: information is high. The cascade does not consult either when assigning a
#: label, so without a warning a standard- or deep-mode profile could report a
#: strong dependence signal and a "weak"/"ambiguous" label side by side.
_DEPENDENCE_WARNING_LABELS = frozenset(
    {"weak_or_no_relationship", "mixed_or_ambiguous"}
)

#: Heuristic labels that can only be assigned when standard-mode metrics
#: (distance correlation, mutual information) are available. Bootstrap stability
#: computed on lite metrics cannot fully test these labels.
STANDARD_ONLY_LABELS = frozenset({"nonmonotonic_dependence"})


def _finite_metric_value(
    metric: MetricResult | None, *, require_available: bool = False
) -> float | None:
    """Return a metric's value, or ``None`` when it is missing, unavailable,
    ``None``, or ``NaN``.

    NaN is treated as "no value" so it can never slip past the ``is None``
    guards in the cascade: every comparison with NaN is ``False``, which would
    otherwise misroute a degenerate pair to ``mixed_or_ambiguous`` instead of
    ``not_computable``.
    """
    if metric is None or metric.value is None:
        return None
    if require_available and not metric.available:
        return None
    value = metric.value
    if value != value:  # NaN is the only value not equal to itself
        return None
    return value


def apply_heuristics(
    metrics: dict[str, MetricResult], flags: list[str], n_used: int
) -> HeuristicResult:
    """Apply the heuristic priority cascade to assign a primary label.

    The cascade evaluates labels in priority order and returns the first that
    matches: ``not_computable``, ``low_power_or_uncertain``,
    ``possible_outlier_or_leverage``, ``nonmonotonic_dependence``,
    ``monotonic_nonlinear``, ``near_linear``, ``weak_or_no_relationship``,
    falling back to ``mixed_or_ambiguous``. Trim-sensitivity flags
    (``pearson_trim_sensitive``, ``outlier_sensitivity_unavailable``) gate the
    ``possible_outlier_or_leverage`` label so it is only assigned when there is
    independent evidence of leverage.

    ``metrics`` may include two shape diagnostics (see ``metrics/shape.py``) in
    addition to the primary correlation metrics: ``bin_lof_r2_gain`` (an
    alternate route into ``monotonic_nonlinear``, for smooth monotonic curves
    and step functions the Spearman-vs-Pearson gap misses) and ``sq_corr`` (an
    alternate route into ``nonmonotonic_dependence``, for magnitude/radial
    dependence distance correlation under-reads). Both are optional; their
    absence never blocks a label the other metrics would otherwise assign.
    """
    m_p = metrics.get("pearson")
    m_s = metrics.get("spearman")
    m_k = metrics.get("kendall_tau_b")
    m_dc = metrics.get("distance_correlation")
    m_bin_lof = metrics.get("bin_lof_r2_gain")
    m_sq_corr = metrics.get("sq_corr")

    p_val = _finite_metric_value(m_p)
    s_val = _finite_metric_value(m_s)
    k_val = _finite_metric_value(m_k)
    p = abs(p_val) if p_val is not None else None
    s = abs(s_val) if s_val is not None else None
    k = abs(k_val) if k_val is not None else None
    dc = _finite_metric_value(m_dc, require_available=True)
    bin_lof = _finite_metric_value(m_bin_lof)
    sq_corr = _finite_metric_value(m_sq_corr)

    # Pearson and Spearman pointing in opposite directions, both non-trivial, is
    # a directional conflict — not a clean linear or monotone signal, and almost
    # always leverage-driven (a few points flip the linear fit's sign relative to
    # the rank trend). It must be detected on the *signed* values: the
    # magnitude-based gaps below would read |+0.8| vs |-0.8| as a gap of 0 and
    # mislabel the pair near_linear. (assess_outlier_sensitivity makes the same
    # signed-comparison choice for exactly this reason.)
    pearson_spearman_conflict = (
        p_val is not None
        and s_val is not None
        and p_val * s_val < 0
        and abs(p_val) >= CONFLICTING_SIGN_THRESHOLD
        and abs(s_val) >= CONFLICTING_SIGN_THRESHOLD
    )

    label = "mixed_or_ambiguous"

    # 1. not_computable
    if "constant_input" in flags or p is None or s is None or k is None:
        label = "not_computable"
    # 2. low_power_or_uncertain  (low_n is set in validation iff n_used < 30)
    elif "low_n" in flags:
        label = "low_power_or_uncertain"
    # 3. possible_outlier_or_leverage
    elif (
        p > STRONG_MAGNITUDE_THRESHOLD
        and (
            p - s > RANK_LINEAR_GAP_THRESHOLD
            or p - k > PEARSON_KENDALL_GAP_THRESHOLD
            or pearson_spearman_conflict
        )
        and (
            "pearson_trim_sensitive" in flags
            or "outlier_sensitivity_unavailable" in flags
        )
    ):
        label = "possible_outlier_or_leverage"
    # 4. nonmonotonic_dependence
    # Two independent routes to the same conclusion: distance correlation
    # clearing its floor (any form of dependence), or |corr(X^2, Y^2)| clearing
    # its floor (magnitude/radial dependence — e.g. points on a circle — that
    # dCor itself can under-read; see BIN_LOF_R2_GAIN_THRESHOLD /
    # SQ_CORR_THRESHOLD module docs). Either is only trusted once Pearson and
    # Spearman are both already weak, so this never competes with rules 5/6.
    elif (
        p < NONMONOTONIC_MONOTONE_CEILING
        and s < NONMONOTONIC_MONOTONE_CEILING
        and (
            (dc is not None and dc > NONMONOTONIC_DC_THRESHOLD)
            or (sq_corr is not None and abs(sq_corr) > SQ_CORR_THRESHOLD)
        )
    ):
        label = "nonmonotonic_dependence"
    # 5. monotonic_nonlinear
    # Gated on Spearman alone (no Kendall fallback, unlike the leverage rule):
    # Spearman is the primary monotone measure here, and tau-b is numerically
    # smaller for the same signal, so adding an OR on tau would only loosen the
    # rule. A borderline-Spearman case deliberately falls through to
    # mixed_or_ambiguous rather than overclaiming nonlinearity. The bin
    # lack-of-fit gain is a second, independent route: it catches smooth
    # monotonic curves (exponential, logarithmic) and step/threshold functions
    # whose Pearson stays close enough to Spearman that the rank-linear gap
    # alone misses them.
    elif (
        s > STRONG_MAGNITUDE_THRESHOLD
        and (
            s - p > RANK_LINEAR_GAP_THRESHOLD
            or (bin_lof is not None and bin_lof > BIN_LOF_R2_GAIN_THRESHOLD)
        )
        and not pearson_spearman_conflict
    ):
        label = "monotonic_nonlinear"
    # 6. near_linear
    elif (
        p > STRONG_MAGNITUDE_THRESHOLD
        and s > STRONG_MAGNITUDE_THRESHOLD
        and abs(p - s) < NEAR_LINEAR_GAP_THRESHOLD
        and not pearson_spearman_conflict
    ):
        label = "near_linear"
    # 7. weak_or_no_relationship
    # Mirrors the distance-correlation ceiling with the same ceiling on
    # |corr(X^2, Y^2)|, so a moderate magnitude-linked signal (below
    # SQ_CORR_THRESHOLD, so rule 4 didn't fire, but above WEAK_DC_THRESHOLD)
    # falls through to mixed_or_ambiguous instead of being called "no
    # relationship" — the same conservative buffer the dc check already gets.
    elif (
        p < WEAK_MAGNITUDE_THRESHOLD
        and s < WEAK_MAGNITUDE_THRESHOLD
        and (dc is None or dc < WEAK_DC_THRESHOLD)
        and (sq_corr is None or abs(sq_corr) < WEAK_DC_THRESHOLD)
    ):
        label = "weak_or_no_relationship"

    return HeuristicResult(
        label=label,
        recommendations=generate_recommendations(label),
    )


def detect_metric_warnings(
    metrics: dict[str, MetricResult], label: str | None = None
) -> list[str]:
    """Return cautionary warnings derived from metric agreement patterns.

    These warnings supplement validation warnings; they do not override the
    primary label. Flags conflicting Pearson/Spearman directionality when both
    magnitudes exceed :data:`CONFLICTING_SIGN_THRESHOLD`; non-constant residual
    variance around an adequately linear mean (the ``variance_shape`` axis); and
    — when ``label`` is provided — high Chatterjee's xi or high mutual
    information alongside a weak or ambiguous label, since the cascade does not
    consult either when assigning labels. Mutual information (nats) is converted
    to a bounded, correlation-like scale via the Gaussian-equivalent-correlation
    transform ``sqrt(1 - exp(-2*MI))`` before comparing against
    :data:`XI_DEPENDENCE_WARN_THRESHOLD`, so both signals share one cut point.
    """
    warnings: list[str] = []

    pearson = _finite_metric_value(metrics.get("pearson"))
    spearman = _finite_metric_value(metrics.get("spearman"))

    if (
        pearson is not None
        and spearman is not None
        and abs(pearson) > CONFLICTING_SIGN_THRESHOLD
        and abs(spearman) > CONFLICTING_SIGN_THRESHOLD
        and pearson * spearman < 0
    ):
        warnings.append(
            "Pearson and Spearman have conflicting directions; inspect the scatter "
            "plot and check for nonlinearity, segments, or leverage points."
        )

    # Heteroscedasticity: the mean trend can be fine while the residual spread
    # changes with x, which quietly breaks homoscedastic inference. Reuse the
    # variance_shape axis logic so the warning and the axis never disagree.
    bp_pvalue = _finite_metric_value(metrics.get("bp_pvalue"))
    gq_ratio = _finite_metric_value(metrics.get("gq_ratio"))
    bin_lof = _finite_metric_value(metrics.get("bin_lof_r2_gain"))
    variance_shape = _variance_shape_axis(bp_pvalue, gq_ratio, bin_lof)
    if variance_shape in ("increasing_spread", "decreasing_spread") and (
        bp_pvalue is not None and gq_ratio is not None
    ):
        direction = "grows" if variance_shape == "increasing_spread" else "shrinks"
        warnings.append(
            f"The mean relationship is approximately linear, but the residual "
            f"spread {direction} across x (Breusch-Pagan p={bp_pvalue:.3g}; "
            f"variance {gq_ratio:.1f}x between the high- and low-x thirds). Pearson "
            f"describes the center trend, but homoscedastic inference (standard "
            f"errors, prediction intervals) may be unreliable."
        )

    if label in _DEPENDENCE_WARNING_LABELS:
        candidates = [
            (name, value, value)
            for name in ("chatterjee_xi", "chatterjee_xi_reverse")
            if (value := _finite_metric_value(metrics.get(name))) is not None
        ]
        mi_value = _finite_metric_value(metrics.get("mutual_information"))
        if mi_value is not None and mi_value >= 0:
            mi_compare = math.sqrt(1.0 - math.exp(-2.0 * mi_value))
            candidates.append(("mutual_information", mi_value, mi_compare))

        if candidates:
            name, display_value, compare_value = max(candidates, key=lambda c: c[2])
            if compare_value > XI_DEPENDENCE_WARN_THRESHOLD:
                unit = " nats" if name == "mutual_information" else ""
                warnings.append(
                    f"{name} ({display_value:.3f}{unit}) is high while linear and "
                    f"rank metrics are weak, which is evidence of nonmonotonic or "
                    f"functional dependence that the '{label}' label may "
                    f"understate. Inspect the scatter plot, or use "
                    f"mode='standard' to check distance correlation."
                )

    return warnings


# ---------------------------------------------------------------------------
# Secondary diagnostic axes
#
# The primary ``pattern`` label answers one question — the dominant shape of the
# relationship — and the cascade is deliberately conservative about it. These
# axes describe *orthogonal* properties a single label cannot carry without
# overloading it (a pair can be linear in mean AND heteroscedastic in variance
# AND driven by one row, all at once). Each axis is a coarse categorical
# *summary* derived from the numeric diagnostics already computed; the raw
# numbers stay the source of truth on ``MetricDiagnostics`` beside these labels.
# See docs/interpretation-guide.md ("Secondary diagnostic fields").
# ---------------------------------------------------------------------------


def _mean_shape_axis(
    p: float | None,
    s: float | None,
    bin_lof: float | None,
    segment_stepness: float | None,
) -> str | None:
    """Is E[Y|X] a straight line, a smooth curve, or a step? (``None`` when not
    assessable.)"""
    if p is None or s is None:
        return None
    # Curvature via either route the cascade uses for monotonic_nonlinear: a
    # positive bin lack-of-fit gain, or a strong Spearman meaningfully above
    # Pearson. Either means the conditional mean is not a straight line.
    curved = (bin_lof is not None and bin_lof > BIN_LOF_R2_GAIN_THRESHOLD) or (
        s > STRONG_MAGNITUDE_THRESHOLD and s - p > RANK_LINEAR_GAP_THRESHOLD
    )
    if curved:
        # Refine a *monotone* curve (strong Spearman) into a step/threshold jump
        # vs a smooth (or piecewise) bend, from the single-breakpoint stepness.
        # A non-monotone curve (weak Spearman, e.g. a U-shape) stays the generic
        # "curved" — smooth-vs-step does not apply, and dependence_type carries
        # its shape instead.
        if s >= STRONG_MAGNITUDE_THRESHOLD and segment_stepness is not None:
            return (
                "step_or_threshold"
                if segment_stepness > SEGMENT_STEPNESS_THRESHOLD
                else "smooth_curve"
            )
        return "curved"
    # Assert "linear" only when we actually checked for curvature (the
    # lack-of-fit test ran) and there is a real trend, or when both coefficients
    # are strong and close (the near_linear regime). Otherwise the shape of the
    # mean is unknown (weak trend, or n below the lack-of-fit floor).
    linear = (
        bin_lof is not None
        and bin_lof <= BIN_LOF_R2_GAIN_THRESHOLD
        and max(p, s) >= WEAK_MAGNITUDE_THRESHOLD
    ) or (
        p > STRONG_MAGNITUDE_THRESHOLD
        and s > STRONG_MAGNITUDE_THRESHOLD
        and abs(p - s) < NEAR_LINEAR_GAP_THRESHOLD
    )
    if linear:
        return "linear"
    return None


def _variance_shape_axis(
    bp_pvalue: float | None, gq_ratio: float | None, bin_lof: float | None
) -> str | None:
    """Does the spread of Y change with X? (``None`` when not assessable.)

    Only assessed when the conditional mean is adequately linear: a *curved*
    mean makes the linear-fit residuals heteroscedastic as an artifact of
    misspecification (a line fit to an exponential has small residuals where the
    curve is flat and large ones where it steepens), which is not a statement
    about the noise variance. So a curved mean (``bin_lof`` above the lack-of-fit
    threshold) yields ``None`` rather than a spurious spread verdict.
    """
    if bp_pvalue is None or gq_ratio is None:
        return None
    if bin_lof is None or bin_lof > BIN_LOF_R2_GAIN_THRESHOLD:
        return None
    if bp_pvalue >= HETEROSCEDASTICITY_PVALUE_THRESHOLD:
        return "constant"
    # Breusch-Pagan rejects; require a meaningful effect (guards against the
    # large-n test flagging negligible heteroscedasticity), and take the
    # direction from which side of the x-range carries the larger spread.
    if gq_ratio > HETEROSCEDASTICITY_RATIO_FLOOR:
        return "increasing_spread"
    if gq_ratio < 1.0 / HETEROSCEDASTICITY_RATIO_FLOOR:
        return "decreasing_spread"
    return "constant"


def _dependence_type_axis(
    p: float | None,
    s: float | None,
    dc: float | None,
    sq_corr: float | None,
    xi_fwd: float | None,
    xi_rev: float | None,
) -> str | None:
    """How do the variables depend on each other — monotonically, through
    magnitude, or as a closed loop? (``None`` when nothing is detected.)"""
    if p is None or s is None:
        return None
    monotone_weak = (
        p < NONMONOTONIC_MONOTONE_CEILING and s < NONMONOTONIC_MONOTONE_CEILING
    )
    sq_dependence = (
        monotone_weak and sq_corr is not None and abs(sq_corr) > SQ_CORR_THRESHOLD
    )
    dc_dependence = monotone_weak and dc is not None and dc > NONMONOTONIC_DC_THRESHOLD
    if sq_dependence or dc_dependence:
        # Deep-mode refinement: if neither variable is a function of the other
        # (both Chatterjee directions weak), this is a closed-loop / multivalued
        # relationship (a circle), not a plain magnitude link (a U-shape, where
        # Y is still a function of X). Needs xi, so only reachable in deep mode.
        if (
            xi_fwd is not None
            and xi_rev is not None
            and max(xi_fwd, xi_rev) < XI_DEPENDENCE_WARN_THRESHOLD
        ):
            return "closed_loop_or_multivalued"
        return "magnitude_linked" if sq_dependence else "nonmonotone"
    if max(p, s) >= WEAK_MAGNITUDE_THRESHOLD:
        return "monotone"
    return None


def _outlier_sensitivity_axis(
    outlier_status: str | None, n_influential: float | None
) -> str | None:
    """Is the summary driven by a few rows, and how many?

    Cook's-distance influence (``n_influential``) refines the answer first: it
    localizes the influence *and*, unlike the 1%-trim check, has no blind spot
    for a mid-range leverage cluster larger than the trimmed fraction, so it can
    fire even when the trim check called the pair stable. Falls back to the
    trim-sensitivity verdict (the leverage rule's own signal) when Cook's
    distance is unavailable or finds no influential row."""
    if n_influential is not None and n_influential >= 1:
        return "single_point_driven" if n_influential == 1 else "high_leverage_cluster"
    if outlier_status == "sensitive":
        return "high"
    if outlier_status == "stable":
        return "low"
    if outlier_status == "unavailable":
        return "unavailable"
    return None


def _functional_direction_axis(
    xi_fwd: float | None, xi_rev: float | None
) -> str | None:
    """Is Y a function of X, X of Y, both, or neither? Derived from Chatterjee's
    xi, so only populated in deep mode (``None`` otherwise)."""
    if xi_fwd is None or xi_rev is None:
        return None
    fwd = xi_fwd >= XI_DEPENDENCE_WARN_THRESHOLD
    rev = xi_rev >= XI_DEPENDENCE_WARN_THRESHOLD
    if fwd and rev:
        return "both_directions"
    if fwd:
        return "y_of_x"
    if rev:
        return "x_of_y"
    return "neither_direction"


def derive_diagnostic_axes(
    metrics: dict[str, MetricResult], label: str, outlier_status: str | None
) -> dict[str, str | None]:
    """Derive the five secondary diagnostic axes from the computed metrics.

    Returns a dict with keys ``mean_shape``, ``variance_shape``,
    ``dependence_type``, ``outlier_sensitivity``, and ``functional_direction``.
    Each value is a coarse categorical summary (or ``None`` when the axis is not
    assessable from the available metrics); the underlying numeric diagnostics
    remain on :class:`~corrsleuth.result.MetricDiagnostics` alongside them.

    These axes are orthogonal to the primary ``label`` — a pair can be, e.g.,
    ``near_linear`` in mean shape while still being magnitude-linked or
    outlier-driven — so they are derived from the numeric evidence rather than
    read off the label.
    """
    p_val = _finite_metric_value(metrics.get("pearson"))
    s_val = _finite_metric_value(metrics.get("spearman"))
    p = abs(p_val) if p_val is not None else None
    s = abs(s_val) if s_val is not None else None
    dc = _finite_metric_value(
        metrics.get("distance_correlation"), require_available=True
    )
    bin_lof = _finite_metric_value(metrics.get("bin_lof_r2_gain"))
    sq_corr = _finite_metric_value(metrics.get("sq_corr"))
    xi_fwd = _finite_metric_value(metrics.get("chatterjee_xi"))
    xi_rev = _finite_metric_value(metrics.get("chatterjee_xi_reverse"))
    bp_pvalue = _finite_metric_value(metrics.get("bp_pvalue"))
    gq_ratio = _finite_metric_value(metrics.get("gq_ratio"))
    segment_stepness = _finite_metric_value(metrics.get("segment_stepness"))
    n_influential = _finite_metric_value(metrics.get("n_influential_points"))

    return {
        "mean_shape": _mean_shape_axis(p, s, bin_lof, segment_stepness),
        "variance_shape": _variance_shape_axis(bp_pvalue, gq_ratio, bin_lof),
        "dependence_type": _dependence_type_axis(p, s, dc, sq_corr, xi_fwd, xi_rev),
        "outlier_sensitivity": _outlier_sensitivity_axis(outlier_status, n_influential),
        "functional_direction": _functional_direction_axis(xi_fwd, xi_rev),
    }
