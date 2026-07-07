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

#: Degrees-of-freedom-adjusted bin-lack-of-fit R² gain (see metrics/shape.py)
#: above which real curvature exists that the Spearman-vs-Pearson gap misses —
#: smooth monotonic curves (exponential, logarithmic) and step/threshold
#: functions whose Pearson stays nearly as high as Spearman. An alternate
#: trigger for monotonic_nonlinear, alongside RANK_LINEAR_GAP_THRESHOLD. Because
#: the gain is df-adjusted (metrics/shape.py), its null expectation is ~0
#: regardless of n, so this cut point no longer drifts with sample size. On the
#: calibration sweep (validation/bin_lof_sweep.py): the adjusted gain of
#: linear/no-trend controls — including a bivariate normal at moderate rho, the
#: case the old unadjusted statistic mislabeled — had a 95th percentile ~0.02
#: and never exceeded 0.05 for the moderate-rho family, while real curvature
#: measured a mean ~0.20; at 0.05 the false-positive rate is ~1% and curvature
#: detection ~95%. Still the thinnest cascade gap, so it leans on the
#: simulations.py regression coverage more than most.
BIN_LOF_R2_GAIN_THRESHOLD = 0.05

#: |corr((X−x̄)², (Y−ȳ)²)| — the correlation between the squared *mean-centered*
#: X and Y (see metrics/shape.py) — above which weak Pearson/Spearman (both under
#: NONMONOTONIC_MONOTONE_CEILING) is read as magnitude/radial dependence — the
#: signature of points scattered around a circle or similar radial structure
#: (roughly constant (X−x̄)²+(Y−ȳ)²) — rather than no relationship. An alternate trigger for
#: nonmonotonic_dependence, alongside NONMONOTONIC_DC_THRESHOLD, for cases
#: where distance correlation itself is structurally capped (a true circular
#: relationship measures dCor ~0.19-0.20 even noiseless). Set equal to
#: NONMONOTONIC_DC_THRESHOLD for consistency; null pairs on the bundled test
#: scenarios measured <=0.11, real magnitude-linked dependence measured >=0.30.
SQ_CORR_THRESHOLD = 0.35

#: Floor the *robust* sq_corr (``sq_corr_robust`` — the |corr| of the squared
#: values after dropping the few most extreme points, see metrics/shape.py) must
#: clear before a raw sq_corr above :data:`SQ_CORR_THRESHOLD` is trusted. A
#: heavy-tailed variable can manufacture a raw sq_corr over 0.35 with a handful of
#: extreme squared values; removing them collapses it, so the robust value falls
#: below this floor. Genuine magnitude links (circle, U-shape) are spread over
#: many points and keep a robust value above it. Deliberately *lower* than
#: SQ_CORR_THRESHOLD (an asymmetric gate). On the calibration sweep
#: (validation/sq_corr_sweep.py), 0.20 removes ~7/8 of the heavy-tailed-Y
#: artifacts while keeping 719/719 circle and 975/978 u_shape detections — the
#: irreducible-residual point: a *higher* floor (e.g. 0.25) removes no additional
#: artifact but costs ~10x more genuine u_shape detections. Set equal to
#: :data:`WEAK_DC_THRESHOLD` so a collapsed sq_corr lands the pair cleanly in
#: weak_or_no_relationship with no ambiguous gap. The one residual artifact —
#: whose *bulk* correlation survives the drop — is indistinguishable from a weak
#: real link, so it cannot be removed without cutting genuine detections.
SQ_CORR_ROBUST_FLOOR = 0.20

#: Minimum bin-mean direction reversals (see metrics/shape.py's
#: ``bin_reversal_count``) before dependence is read as *oscillating* — a
#: sinusoid or any relationship with more than one bend. A single bend
#: (U-shape) measures exactly 1, so 2 is the smallest count that separates
#: oscillation from it; a 1.5-cycle sinusoid measures >= 2 (usually 3) across
#: the validation sweep. Always applied jointly with
#: :data:`OSCILLATION_BIN_LOF_FLOOR` below — the count alone is meaningless.
OSCILLATION_MIN_REVERSALS = 2

#: ``bin_lof_r2_gain`` floor for the oscillation gate. Deliberately higher than
#: :data:`BIN_LOF_R2_GAIN_THRESHOLD` (0.05): the reversal count is only
#: trustworthy once there is *substantial* bin structure. This floor is what
#: keeps pure noise out — noise produces many reversals but a small bin-fit
#: gain. On the calibration sweep (validation/bin_lof_sweep.py), the
#: df-adjusted gain of pure noise never exceeded ~0.05 while real sinusoids
#: measured a minimum ~0.16 (mean ~0.6), so 0.15 sits ~3x above the noise
#: ceiling and below every sinusoid: across the 13-shape × 4-size × 4-noise ×
#: 10-seed grid it gives zero noise false positives and 100% sinusoid detection
#: at every sample size, including n=100. (The value dropped from 0.30 when the
#: bin gain became df-adjusted — the adjustment shifts a sinusoid's small-n gain
#: down, so the old 0.30 floor, calibrated on the unadjusted statistic, would
#: now miss ~13% of sinusoids.)
OSCILLATION_BIN_LOF_FLOOR = 0.15

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

#: Edge-vs-middle ("bowtie") residual-variance ratio (see metrics/variance.py)
#: that must be cleared — above it, or below its reciprocal — before a
#: symmetric variance pattern (spread high at both extremes of x and calm in
#: the middle, or the reverse) is reported. Complements
#: HETEROSCEDASTICITY_RATIO_FLOOR: a bowtie's low-x and high-x groups have
#: *similar* variance (so gq_ratio reads ~1, missing it by construction), but
#: its edges-combined-vs-middle ratio is large. Set conservatively above the
#: 1.5x funnel floor since clean linear and one-directional-funnel data both
#: measured ~1.0-1.2 on the bundled/blind test data, while a real bowtie
#: measured ~11x — a wide margin with room to be conservative.
BOWTIE_RATIO_FLOOR = 2.5

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

#: Heuristic labels that have *at least one* standard-only route (distance
#: correlation, mode="standard"). They are no longer standard-*only* to assign:
#: since the shape diagnostics are lite-computable, ``nonmonotonic_dependence``
#: is also reachable via ``sq_corr`` or the bin-reversal oscillation route (see
#: rule 4). So a lite-mode bootstrap can fully test a shape-diagnostic-driven
#: assignment but not a distance-correlation-driven one. The consumer
#: (``compute_bootstrap`` / ``explain()``) therefore treats the lite-metric
#: caveat as *conservative*: it fires only when dcor was absent from the replicate
#: cascade (``BootstrapStability.dcor_in_cascade``), which may over-warn on a
#: label that was in fact lite-testable. Distinguishing the exact route that
#: fired is out of scope (see docs/shape-diagnostics-design.md).
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

    ``metrics`` may include three shape diagnostics (see ``metrics/shape.py``)
    in addition to the primary correlation metrics: ``bin_lof_r2_gain`` (an
    alternate route into ``monotonic_nonlinear``, for smooth monotonic curves
    and step functions the Spearman-vs-Pearson gap misses), ``sq_corr`` (an
    alternate route into ``nonmonotonic_dependence``, for magnitude/radial
    dependence distance correlation under-reads), and ``bin_reversal_count``
    (a third route into ``nonmonotonic_dependence``, jointly with
    ``bin_lof_r2_gain``, for oscillating/periodic dependence neither of the
    other two reliably catches). All are optional; their absence never blocks
    a label the other metrics would otherwise assign.
    """
    m_p = metrics.get("pearson")
    m_s = metrics.get("spearman")
    m_k = metrics.get("kendall_tau_b")
    m_dc = metrics.get("distance_correlation")
    m_bin_lof = metrics.get("bin_lof_r2_gain")
    m_sq_corr = metrics.get("sq_corr")
    m_reversals = metrics.get("bin_reversal_count")

    p_val = _finite_metric_value(m_p)
    s_val = _finite_metric_value(m_s)
    k_val = _finite_metric_value(m_k)
    p = abs(p_val) if p_val is not None else None
    s = abs(s_val) if s_val is not None else None
    k = abs(k_val) if k_val is not None else None
    dc = _finite_metric_value(m_dc, require_available=True)
    bin_lof = _finite_metric_value(m_bin_lof)
    # Leave-one-bin-out gain (see metrics/shape.py): the oscillation route reads
    # this, not the raw gain, so a lone extreme-Y bin cannot manufacture a
    # spurious oscillation on a structureless predictor.
    bin_lof_robust = _finite_metric_value(metrics.get("bin_lof_r2_gain_robust"))
    sq_corr = _finite_metric_value(m_sq_corr)
    # Robust sq_corr (see metrics/shape.py): the sq_corr routes read this so a
    # heavy-tailed variable's few extreme squared values cannot fake a
    # magnitude-linked signal on a structureless predictor.
    sq_corr_robust = _finite_metric_value(metrics.get("sq_corr_robust"))
    reversals = _finite_metric_value(m_reversals)

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
    # Three independent routes to the same conclusion: distance correlation
    # clearing its floor (any form of dependence); |corr((X−x̄)², (Y−ȳ)²)| clearing
    # its floor (magnitude/radial dependence — e.g. points on a circle — that
    # dCor itself can under-read); or the bin-mean reversal count jointly with
    # a high bin lack-of-fit gain (oscillating/periodic dependence — e.g. a
    # sinusoid — which dCor reads only marginally above its floor and sq_corr
    # misses entirely; the joint gate is essential because pure noise produces
    # many reversals with near-zero gain, and the *robust* (leave-one-bin-out)
    # gain must also clear the floor so a lone extreme-Y bin cannot fake an
    # oscillation on a structureless predictor). The last two are lite-computable, so
    # this label is reachable in every mode for those shapes. Any route is only
    # trusted once Pearson and Spearman are both already weak, so this never
    # competes with rules 5/6. See BIN_LOF_R2_GAIN_THRESHOLD / SQ_CORR_THRESHOLD
    # / OSCILLATION_* module docs.
    elif (
        p < NONMONOTONIC_MONOTONE_CEILING
        and s < NONMONOTONIC_MONOTONE_CEILING
        and (
            (dc is not None and dc > NONMONOTONIC_DC_THRESHOLD)
            or (
                sq_corr is not None
                and abs(sq_corr) > SQ_CORR_THRESHOLD
                and sq_corr_robust is not None
                and sq_corr_robust > SQ_CORR_ROBUST_FLOOR
            )
            or (
                reversals is not None
                and reversals >= OSCILLATION_MIN_REVERSALS
                and bin_lof is not None
                and bin_lof > OSCILLATION_BIN_LOF_FLOOR
                and bin_lof_robust is not None
                and bin_lof_robust > OSCILLATION_BIN_LOF_FLOOR
            )
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
    # |corr((X−x̄)², (Y−ȳ)²)|, so a moderate magnitude-linked signal (below
    # SQ_CORR_THRESHOLD, so rule 4 didn't fire, but above WEAK_DC_THRESHOLD)
    # falls through to mixed_or_ambiguous instead of being called "no
    # relationship" — the same conservative buffer the dc check already gets.
    # Uses the *robust* sq_corr: a heavy-tailed variable's spurious sq_corr
    # collapses once its few extreme squared values are removed, so the pair
    # correctly reads as weak rather than being held out of "no relationship" by
    # an artifact (a genuine magnitude link keeps a robust value above the floor).
    elif (
        p < WEAK_MAGNITUDE_THRESHOLD
        and s < WEAK_MAGNITUDE_THRESHOLD
        and (dc is None or dc < WEAK_DC_THRESHOLD)
        and (sq_corr_robust is None or abs(sq_corr_robust) < WEAK_DC_THRESHOLD)
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

    A variance-shape signal is checked against ``n_influential_points`` before
    being reported as independent evidence: when Cook's distance already flags
    an influential row (``n_influential_points >= 1``) and the ``*_excl_influential``
    metrics (``api.py`` recomputes heteroscedasticity excluding that row —
    Ticket 1.5) show the signal vanishes on the remainder, the warning is
    reworded to attribute it to that same row instead of reporting it as a
    second, independent-sounding problem.
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
    bowtie_ratio = _finite_metric_value(metrics.get("bowtie_ratio"))
    variance_shape = _variance_shape_axis(bp_pvalue, gq_ratio, bin_lof, bowtie_ratio)

    if variance_shape in (
        "increasing_spread",
        "decreasing_spread",
        "edge_high_spread",
        "center_high_spread",
    ):
        # Is this signal just an echo of the same row(s) outlier_sensitivity
        # already flags (e.g. one outlier both inflating Goldfeld-Quandt's
        # high-x group and manufacturing the whole "relationship")? Only
        # concluded when the recomputed *_excl_influential values are present
        # (api.py only computes them once n_influential_points >= 1) and the
        # signal actually disappears on the remainder -- an inconclusive or
        # missing recomputation defaults to treating the signal as
        # independent, the safer default.
        n_influential = _finite_metric_value(metrics.get("n_influential_points"))
        is_leverage_artifact = False
        if n_influential is not None and n_influential >= 1:
            bp_excl = _finite_metric_value(metrics.get("bp_pvalue_excl_influential"))
            gq_excl = _finite_metric_value(metrics.get("gq_ratio_excl_influential"))
            bowtie_excl = _finite_metric_value(
                metrics.get("bowtie_ratio_excl_influential")
            )
            # For a bowtie original the excl recompute must also carry a
            # bowtie_ratio: without it, _variance_shape_axis skips the bowtie
            # check and returns "constant" for a test that never re-ran, wrongly
            # attributing the signal to leverage. Funnel shapes only need bp/gq.
            bowtie_shape = variance_shape in ("edge_high_spread", "center_high_spread")
            if (
                bp_excl is not None
                and gq_excl is not None
                and (not bowtie_shape or bowtie_excl is not None)
            ):
                variance_shape_excl = _variance_shape_axis(
                    bp_excl, gq_excl, bin_lof, bowtie_excl
                )
                is_leverage_artifact = variance_shape_excl == "constant"

        if variance_shape in ("increasing_spread", "decreasing_spread") and (
            bp_pvalue is not None and gq_ratio is not None
        ):
            direction = "grows" if variance_shape == "increasing_spread" else "shrinks"
            if is_leverage_artifact:
                grow_or_shrink = (
                    "grow" if variance_shape == "increasing_spread" else "shrink"
                )
                warnings.append(
                    f"Residual spread appears to {grow_or_shrink} across x "
                    f"(Breusch-Pagan p={bp_pvalue:.3g}; variance {gq_ratio:.1f}x "
                    f"between the upper and lower 40% of x), but this signal "
                    f"disappears once the influential row(s) flagged by "
                    f"outlier_sensitivity are excluded -- it is very likely the "
                    f"same leverage issue, not independent heteroscedasticity."
                )
            else:
                warnings.append(
                    f"The mean relationship is approximately linear, but the residual "
                    f"spread {direction} across x (Breusch-Pagan p={bp_pvalue:.3g}; "
                    f"variance {gq_ratio:.1f}x between the upper and lower 40% of x). Pearson "
                    f"describes the center trend, but homoscedastic inference (standard "
                    f"errors, prediction intervals) may be unreliable."
                )
        elif variance_shape in ("edge_high_spread", "center_high_spread") and (
            bowtie_ratio is not None
        ):
            if variance_shape == "edge_high_spread":
                location = "highest at both extremes of x and lowest near the center"
                ratio = bowtie_ratio
            else:
                location = "highest near the center and lowest at both extremes of x"
                ratio = 1.0 / bowtie_ratio
            if is_leverage_artifact:
                warnings.append(
                    f"Residual spread appears {location} (edge/middle variance "
                    f"ratio {ratio:.1f}x), but this signal disappears once the "
                    f"influential row(s) flagged by outlier_sensitivity are "
                    f"excluded -- it is very likely the same leverage issue, not "
                    f"an independent variance pattern."
                )
            else:
                warnings.append(
                    f"The mean relationship is approximately linear, but residual spread "
                    f"is {location} (edge/middle variance ratio {ratio:.1f}x). This "
                    f"symmetric pattern is invisible to a simple increasing/decreasing "
                    f"spread check; consider a variance model that allows for this shape."
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
    bin_lof_robust: float | None = None,
) -> str | None:
    """Is E[Y|X] a straight line, a smooth curve, or a step? (``None`` when not
    assessable.)"""
    if p is None or s is None:
        return None
    # Curvature via either route the cascade uses for monotonic_nonlinear: a
    # positive bin lack-of-fit gain, or a strong Spearman meaningfully above
    # Pearson. Either means the conditional mean is not a straight line.
    #
    # The bin-lack-of-fit route takes a higher bar in the *weak-trend* regime
    # (both |p| and |s| below the weak floor): with no linear or rank trend to
    # corroborate it, a gain just over BIN_LOF_R2_GAIN_THRESHOLD is as plausibly
    # finite-sample noise as real curvature, so require the substantial bin
    # structure the oscillation gate demands (a genuine U-shape or bend clears it
    # comfortably; noise does not) — and, like that gate, require it of the
    # *robust* (leave-one-bin-out) gain too, so a single extreme-Y bin does not
    # read as curvature on a structureless predictor. With a real trend present
    # the ordinary threshold applies (curvature legitimately concentrates in the
    # extreme bins there). This keeps ``mean_shape="curved"`` off pure-noise pairs
    # that carry a ``weak_or_no_relationship`` label.
    has_trend = max(p, s) >= WEAK_MAGNITUDE_THRESHOLD
    if has_trend:
        bin_curved = bin_lof is not None and bin_lof > BIN_LOF_R2_GAIN_THRESHOLD
    else:
        bin_curved = (
            bin_lof is not None
            and bin_lof > OSCILLATION_BIN_LOF_FLOOR
            and bin_lof_robust is not None
            and bin_lof_robust > OSCILLATION_BIN_LOF_FLOOR
        )
    curved = bin_curved or (
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
    bp_pvalue: float | None,
    gq_ratio: float | None,
    bin_lof: float | None,
    bowtie_ratio: float | None = None,
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
    if bp_pvalue < HETEROSCEDASTICITY_PVALUE_THRESHOLD:
        # Breusch-Pagan rejects; require a meaningful effect (guards against the
        # large-n test flagging negligible heteroscedasticity), and take the
        # direction from which side of the x-range carries the larger spread.
        if gq_ratio > HETEROSCEDASTICITY_RATIO_FLOOR:
            return "increasing_spread"
        if gq_ratio < 1.0 / HETEROSCEDASTICITY_RATIO_FLOOR:
            return "decreasing_spread"
    # Either Breusch-Pagan did not reject, or it did but gq_ratio was
    # inconclusive. Neither rules out a symmetric ("bowtie") pattern: its
    # low-x and high-x groups have similar variance (so gq_ratio reads ~1),
    # and the squared-residuals-vs-x relationship it drives is not linear
    # (it's U- or hill-shaped in x), so Breusch-Pagan's linear auxiliary
    # regression can also miss it. Checked independently via the
    # edges-combined-vs-middle ratio.
    if bowtie_ratio is not None:
        if bowtie_ratio > BOWTIE_RATIO_FLOOR:
            return "edge_high_spread"
        if bowtie_ratio < 1.0 / BOWTIE_RATIO_FLOOR:
            return "center_high_spread"
    # A small/inconclusive effect size (Goldfeld-Quandt ~1, bowtie ~1) is itself
    # the evidence that a Breusch-Pagan rejection is a large-n false positive, so
    # "constant" is the intended verdict here — see the paired-floor rationale in
    # the module constants and the ``variance_shape`` axis test.
    return "constant"


def _dependence_type_axis(
    p: float | None,
    s: float | None,
    dc: float | None,
    sq_corr: float | None,
    xi_fwd: float | None,
    xi_rev: float | None,
    bin_lof: float | None = None,
    reversals: float | None = None,
    bin_lof_robust: float | None = None,
    sq_corr_robust: float | None = None,
) -> str | None:
    """How do the variables depend on each other — monotonically, through
    magnitude, as an oscillation, or as a closed loop? (``None`` when nothing
    is detected.)"""
    if p is None or s is None:
        return None
    monotone_weak = (
        p < NONMONOTONIC_MONOTONE_CEILING and s < NONMONOTONIC_MONOTONE_CEILING
    )
    # Oscillation is checked first: it is the most specific description (a
    # sinusoid also clears the dc floor in standard mode, but "nonmonotone"
    # would undersell its cyclical structure — an analyst should look for
    # periodicity, not a single inflection point). The joint gate mirrors the
    # cascade's rule-4 oscillation route exactly; a shape that qualifies here
    # cannot be a closed loop (a multivalued loop's bin means average the
    # branches, flattening the gain below the floor — a circle measures ~0.05).
    if (
        monotone_weak
        and reversals is not None
        and reversals >= OSCILLATION_MIN_REVERSALS
        and bin_lof is not None
        and bin_lof > OSCILLATION_BIN_LOF_FLOOR
        and bin_lof_robust is not None
        and bin_lof_robust > OSCILLATION_BIN_LOF_FLOOR
    ):
        return "oscillating"
    sq_dependence = (
        monotone_weak
        and sq_corr is not None
        and abs(sq_corr) > SQ_CORR_THRESHOLD
        and sq_corr_robust is not None
        and sq_corr_robust > SQ_CORR_ROBUST_FLOOR
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
    # "monotone" describes a rank trend, so gate on Spearman, not max(|p|, |s|):
    # a leverage pair (strong Pearson, near-zero Spearman) has no monotone trend
    # and must not be called "monotone" on the strength of the linear artifact —
    # it falls through to None, and the outlier_sensitivity axis carries that story.
    if s >= WEAK_MAGNITUDE_THRESHOLD:
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
    xi_fwd: float | None,
    xi_rev: float | None,
    p: float | None = None,
    s: float | None = None,
) -> str | None:
    """Is Y a function of X, X of Y, both, or neither? Derived from Chatterjee's
    xi, so only populated in deep mode (``None`` otherwise).

    ``neither_direction`` is reserved for pairs that genuinely lack a functional
    direction — a circle, say, where dependence exists but neither variable is a
    function of the other. It is *not* reported for a strong monotone pair whose
    xi merely sits below the 0.35 bar: for a bivariate normal xi is only ~0.30 at
    rho=0.7, so an obviously functional noisy-linear pair would otherwise read
    "neither_direction". When |p| or |s| is strong yet neither xi clears the bar,
    the axis is uninformative, so it returns ``None`` rather than that misleading
    label.
    """
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
    if p is not None and s is not None and max(p, s) >= STRONG_MAGNITUDE_THRESHOLD:
        return None
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
    bin_lof_robust = _finite_metric_value(metrics.get("bin_lof_r2_gain_robust"))
    reversals = _finite_metric_value(metrics.get("bin_reversal_count"))
    sq_corr = _finite_metric_value(metrics.get("sq_corr"))
    sq_corr_robust = _finite_metric_value(metrics.get("sq_corr_robust"))
    xi_fwd = _finite_metric_value(metrics.get("chatterjee_xi"))
    xi_rev = _finite_metric_value(metrics.get("chatterjee_xi_reverse"))
    bp_pvalue = _finite_metric_value(metrics.get("bp_pvalue"))
    gq_ratio = _finite_metric_value(metrics.get("gq_ratio"))
    bowtie_ratio = _finite_metric_value(metrics.get("bowtie_ratio"))
    segment_stepness = _finite_metric_value(metrics.get("segment_stepness"))
    n_influential = _finite_metric_value(metrics.get("n_influential_points"))

    return {
        "mean_shape": _mean_shape_axis(p, s, bin_lof, segment_stepness, bin_lof_robust),
        "variance_shape": _variance_shape_axis(
            bp_pvalue, gq_ratio, bin_lof, bowtie_ratio
        ),
        "dependence_type": _dependence_type_axis(
            p,
            s,
            dc,
            sq_corr,
            xi_fwd,
            xi_rev,
            bin_lof,
            reversals,
            bin_lof_robust,
            sq_corr_robust,
        ),
        "outlier_sensitivity": _outlier_sensitivity_axis(outlier_status, n_influential),
        "functional_direction": _functional_direction_axis(xi_fwd, xi_rev, p, s),
    }
