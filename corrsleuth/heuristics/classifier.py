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

#: Magnitude above which Pearson and Spearman having *opposite signs* is worth
#: a directionality warning. Below this both coefficients are near zero and a
#: sign disagreement is just noise, so the warning would be spurious.
CONFLICTING_SIGN_THRESHOLD = 0.3

#: Chatterjee's xi value above which an otherwise weak/ambiguous label gets a
#: dependence warning. Matches :data:`NONMONOTONIC_DC_THRESHOLD`, the
#: distance-correlation threshold used by the nonmonotonic_dependence rule in
#: the cascade.
XI_DEPENDENCE_WARN_THRESHOLD = 0.35

#: Labels that understate the relationship when Chatterjee's xi is high. The
#: cascade does not consult xi, so without a warning a deep-mode profile could
#: report a strong functional dependence and a "weak" label side by side.
_XI_CONTRADICTED_LABELS = frozenset({"weak_or_no_relationship", "mixed_or_ambiguous"})

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
    """
    m_p = metrics.get("pearson")
    m_s = metrics.get("spearman")
    m_k = metrics.get("kendall_tau_b")
    m_dc = metrics.get("distance_correlation")

    p_val = _finite_metric_value(m_p)
    s_val = _finite_metric_value(m_s)
    k_val = _finite_metric_value(m_k)
    p = abs(p_val) if p_val is not None else None
    s = abs(s_val) if s_val is not None else None
    k = abs(k_val) if k_val is not None else None
    dc = _finite_metric_value(m_dc, require_available=True)

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
    elif (
        dc is not None
        and p < NONMONOTONIC_MONOTONE_CEILING
        and s < NONMONOTONIC_MONOTONE_CEILING
        and dc > NONMONOTONIC_DC_THRESHOLD
    ):
        label = "nonmonotonic_dependence"
    # 5. monotonic_nonlinear
    # Gated on Spearman alone (no Kendall fallback, unlike the leverage rule):
    # Spearman is the primary monotone measure here, and tau-b is numerically
    # smaller for the same signal, so adding an OR on tau would only loosen the
    # rule. A borderline-Spearman case deliberately falls through to
    # mixed_or_ambiguous rather than overclaiming nonlinearity.
    elif (
        s > STRONG_MAGNITUDE_THRESHOLD
        and (s - p > RANK_LINEAR_GAP_THRESHOLD)
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
    elif (
        p < WEAK_MAGNITUDE_THRESHOLD
        and s < WEAK_MAGNITUDE_THRESHOLD
        and (dc is None or dc < WEAK_DC_THRESHOLD)
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
    magnitudes exceed :data:`CONFLICTING_SIGN_THRESHOLD`, and — when ``label``
    is provided — high Chatterjee's xi alongside a weak or ambiguous label,
    since the cascade does not consult xi when assigning labels.
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

    if label in _XI_CONTRADICTED_LABELS:
        xi_candidates = [
            (name, value)
            for name in ("chatterjee_xi", "chatterjee_xi_reverse")
            if (value := _finite_metric_value(metrics.get(name))) is not None
        ]
        if xi_candidates:
            xi_name, xi_value = max(xi_candidates, key=lambda item: item[1])
            if xi_value > XI_DEPENDENCE_WARN_THRESHOLD:
                warnings.append(
                    f"{xi_name} ({xi_value:.3f}) is high while linear and rank "
                    f"metrics are weak, which is evidence of nonmonotonic or "
                    f"functional dependence that the '{label}' label may "
                    f"understate. Inspect the scatter plot, or use "
                    f"mode='standard' to check distance correlation."
                )

    return warnings
