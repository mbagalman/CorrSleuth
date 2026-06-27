from typing import Dict, List, Optional

from corrsleuth.result import MetricResult, HeuristicResult
from .explanations import generate_recommendations

CONFLICTING_SIGN_THRESHOLD = 0.3

#: Chatterjee's xi value above which an otherwise weak/ambiguous label gets a
#: dependence warning. Matches the distance-correlation threshold used by the
#: nonmonotonic_dependence rule in the cascade.
XI_DEPENDENCE_WARN_THRESHOLD = 0.35

#: Labels that understate the relationship when Chatterjee's xi is high. The
#: cascade does not consult xi, so without a warning a deep-mode profile could
#: report a strong functional dependence and a "weak" label side by side.
_XI_CONTRADICTED_LABELS = frozenset(
    {"weak_or_no_relationship", "mixed_or_ambiguous"}
)

#: Heuristic labels that can only be assigned when standard-mode metrics
#: (distance correlation, mutual information) are available. Bootstrap stability
#: computed on lite metrics cannot fully test these labels.
STANDARD_ONLY_LABELS = frozenset({"nonmonotonic_dependence"})


def _finite_metric_value(
    metric: Optional[MetricResult], *, require_available: bool = False
) -> Optional[float]:
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
    metrics: Dict[str, MetricResult], flags: List[str], n_used: int
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

    label = "mixed_or_ambiguous"

    # 1. not_computable
    if "constant_input" in flags or p is None or s is None or k is None:
        label = "not_computable"
    # 2. low_power_or_uncertain  (low_n is set in validation iff n_used < 30)
    elif "low_n" in flags:
        label = "low_power_or_uncertain"
    # 3. possible_outlier_or_leverage
    elif (
        p > 0.50
        and (p - s > 0.20 or p - k > 0.25)
        and (
            "pearson_trim_sensitive" in flags
            or "outlier_sensitivity_unavailable" in flags
        )
    ):
        label = "possible_outlier_or_leverage"
    # 4. nonmonotonic_dependence
    elif dc is not None and p < 0.25 and s < 0.25 and dc > 0.35:
        label = "nonmonotonic_dependence"
    # 5. monotonic_nonlinear
    elif s > 0.50 and (s - p > 0.20):
        label = "monotonic_nonlinear"
    # 6. near_linear
    elif p > 0.50 and s > 0.50 and abs(p - s) < 0.15:
        label = "near_linear"
    # 7. weak_or_no_relationship
    elif p < 0.20 and s < 0.20 and (dc is None or dc < 0.20):
        label = "weak_or_no_relationship"

    return HeuristicResult(
        label=label,
        recommendations=generate_recommendations(label),
    )


def detect_metric_warnings(
    metrics: Dict[str, MetricResult], label: Optional[str] = None
) -> List[str]:
    """Return cautionary warnings derived from metric agreement patterns.

    These warnings supplement validation warnings; they do not override the
    primary label. Flags conflicting Pearson/Spearman directionality when both
    magnitudes exceed :data:`CONFLICTING_SIGN_THRESHOLD`, and — when ``label``
    is provided — high Chatterjee's xi alongside a weak or ambiguous label,
    since the cascade does not consult xi when assigning labels.
    """
    warnings: List[str] = []

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
