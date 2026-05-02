from typing import Dict, List

from corrsleuth.result import MetricResult, HeuristicResult
from .explanations import generate_recommendations

CONFLICTING_SIGN_THRESHOLD = 0.3

#: Heuristic labels that can only be assigned when standard-mode metrics
#: (distance correlation, mutual information) are available. Bootstrap stability
#: computed on lite metrics cannot fully test these labels.
STANDARD_ONLY_LABELS = frozenset({"nonmonotonic_dependence"})


def apply_heuristics(
    metrics: Dict[str, MetricResult], flags: List[str], n_used: int
) -> HeuristicResult:
    """Apply the 8-level heuristic priority cascade to assign a primary label.

    See ``AGENTS.md`` for the cascade definition. Trim-sensitivity flags
    (``pearson_trim_sensitive``, ``outlier_sensitivity_unavailable``) gate the
    ``possible_outlier_or_leverage`` label so it is only assigned when there is
    independent evidence of leverage.
    """
    m_p = metrics.get("pearson")
    m_s = metrics.get("spearman")
    m_k = metrics.get("kendall_tau_b")
    m_dc = metrics.get("distance_correlation")

    p = abs(m_p.value) if m_p and m_p.value is not None else None
    s = abs(m_s.value) if m_s and m_s.value is not None else None
    k = abs(m_k.value) if m_k and m_k.value is not None else None
    dc = m_dc.value if m_dc and m_dc.available and m_dc.value is not None else None

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


def detect_metric_warnings(metrics: Dict[str, MetricResult]) -> List[str]:
    """Return cautionary warnings derived from metric agreement patterns.

    These warnings supplement validation warnings; they do not override the
    primary label. Currently flags conflicting Pearson/Spearman directionality
    when both magnitudes exceed :data:`CONFLICTING_SIGN_THRESHOLD`.
    """
    warnings: List[str] = []

    m_p = metrics.get("pearson")
    m_s = metrics.get("spearman")
    pearson = m_p.value if m_p and m_p.value is not None else None
    spearman = m_s.value if m_s and m_s.value is not None else None

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

    return warnings
