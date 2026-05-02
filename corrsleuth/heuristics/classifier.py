from typing import Dict, List
from corrsleuth.result import MetricResult, HeuristicResult
from .explanations import generate_recommendations

def apply_heuristics(metrics: Dict[str, MetricResult], flags: List[str], n_used: int) -> HeuristicResult:
    """
    Applies an 8-level heuristic priority cascade to assign a primary diagnostic label.
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
    # 2. low_power_or_uncertain
    elif "low_n" in flags or n_used < 30:
        label = "low_power_or_uncertain"
    # 3. possible_outlier_or_leverage
    elif p > 0.50 and (p - s > 0.20 or p - k > 0.25):
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
        
    rank_linear_gap = abs(p - s) if p is not None and s is not None else 0.0
    nonmonotonic_gap = (dc - max(p, s)) if dc is not None and p is not None and s is not None else 0.0
    
    return HeuristicResult(
        label=label,
        disagreement_components={
            "rank_linear_gap": rank_linear_gap,
            "nonmonotonic_gap": nonmonotonic_gap
        },
        recommendations=generate_recommendations(label)
    )
