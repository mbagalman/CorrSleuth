import pandas as pd
from typing import Optional
from corrsleuth.result import CorrSleuthResult, MetricDiagnostics
from corrsleuth.validation.input import validate_pair
from corrsleuth.metrics import (
    compute_pearson,
    compute_spearman,
    compute_kendall,
    compute_distance_correlation,
    compute_mutual_information
)
from corrsleuth.heuristics import apply_heuristics


def _metric_value(metrics_map, metric_name: str) -> Optional[float]:
    metric = metrics_map.get(metric_name)
    return metric.value if metric and metric.value is not None else None


def _build_diagnostics(metrics_map, disagreement_score: float) -> MetricDiagnostics:
    pearson = _metric_value(metrics_map, "pearson")
    spearman = _metric_value(metrics_map, "spearman")
    kendall = _metric_value(metrics_map, "kendall_tau_b")
    dcor = _metric_value(metrics_map, "distance_correlation")

    rank_linear_gap = (
        abs(abs(pearson) - abs(spearman))
        if pearson is not None and spearman is not None
        else None
    )
    pearson_spearman_signed_gap = (
        pearson - spearman if pearson is not None and spearman is not None else None
    )
    pearson_kendall_gap = (
        abs(abs(pearson) - abs(kendall))
        if pearson is not None and kendall is not None
        else None
    )
    nonmonotonic_gap = (
        dcor - max(abs(pearson), abs(spearman))
        if dcor is not None and pearson is not None and spearman is not None
        else None
    )

    return MetricDiagnostics(
        rank_linear_gap=rank_linear_gap,
        pearson_spearman_signed_gap=pearson_spearman_signed_gap,
        nonmonotonic_gap=nonmonotonic_gap,
        pearson_kendall_gap=pearson_kendall_gap,
        disagreement_score=disagreement_score,
    )

def profile_pair(
    data: pd.DataFrame,
    x: str,
    y: str,
    mode: str = "lite",
    missing: str = "pairwise",
    include_caveat: bool = True,
    max_n_for_dcor: Optional[int] = 20000
) -> CorrSleuthResult:
    """
    Profiles the relationship between two numeric variables.
    """
    if mode not in ["lite", "standard"]:
        if mode == "deep":
            raise NotImplementedError("mode='deep' is not implemented in v0.1.")
        raise ValueError(f"Unknown mode: {mode}")

    # 1. Validation
    pair = validate_pair(data, x, y, missing=missing)
    
    # 2. Compute Metrics
    metrics_map = {}
    metrics_map["pearson"] = compute_pearson(pair)
    metrics_map["spearman"] = compute_spearman(pair)
    metrics_map["kendall_tau_b"] = compute_kendall(pair)
    
    if mode == "standard":
        metrics_map["distance_correlation"] = compute_distance_correlation(pair, mode=mode, max_n_for_dcor=max_n_for_dcor)
        metrics_map["mutual_information"] = compute_mutual_information(pair, mode=mode)
        
    # 3. Apply Heuristics
    heuristic_result = apply_heuristics(metrics_map, pair.flags, pair.n_used)
    
    p = _metric_value(metrics_map, "pearson")
    s = _metric_value(metrics_map, "spearman")
    
    if p is not None and s is not None and abs(p) > 0.3 and abs(s) > 0.3:
        if (p > 0 and s < 0) or (p < 0 and s > 0):
            pair.warnings.append("Pearson and Spearman have conflicting directions; inspect the scatter plot and check for nonlinearity, segments, or leverage points.")
            
    # 4. Construct Result
    # Create the DataFrame
    records = []
    for k, v in metrics_map.items():
        if v.available:
            records.append({"metric": k, "value": v.value})
    metrics_df = pd.DataFrame(records)
    
    # Calculate disagreement score
    p = abs(_metric_value(metrics_map, "pearson") or 0.0)
    s = abs(_metric_value(metrics_map, "spearman") or 0.0)
    dc = _metric_value(metrics_map, "distance_correlation") or 0.0
    
    disagreement_score = abs(p - s) + max(0.0, dc - s)
    diagnostics = _build_diagnostics(metrics_map, disagreement_score)
    
    return CorrSleuthResult(
        x_name=x,
        y_name=y,
        metrics=metrics_df,
        pattern=heuristic_result.label,
        warnings=pair.warnings,
        recommendations=heuristic_result.recommendations,
        disagreement_score=disagreement_score,
        diagnostics=diagnostics,
        _clean_x=pair.x,
        _clean_y=pair.y,
        _include_caveat=include_caveat
    )
