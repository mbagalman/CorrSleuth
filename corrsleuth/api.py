import pandas as pd
from typing import Optional
from corrsleuth.result import CorrSleuthResult
from corrsleuth.validation.input import validate_pair
from corrsleuth.metrics import (
    compute_pearson,
    compute_spearman,
    compute_kendall,
    compute_distance_correlation,
    compute_mutual_information
)
from corrsleuth.heuristics import apply_heuristics

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
    
    p = metrics_map.get("pearson").value if metrics_map.get("pearson") else None
    s = metrics_map.get("spearman").value if metrics_map.get("spearman") else None
    
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
    p = abs(metrics_map.get("pearson").value) if metrics_map.get("pearson") and metrics_map.get("pearson").value is not None else 0.0
    s = abs(metrics_map.get("spearman").value) if metrics_map.get("spearman") and metrics_map.get("spearman").value is not None else 0.0
    dc = metrics_map.get("distance_correlation").value if "distance_correlation" in metrics_map and metrics_map["distance_correlation"].value is not None else 0.0
    
    disagreement_score = abs(p - s) + max(0.0, dc - s)
    
    return CorrSleuthResult(
        x_name=x,
        y_name=y,
        metrics=metrics_df,
        pattern=heuristic_result.label,
        warnings=pair.warnings,
        recommendations=heuristic_result.recommendations,
        disagreement_score=disagreement_score,
        _clean_x=pair.x,
        _clean_y=pair.y,
        _include_caveat=include_caveat
    )
