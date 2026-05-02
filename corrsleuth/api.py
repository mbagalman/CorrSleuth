import pandas as pd
from typing import Optional
import scipy.stats as stats
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


def _compute_outlier_sensitivity(pair, baseline_pearson: Optional[float]) -> dict[str, Optional[float | str]]:
    if baseline_pearson is None:
        return {"status": "unavailable", "trimmed": None, "delta": None}

    min_n_for_trim = 50
    min_n_after_trim = 30
    if pair.n_used < min_n_for_trim:
        return {"status": "unavailable", "trimmed": None, "delta": None}

    x_low = pair.x.quantile(0.01)
    x_high = pair.x.quantile(0.99)
    y_low = pair.y.quantile(0.01)
    y_high = pair.y.quantile(0.99)
    mask = (
        pair.x.between(x_low, x_high)
        & pair.y.between(y_low, y_high)
    )
    x_trimmed = pair.x[mask]
    y_trimmed = pair.y[mask]

    if len(x_trimmed) < min_n_after_trim:
        return {"status": "unavailable", "trimmed": None, "delta": None}
    if x_trimmed.nunique() <= 1 or y_trimmed.nunique() <= 1:
        return {"status": "unavailable", "trimmed": None, "delta": None}

    trimmed_pearson, _ = stats.pearsonr(x_trimmed, y_trimmed)
    trimmed_pearson = float(trimmed_pearson)
    delta = abs(abs(baseline_pearson) - abs(trimmed_pearson))
    status = "sensitive" if delta > 0.20 else "stable"
    return {"status": status, "trimmed": trimmed_pearson, "delta": delta}


def _build_diagnostics(metrics_map, disagreement_score: float, outlier_sensitivity=None) -> MetricDiagnostics:
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
        pearson_trimmed=outlier_sensitivity.get("trimmed") if outlier_sensitivity else None,
        pearson_trim_delta=outlier_sensitivity.get("delta") if outlier_sensitivity else None,
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
        
    baseline_pearson = _metric_value(metrics_map, "pearson")
    outlier_sensitivity = _compute_outlier_sensitivity(pair, baseline_pearson)
    if outlier_sensitivity["status"] == "sensitive":
        pair.flags.append("pearson_trim_sensitive")
        pair.warnings.append(
            "Pearson changes materially after trimming extreme x/y values; the apparent linear association may be leverage-sensitive."
        )
    elif outlier_sensitivity["status"] == "stable":
        pair.flags.append("pearson_trim_stable")
    else:
        pair.flags.append("outlier_sensitivity_unavailable")

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
    diagnostics = _build_diagnostics(metrics_map, disagreement_score, outlier_sensitivity)
    
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
