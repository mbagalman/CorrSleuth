from typing import Optional, Sequence

import pandas as pd
import scipy.stats as stats

from corrsleuth.result import CorrSleuthResult, MetricDiagnostics, MetricResult
from corrsleuth.validation.input import validate_pair
from corrsleuth.metrics import (
    compute_pearson,
    compute_spearman,
    compute_kendall,
    compute_distance_correlation,
    compute_mutual_information,
    ROBUST_METRIC_MIN_N,
    compute_winsorized_pearson,
    compute_biweight_midcorrelation,
    compute_median_clipped_pearson,
    compute_chatterjee_xi,
    compute_chatterjee_xi_reverse,
    compute_bootstrap,
)
from corrsleuth.heuristics import apply_heuristics, detect_metric_warnings
from corrsleuth.exceptions import InputError


def _metric_value(metrics_map, metric_name: str) -> Optional[float]:
    metric = metrics_map.get(metric_name)
    return metric.value if metric and metric.value is not None else None


def _compute_outlier_sensitivity(
    pair, baseline_pearson: Optional[float]
) -> dict:
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
    mask = pair.x.between(x_low, x_high) & pair.y.between(y_low, y_high)
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


def _build_diagnostics(
    metrics_map, disagreement_score: float, outlier_sensitivity=None
) -> MetricDiagnostics:
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
    max_n_for_dcor: Optional[int] = 20000,
    random_state: int = 42,
    bootstrap: Optional[int] = None,
    bootstrap_metrics: str | Sequence[str] = "lite",
    max_n_for_bootstrap: Optional[int] = 5000,
) -> CorrSleuthResult:
    """Profile the pairwise relationship between two numeric variables.

    Parameters
    ----------
    data : pd.DataFrame
        Source data containing both columns.
    x, y : str
        Column names of the numeric variables to profile.
    mode : {"lite", "standard", "deep"}, default "lite"
        ``"lite"`` computes Pearson, Spearman, and Kendall tau-b.
        ``"standard"`` additionally computes distance correlation and mutual
        information; requires the ``corrsleuth[standard]`` extras.
        ``"deep"`` adds lightweight robust correlation diagnostics without
        requiring optional dependencies.
    missing : {"pairwise", "listwise", "raise"}, default "pairwise"
        Missing-value policy for the selected pair.
    include_caveat : bool, default True
        Whether ``summary()`` and ``explain()`` include the non-causal caveat
        by default.
    max_n_for_dcor : int or None, default 20000
        Cap above which distance-correlation input is downsampled. ``None``
        disables the cap.
    random_state : int, default 42
        Seed used for distance-correlation downsampling and the
        mutual-information estimator. Held fixed by default so repeated runs on
        the same input return the same numbers.
    bootstrap : int or None, default None
        Number of bootstrap resamples to use for approximate metric intervals.
        Disabled by default.
    bootstrap_metrics : {"lite", "standard"} or sequence of str, default "lite"
        Metric set to bootstrap. ``"lite"`` bootstraps Pearson, Spearman, and
        Kendall tau-b even when the main profile uses ``mode="standard"``.
        ``"standard"`` explicitly opts in to bootstrapping distance correlation
        and mutual information.
    max_n_for_bootstrap : int or None, default 5000
        Cap on rows sampled per bootstrap replicate. ``None`` disables the cap.
    """
    if mode not in ("lite", "standard", "deep"):
        raise InputError(
            f"Unknown mode: '{mode}'. Supported modes are 'lite', 'standard', and 'deep'."
        )

    # 1. Validation
    pair = validate_pair(data, x, y, missing=missing)

    # 2. Compute Metrics
    metrics_map = {
        "pearson": compute_pearson(pair),
        "spearman": compute_spearman(pair),
        "kendall_tau_b": compute_kendall(pair),
    }

    if mode == "standard":
        metrics_map["distance_correlation"] = compute_distance_correlation(
            pair,
            mode=mode,
            max_n_for_dcor=max_n_for_dcor,
            random_state=random_state,
        )
        metrics_map["mutual_information"] = compute_mutual_information(
            pair, mode=mode, random_state=random_state
        )

    # 3. Outlier sensitivity check (informs the leverage label)
    baseline_pearson = _metric_value(metrics_map, "pearson")
    outlier_sensitivity = _compute_outlier_sensitivity(pair, baseline_pearson)
    if outlier_sensitivity["status"] == "sensitive":
        pair.flags.append("pearson_trim_sensitive")
        pair.warnings.append(
            "Pearson changes materially after trimming extreme x/y values; "
            "the apparent linear association may be leverage-sensitive."
        )
    elif outlier_sensitivity["status"] == "stable":
        pair.flags.append("pearson_trim_stable")
    else:
        pair.flags.append("outlier_sensitivity_unavailable")

    if mode == "deep":
        if pair.n_used < ROBUST_METRIC_MIN_N:
            pair.warnings.append(
                f"n_used < {ROBUST_METRIC_MIN_N}; deep-mode robust correlation "
                "diagnostics are not computed."
            )
        metrics_map["pearson_trimmed_1pct"] = MetricResult(
            name="pearson_trimmed_1pct",
            value=outlier_sensitivity["trimmed"],
            available=True,
        )
        metrics_map["pearson_winsorized_1pct"] = compute_winsorized_pearson(pair)
        metrics_map["biweight_midcorrelation"] = compute_biweight_midcorrelation(pair)
        metrics_map["pearson_median_clipped_20pct"] = (
            compute_median_clipped_pearson(pair)
        )
        # chatterjee_xi has its own (lower) min-n threshold than the robust
        # correlations, so it can produce a value even when the robust metrics
        # above are not computable. Both directions are computed because the
        # statistic is asymmetric — for target scans, callers usually want the
        # reverse direction (candidate -> target).
        metrics_map["chatterjee_xi"] = compute_chatterjee_xi(pair)
        metrics_map["chatterjee_xi_reverse"] = compute_chatterjee_xi_reverse(pair)

    # 4. Apply heuristics and metric-agreement warnings
    heuristic_result = apply_heuristics(metrics_map, pair.flags, pair.n_used)
    pair.warnings.extend(detect_metric_warnings(metrics_map))

    # 5. Build outputs
    records = [
        {"metric": name, "value": metric.value}
        for name, metric in metrics_map.items()
        if metric.available
    ]
    metrics_df = pd.DataFrame(records)

    abs_p = abs(_metric_value(metrics_map, "pearson") or 0.0)
    abs_s = abs(_metric_value(metrics_map, "spearman") or 0.0)
    dc = _metric_value(metrics_map, "distance_correlation") or 0.0

    disagreement_score = abs(abs_p - abs_s) + max(0.0, dc - max(abs_p, abs_s))
    diagnostics = _build_diagnostics(metrics_map, disagreement_score, outlier_sensitivity)
    bootstrap_result = compute_bootstrap(
        pair,
        bootstrap=bootstrap,
        bootstrap_metrics=bootstrap_metrics,
        random_state=random_state,
        max_n_for_bootstrap=max_n_for_bootstrap,
        original_pattern=heuristic_result.label,
    )

    return CorrSleuthResult(
        x_name=x,
        y_name=y,
        metrics=metrics_df,
        pattern=heuristic_result.label,
        warnings=pair.warnings,
        recommendations=heuristic_result.recommendations,
        disagreement_score=disagreement_score,
        diagnostics=diagnostics,
        bootstrap_intervals=bootstrap_result.intervals,
        bootstrap_stability=bootstrap_result.stability,
        _clean_x=pair.x,
        _clean_y=pair.y,
        _include_caveat=include_caveat,
    )
