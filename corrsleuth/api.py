from collections.abc import Sequence

import pandas as pd

from corrsleuth.exceptions import InputError
from corrsleuth.heuristics import apply_heuristics, detect_metric_warnings
from corrsleuth.metrics import (
    ROBUST_METRIC_MIN_N,
    assess_outlier_sensitivity,
    compute_biweight_midcorrelation,
    compute_bootstrap,
    compute_chatterjee_xi,
    compute_chatterjee_xi_reverse,
    compute_distance_correlation,
    compute_kendall,
    compute_median_clipped_pearson,
    compute_mutual_information,
    compute_pearson,
    compute_spearman,
    compute_winsorized_pearson,
)
from corrsleuth.result import CorrSleuthResult, MetricDiagnostics
from corrsleuth.validation.input import validate_pair


def _metric_value(metrics_map, metric_name: str) -> float | None:
    metric = metrics_map.get(metric_name)
    if metric is None or metric.value is None:
        return None
    value = metric.value
    # Treat NaN as "no value", mirroring the classifier's _finite_metric_value so
    # the disagreement score and diagnostics can never diverge from the label
    # cascade by letting a NaN slip through abs()/max() comparisons.
    if value != value:  # NaN is the only value not equal to itself
        return None
    return value


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
        pearson_trimmed=outlier_sensitivity.trimmed.value
        if outlier_sensitivity
        else None,
        pearson_trim_delta=outlier_sensitivity.delta if outlier_sensitivity else None,
    )


def profile_pair(
    data: pd.DataFrame,
    x: str,
    y: str,
    mode: str = "lite",
    missing: str = "pairwise",
    include_caveat: bool = True,
    max_n_for_dcor: int | None = 20000,
    random_state: int = 42,
    bootstrap: int | None = None,
    bootstrap_metrics: str | Sequence[str] = "lite",
    max_n_for_bootstrap: int | None = 5000,
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
        Missing-value policy. ``"pairwise"`` drops rows missing in ``x`` or
        ``y`` only. ``"listwise"`` drops rows missing in *any* column of
        ``data`` (complete-case deletion) before selecting the pair, so a row
        with a missing value in an unrelated column is excluded; the two
        coincide when ``data`` contains only ``x`` and ``y``. ``"raise"`` errors
        if the pair contains any missing values.
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
        Cap on rows sampled per bootstrap replicate. When ``n_used`` exceeds the
        cap, each replicate resamples only this many rows (an m-out-of-n
        bootstrap), which widens the intervals and makes them conservative
        relative to the full-sample sampling variability; a warning is emitted
        when this happens. ``None`` disables the cap so every replicate uses all
        rows.

    Returns
    -------
    CorrSleuthResult
        The diagnostic profile: the computed metrics, the assigned pattern
        label, diagnostics, warnings, recommendations, and (when requested)
        bootstrap intervals and pattern stability. Render it with
        ``.summary()``, ``.explain()``, ``.to_markdown()``, ``.to_dict()``,
        ``.to_frame()``, or ``.plot()``.

    Raises
    ------
    InputError
        If ``mode`` is not one of ``"lite"``, ``"standard"``, or ``"deep"``, or
        if the input fails validation (``x`` or ``y`` missing or non-numeric,
        ``x == y``, duplicate column names, infinite values in the rows used,
        missing values when ``missing="raise"``, or fewer than two valid
        observations after applying the missing-value policy).
    OptionalDependencyError
        If ``mode="standard"`` and the ``corrsleuth[standard]`` extras
        (``dcor``, ``scikit-learn``) are not installed.
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
    #
    # Steps 3-4 enrich `pair` in place — appending to `pair.flags` and
    # `pair.warnings` as new evidence (trim sensitivity, robust-metric n,
    # metric-agreement warnings) accumulates. This is intentional: `pair` is an
    # internal, single-use CleanPair owned by this call and never shared, so a
    # mutable builder reads more clearly than threading a growing immutable copy
    # through each step. The flags/warnings lists are read once at step 4/5.
    baseline_pearson = _metric_value(metrics_map, "pearson")
    outlier_sensitivity = assess_outlier_sensitivity(pair, baseline_pearson)
    if outlier_sensitivity.status == "sensitive":
        pair.flags.append("pearson_trim_sensitive")
        pair.warnings.append(
            "Pearson changes materially after trimming extreme x/y values; "
            "the apparent linear association may be leverage-sensitive."
        )
    elif outlier_sensitivity.status == "stable":
        pair.flags.append("pearson_trim_stable")
    else:
        pair.flags.append("outlier_sensitivity_unavailable")

    if mode == "deep":
        if pair.n_used < ROBUST_METRIC_MIN_N:
            pair.warnings.append(
                f"n_used < {ROBUST_METRIC_MIN_N}; deep-mode robust correlation "
                "diagnostics are not computed."
            )
        # Reuse the trimmed Pearson already computed for the leverage check, so
        # the metric and the flag never disagree.
        metrics_map["pearson_trimmed_1pct"] = outlier_sensitivity.trimmed
        metrics_map["pearson_winsorized_1pct"] = compute_winsorized_pearson(pair)
        metrics_map["biweight_midcorrelation"] = compute_biweight_midcorrelation(pair)
        metrics_map["pearson_median_clipped_20pct"] = compute_median_clipped_pearson(
            pair
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
    pair.warnings.extend(
        detect_metric_warnings(metrics_map, label=heuristic_result.label)
    )

    # 5. Build outputs
    records = [
        {"metric": name, "value": metric.value}
        for name, metric in metrics_map.items()
        if metric.available
    ]
    metrics_df = pd.DataFrame(records)

    # Build the disagreement score from explicit None checks rather than
    # `value or 0.0`: an unavailable metric (None, e.g. constant input) must
    # contribute nothing, whereas a genuine 0.0 is a real measurement. The
    # `or 0.0` idiom conflated the two — harmless today because pearson,
    # spearman, and dcor all go None together on constant input, but a trap if
    # that ever changes.
    pearson = _metric_value(metrics_map, "pearson")
    spearman = _metric_value(metrics_map, "spearman")
    dcor = _metric_value(metrics_map, "distance_correlation")

    abs_p = abs(pearson) if pearson is not None else None
    abs_s = abs(spearman) if spearman is not None else None

    # Rank-vs-linear gap: only defined when both metrics are available.
    rank_gap = abs(abs_p - abs_s) if abs_p is not None and abs_s is not None else 0.0
    # Nonmonotonic contribution: distance correlation in excess of the strongest
    # linear/rank signal. Absent dcor contributes nothing.
    linear_signal = max([v for v in (abs_p, abs_s) if v is not None], default=0.0)
    nonmonotonic = max(0.0, dcor - linear_signal) if dcor is not None else 0.0

    disagreement_score = rank_gap + nonmonotonic
    diagnostics = _build_diagnostics(
        metrics_map, disagreement_score, outlier_sensitivity
    )
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
