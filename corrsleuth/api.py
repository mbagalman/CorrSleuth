from collections.abc import Sequence

import pandas as pd

from corrsleuth.exceptions import InputError
from corrsleuth.heuristics import (
    apply_heuristics,
    derive_diagnostic_axes,
    detect_metric_warnings,
)
from corrsleuth.metrics import (
    ROBUST_METRIC_MIN_N,
    assess_outlier_sensitivity,
    compute_bin_lof,
    compute_biweight_midcorrelation,
    compute_bootstrap,
    compute_chatterjee_xi,
    compute_chatterjee_xi_reverse,
    compute_cluster_split,
    compute_distance_correlation,
    compute_heteroscedasticity,
    compute_heteroscedasticity_excluding,
    compute_influence,
    compute_influential_mask,
    compute_kendall,
    compute_median_clipped_pearson,
    compute_mutual_information,
    compute_pearson,
    compute_segmentation,
    compute_spearman,
    compute_squared_correlation,
    compute_squared_correlation_robust,
    compute_winsorized_pearson,
)
from corrsleuth.metrics.bootstrap import _validate_bootstrap_inputs
from corrsleuth.result import CorrSleuthResult, MetricDiagnostics
from corrsleuth.validation.input import validate_pair


def _validate_profile_pair_options(
    mode: str = "lite",
    missing: str = "pairwise",
    max_n_for_dcor: int | None = 20000,
    bootstrap: int | None = None,
    bootstrap_metrics: str | Sequence[str] = "lite",
    max_n_for_bootstrap: int | None = 5000,
) -> None:
    """Validate :func:`profile_pair`'s column-independent (configuration) options.

    Factored out of :func:`profile_pair` so :func:`~corrsleuth.scan.core.scan_target`
    can preflight the same checks once before its per-column loop: under
    ``errors="warn"`` a per-column :class:`InputError` is captured as a column
    failure, so a *shared-configuration* mistake (bad ``mode``, ``missing``,
    ``max_n_for_dcor``, or bootstrap option) validated only inside the loop
    would surface as N identical column failures instead of one actionable
    error. Defaults mirror :func:`profile_pair`'s, so a caller can pass just
    the options it received. Raises :class:`InputError` for a bad option (or
    :class:`OptionalDependencyError` when ``bootstrap_metrics`` needs missing
    extras).
    """
    if mode not in ("lite", "standard", "deep"):
        raise InputError(
            f"Unknown mode: '{mode}'. Supported modes are 'lite', 'standard', and 'deep'."
        )
    # Same check (and message) as validate_pair, which still guards direct
    # callers of the validation layer; here it fails before any data work.
    if missing not in ("pairwise", "listwise", "raise"):
        raise InputError(
            f"Unsupported missing mode: '{missing}'. Supported modes are 'pairwise', 'listwise', and 'raise'."
        )
    if max_n_for_dcor is not None and (
        isinstance(max_n_for_dcor, bool)
        or not isinstance(max_n_for_dcor, int)
        or max_n_for_dcor < 1
    ):
        # Validate here rather than let a negative value reach rng.choice(n, -1)
        # in optional.py, which raises a bare numpy ValueError outside the
        # MetricComputationError wrapper (C4 #3).
        raise InputError("max_n_for_dcor must be a positive integer or None.")
    _validate_bootstrap_inputs(
        bootstrap=bootstrap,
        bootstrap_metrics=bootstrap_metrics,
        max_n_for_bootstrap=max_n_for_bootstrap,
    )


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
    metrics_map,
    disagreement_score: float,
    outlier_sensitivity=None,
    bin_lof=None,
    sq_corr=None,
    sq_corr_robust=None,
    heteroscedasticity=None,
    segmentation=None,
    influence=None,
    cluster_split=None,
    axes: dict[str, str | None] | None = None,
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

    axes = axes or {}
    bin_lof = bin_lof or {}
    bin_lof_result = bin_lof.get("bin_lof_r2_gain")
    reversal_result = bin_lof.get("bin_reversal_count")
    heteroscedasticity = heteroscedasticity or {}
    bp_result = heteroscedasticity.get("bp_pvalue")
    gq_result = heteroscedasticity.get("gq_ratio")
    bowtie_result = heteroscedasticity.get("bowtie_ratio")
    segmentation = segmentation or {}
    segment_gain_result = segmentation.get("segment_gain")
    segment_stepness_result = segmentation.get("segment_stepness")
    segment_jump_result = segmentation.get("segment_jump_ratio")
    breakpoint_result = segmentation.get("breakpoint_x")
    # The breakpoint location is only meaningful — and only reported — when the
    # mean reads as a step/threshold or a discontinuous jump; for a smooth
    # curve the "break" is an artifact of forcing a single split onto a
    # gradual bend.
    report_breakpoint = axes.get("mean_shape") in (
        "step_or_threshold",
        "discontinuous_jump",
    )
    influence = influence or {}
    max_cook_result = influence.get("max_cook_distance")
    n_influential_result = influence.get("n_influential_points")
    cluster_split = cluster_split or {}
    cluster_r2_result = cluster_split.get("cluster_split_r2")
    cluster_valley_result = cluster_split.get("cluster_valley_share")
    cluster_share_result = cluster_split.get("cluster_min_share")
    within_cluster_result = cluster_split.get("pearson_within_cluster")
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
        bin_lof_r2_gain=bin_lof_result.value if bin_lof_result else None,
        bin_reversal_count=int(reversal_result.value)
        if (reversal_result and reversal_result.value is not None)
        else None,
        sq_corr=sq_corr.value if sq_corr else None,
        sq_corr_robust=sq_corr_robust.value if sq_corr_robust else None,
        bp_pvalue=bp_result.value if bp_result else None,
        gq_ratio=gq_result.value if gq_result else None,
        bowtie_ratio=bowtie_result.value if bowtie_result else None,
        segment_gain=segment_gain_result.value if segment_gain_result else None,
        segment_stepness=segment_stepness_result.value
        if segment_stepness_result
        else None,
        segment_jump_ratio=segment_jump_result.value if segment_jump_result else None,
        breakpoint_x=breakpoint_result.value
        if (report_breakpoint and breakpoint_result)
        else None,
        max_cook_distance=max_cook_result.value if max_cook_result else None,
        n_influential_points=int(n_influential_result.value)
        if (n_influential_result and n_influential_result.value is not None)
        else None,
        cluster_split_r2=cluster_r2_result.value if cluster_r2_result else None,
        cluster_valley_share=cluster_valley_result.value
        if cluster_valley_result
        else None,
        cluster_min_share=cluster_share_result.value if cluster_share_result else None,
        pearson_within_cluster=within_cluster_result.value
        if within_cluster_result
        else None,
        mean_shape=axes.get("mean_shape"),
        variance_shape=axes.get("variance_shape"),
        dependence_type=axes.get("dependence_type"),
        outlier_sensitivity=axes.get("outlier_sensitivity"),
        functional_direction=axes.get("functional_direction"),
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
        Column names of the real-valued numeric variables to profile. Complex
        dtypes are rejected (cast to the real part or magnitude first).
    mode : {"lite", "standard", "deep"}, default "lite"
        ``"lite"`` computes Pearson, Spearman, and Kendall tau-b.
        ``"standard"`` additionally computes distance correlation and mutual
        information; requires the ``corrsleuth[standard]`` extras.
        ``"deep"`` is a strict superset of ``"standard"`` — it computes
        everything standard does *plus* robust correlation diagnostics and
        Chatterjee's xi — and therefore also requires the ``corrsleuth[standard]``
        extras.
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
        Seed for every stochastic step so repeated runs on the same input return
        the same numbers: distance-correlation downsampling, the
        mutual-information estimator, bootstrap resampling, and the random
        tie-break used by Chatterjee's xi when the sort variable has ties. Held
        fixed by default.
    bootstrap : int or None, default None
        Number of bootstrap resamples to use for approximate metric intervals.
        Disabled by default. Intervals are only computed when the effective
        per-replicate size is ``>= 20`` (see ``max_n_for_bootstrap``); below that
        the percentile bootstrap is too unreliable to report, so
        ``bootstrap_intervals`` is ``None`` (with a warning).
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
        rows. The interval floor and pattern stability both key off this
        effective per-replicate size: a cap below 20 suppresses intervals, and a
        cap below 30 (when ``n_used`` is larger) suppresses pattern stability —
        every replicate would otherwise be judged low-power, making stability
        meaningless against the full-sample label.

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
        if the input fails validation (``x`` or ``y`` missing, non-numeric, or
        complex-valued, ``x == y``, duplicate column names, infinite values in
        the rows used, missing values when ``missing="raise"``, or fewer than
        two valid observations after applying the missing-value policy).
    OptionalDependencyError
        If ``mode`` is ``"standard"`` or ``"deep"`` and the
        ``corrsleuth[standard]`` extras (``dcor``, ``scikit-learn``) are not
        installed.
    """
    _validate_profile_pair_options(
        mode=mode,
        missing=missing,
        max_n_for_dcor=max_n_for_dcor,
        bootstrap=bootstrap,
        bootstrap_metrics=bootstrap_metrics,
        max_n_for_bootstrap=max_n_for_bootstrap,
    )

    # 1. Validation
    pair = validate_pair(data, x, y, missing=missing)

    # 2. Compute Metrics
    metrics_map = {
        "pearson": compute_pearson(pair),
        "spearman": compute_spearman(pair),
        "kendall_tau_b": compute_kendall(pair),
    }

    # Distance correlation and mutual information are computed in both standard
    # and deep mode -- deep is a strict superset of standard (it adds robust
    # correlations and Chatterjee's xi on top). Passing mode through means both
    # raise OptionalDependencyError when the [standard] extras are missing.
    if mode in ("standard", "deep"):
        metrics_map["distance_correlation"] = compute_distance_correlation(
            pair,
            mode=mode,
            max_n_for_dcor=max_n_for_dcor,
            random_state=random_state,
        )
        metrics_map["mutual_information"] = compute_mutual_information(
            pair, mode=mode, random_state=random_state
        )

    # Shape and variance diagnostics: pure numpy/scipy, no mode gate, cheap
    # enough for every mode. Feed the label cascade / secondary axes /
    # MetricDiagnostics only — kept out of metrics_map so they never appear in
    # the public metrics table alongside primary association coefficients like
    # pearson/dcor/MI. compute_bin_lof and compute_heteroscedasticity return
    # {name: MetricResult} dicts (bin_lof_r2_gain, bin_reversal_count,
    # bin_lof_r2_gain_robust; bp_pvalue, gq_ratio, bowtie_ratio).
    bin_lof = compute_bin_lof(pair)
    sq_corr = compute_squared_correlation(pair)
    # Leave-the-top-out companion: the classifier gates the sq_corr routes on it
    # so a heavy-tailed variable's few extreme squared values cannot manufacture a
    # magnitude-linked label. Unlike bin_lof_r2_gain_robust (which stays internal
    # to the cascade), this value is also surfaced on MetricDiagnostics.
    sq_corr_robust = compute_squared_correlation_robust(pair)
    heteroscedasticity = compute_heteroscedasticity(pair)
    segmentation = compute_segmentation(pair)
    influence = compute_influence(pair)
    # Two-group / mixture split diagnostics (metrics/mixture.py): is the pooled
    # correlation carried by a between-group mean shift? Feeds the
    # dependence_type axis and the two-group warning; lite-computable.
    cluster_split = compute_cluster_split(pair)

    # Re-test heteroscedasticity excluding the Cook's-flagged row(s), but only
    # when there is one to exclude (n_influential_points >= 1) -- this keeps
    # the common case (no elevated leverage) free of the extra computation.
    # Feeds detect_metric_warnings so it can tell a genuine variance-shape
    # signal apart from one that's just an echo of the same row(s)
    # outlier_sensitivity already flags (see Ticket 1.5 / X13, X16).
    n_influential = _metric_value(influence, "n_influential_points")
    heteroscedasticity_excl_influential = {}
    if n_influential is not None and n_influential >= 1:
        influential_mask = compute_influential_mask(pair)
        if influential_mask is not None and influential_mask.any():
            heteroscedasticity_excl_influential = {
                f"{name}_excl_influential": result
                for name, result in compute_heteroscedasticity_excluding(
                    pair, influential_mask
                ).items()
            }

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
        # statistic is asymmetric. scan_target profiles each pair as
        # profile_pair(data, candidate, target), so the *forward* direction is
        # candidate -> target -- usually the one callers want; the reverse is
        # target -> candidate.
        metrics_map["chatterjee_xi"] = compute_chatterjee_xi(
            pair, random_state=random_state
        )
        metrics_map["chatterjee_xi_reverse"] = compute_chatterjee_xi_reverse(
            pair, random_state=random_state
        )

    # 4. Apply heuristics and metric-agreement warnings
    #
    # The shape/variance diagnostics are merged in only for the
    # cascade/warning/axis calls, never into metrics_map itself, so they stay
    # out of metrics_df. (apply_heuristics ignores the extra keys — variance is
    # a modifier, not a label input.)
    cascade_metrics = {
        **metrics_map,
        **bin_lof,
        "sq_corr": sq_corr,
        "sq_corr_robust": sq_corr_robust,
        **heteroscedasticity,
        **segmentation,
        **influence,
        **cluster_split,
        **heteroscedasticity_excl_influential,
    }
    heuristic_result = apply_heuristics(cascade_metrics, pair.flags, pair.n_used)
    pair.warnings.extend(
        detect_metric_warnings(cascade_metrics, label=heuristic_result.label)
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

    # Rank-vs-linear gap: only defined when both metrics are available. Computed
    # from the *signed* difference, not the gap of magnitudes: when Pearson and
    # Spearman point in opposite directions (e.g. +0.8 vs -0.8, a leverage
    # signature) the magnitude gap is 0 and would report perfect agreement,
    # whereas |pearson - spearman| = 1.6 correctly reflects the conflict. For
    # same-sign metrics the two are identical.
    rank_gap = (
        abs(pearson - spearman) if pearson is not None and spearman is not None else 0.0
    )
    # Nonmonotonic contribution: distance correlation in excess of the strongest
    # linear/rank signal. Absent dcor contributes nothing.
    linear_signal = max([v for v in (abs_p, abs_s) if v is not None], default=0.0)
    nonmonotonic = max(0.0, dcor - linear_signal) if dcor is not None else 0.0

    disagreement_score = rank_gap + nonmonotonic
    # Secondary diagnostic axes: coarse categorical summaries derived from the
    # metrics already computed (using cascade_metrics so the shape diagnostics
    # and, in deep mode, Chatterjee's xi are visible). Orthogonal to the primary
    # label — see heuristics.derive_diagnostic_axes.
    axes = derive_diagnostic_axes(
        cascade_metrics, heuristic_result.label, outlier_sensitivity.status
    )
    diagnostics = _build_diagnostics(
        metrics_map,
        disagreement_score,
        outlier_sensitivity,
        bin_lof=bin_lof,
        sq_corr=sq_corr,
        sq_corr_robust=sq_corr_robust,
        heteroscedasticity=heteroscedasticity,
        segmentation=segmentation,
        influence=influence,
        cluster_split=cluster_split,
        axes=axes,
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
