from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from corrsleuth.exceptions import InputError
from corrsleuth.heuristics import STANDARD_ONLY_LABELS, apply_heuristics
from corrsleuth.metrics.core import compute_kendall, compute_pearson, compute_spearman
from corrsleuth.metrics.optional import (
    compute_distance_correlation,
    compute_mutual_information,
)
from corrsleuth.validation.input import (
    CleanPair,
    compute_heuristic_flags,
    compute_tie_rate,
    is_constant_series,
)

_LITE_BOOTSTRAP_METRICS = ("pearson", "spearman", "kendall_tau_b")
_STANDARD_BOOTSTRAP_METRICS = (
    "pearson",
    "spearman",
    "kendall_tau_b",
    "distance_correlation",
    "mutual_information",
)

_STABILITY_HIGH_THRESHOLD = 0.80
_STABILITY_MEDIUM_THRESHOLD = 0.50


@dataclass
class BootstrapStability:
    pattern_stability: float
    bootstrap_label_counts: dict[str, int]
    stability_label: str
    metric_set: str
    n_bootstrap: int
    n_iterations: int

    def to_dict(self):
        return asdict(self)


@dataclass
class BootstrapResult:
    intervals: Optional[pd.DataFrame]
    stability: Optional[BootstrapStability]


def _resolve_bootstrap_metrics(bootstrap_metrics: str | Sequence[str]) -> tuple[str, ...]:
    if bootstrap_metrics == "lite":
        return _LITE_BOOTSTRAP_METRICS
    if bootstrap_metrics == "standard":
        return _STANDARD_BOOTSTRAP_METRICS
    if isinstance(bootstrap_metrics, str):
        raise InputError(
            "bootstrap_metrics must be 'lite', 'standard', or a sequence of metric names."
        )

    requested = tuple(bootstrap_metrics)
    supported = set(_STANDARD_BOOTSTRAP_METRICS)
    unsupported = sorted(set(requested) - supported)
    if unsupported:
        raise InputError(
            "Unsupported bootstrap metric(s): "
            + ", ".join(unsupported)
            + ". Supported metrics are: "
            + ", ".join(_STANDARD_BOOTSTRAP_METRICS)
            + "."
        )
    return requested


def _metric_set_label(
    bootstrap_metrics: str | Sequence[str], metric_names: Sequence[str]
) -> str:
    if isinstance(bootstrap_metrics, str):
        return bootstrap_metrics
    return ",".join(sorted(metric_names))


def _bootstrap_sample_pair(pair: CleanPair, idx) -> CleanPair:
    x = pd.Series(pair.x.to_numpy()[idx], name=pair.x_name)
    y = pd.Series(pair.y.to_numpy()[idx], name=pair.y_name)
    n_used = len(idx)
    return CleanPair(
        x=x,
        y=y,
        x_name=pair.x_name,
        y_name=pair.y_name,
        n_original=n_used,
        n_used=n_used,
        missing_count=0,
        missing_ratio=0.0,
        x_unique_ratio=x.nunique() / n_used if n_used else 0.0,
        y_unique_ratio=y.nunique() / n_used if n_used else 0.0,
        x_is_constant=is_constant_series(x),
        y_is_constant=is_constant_series(y),
        x_tie_rate=compute_tie_rate(x),
        y_tie_rate=compute_tie_rate(y),
        flags=[],
        warnings=[],
    )


def _bootstrap_flags(pair: CleanPair) -> list[str]:
    flags = compute_heuristic_flags(pair)
    # Bootstrap doesn't recompute Pearson trim sensitivity per replicate; signal
    # that to the heuristic via outlier_sensitivity_unavailable so the
    # possible_outlier_or_leverage rule can still gate when n is large enough.
    if "low_n" not in flags:
        flags.append("outlier_sensitivity_unavailable")
    return flags


def _compute_bootstrap_metric(name: str, pair: CleanPair, random_state: int):
    if name == "pearson":
        return compute_pearson(pair)
    if name == "spearman":
        return compute_spearman(pair)
    if name == "kendall_tau_b":
        return compute_kendall(pair)
    if name == "distance_correlation":
        return compute_distance_correlation(
            pair, mode="standard", max_n_for_dcor=None, random_state=random_state
        )
    if name == "mutual_information":
        return compute_mutual_information(pair, mode="standard", random_state=random_state)
    raise InputError(f"Unsupported bootstrap metric: {name}")


def _validate_bootstrap_inputs(
    bootstrap: Optional[int],
    bootstrap_metrics: str | Sequence[str],
    max_n_for_bootstrap: Optional[int],
) -> Optional[tuple[tuple[str, ...], str]]:
    if bootstrap is None:
        return None
    if isinstance(bootstrap, bool) or not isinstance(bootstrap, int):
        raise InputError("bootstrap must be a positive integer or None.")
    if bootstrap < 1:
        raise InputError("bootstrap must be a positive integer or None.")
    if (
        max_n_for_bootstrap is not None
        and (
            isinstance(max_n_for_bootstrap, bool)
            or not isinstance(max_n_for_bootstrap, int)
            or max_n_for_bootstrap < 1
        )
    ):
        raise InputError("max_n_for_bootstrap must be a positive integer or None.")

    metric_names = _resolve_bootstrap_metrics(bootstrap_metrics)
    if not metric_names:
        raise InputError("bootstrap_metrics must include at least one metric.")
    metric_set = _metric_set_label(bootstrap_metrics, metric_names)
    return metric_names, metric_set


def _stability_label(pattern_stability: float) -> str:
    if pattern_stability >= _STABILITY_HIGH_THRESHOLD:
        return "high"
    if pattern_stability >= _STABILITY_MEDIUM_THRESHOLD:
        return "medium"
    return "low"


def compute_bootstrap(
    pair: CleanPair,
    bootstrap: Optional[int],
    bootstrap_metrics: str | Sequence[str],
    random_state: int,
    max_n_for_bootstrap: Optional[int],
    original_pattern: Optional[str] = None,
) -> BootstrapResult:
    """Compute percentile bootstrap intervals and pattern stability.

    Each replicate resamples ``sample_size`` rows with replacement. When
    ``max_n_for_bootstrap`` is smaller than ``pair.n_used`` the replicates draw
    only that many rows -- an *m-out-of-n* bootstrap. This is not just a
    performance cap: resampling fewer rows than the data contains inflates the
    per-replicate variance, so the reported intervals are wider (more
    conservative) than the true full-sample sampling variability by roughly a
    factor of ``sqrt(n_used / sample_size)``. A warning is emitted whenever the
    cap binds; pass ``max_n_for_bootstrap=None`` to resample all rows.
    """
    resolved = _validate_bootstrap_inputs(
        bootstrap=bootstrap,
        bootstrap_metrics=bootstrap_metrics,
        max_n_for_bootstrap=max_n_for_bootstrap,
    )
    if resolved is None:
        return BootstrapResult(intervals=None, stability=None)

    metric_names, metric_set = resolved

    sample_size = pair.n_used
    if max_n_for_bootstrap is not None and sample_size > max_n_for_bootstrap:
        pair.warnings.append(
            f"n_used > {max_n_for_bootstrap}. Bootstrap samples are capped at "
            f"{max_n_for_bootstrap} rows (random_state={random_state}); "
            f"resampling fewer rows than n_used is an m-out-of-n bootstrap that "
            f"widens the intervals, so they are conservative relative to the "
            f"full-sample sampling variability. Pass max_n_for_bootstrap=None to "
            f"use all rows."
        )
        sample_size = max_n_for_bootstrap

    if pair.n_used < 30:
        pair.warnings.append(
            "Bootstrap intervals requested with n_used < 30; intervals may be unstable."
        )

    generator = np.random.default_rng(random_state)
    values = {name: [] for name in metric_names}
    label_counts: dict[str, int] = {}
    n_iterations = 0

    for i in range(bootstrap):
        idx = generator.choice(pair.n_used, size=sample_size, replace=True)
        sample_pair = _bootstrap_sample_pair(pair, idx)
        sample_metrics = {}
        for name in metric_names:
            metric = _compute_bootstrap_metric(name, sample_pair, random_state + i + 1)
            sample_metrics[name] = metric
            if metric.value is not None and pd.notna(metric.value):
                values[name].append(float(metric.value))

        heuristic = apply_heuristics(
            sample_metrics,
            _bootstrap_flags(sample_pair),
            sample_pair.n_used,
        )
        label_counts[heuristic.label] = label_counts.get(heuristic.label, 0) + 1
        n_iterations += 1

    records = []
    for name in metric_names:
        metric_values = values[name]
        if metric_values:
            ci_low, ci_high = np.percentile(metric_values, [2.5, 97.5])
            ci_low = float(ci_low)
            ci_high = float(ci_high)
        else:
            ci_low = None
            ci_high = None
        records.append(
            {
                "metric": name,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_success": len(metric_values),
                "n_bootstrap": bootstrap,
                "sample_size": sample_size,
                "metric_set": metric_set,
            }
        )

    incomplete_metrics = [
        row["metric"]
        for row in records
        if row["n_success"] == 0 or row["n_success"] / bootstrap < 0.95
    ]
    if incomplete_metrics:
        pair.warnings.append(
            "Bootstrap intervals for "
            + ", ".join(incomplete_metrics)
            + " used fewer than 95% of requested samples because some resamples "
            + "were non-computable."
        )

    intervals = pd.DataFrame(records)
    stability = None
    if original_pattern is not None:
        pattern_stability = (
            label_counts.get(original_pattern, 0) / n_iterations
            if n_iterations
            else 0.0
        )
        stability = BootstrapStability(
            pattern_stability=float(pattern_stability),
            bootstrap_label_counts=label_counts,
            stability_label=_stability_label(pattern_stability),
            metric_set=metric_set,
            n_bootstrap=bootstrap,
            n_iterations=n_iterations,
        )

        if original_pattern in STANDARD_ONLY_LABELS and metric_set == "lite":
            pair.warnings.append(
                f"Pattern stability used lite bootstrap metrics, so it may not "
                f"fully test a standard-mode {original_pattern} label."
            )

    return BootstrapResult(intervals=intervals, stability=stability)


def compute_bootstrap_intervals(
    pair: CleanPair,
    bootstrap: Optional[int],
    bootstrap_metrics: str | Sequence[str],
    random_state: int,
    max_n_for_bootstrap: Optional[int],
) -> Optional[pd.DataFrame]:
    return compute_bootstrap(
        pair=pair,
        bootstrap=bootstrap,
        bootstrap_metrics=bootstrap_metrics,
        random_state=random_state,
        max_n_for_bootstrap=max_n_for_bootstrap,
        original_pattern=None,
    ).intervals
