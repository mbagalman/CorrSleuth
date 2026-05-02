from typing import Optional, Sequence

import numpy as np
import pandas as pd

from corrsleuth.exceptions import InputError
from corrsleuth.metrics.core import compute_kendall, compute_pearson, compute_spearman
from corrsleuth.metrics.optional import (
    compute_distance_correlation,
    compute_mutual_information,
)
from corrsleuth.validation.input import CleanPair, is_constant_series

_LITE_BOOTSTRAP_METRICS = ("pearson", "spearman", "kendall_tau_b")
_STANDARD_BOOTSTRAP_METRICS = (
    "pearson",
    "spearman",
    "kendall_tau_b",
    "distance_correlation",
    "mutual_information",
)


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
        flags=[],
        warnings=[],
    )


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


def compute_bootstrap_intervals(
    pair: CleanPair,
    bootstrap: Optional[int],
    bootstrap_metrics: str | Sequence[str],
    random_state: int,
    max_n_for_bootstrap: Optional[int],
) -> Optional[pd.DataFrame]:
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

    sample_size = pair.n_used
    if max_n_for_bootstrap is not None and sample_size > max_n_for_bootstrap:
        pair.warnings.append(
            f"n_used > {max_n_for_bootstrap}. Bootstrap samples are capped at "
            f"{max_n_for_bootstrap} rows (random_state={random_state})."
        )
        sample_size = max_n_for_bootstrap

    if pair.n_used < 30:
        pair.warnings.append(
            "Bootstrap intervals requested with n_used < 30; intervals may be unstable."
        )

    generator = np.random.default_rng(random_state)
    values = {name: [] for name in metric_names}

    for i in range(bootstrap):
        idx = generator.choice(pair.n_used, size=sample_size, replace=True)
        sample_pair = _bootstrap_sample_pair(pair, idx)
        for name in metric_names:
            metric = _compute_bootstrap_metric(name, sample_pair, random_state + i + 1)
            if metric.value is not None and pd.notna(metric.value):
                values[name].append(float(metric.value))

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

    return pd.DataFrame(records)
