from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from corrsleuth.exceptions import InputError
from corrsleuth.heuristics import STANDARD_ONLY_LABELS, apply_heuristics
from corrsleuth.metrics.core import compute_kendall, compute_pearson, compute_spearman
from corrsleuth.metrics.optional import (
    compute_distance_correlation,
    compute_mutual_information,
)
from corrsleuth.metrics.robust import assess_outlier_sensitivity
from corrsleuth.validation.input import (
    LOW_N_THRESHOLD,
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

#: Pattern stability (the fraction of bootstrap replicates whose label matches
#: the original) at or above which the relationship is labeled "high" stability.
#: 0.80 means at most ~1 in 5 resamples disagreed. These two cut points are
#: presentation bands for a continuous score, not significance tests.
_STABILITY_HIGH_THRESHOLD = 0.80
#: Pattern stability at or above which stability is "medium"; below it, "low".
#: 0.50 is the point at which the modal label no longer holds a majority of
#: replicates, which is the natural boundary for "treat this label as shaky".
_STABILITY_MEDIUM_THRESHOLD = 0.50

#: Minimum rows before percentile bootstrap *intervals* are computed at all.
#: Below this a with-replacement resample of so few points cannot represent the
#: distribution's tails, so the 2.5/97.5 percentiles are erratic and imply false
#: precision — we return ``intervals=None`` with a warning instead. Matches the
#: Chatterjee-xi floor (:data:`~corrsleuth.metrics.nonlinear._MIN_N_FOR_CHATTERJEE_XI`).
#: Pattern stability is *not* gated here (it is a label-agreement signal, and the
#: profile is already labeled low_power_or_uncertain below n=30).
_MIN_N_FOR_INTERVALS = 20


@dataclass
class BootstrapStability:
    pattern_stability: float
    bootstrap_label_counts: dict[str, int]
    stability_label: str
    metric_set: str
    n_bootstrap: int
    n_iterations: int

    def to_dict(self) -> dict:
        """Return the stability fields as a plain dict (for serialization/export)."""
        return asdict(self)


@dataclass
class BootstrapResult:
    intervals: pd.DataFrame | None
    stability: BootstrapStability | None


def _resolve_bootstrap_metrics(
    bootstrap_metrics: str | Sequence[str],
) -> tuple[str, ...]:
    """Normalize the ``bootstrap_metrics`` argument to a tuple of metric names.

    Accepts the ``"lite"`` / ``"standard"`` presets or an explicit sequence of
    metric names. Raises :class:`InputError` for an unknown string preset or for
    any name outside the standard bootstrap set, listing the supported names.
    """
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
    """Return the human-readable label recorded on the stability result.

    Presets keep their name (``"lite"`` / ``"standard"``); an explicit sequence
    is rendered as its sorted, comma-joined names so the label is deterministic
    regardless of the order the caller passed them in.
    """
    if isinstance(bootstrap_metrics, str):
        return bootstrap_metrics
    return ",".join(sorted(metric_names))


def _bootstrap_sample_pair(pair: CleanPair, idx) -> CleanPair:
    """Build a fresh :class:`CleanPair` for one bootstrap replicate.

    ``idx`` is the array of resampled row positions (drawn with replacement).
    Rebuilding the full ``CleanPair`` — rather than just resampling x/y — lets
    each replicate be re-profiled through the same constant/tie/low-n machinery
    as the original pair, so the per-replicate label is computed under identical
    rules. Missing-data fields are zeroed because the source pair is already
    clean; tie/unique/constant fields are recomputed from the resample.
    """
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


def _bootstrap_flags(pair: CleanPair, outlier_status: str) -> list[str]:
    """Return the heuristic flags for a replicate, matching profile_pair's context.

    Mirrors the flags ``profile_pair`` synthesizes — including the Pearson
    trim-sensitivity flag, which is recomputed per replicate and passed in as
    ``outlier_status`` (``"sensitive"`` / ``"stable"`` / ``"unavailable"``)
    rather than assumed unavailable. Blanket-flagging
    ``outlier_sensitivity_unavailable`` would let the leverage rule fire on
    resamples of a relationship the original profile already proved trim-stable,
    biasing pattern stability against a stable, non-leverage label.
    """
    flags = compute_heuristic_flags(pair)
    if outlier_status == "sensitive":
        flags.append("pearson_trim_sensitive")
    elif outlier_status == "stable":
        flags.append("pearson_trim_stable")
    else:
        flags.append("outlier_sensitivity_unavailable")
    return flags


def _compute_bootstrap_metric(name: str, pair: CleanPair, random_state: int):
    """Dispatch a single metric computation by name for one replicate.

    Distance correlation and mutual information are computed with
    ``max_n_for_dcor=None`` (no per-replicate downsampling) so every replicate
    uses its full resampled rows. Raises :class:`InputError` for an unknown
    name (the input is pre-validated, so this is a guard against drift).
    """
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
        return compute_mutual_information(
            pair, mode="standard", random_state=random_state
        )
    raise InputError(f"Unsupported bootstrap metric: {name}")


def _validate_bootstrap_inputs(
    bootstrap: int | None,
    bootstrap_metrics: str | Sequence[str],
    max_n_for_bootstrap: int | None,
) -> tuple[tuple[str, ...], str] | None:
    """Validate bootstrap arguments and resolve the metric set.

    Returns ``None`` when ``bootstrap`` is ``None`` (the no-op case, so callers
    can early-return), otherwise ``(metric_names, metric_set_label)``. Raises
    :class:`InputError` for a non-positive-integer ``bootstrap``, a bad
    ``max_n_for_bootstrap``, or an empty/unsupported metric set.
    """
    if bootstrap is None:
        return None
    if isinstance(bootstrap, bool) or not isinstance(bootstrap, int):
        raise InputError("bootstrap must be a positive integer or None.")
    if bootstrap < 1:
        raise InputError("bootstrap must be a positive integer or None.")
    if max_n_for_bootstrap is not None and (
        isinstance(max_n_for_bootstrap, bool)
        or not isinstance(max_n_for_bootstrap, int)
        or max_n_for_bootstrap < 1
    ):
        raise InputError("max_n_for_bootstrap must be a positive integer or None.")

    metric_names = _resolve_bootstrap_metrics(bootstrap_metrics)
    if not metric_names:
        raise InputError("bootstrap_metrics must include at least one metric.")
    metric_set = _metric_set_label(bootstrap_metrics, metric_names)
    return metric_names, metric_set


def _stability_label(pattern_stability: float) -> str:
    """Bucket a continuous pattern-stability fraction into high/medium/low.

    Boundaries are :data:`_STABILITY_HIGH_THRESHOLD` and
    :data:`_STABILITY_MEDIUM_THRESHOLD`; see their definitions for the rationale.
    """
    if pattern_stability >= _STABILITY_HIGH_THRESHOLD:
        return "high"
    if pattern_stability >= _STABILITY_MEDIUM_THRESHOLD:
        return "medium"
    return "low"


def compute_bootstrap(
    pair: CleanPair,
    bootstrap: int | None,
    bootstrap_metrics: str | Sequence[str],
    random_state: int,
    max_n_for_bootstrap: int | None,
    original_pattern: str | None = None,
) -> BootstrapResult:
    """Compute percentile bootstrap intervals and pattern stability.

    Each replicate resamples rows with replacement, drawing ``pair.n_used`` rows
    by default. When ``max_n_for_bootstrap`` is smaller than ``pair.n_used`` the
    replicates draw only that many rows -- an *m-out-of-n* bootstrap. This is not
    just a performance cap: resampling fewer rows than the data contains inflates
    the per-replicate variance, so the reported intervals are wider (more
    conservative) than the true full-sample sampling variability by roughly a
    factor of ``sqrt(pair.n_used / m)`` (where ``m`` is the capped row count). A
    warning is emitted whenever the cap binds; pass ``max_n_for_bootstrap=None``
    to resample all rows.
    """
    resolved = _validate_bootstrap_inputs(
        bootstrap=bootstrap,
        bootstrap_metrics=bootstrap_metrics,
        max_n_for_bootstrap=max_n_for_bootstrap,
    )
    if resolved is None:
        return BootstrapResult(intervals=None, stability=None)

    metric_names, metric_set = resolved
    # _validate_bootstrap_inputs only returns non-None for a valid positive int.
    assert bootstrap is not None

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

    # Below _MIN_N_FOR_INTERVALS the percentile interval is unreliable enough to
    # be misleading, so we skip it entirely (intervals=None) rather than report
    # false precision; pattern stability is still computed below.
    skip_intervals = pair.n_used < _MIN_N_FOR_INTERVALS
    if skip_intervals:
        pair.warnings.append(
            f"n_used < {_MIN_N_FOR_INTERVALS}: bootstrap intervals are not "
            "computed (too few rows for a reliable percentile bootstrap). "
            "Pattern stability is still reported."
        )
    elif pair.n_used < LOW_N_THRESHOLD:
        pair.warnings.append(
            "Bootstrap intervals requested with n_used < 30; intervals may be unstable."
        )

    # Decouple interval selection from the stability cascade: intervals are
    # reported for the caller's requested metrics (``metric_names``), but the
    # label cascade always needs at least the lite triple
    # (Pearson/Spearman/Kendall) — otherwise apply_heuristics short-circuits to
    # not_computable and stability is meaningless for a custom subset like
    # ["pearson"]. So compute the union per replicate and feed the full set to
    # the cascade, while collecting interval values only for the requested set.
    cascade_metrics = tuple(dict.fromkeys((*_LITE_BOOTSTRAP_METRICS, *metric_names)))

    generator = np.random.default_rng(random_state)
    values: dict[str, list[float]] = {name: [] for name in metric_names}
    interval_metrics = set(metric_names)
    label_counts: dict[str, int] = {}
    n_iterations = 0

    for i in range(bootstrap):
        idx = generator.choice(pair.n_used, size=sample_size, replace=True)
        sample_pair = _bootstrap_sample_pair(pair, idx)
        sample_metrics = {}
        for name in cascade_metrics:
            metric = _compute_bootstrap_metric(name, sample_pair, random_state + i + 1)
            sample_metrics[name] = metric
            if (
                name in interval_metrics
                and metric.value is not None
                and pd.notna(metric.value)
            ):
                values[name].append(float(metric.value))

        # Recompute trim sensitivity on this replicate (same check profile_pair
        # runs) so the leverage rule gates on real per-resample evidence rather
        # than a blanket "unavailable".
        baseline_pearson = sample_metrics["pearson"].value
        outlier_status = assess_outlier_sensitivity(
            sample_pair, baseline_pearson
        ).status
        heuristic = apply_heuristics(
            sample_metrics,
            _bootstrap_flags(sample_pair, outlier_status),
            sample_pair.n_used,
        )
        label_counts[heuristic.label] = label_counts.get(heuristic.label, 0) + 1
        n_iterations += 1

    if skip_intervals:
        intervals = None
    else:
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
                + " are based on fewer than 95% of the requested resamples because "
                + "the metric was undefined on some resamples (e.g. a resample drew "
                + "a near-constant column); treat these intervals as less reliable."
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

        # A standard-only label (e.g. nonmonotonic_dependence) needs distance
        # correlation to be tested; warn whenever the cascade did not see it,
        # regardless of how the interval metric set was labeled.
        if (
            original_pattern in STANDARD_ONLY_LABELS
            and "distance_correlation" not in cascade_metrics
        ):
            pair.warnings.append(
                f"Pattern stability used lite bootstrap metrics, so it may not "
                f"fully test a standard-mode {original_pattern} label."
            )

    return BootstrapResult(intervals=intervals, stability=stability)


def compute_bootstrap_intervals(
    pair: CleanPair,
    bootstrap: int | None,
    bootstrap_metrics: str | Sequence[str],
    random_state: int,
    max_n_for_bootstrap: int | None,
) -> pd.DataFrame | None:
    return compute_bootstrap(
        pair=pair,
        bootstrap=bootstrap,
        bootstrap_metrics=bootstrap_metrics,
        random_state=random_state,
        max_n_for_bootstrap=max_n_for_bootstrap,
        original_pattern=None,
    ).intervals
