"""Target-scan orchestration.

Drives the scan: resolves which numeric columns are eligible candidates against
the target, profiles each with :func:`profile_pair`, captures per-column
failures, and assembles a :class:`~corrsleuth.scan.report.CorrSleuthTargetReport`.
Report rendering lives in :mod:`corrsleuth.scan.report`; plotting lives in
:mod:`corrsleuth.scan.plot`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from corrsleuth.api import _validate_profile_pair_options, profile_pair
from corrsleuth.exceptions import InputError, OptionalDependencyError
from corrsleuth.result import CorrSleuthResult
from corrsleuth.validation.input import is_real_numeric_dtype, real_numeric_problem

if TYPE_CHECKING:
    from corrsleuth.scan.report import CorrSleuthTargetReport

_VALID_ERRORS_POLICIES = ("warn", "raise")

#: Exception types that are systemic (configuration-level), never a per-column
#: data condition: a missing optional dependency (``mode="standard"``/``"deep"``
#: without the extras) or a bad/misspelled ``profile_pair`` keyword. These are
#: propagated even under ``errors="warn"`` so one config mistake surfaces as a
#: single actionable error instead of N identical per-column "error" entries
#: wrapping a scan that "completes" with zero successes.
#:
#: ``InputError`` is deliberately excluded: ``validate_pair`` raises it *per
#: column* for genuine data problems (all-NaN/constant columns), which
#: ``errors="warn"`` must keep capturing, and a config ``InputError`` (e.g. a bad
#: kwarg value) is not reliably distinguishable from a data one *at catch time*.
#: Shared-configuration ``InputError``s are instead prevented from reaching the
#: loop at all: ``scan_target`` preflights the column-independent options via
#: :func:`corrsleuth.api._validate_profile_pair_options` before iterating.
_CONFIG_CLASS_EXCEPTIONS = (OptionalDependencyError, TypeError)

_VALID_DIRECTIONS = ("forward", "reverse", "both")


@dataclass
class TargetScanEntry:
    """One column's outcome from a target scan.

    ``status="ok"`` entries have a populated ``result``. ``status="skipped"``
    entries describe columns that were filtered out before profiling (for
    example, a non-numeric column the caller listed in ``columns=``).
    ``status="error"`` entries describe profile_pair failures captured under
    ``errors="warn"``.

    ``result`` holds the primary profile — ``profile_pair(candidate, target)`` for
    ``direction="forward"`` (describes ``E[target | candidate]``) or
    ``profile_pair(target, candidate)`` for ``direction="reverse"`` (describes
    ``E[candidate | target]``). ``reverse_result`` is populated only for
    ``direction="both"``, carrying the reverse-orientation profile so the report
    can show how the candidate's *shape* looks as a function of the target.
    """

    column: str
    status: str
    result: CorrSleuthResult | None = None
    error_type: str | None = None
    error_message: str | None = None
    reverse_result: CorrSleuthResult | None = None

    @property
    def result_data(self) -> CorrSleuthResult:
        """Return the profiling result, which is always present for ``status="ok"``.

        The report and plot modules operate on ``successes`` (``status="ok"``),
        where ``result`` is guaranteed populated. Accessing it on a
        skipped/errored entry is a programming error, so this raises rather than
        returning ``None`` — which also lets type checkers narrow away the
        ``Optional`` at every render site.
        """
        if self.result is None:
            raise ValueError(
                f"column {self.column!r} (status={self.status!r}) has no result"
            )
        return self.result


def metrics_map(entry: TargetScanEntry) -> dict[str, Any]:
    """Return ``{metric_name: value}`` for a successful entry's result.

    Shared by the report and plot modules, which both need ad-hoc lookups of a
    single metric (e.g. Pearson, Spearman) by name without re-iterating the
    metrics frame at every call site.
    """
    return {
        row["metric"]: row["value"] for _, row in entry.result_data.metrics.iterrows()
    }


def _iter_with_progress(items: Sequence[str], progress: bool):
    """Yield from ``items``, optionally wrapping with tqdm when available.

    When ``progress=True`` and tqdm is not installed, this is a documented
    no-op rather than printing a homemade progress bar — keeps script and
    test output clean by default.
    """
    if not progress:
        return iter(items)
    try:
        from tqdm import tqdm
    except ImportError:
        return iter(items)
    return tqdm(items, desc="scan_target")


def _resolve_candidate_columns(
    data: pd.DataFrame,
    target: str,
    columns: Sequence[str] | None,
) -> tuple[list[str], list[TargetScanEntry], list[str]]:
    """Return (numeric candidates, pre-skipped entries, resolution order)
    honoring ``columns=``.

    If ``columns`` is None, every real-valued numeric column except the target
    is a candidate; complex columns are excluded (pandas classifies them as
    numeric, but CorrSleuth's metrics are defined for real-valued data). When
    the caller passes an explicit list, we preserve their order and emit
    ``status="skipped"`` entries for missing, non-numeric, or complex names so
    the report still reflects the request. The third element interleaves
    candidate and skipped names in the order they were requested (or appear in
    ``data``), so the final report can present entries in that order rather
    than grouping all skips first.

    Duplicate column names are surfaced as ``DuplicateColumn`` skips in either
    mode (rather than silently dropped), because ``data[col]`` returns a
    DataFrame for a repeated name, which is ambiguous to profile and would
    otherwise look non-numeric.
    """
    duplicated = set(data.columns[data.columns.duplicated(keep=False)])

    def _duplicate_skip(col: str) -> TargetScanEntry:
        return TargetScanEntry(
            column=col,
            status="skipped",
            error_type="DuplicateColumn",
            error_message=(
                f"Column '{col}' matches multiple columns in data; "
                f"column names must be unique."
            ),
        )

    if columns is None:
        candidates: list[str] = []
        skipped: list[TargetScanEntry] = []
        order: list[str] = []
        seen_duplicates: set = set()
        for col in data.columns:
            if col == target:
                continue
            if col in duplicated:
                # Emit one skip entry per duplicated name, not per occurrence.
                if col not in seen_duplicates:
                    seen_duplicates.add(col)
                    skipped.append(_duplicate_skip(col))
                    order.append(col)
                continue
            if is_real_numeric_dtype(data[col]):
                candidates.append(col)
                order.append(col)
        return candidates, skipped, order

    candidates = []
    skipped = []
    order = []
    seen: set[str] = set()
    for col in columns:
        # Skip names the caller listed more than once so a column is profiled
        # (and reported) at most once, mirroring the columns=None branch.
        if col in seen:
            continue
        seen.add(col)
        order.append(col)
        if col == target:
            skipped.append(
                TargetScanEntry(
                    column=col,
                    status="skipped",
                    error_type="TargetExcluded",
                    error_message="Target column cannot be scanned against itself.",
                )
            )
            continue
        if col not in data.columns:
            skipped.append(
                TargetScanEntry(
                    column=col,
                    status="skipped",
                    error_type="ColumnNotFound",
                    error_message=f"Column '{col}' not found in data.",
                )
            )
            continue
        if col in duplicated:
            skipped.append(_duplicate_skip(col))
            continue
        problem = real_numeric_problem(data[col], f"Column '{col}'", context="scanning")
        if problem is not None:
            skipped.append(
                TargetScanEntry(
                    column=col,
                    status="skipped",
                    error_type=problem.error_type,
                    error_message=problem.message,
                )
            )
            continue
        candidates.append(col)
    return candidates, skipped, order


def scan_target(
    data: pd.DataFrame,
    target: str,
    *,
    columns: Sequence[str] | None = None,
    mode: str = "lite",
    missing: str = "pairwise",
    errors: str = "warn",
    direction: str = "forward",
    max_pairs: int | None = None,
    sample_size: int | None = None,
    progress: bool = False,
    random_state: int = 42,
    **profile_pair_kwargs: Any,
) -> CorrSleuthTargetReport:
    """Profile every eligible real-valued numeric predictor against ``target``.

    Parameters
    ----------
    data : pd.DataFrame
        Source data. Must contain ``target``.
    target : str
        Real-valued numeric target column, profiled against every other
        real-valued numeric column. A complex-dtype target raises
        :class:`InputError` (cast to the real part or magnitude first).
    columns : sequence of str, optional
        Restrict the scan to these columns. Non-numeric, complex, or missing
        names are recorded as ``skipped`` entries rather than raising. When
        ``None`` (default), all real-valued numeric columns except the target
        are scanned; complex columns are excluded.
    mode : {"lite", "standard", "deep"}, default "lite"
        Forwarded to :func:`profile_pair`.
    missing : {"pairwise", "listwise", "raise"}, default "pairwise"
        Forwarded to :func:`profile_pair`. Note ``"listwise"`` is complete-case
        across *all* columns of ``data``, so every pair is profiled on the same
        rows (those with no missing value anywhere); ``"pairwise"`` instead uses
        each candidate's own non-missing overlap with the target.
    errors : {"warn", "raise"}, default "warn"
        ``"warn"`` captures per-column ``profile_pair`` exceptions as
        ``error`` entries so the scan continues. ``"raise"`` propagates the
        first exception. Systemic, configuration-level failures — a missing
        optional dependency (``OptionalDependencyError`` from ``mode="standard"``
        / ``"deep"`` without the extras) or a misspelled ``profile_pair`` keyword
        (``TypeError``) — are propagated even under ``"warn"``, so one config
        mistake surfaces once rather than as N identical errors. Shared
        (column-independent) option values — a bad ``mode``, ``missing``,
        ``max_n_for_dcor``, or bootstrap option — are validated once before the
        scan starts and raise ``InputError`` immediately, for the same reason.
        Genuine per-column data failures (e.g. an all-NaN or constant column,
        which raise ``InputError``) remain captured.
    direction : {"forward", "reverse", "both"}, default "forward"
        Which orientation each pair is profiled in. The primary association
        metrics (Pearson/Spearman/Kendall/dCor/MI) are symmetric and identical
        either way; only the **shape** diagnostics (mean shape, curvature,
        oscillation, variance, segmentation) depend on direction.

        - ``"forward"`` — ``profile_pair(candidate, target)``, describing
          ``E[target | candidate]``: the feature-screening question, "does this
          predictor drive the target?"
        - ``"reverse"`` — ``profile_pair(target, candidate)``, describing
          ``E[candidate | target]``: "is this candidate a function of the target?"
          Useful when the data was engineered as ``candidate = f(target)``, where
          the shape only shows in this orientation.
        - ``"both"`` — profiles forward (the primary result) *and* reverse, and
          adds a "Shape differs by direction" report section flagging candidates
          whose reverse shape is structured (nonlinear) while their forward shape
          is not — the signature of ``candidate = f(target)``. Costs two
          ``profile_pair`` calls per candidate.
    max_pairs : int, optional
        Cap on the number of columns profiled. Applied after ``columns=``.
        Must be a positive integer when provided. Columns beyond the cap are
        recorded as ``skipped`` entries (``error_type="MaxPairsExceeded"``) so
        the report reflects that coverage was truncated.
    sample_size : int, optional
        If set and ``len(data) > sample_size``, downsample once with
        ``random_state`` before scanning. Skipped/errored entries still reflect
        the original column list. Must be a positive integer when provided.
    progress : bool, default False
        When True and ``tqdm`` is installed, wrap the iteration with a progress
        bar. Without ``tqdm``, this is a documented no-op. Install via the
        ``progress`` extra (``pip install corrsleuth[progress]``).
    random_state : int, default 42
        Seed for ``sample_size`` downsampling and forwarded to ``profile_pair``.
    **profile_pair_kwargs
        Additional keyword arguments forwarded to :func:`profile_pair` (for
        example, ``bootstrap``, ``bootstrap_metrics``, ``include_caveat``).

    Notes
    -----
    **No multiple-testing correction.** The scan profiles every candidate
    independently and applies no family-wise or FDR adjustment, so across many
    columns some variables will surface in the pattern/underrate sections purely
    by chance. Treat the rankings as *hypothesis-generating*, not as confirmed
    findings, and validate promising candidates with a targeted analysis.

    The scan is **sequential**: columns are profiled one at a time. This is fine
    for typical EDA, but for very wide DataFrames (hundreds/thousands of
    columns) combined with ``mode="deep"`` or ``bootstrap=...`` it can be slow.
    Subset with ``columns=`` / ``max_pairs=`` or downsample with
    ``sample_size=`` to bound the cost. See the performance note in the
    interpretation guide.

    Returns
    -------
    CorrSleuthTargetReport
        Aggregated per-column results. Inspect via ``to_frame()``,
        ``summary()``, ``to_markdown()``, ``pearson_underrated()``, and
        ``plot_top()``.
    """
    # Imported here (not at module top) to avoid a core <-> report import cycle:
    # report imports TargetScanEntry/metrics_map from this module.
    from corrsleuth.scan.report import CorrSleuthTargetReport

    if errors not in _VALID_ERRORS_POLICIES:
        raise InputError(
            f"Unknown errors policy: '{errors}'. Supported policies are "
            f"{_VALID_ERRORS_POLICIES}."
        )
    if direction not in _VALID_DIRECTIONS:
        raise InputError(
            f"Unknown direction: '{direction}'. Supported directions are "
            f"{_VALID_DIRECTIONS}."
        )
    if max_pairs is not None and (
        isinstance(max_pairs, bool) or not isinstance(max_pairs, int) or max_pairs < 1
    ):
        raise InputError("max_pairs must be a positive integer or None.")
    if sample_size is not None and (
        isinstance(sample_size, bool)
        or not isinstance(sample_size, int)
        or sample_size < 1
    ):
        raise InputError("sample_size must be a positive integer or None.")
    if target not in data.columns:
        raise InputError(f"Target column '{target}' not found in data.")
    if isinstance(data[target], pd.DataFrame):
        raise InputError(
            f"Target column '{target}' matches multiple columns in data; "
            f"column names must be unique."
        )
    target_problem = real_numeric_problem(
        data[target], f"Target column '{target}'", context="scanning"
    )
    if target_problem is not None:
        raise InputError(target_problem.message)

    if isinstance(columns, str):
        columns = [columns]

    # Preflight the shared (column-independent) profile_pair configuration so a
    # bad mode / missing policy / max_n_for_dcor / bootstrap option raises once,
    # here — under errors="warn" the per-column loop captures InputError as a
    # column failure, which would otherwise turn one config mistake into N
    # identical error entries and zero successes.
    _validate_profile_pair_options(
        mode=mode,
        missing=missing,
        **{
            key: profile_pair_kwargs[key]
            for key in (
                "max_n_for_dcor",
                "bootstrap",
                "bootstrap_metrics",
                "max_n_for_bootstrap",
            )
            if key in profile_pair_kwargs
        },
    )

    if sample_size is not None and len(data) > sample_size:
        data = data.sample(n=sample_size, random_state=random_state)

    candidates, pre_skipped, resolution_order = _resolve_candidate_columns(
        data, target, columns
    )
    if max_pairs is not None and len(candidates) > max_pairs:
        # Record the columns dropped by the cap as explicit skips rather than
        # letting them vanish — otherwise summary() reports "profiled: N,
        # skipped: 0" on a wider frame as if coverage were complete, and *which*
        # columns were dropped (data-order-dependent) goes unrecorded.
        for col in candidates[max_pairs:]:
            pre_skipped.append(
                TargetScanEntry(
                    column=col,
                    status="skipped",
                    error_type="MaxPairsExceeded",
                    error_message=(
                        f"Not profiled: candidate count exceeded max_pairs={max_pairs}."
                    ),
                )
            )
        candidates = candidates[:max_pairs]

    want_forward = direction in ("forward", "both")
    want_reverse = direction in ("reverse", "both")

    def _profile(x: str, y: str) -> CorrSleuthResult:
        # Direction-sensitive diagnostics (bin lack-of-fit, variance shape,
        # segmentation/breakpoint_x, Cook's influence, forward Chatterjee's xi)
        # describe E[y | x]; the symmetric metrics (Pearson/Spearman/Kendall/
        # dCor/MI/sq_corr) are unaffected by the argument order.
        return profile_pair(
            data,
            x,
            y,
            mode=mode,
            missing=missing,
            random_state=random_state,
            **profile_pair_kwargs,
        )

    # Collect per-column outcomes keyed by name, then emit them in resolution
    # order — extending a pre_skipped list with profiled entries would group
    # every skip before every success, losing the caller's requested order
    # (columns=["a", "missing", "b"] must come back a, missing, b).
    entry_by_column: dict[str, TargetScanEntry] = {e.column: e for e in pre_skipped}
    for col in _iter_with_progress(candidates, progress):
        try:
            # Forward = profile_pair(candidate, target) → E[target | candidate]
            # (feature screening). Reverse = profile_pair(target, candidate) →
            # E[candidate | target] (is the candidate a function of the target?).
            forward = _profile(col, target) if want_forward else None
            reverse = _profile(target, col) if want_reverse else None
        except _CONFIG_CLASS_EXCEPTIONS:
            # Systemic, not per-column (see _CONFIG_CLASS_EXCEPTIONS): propagate
            # regardless of the errors policy so the actionable install/config
            # hint is not buried under N identical entries and zero successes.
            raise
        except Exception as exc:
            if errors == "raise":
                raise
            entry_by_column[col] = TargetScanEntry(
                column=col,
                status="error",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            continue
        # For "reverse" the reverse profile IS the primary result; for "forward"/
        # "both" the forward profile is primary and the reverse (if any) rides
        # alongside for the shape-comparison section.
        primary = reverse if direction == "reverse" else forward
        entry_by_column[col] = TargetScanEntry(
            column=col,
            status="ok",
            result=primary,
            reverse_result=reverse if direction == "both" else None,
        )

    entries = [entry_by_column[col] for col in resolution_order]
    return CorrSleuthTargetReport(target=target, entries=entries, direction=direction)
