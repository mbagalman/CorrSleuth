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

from corrsleuth.api import profile_pair
from corrsleuth.exceptions import InputError
from corrsleuth.result import CorrSleuthResult

if TYPE_CHECKING:
    from corrsleuth.scan.report import CorrSleuthTargetReport

_VALID_ERRORS_POLICIES = ("warn", "raise")


@dataclass
class TargetScanEntry:
    """One column's outcome from a target scan.

    ``status="ok"`` entries have a populated ``result``. ``status="skipped"``
    entries describe columns that were filtered out before profiling (for
    example, a non-numeric column the caller listed in ``columns=``).
    ``status="error"`` entries describe profile_pair failures captured under
    ``errors="warn"``.
    """

    column: str
    status: str
    result: CorrSleuthResult | None = None
    error_type: str | None = None
    error_message: str | None = None

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
) -> tuple[list[str], list[TargetScanEntry]]:
    """Return (numeric candidates, pre-skipped entries) honoring ``columns=``.

    If ``columns`` is None, every numeric column except the target is a
    candidate. When the caller passes an explicit list, we preserve their order
    and emit ``status="skipped"`` entries for missing or non-numeric names so
    the report still reflects the request.

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
        seen_duplicates: set = set()
        for col in data.columns:
            if col == target:
                continue
            if col in duplicated:
                # Emit one skip entry per duplicated name, not per occurrence.
                if col not in seen_duplicates:
                    seen_duplicates.add(col)
                    skipped.append(_duplicate_skip(col))
                continue
            if pd.api.types.is_numeric_dtype(data[col]):
                candidates.append(col)
        return candidates, skipped

    candidates = []
    skipped = []
    seen: set[str] = set()
    for col in columns:
        # Skip names the caller listed more than once so a column is profiled
        # (and reported) at most once, mirroring the columns=None branch.
        if col in seen:
            continue
        seen.add(col)
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
        if not pd.api.types.is_numeric_dtype(data[col]):
            skipped.append(
                TargetScanEntry(
                    column=col,
                    status="skipped",
                    error_type="NonNumeric",
                    error_message=f"Column '{col}' is not numeric.",
                )
            )
            continue
        candidates.append(col)
    return candidates, skipped


def scan_target(
    data: pd.DataFrame,
    target: str,
    *,
    columns: Sequence[str] | None = None,
    mode: str = "lite",
    missing: str = "pairwise",
    errors: str = "warn",
    max_pairs: int | None = None,
    sample_size: int | None = None,
    progress: bool = False,
    random_state: int = 42,
    **profile_pair_kwargs: Any,
) -> CorrSleuthTargetReport:
    """Profile every eligible numeric predictor against ``target``.

    Parameters
    ----------
    data : pd.DataFrame
        Source data. Must contain ``target``.
    target : str
        Numeric target column. Profiled against every other numeric column.
    columns : sequence of str, optional
        Restrict the scan to these columns. Non-numeric or missing names are
        recorded as ``skipped`` entries rather than raising. When ``None``
        (default), all numeric columns except the target are scanned.
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
        first exception.
    max_pairs : int, optional
        Cap on the number of columns profiled. Applied after ``columns=``.
        Must be a positive integer when provided.
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
    if not pd.api.types.is_numeric_dtype(data[target]):
        raise InputError(f"Target column '{target}' is not numeric.")

    if isinstance(columns, str):
        columns = [columns]

    if sample_size is not None and len(data) > sample_size:
        data = data.sample(n=sample_size, random_state=random_state)

    candidates, pre_skipped = _resolve_candidate_columns(data, target, columns)
    if max_pairs is not None:
        candidates = candidates[:max_pairs]

    entries: list[TargetScanEntry] = list(pre_skipped)
    for col in _iter_with_progress(candidates, progress):
        try:
            result = profile_pair(
                data,
                target,
                col,
                mode=mode,
                missing=missing,
                random_state=random_state,
                **profile_pair_kwargs,
            )
        except Exception as exc:
            if errors == "raise":
                raise
            entries.append(
                TargetScanEntry(
                    column=col,
                    status="error",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )
            continue
        entries.append(TargetScanEntry(column=col, status="ok", result=result))

    return CorrSleuthTargetReport(target=target, entries=entries)
