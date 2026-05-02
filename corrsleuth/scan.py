"""Target-oriented scanning workflow.

Profiles every eligible numeric predictor in a DataFrame against a single
numeric target column, returning a report that can be inspected as a tidy
DataFrame. Per-column failures are captured rather than aborting the scan
unless the caller asks for ``errors="raise"``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from corrsleuth.api import profile_pair
from corrsleuth.exceptions import InputError
from corrsleuth.result import CorrSleuthResult


_VALID_ERRORS_POLICIES = ("warn", "raise")

_STATIC_FRAME_COLUMNS = (
    "variable",
    "target",
    "status",
    "error_type",
    "error_message",
    "pattern",
    "disagreement_score",
    "warnings",
    "recommendations",
)

_DEFAULT_METRIC_COLUMNS = (
    "metric_pearson",
    "metric_spearman",
    "metric_kendall_tau_b",
)

#: Pattern → summary section title, in display order. Patterns not listed here
#: (``low_power_or_uncertain``, ``mixed_or_ambiguous``, ``not_computable``) fall
#: through to the catch-all "Other or inconclusive" section so no profiled
#: variable disappears from the summary.
_PATTERN_SECTIONS: tuple[tuple[str, str], ...] = (
    ("near_linear", "Strongest near-linear relationships"),
    ("monotonic_nonlinear", "Potential monotonic nonlinear relationships"),
    ("nonmonotonic_dependence", "Potential nonmonotonic relationships"),
    ("possible_outlier_or_leverage", "Possible outlier-driven relationships"),
    ("weak_or_no_relationship", "Weak or no pairwise relationships"),
)

#: Threshold above which `rank_linear_gap` or `nonmonotonic_gap` qualifies a
#: variable for the cross-cutting "Pearson may underrate" section.
_PEARSON_UNDERRATE_GAP = 0.20

#: Substrings used to surface a variable in the "missingness or tie warnings"
#: cross-cutting section. Matched against `result.warnings` text emitted by
#: validate_pair / profile_pair.
_RELIABILITY_WARNING_KEYWORDS: tuple[str, ...] = (
    "tie rate",
    "Low unique value ratio",
    "missing data",
    "Small sample size",
    "constant",
)

_SUMMARY_CAVEAT = (
    "Caveat: Pairwise association does not imply causation or predictive "
    "usefulness by itself. Always inspect the diagnostic plots and validate "
    "with proper analysis."
)


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
    result: Optional[CorrSleuthResult] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None


class CorrSleuthTargetReport:
    """Aggregate output of :func:`scan_target`.

    Stores one :class:`TargetScanEntry` per inspected column. Use
    :meth:`to_frame` for downstream pandas workflows or :meth:`summary` for
    a quick text overview.
    """

    def __init__(self, target: str, entries: List[TargetScanEntry]) -> None:
        self.target = target
        self.entries = list(entries)

    @property
    def successes(self) -> List[TargetScanEntry]:
        return [e for e in self.entries if e.status == "ok"]

    @property
    def failures(self) -> List[TargetScanEntry]:
        return [e for e in self.entries if e.status != "ok"]

    def to_frame(self) -> pd.DataFrame:
        """Return one row per inspected column.

        The frame always includes the documented static columns (``variable``,
        ``target``, ``status``, ``error_type``, ``error_message``, ``pattern``,
        ``disagreement_score``, ``warnings``, ``recommendations``) plus the
        lite metric columns (``metric_pearson``, ``metric_spearman``,
        ``metric_kendall_tau_b``). Additional metric columns are appended when
        any successful row produced them. Skipped/errored rows leave the
        result-dependent fields NaN and populate ``error_type`` /
        ``error_message`` instead.
        """
        metric_columns: List[str] = list(_DEFAULT_METRIC_COLUMNS)
        for entry in self.successes:
            for metric_name in entry.result.metrics["metric"]:
                col = f"metric_{metric_name}"
                if col not in metric_columns:
                    metric_columns.append(col)
        all_columns = list(_STATIC_FRAME_COLUMNS) + metric_columns

        rows: List[Dict[str, Any]] = []
        for entry in self.entries:
            row: Dict[str, Any] = {col: None for col in all_columns}
            row["variable"] = entry.column
            row["target"] = self.target
            row["status"] = entry.status
            row["error_type"] = entry.error_type
            row["error_message"] = entry.error_message
            res = entry.result
            if res is not None:
                row["pattern"] = res.pattern
                row["disagreement_score"] = res.disagreement_score
                row["warnings"] = "; ".join(res.warnings) if res.warnings else ""
                row["recommendations"] = (
                    "; ".join(res.recommendations) if res.recommendations else ""
                )
                for _, metric_row in res.metrics.iterrows():
                    row[f"metric_{metric_row['metric']}"] = metric_row["value"]
            rows.append(row)
        return pd.DataFrame(rows, columns=all_columns)

    def summary(self, top_n: int = 5, include_caveat: bool = True) -> str:
        """Return a section-structured overview of the scan outcome.

        Sections are emitted in this fixed order, each capped at ``top_n``:

        1. Pattern sections (``near_linear``, ``monotonic_nonlinear``,
           ``nonmonotonic_dependence``, ``possible_outlier_or_leverage``,
           ``weak_or_no_relationship``).
        2. ``Other or inconclusive`` — variables with patterns outside the
           explicit set (e.g., ``low_power_or_uncertain``).
        3. ``Variables Pearson may underrate`` — cross-cutting; entries whose
           ``rank_linear_gap`` or ``nonmonotonic_gap`` exceeds 0.20.
        4. ``Variables with missingness or tie warnings`` — cross-cutting;
           entries whose ``warnings`` mention ties, missing data, low unique
           ratio, small samples, or constant inputs.
        5. ``Skipped or failed`` — non-numeric / missing columns and per-column
           profile failures captured under ``errors="warn"``.

        Within each section, entries are sorted by ``disagreement_score``
        descending. Empty sections are omitted. The caveat is appended unless
        ``include_caveat=False``.
        """
        lines = [
            f"Target scan: {self.target}",
            f"  profiled : {len(self.successes)}",
            f"  errored  : {sum(1 for e in self.entries if e.status == 'error')}",
            f"  skipped  : {sum(1 for e in self.entries if e.status == 'skipped')}",
        ]

        for pattern, title in _PATTERN_SECTIONS:
            section_entries = [e for e in self.successes if e.result.pattern == pattern]
            if not section_entries:
                continue
            section_entries.sort(
                key=lambda e: (-e.result.disagreement_score, e.column)
            )
            lines.extend(["", f"{title}:"])
            for entry in section_entries[:top_n]:
                lines.append(f"  {self._format_pattern_entry(entry)}")

        listed_patterns = {pattern for pattern, _ in _PATTERN_SECTIONS}
        other_entries = [
            e for e in self.successes if e.result.pattern not in listed_patterns
        ]
        if other_entries:
            other_entries.sort(
                key=lambda e: (-e.result.disagreement_score, e.column)
            )
            lines.extend(["", "Other or inconclusive:"])
            for entry in other_entries[:top_n]:
                lines.append(
                    f"  {entry.column} ({entry.result.pattern}, "
                    f"disagreement={entry.result.disagreement_score:.2f})"
                )

        underrate = [e for e in self.successes if self._is_pearson_underrate(e)]
        if underrate:
            underrate.sort(
                key=lambda e: (-self._pearson_underrate_gap(e), e.column)
            )
            lines.extend(["", "Variables Pearson may underrate:"])
            for entry in underrate[:top_n]:
                gap = self._pearson_underrate_gap(entry)
                lines.append(f"  {entry.column} (gap={gap:.2f})")

        warned = [e for e in self.successes if self._has_reliability_warning(e)]
        if warned:
            warned.sort(key=lambda e: e.column)
            lines.extend(["", "Variables with missingness or tie warnings:"])
            for entry in warned[:top_n]:
                lines.append(
                    f"  {entry.column}: "
                    f"{self._primary_reliability_warning(entry)}"
                )

        skipped_or_failed = [e for e in self.entries if e.status != "ok"]
        if skipped_or_failed:
            skipped_or_failed.sort(key=lambda e: (e.status, e.column))
            lines.extend(["", "Skipped or failed:"])
            for entry in skipped_or_failed[:top_n]:
                detail = entry.error_type or "unknown"
                lines.append(f"  {entry.column} ({entry.status}: {detail})")

        if include_caveat:
            lines.extend(["", _SUMMARY_CAVEAT])

        return "\n".join(lines)

    @staticmethod
    def _format_pattern_entry(entry: TargetScanEntry) -> str:
        result = entry.result
        metrics = {
            row["metric"]: row["value"] for _, row in result.metrics.iterrows()
        }

        def _fmt(value: Any) -> str:
            if value is None or pd.isna(value):
                return "NA"
            return f"{value:.2f}"

        return (
            f"{entry.column} "
            f"(pearson={_fmt(metrics.get('pearson'))}, "
            f"spearman={_fmt(metrics.get('spearman'))}, "
            f"disagreement={result.disagreement_score:.2f})"
        )

    @staticmethod
    def _pearson_underrate_gap(entry: TargetScanEntry) -> float:
        diag = entry.result.diagnostics
        candidates = []
        if diag.rank_linear_gap is not None:
            candidates.append(diag.rank_linear_gap)
        if diag.nonmonotonic_gap is not None:
            candidates.append(diag.nonmonotonic_gap)
        return max(candidates) if candidates else 0.0

    @classmethod
    def _is_pearson_underrate(cls, entry: TargetScanEntry) -> bool:
        return cls._pearson_underrate_gap(entry) > _PEARSON_UNDERRATE_GAP

    @staticmethod
    def _has_reliability_warning(entry: TargetScanEntry) -> bool:
        return any(
            keyword in warning
            for warning in entry.result.warnings
            for keyword in _RELIABILITY_WARNING_KEYWORDS
        )

    @staticmethod
    def _primary_reliability_warning(entry: TargetScanEntry) -> str:
        for warning in entry.result.warnings:
            if any(k in warning for k in _RELIABILITY_WARNING_KEYWORDS):
                return warning
        return entry.result.warnings[0] if entry.result.warnings else ""


def _iter_with_progress(items: Sequence[str], progress: bool):
    """Yield from ``items``, optionally wrapping with tqdm when available.

    When ``progress=True`` and tqdm is not installed, this is a documented
    no-op rather than printing a homemade progress bar — keeps script and
    test output clean by default.
    """
    if not progress:
        return iter(items)
    try:
        from tqdm import tqdm  # type: ignore[import-not-found]
    except ImportError:
        return iter(items)
    return tqdm(items, desc="scan_target")


def _resolve_candidate_columns(
    data: pd.DataFrame,
    target: str,
    columns: Optional[Sequence[str]],
) -> tuple[List[str], List[TargetScanEntry]]:
    """Return (numeric candidates, pre-skipped entries) honoring ``columns=``.

    If ``columns`` is None, every numeric column except the target is a
    candidate. When the caller passes an explicit list, we preserve their order
    and emit ``status="skipped"`` entries for missing or non-numeric names so
    the report still reflects the request.
    """
    if columns is None:
        candidates = [
            col
            for col in data.columns
            if col != target and pd.api.types.is_numeric_dtype(data[col])
        ]
        return candidates, []

    candidates: List[str] = []
    skipped: List[TargetScanEntry] = []
    for col in columns:
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
    columns: Optional[Sequence[str]] = None,
    mode: str = "lite",
    missing: str = "pairwise",
    errors: str = "warn",
    max_pairs: Optional[int] = None,
    sample_size: Optional[int] = None,
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
    mode : {"lite", "standard"}, default "lite"
        Forwarded to :func:`profile_pair`.
    missing : {"pairwise", "listwise", "raise"}, default "pairwise"
        Forwarded to :func:`profile_pair`.
    errors : {"warn", "raise"}, default "warn"
        ``"warn"`` captures per-column ``profile_pair`` exceptions as
        ``error`` entries so the scan continues. ``"raise"`` propagates the
        first exception.
    max_pairs : int, optional
        Cap on the number of columns profiled. Applied after ``columns=``.
    sample_size : int, optional
        If set and ``len(data) > sample_size``, downsample once with
        ``random_state`` before scanning. Skipped/errored entries still reflect
        the original column list.
    progress : bool, default False
        When True and ``tqdm`` is installed, wrap the iteration with a progress
        bar. Without ``tqdm``, this is a documented no-op.
    random_state : int, default 42
        Seed for ``sample_size`` downsampling and forwarded to ``profile_pair``.
    **profile_pair_kwargs
        Additional keyword arguments forwarded to :func:`profile_pair` (for
        example, ``bootstrap``, ``bootstrap_metrics``, ``include_caveat``).

    Returns
    -------
    CorrSleuthTargetReport
        Aggregated per-column results with ``to_frame()`` and ``summary()``.
    """
    if errors not in _VALID_ERRORS_POLICIES:
        raise InputError(
            f"Unknown errors policy: '{errors}'. Supported policies are "
            f"{_VALID_ERRORS_POLICIES}."
        )
    if target not in data.columns:
        raise InputError(f"Target column '{target}' not found in data.")
    if not pd.api.types.is_numeric_dtype(data[target]):
        raise InputError(f"Target column '{target}' is not numeric.")

    if isinstance(columns, str):
        columns = [columns]

    if sample_size is not None and len(data) > sample_size:
        data = data.sample(n=sample_size, random_state=random_state)

    candidates, pre_skipped = _resolve_candidate_columns(data, target, columns)
    if max_pairs is not None:
        candidates = candidates[:max_pairs]

    entries: List[TargetScanEntry] = list(pre_skipped)
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
