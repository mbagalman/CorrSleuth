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
from corrsleuth.utils.markdown import (
    escape_markdown_cell,
    format_markdown_value,
    markdown_table,
)


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

_SUMMARY_CAVEAT_BODY = (
    "Pairwise association does not imply causation or predictive "
    "usefulness by itself. Always inspect the diagnostic plots and validate "
    "with proper analysis."
)
_SUMMARY_CAVEAT = f"Caveat: {_SUMMARY_CAVEAT_BODY}"

#: Valid ``sort_by`` keys for :meth:`CorrSleuthTargetReport.plot_top`. Metric
#: names are sorted by absolute value descending; ``disagreement_score`` is
#: sorted by raw value descending.
_VALID_SORT_KEYS: tuple[str, ...] = (
    "disagreement_score",
    "pearson",
    "spearman",
    "kendall_tau_b",
    "distance_correlation",
    "mutual_information",
    "pearson_trimmed_1pct",
    "pearson_winsorized_1pct",
    "biweight_midcorrelation",
    "pearson_median_clipped_20pct",
    "chatterjee_xi",
    "chatterjee_xi_reverse",
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

        Parameters
        ----------
        top_n : int, default 5
            Per-section cap on the number of entries displayed. Must be a
            positive integer.
        include_caveat : bool, default True
            Append the non-causal caveat line.

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
        if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
            raise InputError("top_n must be a positive integer.")

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

    def to_markdown(self, top_n: int = 5, include_caveat: bool = True) -> str:
        """Return a deterministic Markdown report for a target scan."""
        if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
            raise InputError("top_n must be a positive integer.")

        lines = [
            f"# CorrSleuth Target Report: `{self.target}`",
            "",
            "## Overview",
            markdown_table(
                ["Profiled", "Errored", "Skipped"],
                [
                    [
                        len(self.successes),
                        sum(1 for e in self.entries if e.status == "error"),
                        sum(1 for e in self.entries if e.status == "skipped"),
                    ]
                ],
            ),
        ]

        for pattern, title in _PATTERN_SECTIONS:
            section_entries = [e for e in self.successes if e.result.pattern == pattern]
            if not section_entries:
                continue
            section_entries.sort(
                key=lambda e: (-e.result.disagreement_score, e.column)
            )
            lines.extend(["", f"## {title}"])
            lines.append(self._markdown_entries_table(section_entries[:top_n]))

        listed_patterns = {pattern for pattern, _ in _PATTERN_SECTIONS}
        other_entries = [
            e for e in self.successes if e.result.pattern not in listed_patterns
        ]
        if other_entries:
            other_entries.sort(
                key=lambda e: (-e.result.disagreement_score, e.column)
            )
            lines.extend(["", "## Other or inconclusive"])
            lines.append(self._markdown_entries_table(other_entries[:top_n]))

        underrate = [e for e in self.successes if self._is_pearson_underrate(e)]
        if underrate:
            underrate.sort(
                key=lambda e: (-self._pearson_underrate_gap(e), e.column)
            )
            lines.extend(["", "## Variables Pearson may underrate"])
            lines.append(
                markdown_table(
                    ["Variable", "Pattern", "Gap", "Pearson", "Spearman"],
                    [
                        [
                            entry.column,
                            entry.result.pattern,
                            format_markdown_value(self._pearson_underrate_gap(entry)),
                            format_markdown_value(
                                self._metrics_map(entry).get("pearson")
                            ),
                            format_markdown_value(
                                self._metrics_map(entry).get("spearman")
                            ),
                        ]
                        for entry in underrate[:top_n]
                    ],
                )
            )

        warned = [e for e in self.successes if self._has_reliability_warning(e)]
        if warned:
            warned.sort(key=lambda e: e.column)
            lines.extend(["", "## Variables with missingness or tie warnings"])
            lines.append(
                markdown_table(
                    ["Variable", "Warning"],
                    [
                        [entry.column, self._primary_reliability_warning(entry)]
                        for entry in warned[:top_n]
                    ],
                )
            )

        skipped_or_failed = [e for e in self.entries if e.status != "ok"]
        if skipped_or_failed:
            skipped_or_failed.sort(key=lambda e: (e.status, e.column))
            lines.extend(["", "## Skipped or failed"])
            lines.append(
                markdown_table(
                    ["Variable", "Status", "Detail", "Message"],
                    [
                        [
                            entry.column,
                            entry.status,
                            entry.error_type or "unknown",
                            entry.error_message or "",
                        ]
                        for entry in skipped_or_failed[:top_n]
                    ],
                )
            )

        if include_caveat:
            lines.extend(["", "## Caveat", _SUMMARY_CAVEAT_BODY])

        return "\n".join(lines)

    def pearson_underrated(
        self, threshold: float = _PEARSON_UNDERRATE_GAP
    ) -> pd.DataFrame:
        """Return variables where Pearson may understate the relationship.

        The ranking is directional: a variable is included only when rank-based
        metrics or standard-mode nonmonotonic evidence exceed Pearson by more
        than ``threshold``. This keeps outlier/leverage cases, where Pearson is
        stronger than rank metrics, out of the ranking.

        Parameters
        ----------
        threshold : float, default 0.20
            Minimum positive gap required for inclusion. The default matches
            the cross-cutting summary section.

        Returns
        -------
        pandas.DataFrame
            One row per included variable, sorted by strongest evidence. The
            frame includes metric values, directional gap values, the primary
            diagnostic pattern, disagreement score, and warnings. The
            ``nonmonotonic_gap`` column stores the raw diagnostic value; only
            its positive portion contributes to ``pearson_underrate_score``.
        """
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or pd.isna(threshold)
            or threshold < 0
        ):
            raise InputError("threshold must be a non-negative number.")

        columns = [
            "variable",
            "target",
            "pattern",
            "pearson_underrate_score",
            "spearman_excess_over_pearson",
            "kendall_excess_over_pearson",
            "nonmonotonic_gap",
            "disagreement_score",
            "metric_pearson",
            "metric_spearman",
            "metric_kendall_tau_b",
            "metric_distance_correlation",
            "metric_mutual_information",
            "warnings",
        ]

        rows: List[Dict[str, Any]] = []
        for entry in self.successes:
            metrics = self._metrics_map(entry)
            components = self._directional_underrate_components(entry)
            score = components["score"]

            if score <= threshold:
                continue

            rows.append(
                {
                    "variable": entry.column,
                    "target": self.target,
                    "pattern": entry.result.pattern,
                    "pearson_underrate_score": score,
                    "spearman_excess_over_pearson": components[
                        "spearman_excess_over_pearson"
                    ],
                    "kendall_excess_over_pearson": components[
                        "kendall_excess_over_pearson"
                    ],
                    "nonmonotonic_gap": components["nonmonotonic_gap"],
                    "disagreement_score": entry.result.disagreement_score,
                    "metric_pearson": metrics.get("pearson"),
                    "metric_spearman": metrics.get("spearman"),
                    "metric_kendall_tau_b": metrics.get("kendall_tau_b"),
                    "metric_distance_correlation": metrics.get("distance_correlation"),
                    "metric_mutual_information": metrics.get("mutual_information"),
                    "warnings": "; ".join(entry.result.warnings)
                    if entry.result.warnings
                    else "",
                }
            )

        frame = pd.DataFrame(rows, columns=columns)
        if not frame.empty:
            frame = frame.sort_values(
                by=["pearson_underrate_score", "disagreement_score", "variable"],
                ascending=[False, False, True],
                kind="mergesort",
            ).reset_index(drop=True)
        return frame

    @staticmethod
    def _format_pattern_entry(entry: TargetScanEntry) -> str:
        result = entry.result
        metrics = CorrSleuthTargetReport._metrics_map(entry)

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
    def _markdown_entries_table(entries: List[TargetScanEntry]) -> str:
        rows = []
        for entry in entries:
            metrics = CorrSleuthTargetReport._metrics_map(entry)
            rows.append(
                [
                    entry.column,
                    entry.result.pattern,
                    format_markdown_value(metrics.get("pearson")),
                    format_markdown_value(metrics.get("spearman")),
                    format_markdown_value(entry.result.disagreement_score),
                    "; ".join(escape_markdown_cell(w) for w in entry.result.warnings)
                    if entry.result.warnings
                    else "",
                ]
            )
        return markdown_table(
            ["Variable", "Pattern", "Pearson", "Spearman", "Disagreement", "Warnings"],
            rows,
        )

    @staticmethod
    def _metrics_map(entry: TargetScanEntry) -> Dict[str, Any]:
        return {
            row["metric"]: row["value"]
            for _, row in entry.result.metrics.iterrows()
        }

    @staticmethod
    def _positive_abs_gap(value: Any, baseline_abs: Optional[float]) -> float:
        if baseline_abs is None or value is None or pd.isna(value):
            return 0.0
        return max(0.0, abs(float(value)) - baseline_abs)

    @classmethod
    def _directional_underrate_components(
        cls, entry: TargetScanEntry
    ) -> Dict[str, Optional[float]]:
        metrics = cls._metrics_map(entry)
        pearson = metrics.get("pearson")
        abs_p = (
            abs(pearson)
            if pearson is not None and not pd.isna(pearson)
            else None
        )

        spearman_gap = cls._positive_abs_gap(metrics.get("spearman"), abs_p)
        kendall_gap = cls._positive_abs_gap(metrics.get("kendall_tau_b"), abs_p)

        raw_nonmonotonic_gap = entry.result.diagnostics.nonmonotonic_gap
        nonmonotonic_gap = (
            None
            if raw_nonmonotonic_gap is None or pd.isna(raw_nonmonotonic_gap)
            else float(raw_nonmonotonic_gap)
        )
        nonmonotonic_contribution = (
            max(0.0, nonmonotonic_gap) if nonmonotonic_gap is not None else 0.0
        )

        score = max(spearman_gap, kendall_gap, nonmonotonic_contribution)
        return {
            "score": score,
            "spearman_excess_over_pearson": spearman_gap,
            "kendall_excess_over_pearson": kendall_gap,
            "nonmonotonic_gap": nonmonotonic_gap,
        }

    @classmethod
    def _directional_underrate_score(cls, entry: TargetScanEntry) -> float:
        return float(cls._directional_underrate_components(entry)["score"])

    @staticmethod
    def _pearson_underrate_gap(entry: TargetScanEntry) -> float:
        """Directional gap: positive only when rank/dCor metrics exceed Pearson.

        ``rank_linear_gap`` is symmetric (``abs(abs(p) - abs(s))``), so it would
        treat ``Pearson >> Spearman`` (often outlier-driven) the same as
        ``Spearman >> Pearson``. This helper delegates to the same clamped
        directional score used by :meth:`pearson_underrated`, so leverage-driven
        entries do not surface here.
        """
        return CorrSleuthTargetReport._directional_underrate_score(entry)

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

    def plot_top(
        self,
        n: int = 12,
        sort_by: str = "disagreement_score",
        patterns: Optional[Sequence[str]] = None,
        ncols: int = 3,
        figsize: Optional[tuple[float, float]] = None,
        show: bool = False,
    ) -> Any:
        """Return a Matplotlib ``Figure`` with up to ``n`` scatter panels.

        Each panel shows the raw scatter for one inspected pair, titled with
        the candidate column name, the assigned diagnostic pattern, and key
        metric values. Empty axes are hidden when fewer than
        ``ncols * ceil(n/ncols)`` panels are filled.

        Parameters
        ----------
        n : int, default 12
            Maximum number of panels. Must be a positive integer.
        sort_by : str, default ``"disagreement_score"``
            Either ``"disagreement_score"`` (sorted by raw value descending) or
            a metric name (``"pearson"``, ``"spearman"``, ``"kendall_tau_b"``,
            ``"distance_correlation"``, ``"mutual_information"``, or a
            deep-mode robust metric); metric keys are sorted by absolute value
            descending.
        patterns : sequence of str, optional
            Only include entries whose ``pattern`` is in this set. A single
            string is normalized to a one-element list. ``None`` (default)
            includes all successful entries.
        ncols : int, default 3
            Panels per row. Must be a positive integer.
        figsize : tuple of (float, float), optional
            Figure size in inches. Defaults to ``(4*ncols, 3*nrows)``.
        show : bool, default False
            If True, call ``plt.show()`` before returning the figure.

        Returns
        -------
        matplotlib.figure.Figure
            A figure even when no variables match — in that case it contains a
            single placeholder axis with a "No variables to plot" message.
        """
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise InputError("n must be a positive integer.")
        if isinstance(ncols, bool) or not isinstance(ncols, int) or ncols < 1:
            raise InputError("ncols must be a positive integer.")
        if sort_by not in _VALID_SORT_KEYS:
            raise InputError(
                f"Unknown sort_by: {sort_by!r}. Supported values are "
                f"{_VALID_SORT_KEYS}."
            )

        if isinstance(patterns, str):
            patterns = [patterns]

        candidates = [
            e for e in self.successes
            if e.result._clean_x is not None and e.result._clean_y is not None
        ]
        if patterns is not None:
            pattern_set = set(patterns)
            candidates = [e for e in candidates if e.result.pattern in pattern_set]

        candidates.sort(key=lambda e: (-self._sort_value(e, sort_by), e.column))
        candidates = candidates[:n]

        import math

        import matplotlib.pyplot as plt

        if not candidates:
            fig, ax = plt.subplots(figsize=figsize or (6, 4))
            ax.text(
                0.5, 0.5,
                "No variables to plot.",
                ha="center", va="center",
                transform=ax.transAxes,
                fontsize=12, color="dimgray",
            )
            ax.set_axis_off()
            fig.suptitle(
                f"Target scan: {self.target}",
                fontsize=12, fontweight="bold",
            )
            if show:
                plt.show()
            return fig

        nrows = math.ceil(len(candidates) / ncols)
        if figsize is None:
            figsize = (4 * ncols, 3 * nrows)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        flat_axes = axes.flatten()

        for ax, entry in zip(flat_axes, candidates):
            result = entry.result
            # scan_target() calls profile_pair(data, target, col), so
            # _clean_x holds the target's data and _clean_y holds the
            # candidate's. EDA convention puts the candidate (predictor) on x
            # and the target on y, so swap here.
            target_data = result._clean_x.values
            candidate_data = result._clean_y.values
            n_pts = len(candidate_data)

            if n_pts > 5000:
                ax.hexbin(
                    candidate_data, target_data,
                    gridsize=30, cmap="Blues", mincnt=1,
                )
            else:
                alpha = min(1.0, 100 / n_pts) if n_pts > 0 else 1.0
                ax.scatter(
                    candidate_data, target_data,
                    alpha=alpha, edgecolor="none",
                    color="steelblue", s=8,
                )

            ax.set_title(self._panel_title(entry), fontsize=9)
            ax.set_xlabel(entry.column, fontsize=8)
            ax.set_ylabel(self.target, fontsize=8)
            ax.tick_params(labelsize=7)

        for ax in flat_axes[len(candidates):]:
            ax.set_axis_off()

        fig.suptitle(
            f"Target scan: {self.target} (top by {sort_by})",
            fontsize=12, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        if show:
            plt.show()

        return fig

    @staticmethod
    def _sort_value(entry: TargetScanEntry, sort_by: str) -> float:
        if sort_by == "disagreement_score":
            return float(entry.result.disagreement_score)
        metrics = CorrSleuthTargetReport._metrics_map(entry)
        value = metrics.get(sort_by)
        if value is None or pd.isna(value):
            return 0.0
        return abs(float(value))

    @staticmethod
    def _panel_title(entry: TargetScanEntry) -> str:
        metrics = CorrSleuthTargetReport._metrics_map(entry)
        pearson = metrics.get("pearson")
        spearman = metrics.get("spearman")

        def _fmt(value: Any) -> str:
            if value is None or pd.isna(value):
                return "NA"
            return f"{value:.2f}"

        return (
            f"{entry.column}\n"
            f"{entry.result.pattern} | "
            f"p={_fmt(pearson)} s={_fmt(spearman)}"
        )


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
    if max_pairs is not None and (
        isinstance(max_pairs, bool)
        or not isinstance(max_pairs, int)
        or max_pairs < 1
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
