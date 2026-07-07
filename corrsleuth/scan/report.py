"""Report object for a target scan and its text/frame rendering.

Holds the per-column :class:`~corrsleuth.scan.core.TargetScanEntry` list and
turns it into a tidy DataFrame (:meth:`CorrSleuthTargetReport.to_frame`), a
section-structured text overview (:meth:`~CorrSleuthTargetReport.summary`),
deterministic Markdown (:meth:`~CorrSleuthTargetReport.to_markdown`), and the
"Pearson may underrate" ranking (:meth:`~CorrSleuthTargetReport.pearson_underrated`).
Plotting is delegated to :mod:`corrsleuth.scan.plot`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import pandas as pd

from corrsleuth.exceptions import InputError
from corrsleuth.result import MetricDiagnostics
from corrsleuth.scan.core import TargetScanEntry, metrics_map
from corrsleuth.utils.markdown import (
    escape_markdown_code_span,
    format_markdown_value,
    markdown_table,
)

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

#: Threshold by which rank (Spearman/Kendall) or nonmonotonic evidence must
#: exceed Pearson for a variable to qualify for the cross-cutting "Pearson may
#: underrate" section. The gap is directional (see
#: ``_directional_underrate_components``), so leverage cases where Pearson is
#: stronger than the rank metrics do not qualify. Set to the same 0.20 as the
#: cascade's ``RANK_LINEAR_GAP_THRESHOLD`` so the scan-level "underrate" callout
#: and the per-pair monotonic_nonlinear label fire on the same gap size.
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
    "with proper analysis. This scan applies no multiple-testing correction, "
    "so with many candidates some patterns will appear by chance — treat the "
    "rankings as hypothesis-generating."
)
_SUMMARY_CAVEAT = f"Caveat: {_SUMMARY_CAVEAT_BODY}"


class CorrSleuthTargetReport:
    """Aggregate output of :func:`scan_target`.

    Stores one :class:`TargetScanEntry` per inspected column. Use
    :meth:`to_frame` for downstream pandas workflows or :meth:`summary` for
    a quick text overview.
    """

    #: Primary patterns considered "structured" (a real nonlinear relationship
    #: shape) for the reverse-direction comparison — a candidate whose reverse
    #: pattern is here but whose forward pattern is not looks like ``f(target)``.
    _STRUCTURED_PATTERNS = frozenset({"monotonic_nonlinear", "nonmonotonic_dependence"})

    def __init__(
        self,
        target: str,
        entries: list[TargetScanEntry],
        direction: str = "forward",
    ) -> None:
        self.target = target
        self.entries = list(entries)
        self.direction = direction

    @property
    def successes(self) -> list[TargetScanEntry]:
        return [e for e in self.entries if e.status == "ok"]

    @property
    def failures(self) -> list[TargetScanEntry]:
        return [e for e in self.entries if e.status != "ok"]

    def _reverse_reveals_shape(self, entry: TargetScanEntry) -> bool:
        """True when ``direction="both"`` and the reverse orientation shows a
        structured (nonlinear) shape the forward orientation does not — the
        ``candidate = f(target)`` signature the shape lives in the other view."""
        if entry.reverse_result is None or entry.result is None:
            return False
        return (
            entry.reverse_result.pattern in self._STRUCTURED_PATTERNS
            and entry.result.pattern not in self._STRUCTURED_PATTERNS
        )

    def to_frame(self) -> pd.DataFrame:
        """Return one row per inspected column.

        The frame always includes the documented static columns (``variable``,
        ``target``, ``status``, ``error_type``, ``error_message``, ``pattern``,
        ``disagreement_score``, ``warnings``, ``recommendations``), the lite
        metric columns (``metric_pearson``, ``metric_spearman``,
        ``metric_kendall_tau_b``), and one ``diagnostic_<field>`` column for every
        field on :class:`~corrsleuth.result.MetricDiagnostics` — the numeric
        diagnostics *and* the five secondary axes (``diagnostic_mean_shape``,
        ``diagnostic_variance_shape``, ``diagnostic_dependence_type``,
        ``diagnostic_outlier_sensitivity``, ``diagnostic_functional_direction``),
        mirroring :meth:`CorrSleuthResult.to_frame`. Extra ``metric_*`` columns
        are appended when any successful row produced them (standard/deep
        metrics). When bootstrapping was requested, ``pattern_stability`` /
        ``stability_label`` / ``stability_metric_set`` columns are added. When the
        scan ran with ``direction="both"``, ``reverse_pattern`` /
        ``reverse_mean_shape`` / ``reverse_dependence_type`` columns carry the
        reverse-orientation (``E[candidate | target]``) shape.
        Skipped/errored rows leave the result-dependent fields NaN and populate
        ``error_type`` / ``error_message`` instead.
        """
        metric_columns: list[str] = list(_DEFAULT_METRIC_COLUMNS)
        for entry in self.successes:
            for metric_name in entry.result_data.metrics["metric"]:
                col = f"metric_{metric_name}"
                if col not in metric_columns:
                    metric_columns.append(col)

        # One column per MetricDiagnostics field: a stable set, since every
        # profiled pair computes the same diagnostics regardless of mode.
        diagnostic_columns = [
            f"diagnostic_{field.name}"
            for field in dataclasses.fields(MetricDiagnostics)
        ]

        include_stability = any(
            e.result is not None and e.result.bootstrap_stability is not None
            for e in self.entries
        )
        stability_columns = (
            ["pattern_stability", "stability_label", "stability_metric_set"]
            if include_stability
            else []
        )

        # direction="both": the reverse-orientation shape (E[candidate | target])
        # rides alongside the forward primary so a single frame carries both views.
        include_reverse = any(e.reverse_result is not None for e in self.entries)
        reverse_columns = (
            ["reverse_pattern", "reverse_mean_shape", "reverse_dependence_type"]
            if include_reverse
            else []
        )

        all_columns = (
            list(_STATIC_FRAME_COLUMNS)
            + metric_columns
            + diagnostic_columns
            + stability_columns
            + reverse_columns
        )

        rows: list[dict[str, Any]] = []
        for entry in self.entries:
            row: dict[str, Any] = {col: None for col in all_columns}
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
                for key, value in res.diagnostics.to_dict().items():
                    row[f"diagnostic_{key}"] = value
                if include_stability and res.bootstrap_stability is not None:
                    stability = res.bootstrap_stability
                    row["pattern_stability"] = stability.pattern_stability
                    row["stability_label"] = stability.stability_label
                    row["stability_metric_set"] = stability.metric_set
            if include_reverse and entry.reverse_result is not None:
                rev = entry.reverse_result
                row["reverse_pattern"] = rev.pattern
                row["reverse_mean_shape"] = rev.diagnostics.mean_shape
                row["reverse_dependence_type"] = rev.diagnostics.dependence_type
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
        3. ``Variables Pearson may underrate`` — cross-cutting; entries where
           rank (Spearman/Kendall) or nonmonotonic evidence exceeds Pearson by
           more than 0.20. The gap is directional, so leverage cases where
           Pearson is stronger than the rank metrics are excluded.
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
            section_entries = [
                e for e in self.successes if e.result_data.pattern == pattern
            ]
            if not section_entries:
                continue
            section_entries.sort(
                key=lambda e: (-e.result_data.disagreement_score, e.column)
            )
            lines.extend(["", f"{title}:"])
            for entry in section_entries[:top_n]:
                lines.append(f"  {self._format_pattern_entry(entry)}")

        listed_patterns = {pattern for pattern, _ in _PATTERN_SECTIONS}
        other_entries = [
            e for e in self.successes if e.result_data.pattern not in listed_patterns
        ]
        if other_entries:
            other_entries.sort(
                key=lambda e: (-e.result_data.disagreement_score, e.column)
            )
            lines.extend(["", "Other or inconclusive:"])
            for entry in other_entries[:top_n]:
                lines.append(
                    f"  {entry.column} ({entry.result_data.pattern}, "
                    f"disagreement={entry.result_data.disagreement_score:.2f})"
                )

        underrate = [e for e in self.successes if self._is_pearson_underrate(e)]
        if underrate:
            underrate.sort(key=lambda e: (-self._pearson_underrate_gap(e), e.column))
            lines.extend(["", "Variables Pearson may underrate:"])
            for entry in underrate[:top_n]:
                gap = self._pearson_underrate_gap(entry)
                lines.append(f"  {entry.column} (gap={gap:.2f})")

        reverse_shape = [e for e in self.successes if self._reverse_reveals_shape(e)]
        if reverse_shape:
            reverse_shape.sort(key=lambda e: e.column)
            lines.extend(
                [
                    "",
                    f"Shape differs by direction (candidate = f({self.target})):",
                ]
            )
            for entry in reverse_shape[:top_n]:
                rev = entry.reverse_result
                assert rev is not None  # guaranteed by _reverse_reveals_shape
                shape = (
                    rev.diagnostics.mean_shape
                    or rev.diagnostics.dependence_type
                    or rev.pattern
                )
                lines.append(
                    f"  {entry.column}: predicts {self.target} as "
                    f"{entry.result_data.pattern}, but {self.target}->{entry.column} "
                    f"is {rev.pattern} ({shape})"
                )

        warned = [e for e in self.successes if self._has_reliability_warning(e)]
        if warned:
            warned.sort(key=lambda e: e.column)
            lines.extend(["", "Variables with missingness or tie warnings:"])
            for entry in warned[:top_n]:
                lines.append(
                    f"  {entry.column}: {self._primary_reliability_warning(entry)}"
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
            f"# CorrSleuth Target Report: `{escape_markdown_code_span(self.target)}`",
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
            section_entries = [
                e for e in self.successes if e.result_data.pattern == pattern
            ]
            if not section_entries:
                continue
            section_entries.sort(
                key=lambda e: (-e.result_data.disagreement_score, e.column)
            )
            lines.extend(["", f"## {title}"])
            lines.append(self._markdown_entries_table(section_entries[:top_n]))

        listed_patterns = {pattern for pattern, _ in _PATTERN_SECTIONS}
        other_entries = [
            e for e in self.successes if e.result_data.pattern not in listed_patterns
        ]
        if other_entries:
            other_entries.sort(
                key=lambda e: (-e.result_data.disagreement_score, e.column)
            )
            lines.extend(["", "## Other or inconclusive"])
            lines.append(self._markdown_entries_table(other_entries[:top_n]))

        underrate = [e for e in self.successes if self._is_pearson_underrate(e)]
        if underrate:
            underrate.sort(key=lambda e: (-self._pearson_underrate_gap(e), e.column))
            lines.extend(["", "## Variables Pearson may underrate"])
            lines.append(
                markdown_table(
                    ["Variable", "Pattern", "Gap", "Pearson", "Spearman"],
                    [
                        [
                            entry.column,
                            entry.result_data.pattern,
                            format_markdown_value(self._pearson_underrate_gap(entry)),
                            format_markdown_value(metrics_map(entry).get("pearson")),
                            format_markdown_value(metrics_map(entry).get("spearman")),
                        ]
                        for entry in underrate[:top_n]
                    ],
                )
            )

        reverse_shape = [e for e in self.successes if self._reverse_reveals_shape(e)]
        if reverse_shape:
            reverse_shape.sort(key=lambda e: e.column)
            lines.extend(
                ["", f"## Shape differs by direction (candidate = f(`{self.target}`))"]
            )
            lines.append(
                "These candidates read as unstructured when used to predict "
                f"`{self.target}`, but `{self.target}` -> candidate is a "
                "structured nonlinear shape — the signature of the candidate being "
                f"generated from `{self.target}`. Read the reverse shape as *how "
                "the candidate depends on the target*, not as predictive."
            )
            reverse_rows = []
            for entry in reverse_shape[:top_n]:
                rev = entry.reverse_result
                assert rev is not None  # guaranteed by _reverse_reveals_shape
                reverse_rows.append(
                    [
                        entry.column,
                        entry.result_data.pattern,
                        rev.pattern,
                        rev.diagnostics.mean_shape
                        or rev.diagnostics.dependence_type
                        or "-",
                    ]
                )
            lines.append(
                markdown_table(
                    [
                        "Variable",
                        f"Predicts {self.target}",
                        "Reverse pattern",
                        "Reverse shape",
                    ],
                    reverse_rows,
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
        metrics, distance-correlation nonmonotonic evidence, or the lite-computable
        ``sq_corr`` (magnitude/radial dependence) exceed Pearson by more than
        ``threshold``. This keeps outlier/leverage cases, where Pearson is
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
            "sq_corr_excess_over_pearson",
            "disagreement_score",
            "metric_pearson",
            "metric_spearman",
            "metric_kendall_tau_b",
            "metric_distance_correlation",
            "metric_mutual_information",
            "warnings",
        ]

        rows: list[dict[str, Any]] = []
        for entry in self.successes:
            metrics = metrics_map(entry)
            components = self._directional_underrate_components(entry)
            # "score" is max(...) of non-negative floats, so never None.
            score = components["score"]
            assert score is not None

            if score <= threshold:
                continue

            rows.append(
                {
                    "variable": entry.column,
                    "target": self.target,
                    "pattern": entry.result_data.pattern,
                    "pearson_underrate_score": score,
                    "spearman_excess_over_pearson": components[
                        "spearman_excess_over_pearson"
                    ],
                    "kendall_excess_over_pearson": components[
                        "kendall_excess_over_pearson"
                    ],
                    "nonmonotonic_gap": components["nonmonotonic_gap"],
                    "sq_corr_excess_over_pearson": components[
                        "sq_corr_excess_over_pearson"
                    ],
                    "disagreement_score": entry.result_data.disagreement_score,
                    "metric_pearson": metrics.get("pearson"),
                    "metric_spearman": metrics.get("spearman"),
                    "metric_kendall_tau_b": metrics.get("kendall_tau_b"),
                    "metric_distance_correlation": metrics.get("distance_correlation"),
                    "metric_mutual_information": metrics.get("mutual_information"),
                    "warnings": "; ".join(entry.result_data.warnings)
                    if entry.result_data.warnings
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
        result = entry.result_data
        metrics = metrics_map(entry)

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
    def _markdown_entries_table(entries: list[TargetScanEntry]) -> str:
        rows = []
        for entry in entries:
            metrics = metrics_map(entry)
            rows.append(
                [
                    entry.column,
                    entry.result_data.pattern,
                    format_markdown_value(metrics.get("pearson")),
                    format_markdown_value(metrics.get("spearman")),
                    format_markdown_value(entry.result_data.disagreement_score),
                    # Join the warnings raw; markdown_table escapes every cell once
                    # (double-escaping here rendered underscores as literal `\_`).
                    "; ".join(entry.result_data.warnings)
                    if entry.result_data.warnings
                    else "",
                ]
            )
        return markdown_table(
            ["Variable", "Pattern", "Pearson", "Spearman", "Disagreement", "Warnings"],
            rows,
        )

    @staticmethod
    def _positive_abs_gap(value: Any, baseline_abs: float | None) -> float:
        if baseline_abs is None or value is None or pd.isna(value):
            return 0.0
        return max(0.0, abs(float(value)) - baseline_abs)

    @classmethod
    def _directional_underrate_components(
        cls, entry: TargetScanEntry
    ) -> dict[str, float | None]:
        metrics = metrics_map(entry)
        pearson = metrics.get("pearson")
        abs_p = abs(pearson) if pearson is not None and not pd.isna(pearson) else None

        spearman_gap = cls._positive_abs_gap(metrics.get("spearman"), abs_p)
        kendall_gap = cls._positive_abs_gap(metrics.get("kendall_tau_b"), abs_p)

        raw_nonmonotonic_gap = entry.result_data.diagnostics.nonmonotonic_gap
        nonmonotonic_gap = (
            None
            if raw_nonmonotonic_gap is None or pd.isna(raw_nonmonotonic_gap)
            else float(raw_nonmonotonic_gap)
        )
        nonmonotonic_contribution = (
            max(0.0, nonmonotonic_gap) if nonmonotonic_gap is not None else 0.0
        )

        # sq_corr is lite-computable (no mode gate), so a lite scan can label a
        # pair nonmonotonic via magnitude/radial dependence with no dcor. Its
        # excess over |Pearson| is on the same correlation scale as the rank gaps,
        # so a magnitude-linked pair now surfaces under "Pearson may underrate"
        # rather than only in its pattern section (C6 #6). A clean linear pair has
        # |sq_corr| ~= |Pearson| (the squares track together), so this adds ~0
        # there — no false surfacing.
        #
        # Only count it when the diagnostics actually concluded genuine
        # magnitude/radial dependence: ``dependence_type`` in
        # {magnitude_linked, closed_loop_or_multivalued} — which (post-robust-gate,
        # see classifier._dependence_type_axis) means the sq_corr signal survived
        # dropping the few most extreme points. Otherwise a heavy-tailed-Y artifact
        # (raw sq_corr over the bar but robust-collapsed, so the cascade already
        # calls it weak_or_no_relationship) would be promoted here on the exact
        # signal the robust gate exists to suppress — the ranking must use the same
        # robust evidence as the cascade.
        dep_type = entry.result_data.diagnostics.dependence_type
        sq_corr_gap = (
            cls._positive_abs_gap(entry.result_data.diagnostics.sq_corr, abs_p)
            if dep_type in ("magnitude_linked", "closed_loop_or_multivalued")
            else 0.0
        )

        score = max(spearman_gap, kendall_gap, nonmonotonic_contribution, sq_corr_gap)
        return {
            "score": score,
            "spearman_excess_over_pearson": spearman_gap,
            "kendall_excess_over_pearson": kendall_gap,
            "nonmonotonic_gap": nonmonotonic_gap,
            "sq_corr_excess_over_pearson": sq_corr_gap,
        }

    @classmethod
    def _directional_underrate_score(cls, entry: TargetScanEntry) -> float:
        score = cls._directional_underrate_components(entry)["score"]
        # "score" is max(...) of non-negative floats, so never None.
        assert score is not None
        return float(score)

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
            for warning in entry.result_data.warnings
            for keyword in _RELIABILITY_WARNING_KEYWORDS
        )

    @staticmethod
    def _primary_reliability_warning(entry: TargetScanEntry) -> str:
        for warning in entry.result_data.warnings:
            if any(k in warning for k in _RELIABILITY_WARNING_KEYWORDS):
                return warning
        return entry.result_data.warnings[0] if entry.result_data.warnings else ""

    def plot_top(
        self,
        n: int = 12,
        sort_by: str = "disagreement_score",
        patterns: Sequence[str] | None = None,
        ncols: int = 3,
        figsize: tuple[float, float] | None = None,
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
        # Deferred import keeps matplotlib (and its expensive pyplot import) out
        # of the hot path for non-plotting workflows.
        from corrsleuth.scan.plot import build_scan_figure

        return build_scan_figure(
            self,
            n=n,
            sort_by=sort_by,
            patterns=patterns,
            ncols=ncols,
            figsize=figsize,
            show=show,
        )
