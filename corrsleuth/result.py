from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from corrsleuth.utils.markdown import (
    escape_markdown_cell,
    escape_markdown_code_span,
    format_markdown_value,
    markdown_table,
)

if TYPE_CHECKING:
    from corrsleuth.metrics.bootstrap import BootstrapStability


def _json_safe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """``frame.to_dict(orient="records")`` with NaN normalized to ``None``.

    A metrics / bootstrap-intervals column that mixes real floats with
    unavailable values is float-typed, so pandas stores the gaps as ``NaN`` —
    which ``to_dict`` would emit as a bare ``float('nan')``, not JSON-compliant
    (``json.dumps(..., allow_nan=False)`` raises). Callers read ``None`` as "no
    value", so normalize before returning the public dict."""

    def _clean(value: Any) -> Any:
        try:
            return None if pd.isna(value) else value
        except (TypeError, ValueError):
            return value  # non-scalar (e.g. a list): leave as-is

    return [
        # str(key): pandas types record keys as Hashable; column names are
        # strings, and JSON object keys must be strings regardless.
        {str(key): _clean(value) for key, value in record.items()}
        for record in frame.to_dict(orient="records")
    ]


@dataclass
class MetricResult:
    """
    Internal contract representing the result of a single association metric computation.
    """

    name: str
    value: float | None
    available: bool

    @classmethod
    def no_value(cls, name: str) -> "MetricResult":
        """Build a result for a metric that *applied* but produced no value.

        Used for the recurring "the check ran, but the data can't support a
        number" cases — a constant input, or ``n_used`` below a metric's
        minimum. ``available=True`` (the metric is part of this profile) with
        ``value=None`` (no usable estimate), as opposed to ``available=False``,
        which marks a metric that was never applicable (e.g. an optional
        dependency missing, or a lower-tier mode).
        """
        return cls(name=name, value=None, available=True)


@dataclass
class HeuristicResult:
    """
    Internal contract representing the outcome of the heuristic classification.
    """

    label: str
    recommendations: list[str]


@dataclass
class MetricDiagnostics:
    """Diagnostic components describing metric agreement and disagreement.

    Carries the absolute rank-vs-linear gap (``rank_linear_gap``), the signed
    Pearson-minus-Spearman gap (``pearson_spearman_signed_gap``, which reveals
    sign disagreement the absolute gap hides), the nonmonotonic and
    Pearson-Kendall gaps, the overall ``disagreement_score``, the
    outlier-sensitivity fields (``pearson_trimmed``, ``pearson_trim_delta``),
    the shape diagnostics (``bin_lof_r2_gain``, the equal-frequency-bin
    lack-of-fit test's R² gain over a linear fit; ``bin_reversal_count``, how
    many times the sequence of bin means changes direction — 0 for a monotone
    trend, 1 for a single bend, 2+ for an oscillation; ``sq_corr``, the
    correlation between the squared mean-centered X and Y; ``sq_corr_robust``,
    the leave-the-top-out companion to ``sq_corr`` — the smallest ``|sq_corr|``
    after dropping the few most extreme squared points, so a heavy-tailed
    variable's spurious magnitude signal collapses while a genuine one survives)
    — see ``corrsleuth/metrics/shape.py`` — and the
    heteroscedasticity diagnostics (``bp_pvalue``, the Breusch-Pagan p-value;
    ``gq_ratio``, the Goldfeld-Quandt high-vs-low-x residual variance ratio;
    ``bowtie_ratio``, the edge-thirds-vs-middle-third residual variance ratio) —
    see ``corrsleuth/metrics/variance.py`` — the segmentation diagnostics
    (``segment_gain``, the R² gain of the best single-breakpoint two-line fit
    over one line; ``segment_stepness``, the fraction of that gain a two-*level*
    (flat-segment) model already captures — ``≈ 1`` for a step/threshold jump,
    ``≤ 0`` for a smooth bend, and the number behind the ``mean_shape``
    step-vs-smooth call; ``breakpoint_x``, the x-location of a detected step,
    reported only when ``mean_shape`` reads as a step/threshold) — and the
    influence
    diagnostics (``max_cook_distance``, the largest Cook's distance;
    ``n_influential_points``, how many rows exceed the influence cutoff) — see
    ``corrsleuth/metrics/influence.py``. Gap, shape, variance, segmentation, and
    influence fields are ``None`` when the metrics they depend on are
    unavailable.

    The final five fields are the **secondary diagnostic axes** — coarse
    categorical summaries describing orthogonal properties of the relationship
    that the single primary ``pattern`` label cannot carry at once: the shape of
    the conditional mean (``mean_shape``), the shape of the conditional variance
    (``variance_shape``), the kind of dependence (``dependence_type``), whether a
    few rows drive the summary (``outlier_sensitivity``), and which variable is a
    function of the other (``functional_direction``, deep mode only). Each is
    derived from the numeric diagnostics/metrics above and is ``None`` when the
    axis is not assessable. See ``corrsleuth/heuristics/classifier.py``
    (``derive_diagnostic_axes``) and docs/interpretation-guide.md.
    """

    rank_linear_gap: float | None
    pearson_spearman_signed_gap: float | None
    nonmonotonic_gap: float | None
    pearson_kendall_gap: float | None
    disagreement_score: float
    pearson_trimmed: float | None = None
    pearson_trim_delta: float | None = None
    bin_lof_r2_gain: float | None = None
    bin_reversal_count: int | None = None
    sq_corr: float | None = None
    sq_corr_robust: float | None = None
    bp_pvalue: float | None = None
    gq_ratio: float | None = None
    bowtie_ratio: float | None = None
    segment_gain: float | None = None
    segment_stepness: float | None = None
    breakpoint_x: float | None = None
    max_cook_distance: float | None = None
    n_influential_points: int | None = None
    mean_shape: str | None = None
    variance_shape: str | None = None
    dependence_type: str | None = None
    outlier_sensitivity: str | None = None
    functional_direction: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CorrSleuthResult:
    """
    Public output object representing the diagnostic profile of a pairwise relationship.
    """

    def __init__(
        self,
        x_name: str,
        y_name: str,
        metrics: pd.DataFrame,
        pattern: str,
        warnings: list[str],
        recommendations: list[str],
        disagreement_score: float,
        diagnostics: MetricDiagnostics | None = None,
        bootstrap_intervals: pd.DataFrame | None = None,
        bootstrap_stability: "BootstrapStability | None" = None,
        _clean_x: pd.Series | None = None,
        _clean_y: pd.Series | None = None,
        _include_caveat: bool = True,
    ):
        self.x_name = x_name
        self.y_name = y_name
        self.metrics = metrics
        self.pattern = pattern
        self.warnings = warnings
        self.recommendations = recommendations
        self.disagreement_score = disagreement_score
        self.diagnostics = diagnostics or MetricDiagnostics(
            rank_linear_gap=None,
            pearson_spearman_signed_gap=None,
            nonmonotonic_gap=None,
            pearson_kendall_gap=None,
            disagreement_score=disagreement_score,
            pearson_trimmed=None,
            pearson_trim_delta=None,
            bin_lof_r2_gain=None,
            bin_reversal_count=None,
            sq_corr=None,
            sq_corr_robust=None,
            bp_pvalue=None,
            gq_ratio=None,
            bowtie_ratio=None,
            segment_gain=None,
            segment_stepness=None,
            breakpoint_x=None,
            max_cook_distance=None,
            n_influential_points=None,
            mean_shape=None,
            variance_shape=None,
            dependence_type=None,
            outlier_sensitivity=None,
            functional_direction=None,
        )
        self.bootstrap_intervals = bootstrap_intervals
        self.bootstrap_stability = bootstrap_stability
        self.pattern_stability = (
            bootstrap_stability.pattern_stability if bootstrap_stability else None
        )
        self.bootstrap_label_counts = (
            bootstrap_stability.bootstrap_label_counts if bootstrap_stability else None
        )
        self.stability_label = (
            bootstrap_stability.stability_label if bootstrap_stability else None
        )
        self._clean_x = _clean_x
        self._clean_y = _clean_y
        self._include_caveat = _include_caveat

    @staticmethod
    def _format_value(value: float | None) -> str:
        return f"{value:.3f}" if value is not None and pd.notna(value) else "NA"

    @staticmethod
    def _format_axis(value: str | None) -> str:
        """Render a categorical secondary-axis value, ``NA`` when not assessable."""
        return value if value else "NA"

    @staticmethod
    def _format_count(value: int | None) -> str:
        """Render an integer count diagnostic, ``NA`` when not assessable."""
        return str(value) if value is not None else "NA"

    def summary(self, include_caveat: bool | None = None) -> str:
        """
        Returns a tabular view of metrics and the primary label.
        """
        if include_caveat is None:
            include_caveat = self._include_caveat

        lines = [
            f"Relationship Profile: {self.x_name} vs {self.y_name}",
            f"Primary pattern: {self.pattern}",
            "",
            "Metrics:",
        ]
        for _, row in self.metrics.iterrows():
            val_str = self._format_value(row["value"])
            lines.append(f"  {row['metric'].ljust(25)}: {val_str}")

        diag = self.diagnostics
        # (label, formatted-value) pairs, laid out with one ljust width shared by
        # every row so the colons align even for the longest label — building the
        # rows programmatically also keeps ``pearson_trimmed`` (the level Pearson
        # moved to) beside its delta on this surface, not only in to_dict/to_frame.
        diag_rows = [
            ("disagreement_score", self._format_value(diag.disagreement_score)),
            ("rank_linear_gap", self._format_value(diag.rank_linear_gap)),
            (
                "pearson_spearman_signed_gap",
                self._format_value(diag.pearson_spearman_signed_gap),
            ),
            ("nonmonotonic_gap", self._format_value(diag.nonmonotonic_gap)),
            ("pearson_kendall_gap", self._format_value(diag.pearson_kendall_gap)),
            ("pearson_trimmed", self._format_value(diag.pearson_trimmed)),
            ("pearson_trim_delta", self._format_value(diag.pearson_trim_delta)),
            ("bin_lof_r2_gain", self._format_value(diag.bin_lof_r2_gain)),
            ("bin_reversal_count", self._format_count(diag.bin_reversal_count)),
            ("sq_corr", self._format_value(diag.sq_corr)),
            ("sq_corr_robust", self._format_value(diag.sq_corr_robust)),
            ("bp_pvalue", self._format_value(diag.bp_pvalue)),
            ("gq_ratio", self._format_value(diag.gq_ratio)),
            ("bowtie_ratio", self._format_value(diag.bowtie_ratio)),
            ("segment_gain", self._format_value(diag.segment_gain)),
            ("segment_stepness", self._format_value(diag.segment_stepness)),
            ("breakpoint_x", self._format_value(diag.breakpoint_x)),
            ("max_cook_distance", self._format_value(diag.max_cook_distance)),
            ("n_influential_points", self._format_count(diag.n_influential_points)),
        ]
        diag_width = max(len(label) for label, _ in diag_rows)
        lines.extend(["", "Diagnostics:"])
        lines.extend(f"  {label.ljust(diag_width)}: {val}" for label, val in diag_rows)
        lines.extend(
            [
                "",
                "Relationship axes:",
                f"  mean_shape           : {self._format_axis(diag.mean_shape)}",
                f"  variance_shape       : {self._format_axis(diag.variance_shape)}",
                f"  dependence_type      : {self._format_axis(diag.dependence_type)}",
                f"  outlier_sensitivity  : {self._format_axis(diag.outlier_sensitivity)}",
                f"  functional_direction : {self._format_axis(diag.functional_direction)}",
            ]
        )

        if self.bootstrap_intervals is not None and not self.bootstrap_intervals.empty:
            lines.extend(["", "Bootstrap intervals:"])
            for _, row in self.bootstrap_intervals.iterrows():
                low = self._format_value(row["ci_low"])
                high = self._format_value(row["ci_high"])
                lines.append(
                    f"  {row['metric'].ljust(25)}: [{low}, {high}] "
                    f"(n={int(row['n_success'])}/{int(row['n_bootstrap'])}, "
                    f"{row['metric_set']})"
                )

        if self.bootstrap_stability is not None:
            stability = self.bootstrap_stability
            lines.extend(
                [
                    "",
                    "Pattern stability:",
                    (
                        f"  {self._format_value(stability.pattern_stability)} "
                        f"({stability.stability_label}, {stability.metric_set}, "
                        f"n={int(stability.n_iterations)}/{int(stability.n_bootstrap)})"
                    ),
                    f"  label_counts: {self._format_label_counts(stability.bootstrap_label_counts)}",
                ]
            )

        if self.warnings:
            lines.append("\nWarnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")

        if self.recommendations:
            lines.append("\nRecommendations:")
            for r in self.recommendations:
                lines.append(f"  - {r}")

        if include_caveat:
            from corrsleuth.heuristics.explanations import _CAVEAT

            lines.append(f"\nCaveat: {_CAVEAT}")

        return "\n".join(lines)

    def explain(self, include_caveat: bool | None = None) -> str:
        """
        Returns a 2-3 sentence narrative explaining metric disagreement and pattern evidence.
        """
        if include_caveat is None:
            include_caveat = self._include_caveat

        from corrsleuth.heuristics.explanations import generate_explanation

        explanation = generate_explanation(
            self.pattern,
            metrics=self.metrics,
            include_caveat=include_caveat,
            diagnostics=self.diagnostics,
        )
        if self.bootstrap_stability is not None:
            stability = self.bootstrap_stability
            explanation += (
                " Bootstrap resampling assigned the same diagnostic label in "
                f"{stability.pattern_stability:.1%} of samples "
                f"({stability.stability_label} stability, {stability.metric_set} metrics)."
            )
            from corrsleuth.heuristics import STANDARD_ONLY_LABELS

            # Gate on whether dcor was actually in the replicate cascade — not on
            # the metric_set string — so this matches the warnings list exactly
            # even for an explicit subset like bootstrap_metrics=["pearson"]
            # (metric_set="pearson" but dcor still absent). See C5 #2.
            if self.pattern in STANDARD_ONLY_LABELS and not stability.dcor_in_cascade:
                explanation += (
                    f" Because stability used lite metrics, it may not fully test a "
                    f"standard-mode {self.pattern} label (this can be conservative "
                    f"if the label was actually driven by a lite-computable shape "
                    f"diagnostic — sq_corr or the bin-reversal oscillation route — "
                    f"rather than distance correlation)."
                )
        return explanation

    def plot(self, show: bool = False) -> Any:
        """Return a multi-panel scatter + rank plot with pattern annotation.

        Parameters
        ----------
        show : bool, default False
            If ``True``, display the figure via ``matplotlib.pyplot.show()``.

        Returns
        -------
        matplotlib.figure.Figure
            The diagnostic figure.

        Raises
        ------
        ValueError
            If the cleaned data was not preserved on this result object (e.g. a
            result reconstructed without ``_clean_x``/``_clean_y``), so there is
            nothing to plot.
        """
        if self._clean_x is None or self._clean_y is None:
            raise ValueError(
                "Cleaned data was not preserved in this result object, so plotting is unavailable."
            )
        from corrsleuth.plotting.pairplot import plot_pair

        return plot_pair(self, show=show)

    def to_markdown(self, include_caveat: bool | None = None) -> str:
        """Return a compact Markdown report for sharing in notebooks or docs."""
        if include_caveat is None:
            include_caveat = self._include_caveat

        lines = [
            f"# CorrSleuth Pair Report: "
            f"`{escape_markdown_code_span(self.x_name)}` vs "
            f"`{escape_markdown_code_span(self.y_name)}`",
            "",
            f"**Primary pattern:** `{self.pattern}`",
            "",
            "## Metrics",
            markdown_table(
                ["Metric", "Value"],
                [
                    [row["metric"], format_markdown_value(row["value"])]
                    for _, row in self.metrics.iterrows()
                ],
            ),
            "",
            "## Diagnostics",
            markdown_table(
                ["Diagnostic", "Value"],
                [
                    [
                        "disagreement_score",
                        format_markdown_value(self.diagnostics.disagreement_score),
                    ],
                    [
                        "rank_linear_gap",
                        format_markdown_value(self.diagnostics.rank_linear_gap),
                    ],
                    [
                        "pearson_spearman_signed_gap",
                        format_markdown_value(
                            self.diagnostics.pearson_spearman_signed_gap
                        ),
                    ],
                    [
                        "nonmonotonic_gap",
                        format_markdown_value(self.diagnostics.nonmonotonic_gap),
                    ],
                    [
                        "pearson_kendall_gap",
                        format_markdown_value(self.diagnostics.pearson_kendall_gap),
                    ],
                    [
                        "pearson_trimmed",
                        format_markdown_value(self.diagnostics.pearson_trimmed),
                    ],
                    [
                        "pearson_trim_delta",
                        format_markdown_value(self.diagnostics.pearson_trim_delta),
                    ],
                    [
                        "bin_lof_r2_gain",
                        format_markdown_value(self.diagnostics.bin_lof_r2_gain),
                    ],
                    [
                        "bin_reversal_count",
                        self._format_count(self.diagnostics.bin_reversal_count),
                    ],
                    [
                        "sq_corr",
                        format_markdown_value(self.diagnostics.sq_corr),
                    ],
                    [
                        "sq_corr_robust",
                        format_markdown_value(self.diagnostics.sq_corr_robust),
                    ],
                    [
                        "bp_pvalue",
                        format_markdown_value(self.diagnostics.bp_pvalue),
                    ],
                    [
                        "gq_ratio",
                        format_markdown_value(self.diagnostics.gq_ratio),
                    ],
                    [
                        "bowtie_ratio",
                        format_markdown_value(self.diagnostics.bowtie_ratio),
                    ],
                    [
                        "segment_gain",
                        format_markdown_value(self.diagnostics.segment_gain),
                    ],
                    [
                        "segment_stepness",
                        format_markdown_value(self.diagnostics.segment_stepness),
                    ],
                    [
                        "breakpoint_x",
                        format_markdown_value(self.diagnostics.breakpoint_x),
                    ],
                    [
                        "max_cook_distance",
                        format_markdown_value(self.diagnostics.max_cook_distance),
                    ],
                    [
                        "n_influential_points",
                        self._format_count(self.diagnostics.n_influential_points),
                    ],
                ],
            ),
            "",
            "## Relationship Axes",
            markdown_table(
                ["Axis", "Value"],
                [
                    ["mean_shape", self._format_axis(self.diagnostics.mean_shape)],
                    [
                        "variance_shape",
                        self._format_axis(self.diagnostics.variance_shape),
                    ],
                    [
                        "dependence_type",
                        self._format_axis(self.diagnostics.dependence_type),
                    ],
                    [
                        "outlier_sensitivity",
                        self._format_axis(self.diagnostics.outlier_sensitivity),
                    ],
                    [
                        "functional_direction",
                        self._format_axis(self.diagnostics.functional_direction),
                    ],
                ],
            ),
        ]

        if self.bootstrap_intervals is not None and not self.bootstrap_intervals.empty:
            lines.extend(
                [
                    "",
                    "## Bootstrap Intervals",
                    markdown_table(
                        [
                            "Metric",
                            "CI low",
                            "CI high",
                            "Successful samples",
                            "Metric set",
                        ],
                        [
                            [
                                row["metric"],
                                format_markdown_value(row["ci_low"]),
                                format_markdown_value(row["ci_high"]),
                                f"{int(row['n_success'])}/{int(row['n_bootstrap'])}",
                                row["metric_set"],
                            ]
                            for _, row in self.bootstrap_intervals.iterrows()
                        ],
                    ),
                ]
            )

        if self.bootstrap_stability is not None:
            stability = self.bootstrap_stability
            lines.extend(
                [
                    "",
                    "## Pattern Stability",
                    markdown_table(
                        ["Stability", "Label", "Metric set", "Samples", "Label counts"],
                        [
                            [
                                format_markdown_value(stability.pattern_stability),
                                stability.stability_label,
                                stability.metric_set,
                                f"{int(stability.n_iterations)}/{int(stability.n_bootstrap)}",
                                self._format_label_counts(
                                    stability.bootstrap_label_counts
                                ),
                            ]
                        ],
                    ),
                ]
            )

        lines.extend(["", "## Warnings"])
        if self.warnings:
            lines.extend(f"- {escape_markdown_cell(w)}" for w in self.warnings)
        else:
            lines.append("- None")

        lines.extend(["", "## Recommendations"])
        if self.recommendations:
            lines.extend(f"- {escape_markdown_cell(r)}" for r in self.recommendations)
        else:
            lines.append("- None")

        if include_caveat:
            from corrsleuth.heuristics.explanations import _CAVEAT

            lines.extend(["", "## Caveat", _CAVEAT])

        return "\n".join(lines)

    @staticmethod
    def _format_label_counts(label_counts: dict[str, int]) -> str:
        items = sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))
        return "; ".join(f"{label}: {count}" for label, count in items)

    def to_dict(self) -> dict[str, Any]:
        """Return the result as a plain dictionary.

        Keys: ``x``, ``y``, ``pattern``, ``metrics`` (list of records),
        ``disagreement_score``, ``diagnostics``, ``bootstrap_intervals``,
        ``bootstrap_stability``, ``pattern_stability``,
        ``bootstrap_label_counts``, ``stability_label``, ``warnings``, and
        ``recommendations``. Bootstrap keys are ``None`` when bootstrapping was
        not requested.
        """
        return {
            "x": self.x_name,
            "y": self.y_name,
            "pattern": self.pattern,
            "metrics": _json_safe_records(self.metrics),
            "disagreement_score": self.disagreement_score,
            "diagnostics": self.diagnostics.to_dict(),
            "bootstrap_intervals": (
                None
                if self.bootstrap_intervals is None
                else _json_safe_records(self.bootstrap_intervals)
            ),
            "bootstrap_stability": (
                None
                if self.bootstrap_stability is None
                else self.bootstrap_stability.to_dict()
            ),
            "pattern_stability": self.pattern_stability,
            "bootstrap_label_counts": self.bootstrap_label_counts,
            "stability_label": self.stability_label,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
        }

    def to_frame(self) -> pd.DataFrame:
        """
        Returns the result as a pandas DataFrame.
        """
        df = self.metrics.copy()
        df["x"] = self.x_name
        df["y"] = self.y_name
        df["pattern"] = self.pattern
        for key, value in self.diagnostics.to_dict().items():
            df[f"diagnostic_{key}"] = value
        if self.bootstrap_intervals is not None and not self.bootstrap_intervals.empty:
            intervals = self.bootstrap_intervals.set_index("metric")
            df["bootstrap_ci_low"] = df["metric"].map(intervals["ci_low"])
            df["bootstrap_ci_high"] = df["metric"].map(intervals["ci_high"])
            df["bootstrap_n_success"] = df["metric"].map(intervals["n_success"])
            df["bootstrap_n"] = df["metric"].map(intervals["n_bootstrap"])
            df["bootstrap_sample_size"] = df["metric"].map(intervals["sample_size"])
        if self.bootstrap_stability is not None:
            df["pattern_stability"] = self.bootstrap_stability.pattern_stability
            df["stability_label"] = self.bootstrap_stability.stability_label
            df["stability_metric_set"] = self.bootstrap_stability.metric_set
        return df
