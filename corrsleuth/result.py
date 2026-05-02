from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import pandas as pd

if TYPE_CHECKING:
    from corrsleuth.metrics.bootstrap import BootstrapStability


@dataclass
class MetricResult:
    """
    Internal contract representing the result of a single association metric computation.
    """
    name: str
    value: Optional[float]
    available: bool


@dataclass
class HeuristicResult:
    """
    Internal contract representing the outcome of the heuristic classification.
    """
    label: str
    recommendations: List[str]


@dataclass
class MetricDiagnostics:
    """
    Public diagnostic components that describe metric agreement and disagreement.
    """
    rank_linear_gap: Optional[float]
    pearson_spearman_signed_gap: Optional[float]
    nonmonotonic_gap: Optional[float]
    pearson_kendall_gap: Optional[float]
    disagreement_score: float
    pearson_trimmed: Optional[float] = None
    pearson_trim_delta: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
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
        warnings: List[str],
        recommendations: List[str],
        disagreement_score: float,
        diagnostics: Optional[MetricDiagnostics] = None,
        bootstrap_intervals: Optional[pd.DataFrame] = None,
        bootstrap_stability: Optional["BootstrapStability"] = None,
        _clean_x: Optional[pd.Series] = None,
        _clean_y: Optional[pd.Series] = None,
        _include_caveat: bool = True
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
    def _format_value(value: Optional[float]) -> str:
        return f"{value:.3f}" if value is not None and pd.notna(value) else "NA"

    def summary(self, include_caveat: Optional[bool] = None) -> str:
        """
        Returns a tabular view of metrics and the primary label.
        """
        if include_caveat is None:
            include_caveat = self._include_caveat
            
        lines = [
            f"Relationship Profile: {self.x_name} vs {self.y_name}",
            f"Primary pattern: {self.pattern}",
            "",
            "Metrics:"
        ]
        for _, row in self.metrics.iterrows():
            val_str = self._format_value(row["value"])
            lines.append(f"  {row['metric'].ljust(25)}: {val_str}")

        lines.extend([
            "",
            "Diagnostics:",
            f"  disagreement_score       : {self._format_value(self.diagnostics.disagreement_score)}",
            f"  rank_linear_gap          : {self._format_value(self.diagnostics.rank_linear_gap)}",
            f"  nonmonotonic_gap         : {self._format_value(self.diagnostics.nonmonotonic_gap)}",
            f"  pearson_kendall_gap      : {self._format_value(self.diagnostics.pearson_kendall_gap)}",
            f"  pearson_trim_delta       : {self._format_value(self.diagnostics.pearson_trim_delta)}",
        ])

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
                    f"  label_counts: {stability.bootstrap_label_counts}",
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

    def explain(self, include_caveat: Optional[bool] = None) -> str:
        """
        Returns a 2-3 sentence narrative explaining metric disagreement and pattern evidence.
        """
        if include_caveat is None:
            include_caveat = self._include_caveat
            
        from corrsleuth.heuristics.explanations import generate_explanation
        explanation = generate_explanation(
            self.pattern, metrics=self.metrics, include_caveat=include_caveat
        )
        if self.bootstrap_stability is not None:
            stability = self.bootstrap_stability
            explanation += (
                " Bootstrap resampling assigned the same diagnostic label in "
                f"{stability.pattern_stability:.1%} of samples "
                f"({stability.stability_label} stability, {stability.metric_set} metrics)."
            )
            from corrsleuth.heuristics import STANDARD_ONLY_LABELS
            if self.pattern in STANDARD_ONLY_LABELS and stability.metric_set == "lite":
                explanation += (
                    f" Because stability used lite metrics, it may not fully test a "
                    f"standard-mode {self.pattern} label."
                )
        return explanation

    def plot(self, show: bool = False) -> Any:
        """
        Returns a multi-panel scatter + rank plot with pattern annotation.
        Returns a matplotlib.figure.Figure.
        """
        if self._clean_x is None or self._clean_y is None:
            raise ValueError("Cleaned data was not preserved in this result object, so plotting is unavailable.")
        from corrsleuth.plotting.pairplot import plot_pair
        return plot_pair(self, show=show)

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns the result as a dictionary.
        """
        return {
            "x": self.x_name,
            "y": self.y_name,
            "pattern": self.pattern,
            "metrics": self.metrics.to_dict(orient="records"),
            "disagreement_score": self.disagreement_score,
            "diagnostics": self.diagnostics.to_dict(),
            "bootstrap_intervals": (
                None
                if self.bootstrap_intervals is None
                else self.bootstrap_intervals.to_dict(orient="records")
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
            "recommendations": self.recommendations
        }

    def to_frame(self) -> pd.DataFrame:
        """
        Returns the result as a pandas DataFrame.
        """
        df = self.metrics.copy()
        df['x'] = self.x_name
        df['y'] = self.y_name
        df['pattern'] = self.pattern
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
