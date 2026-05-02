from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
import pandas as pd


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
    disagreement_components: Dict[str, float]
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

    def to_dict(self) -> Dict[str, Optional[float]]:
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
        )
        self._clean_x = _clean_x
        self._clean_y = _clean_y
        self._include_caveat = _include_caveat

    def summary(self, include_caveat: bool = None) -> str:
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
            val_str = f"{row['value']:.3f}" if pd.notna(row['value']) else "NA"
            lines.append(f"  {row['metric'].ljust(25)}: {val_str}")
            
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

    def explain(self, include_caveat: bool = None) -> str:
        """
        Returns a 2-3 sentence narrative explaining metric disagreement and pattern evidence.
        """
        if include_caveat is None:
            include_caveat = self._include_caveat
            
        from corrsleuth.heuristics.explanations import generate_explanation
        return generate_explanation(self.pattern, metrics=self.metrics, include_caveat=include_caveat)

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
        return df
