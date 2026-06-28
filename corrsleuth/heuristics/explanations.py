import pandas as pd

_CAVEAT = (
    "Do not interpret this association causally without proper design or controls."
)

_EXPLANATIONS = {
    "not_computable": "The metrics could not be computed. This usually happens when one or both variables are entirely constant, or there are no valid overlapping data points.",
    "low_power_or_uncertain": "The evidence is too uncertain to confidently describe the relationship shape, often due to a very small sample size.",
    "possible_outlier_or_leverage": "The apparent linear association may be artificially inflated or driven by outliers and high-leverage observations.",
    "nonmonotonic_dependence": "Evidence consistent with a relationship that is not simply increasing or decreasing (e.g., U-shaped or cyclical). Standard linear and rank metrics may understate this relationship.",
    "monotonic_nonlinear": "Evidence consistent with a directional relationship that is not well summarized by a straight line. Pearson may significantly understate the relationship compared to rank-based measures.",
    "near_linear": "Evidence consistent with an approximately linear or near-linear relationship. Both variables appear to scale together smoothly.",
    "weak_or_no_relationship": "Little to no evidence of a pairwise association in the observed data.",
    "mixed_or_ambiguous": "The metrics disagree in a way that doesn't strongly match a canonical pattern. The relationship may be complex or noisy.",
}

_RECOMMENDATIONS = {
    "not_computable": [
        "Check for constant variables (zero variance).",
        "Check for data misalignment or missingness.",
    ],
    "low_power_or_uncertain": [
        "Collect more data.",
        "Rely on domain knowledge rather than statistical significance here.",
    ],
    "possible_outlier_or_leverage": [
        "Inspect scatter plots for extreme points.",
        "Consider robust or winsorized sensitivity checks.",
    ],
    "nonmonotonic_dependence": [
        "Inspect the relationship visually (e.g., scatter plot with a smoother).",
        "Consider modeling with polynomials, splines, or tree-based methods.",
    ],
    "monotonic_nonlinear": [
        "Inspect the scatter plot for curvature.",
        "Consider logarithmic or other monotonic transformations.",
    ],
    "near_linear": [
        "A standard linear model or Pearson correlation is likely appropriate here."
    ],
    "weak_or_no_relationship": [
        "Consider whether the relationship might be conditionally masked by a third variable.",
        "This feature may not be a strong linear predictor on its own.",
    ],
    "mixed_or_ambiguous": [
        "Inspect the data visually.",
        "Check whether this pattern holds within important segments or clusters.",
    ],
}


def _metric_map(metrics: pd.DataFrame | None) -> dict[str, float]:
    if metrics is None:
        return {}

    values = {}
    for _, row in metrics.iterrows():
        value = row["value"]
        if pd.notna(value):
            values[str(row["metric"])] = float(value)
    return values


def _fmt(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.3f}"


def _metric_context(pattern: str, metrics: pd.DataFrame | None) -> list[str]:
    values = _metric_map(metrics)
    if not values:
        return []

    pearson = values.get("pearson")
    spearman = values.get("spearman")
    kendall = values.get("kendall_tau_b")
    dcor = values.get("distance_correlation")

    abs_p = abs(pearson) if pearson is not None else None
    abs_s = abs(spearman) if spearman is not None else None
    abs_k = abs(kendall) if kendall is not None else None

    if pattern == "near_linear" and abs_p is not None and abs_s is not None:
        return [
            (
                f"Pearson ({_fmt(pearson)}) and Spearman ({_fmt(spearman)}) "
                "are both relatively strong and closely aligned, so the metrics "
                "appear to agree on a mostly linear association."
            )
        ]

    if pattern == "monotonic_nonlinear" and abs_p is not None and abs_s is not None:
        context = [
            (
                f"Spearman ({_fmt(spearman)}) is meaningfully stronger than "
                f"Pearson ({_fmt(pearson)}), which may suggest a directional "
                "relationship that is not well summarized by a straight line."
            )
        ]
        if abs_k is not None:
            context.append(
                f"Kendall tau-b ({_fmt(kendall)}) provides an additional rank-based check on that directional pattern."
            )
        return context

    if pattern == "nonmonotonic_dependence" and abs_p is not None and abs_s is not None:
        if dcor is not None:
            return [
                (
                    f"Pearson ({_fmt(pearson)}) and Spearman ({_fmt(spearman)}) "
                    f"are weak, while distance correlation ({_fmt(dcor)}) is higher; "
                    "that disagreement is evidence consistent with dependence that is not simply increasing or decreasing."
                )
            ]
        return [
            (
                f"Pearson ({_fmt(pearson)}) and Spearman ({_fmt(spearman)}) "
                "are weak, and standard-mode nonlinear metrics are unavailable to further check for nonmonotonic dependence."
            )
        ]

    if (
        pattern == "possible_outlier_or_leverage"
        and abs_p is not None
        and abs_s is not None
    ):
        return [
            (
                f"Pearson ({_fmt(pearson)}) is much stronger than Spearman "
                f"({_fmt(spearman)}), so the linear association may be more sensitive "
                "to extreme values than the rank-based metrics."
            )
        ]

    if pattern == "weak_or_no_relationship" and abs_p is not None and abs_s is not None:
        if dcor is not None:
            return [
                (
                    f"Pearson ({_fmt(pearson)}), Spearman ({_fmt(spearman)}), "
                    f"and distance correlation ({_fmt(dcor)}) are all weak, so the available metrics do not show much pairwise association."
                )
            ]
        return [
            (
                f"Pearson ({_fmt(pearson)}) and Spearman ({_fmt(spearman)}) are weak. "
                "Because this result was computed without standard-mode nonlinear metrics, complex dependence may require further visual inspection."
            )
        ]

    return []


def generate_explanation(
    pattern: str,
    metrics: pd.DataFrame | None = None,
    include_caveat: bool = True,
) -> str:
    exp = _EXPLANATIONS.get(pattern, _EXPLANATIONS["mixed_or_ambiguous"])

    context_sentences = _metric_context(pattern, metrics)
    if context_sentences:
        exp += " " + " ".join(context_sentences)

    if include_caveat:
        exp = f"{exp} {_CAVEAT}"
    return exp


def generate_recommendations(pattern: str) -> list[str]:
    return _RECOMMENDATIONS.get(pattern, _RECOMMENDATIONS["mixed_or_ambiguous"])
