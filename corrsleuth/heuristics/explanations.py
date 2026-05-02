from typing import List

_CAVEAT = "Do not interpret this association causally without proper design or controls."

_EXPLANATIONS = {
    "not_computable": "The metrics could not be computed. This usually happens when one or both variables are entirely constant, or there are no valid overlapping data points.",
    "low_power_or_uncertain": "The evidence is too uncertain to confidently describe the relationship shape, often due to a very small sample size.",
    "possible_outlier_or_leverage": "The apparent linear association may be artificially inflated or driven by outliers and high-leverage observations.",
    "nonmonotonic_dependence": "Evidence consistent with a relationship that is not simply increasing or decreasing (e.g., U-shaped or cyclical). Standard linear and rank metrics may understate this relationship.",
    "monotonic_nonlinear": "Evidence consistent with a directional relationship that is not well summarized by a straight line. Pearson may significantly understate the relationship compared to rank-based measures.",
    "near_linear": "Evidence consistent with an approximately linear or near-linear relationship. Both variables appear to scale together smoothly.",
    "weak_or_no_relationship": "Little to no evidence of a pairwise association in the observed data.",
    "mixed_or_ambiguous": "The metrics disagree in a way that doesn't strongly match a canonical pattern. The relationship may be complex or noisy."
}

_RECOMMENDATIONS = {
    "not_computable": [
        "Check for constant variables (zero variance).",
        "Check for data misalignment or missingness."
    ],
    "low_power_or_uncertain": [
        "Collect more data.",
        "Rely on domain knowledge rather than statistical significance here."
    ],
    "possible_outlier_or_leverage": [
        "Inspect scatter plots for extreme points.",
        "Consider robust or winsorized sensitivity checks."
    ],
    "nonmonotonic_dependence": [
        "Inspect the relationship visually (e.g., scatter plot with a smoother).",
        "Consider modeling with polynomials, splines, or tree-based methods."
    ],
    "monotonic_nonlinear": [
        "Inspect the scatter plot for curvature.",
        "Consider logarithmic or other monotonic transformations."
    ],
    "near_linear": [
        "A standard linear model or Pearson correlation is likely appropriate here."
    ],
    "weak_or_no_relationship": [
        "Consider whether the relationship might be conditionally masked by a third variable.",
        "This feature may not be a strong linear predictor on its own."
    ],
    "mixed_or_ambiguous": [
        "Inspect the data visually.",
        "Check whether this pattern holds within important segments or clusters."
    ]
}

def generate_explanation(pattern: str, include_caveat: bool = True) -> str:
    exp = _EXPLANATIONS.get(pattern, _EXPLANATIONS["mixed_or_ambiguous"])
    if include_caveat:
        exp = f"{exp} {_CAVEAT}"
    return exp

def generate_recommendations(pattern: str) -> List[str]:
    return _RECOMMENDATIONS.get(pattern, _RECOMMENDATIONS["mixed_or_ambiguous"])
