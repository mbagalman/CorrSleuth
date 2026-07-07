from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from corrsleuth.result import MetricDiagnostics

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


def _nonmonotonic_context(
    pearson: float,
    spearman: float,
    dcor: float | None,
    diagnostics: "MetricDiagnostics | None",
) -> list[str]:
    """Describe *why* a pair reads as nonmonotonic, by the route that fired.

    The label has three routes (distance correlation, the squared-value
    correlation ``sq_corr``, and the bin-reversal oscillation gate), so crediting
    distance correlation unconditionally is wrong: for a circle dCor sits ~0.2
    (below its floor) and the label is driven by ``sq_corr`` instead, and in
    ``lite``/``deep`` mode dCor may be absent entirely while ``sq_corr`` or the
    oscillation gate fires. The secondary ``dependence_type`` axis already
    records which mechanism fired, so key off it — this keeps the explanation
    consistent with the axis and quotes the evidence that actually matched.
    """
    base = f"Pearson ({_fmt(pearson)}) and Spearman ({_fmt(spearman)}) are weak"

    if diagnostics is None:
        dep_type = sq_corr = reversals = bin_lof = None
    else:
        dep_type = diagnostics.dependence_type
        sq_corr = diagnostics.sq_corr
        reversals = diagnostics.bin_reversal_count
        bin_lof = diagnostics.bin_lof_r2_gain

    if dep_type == "oscillating" and reversals is not None:
        detail = (
            f"the binned conditional mean reverses direction {int(reversals)} "
            "times with substantial bin structure"
        )
        if bin_lof is not None:
            detail += f" (lack-of-fit gain {_fmt(bin_lof)})"
        return [
            f"{base}, but {detail} — evidence of an oscillating or periodic "
            "relationship that a simple increasing/decreasing summary misses."
        ]
    if dep_type == "closed_loop_or_multivalued":
        num = (
            f" (squared-value correlation {_fmt(sq_corr)})"
            if sq_corr is not None
            else ""
        )
        return [
            f"{base}, but the variables trace a closed loop{num} — neither is a "
            "function of the other, as with points scattered around a ring, a "
            "dependence the monotone metrics cannot see."
        ]
    if dep_type == "magnitude_linked" and sq_corr is not None:
        return [
            f"{base}, but the correlation of the mean-centered squared values "
            f"({_fmt(sq_corr)}) is strong — the variables are linked through "
            "magnitude, a nonmonotonic pattern the linear and rank metrics miss."
        ]
    if dcor is not None:
        return [
            f"{base}, while distance correlation ({_fmt(dcor)}) is higher; that "
            "disagreement is evidence consistent with dependence that is not "
            "simply increasing or decreasing."
        ]
    # Defensive fallback: each genuine route sets one of the branches above once
    # the diagnostics are present; this covers a result assembled without them.
    return [
        f"{base}, yet the pair reads as nonmonotonic — inspect the scatter plot "
        "for U-shaped, cyclical, or radial structure the monotone metrics miss."
    ]


def _metric_context(
    pattern: str,
    metrics: pd.DataFrame | None,
    diagnostics: "MetricDiagnostics | None" = None,
) -> list[str]:
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

    if (
        pattern == "nonmonotonic_dependence"
        and pearson is not None
        and spearman is not None
    ):
        return _nonmonotonic_context(pearson, spearman, dcor, diagnostics)

    if (
        pattern == "possible_outlier_or_leverage"
        and pearson is not None
        and spearman is not None
    ):
        # This label has two routes, with opposite statistical evidence, so the
        # explanation must match the one that fired:
        #  - direction conflict: Pearson and the rank metrics point opposite ways
        #    (leverage flips the linear sign relative to the monotone trend);
        #  - magnitude dominance: Pearson is much larger than the rank metrics in
        #    the same direction (leverage inflates the linear correlation).
        # Read the sign-conflict magnitude bar from the live classifier constant
        # (lazy import breaks the module cycle: classifier imports this module),
        # so a user overriding CONFLICTING_SIGN_THRESHOLD gets a consistent
        # branch here rather than a hard-coded duplicate.
        from corrsleuth.heuristics import classifier

        if (
            pearson * spearman < 0
            and abs(pearson) >= classifier.CONFLICTING_SIGN_THRESHOLD
            and abs(spearman) >= classifier.CONFLICTING_SIGN_THRESHOLD
        ):
            # Stated from the metrics alone: this label is also reachable via the
            # "sensitivity could not be computed" flag, where the trim/robust
            # check never produced a verdict — so the explanation must not assert
            # that it did. The opposite-sign shape is itself the leverage signature.
            return [
                (
                    f"Pearson ({_fmt(pearson)}) and the rank metrics (Spearman "
                    f"{_fmt(spearman)}, Kendall tau-b {_fmt(kendall)}) point in "
                    "opposite directions — a signature of high-leverage points "
                    "driving the linear correlation against the monotone trend."
                )
            ]
        # Phrased in terms of rank-based metrics generally (not just Spearman):
        # this route can be reached via the Pearson-vs-Kendall gap alone, so
        # naming Spearman specifically could overstate that particular gap.
        return [
            (
                f"Pearson ({_fmt(pearson)}) is much stronger than the rank-based "
                f"metrics (Spearman {_fmt(spearman)}, Kendall tau-b {_fmt(kendall)}), "
                "so the linear association may be more sensitive to extreme values "
                "than the rank-based metrics."
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
    diagnostics: "MetricDiagnostics | None" = None,
) -> str:
    exp = _EXPLANATIONS.get(pattern, _EXPLANATIONS["mixed_or_ambiguous"])

    context_sentences = _metric_context(pattern, metrics, diagnostics)
    if context_sentences:
        exp += " " + " ".join(context_sentences)

    if include_caveat:
        exp = f"{exp} {_CAVEAT}"
    return exp


def generate_recommendations(pattern: str) -> list[str]:
    return _RECOMMENDATIONS.get(pattern, _RECOMMENDATIONS["mixed_or_ambiguous"])
