import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from corrsleuth.api import profile_pair
from corrsleuth.datasets import make_relationship
from corrsleuth.exceptions import InputError
from corrsleuth.result import CorrSleuthResult
from corrsleuth.utils.markdown import escape_markdown_cell, markdown_table


def _result_for_explanation(pattern, metric_values):
    return CorrSleuthResult(
        x_name="x",
        y_name="y",
        metrics=pd.DataFrame(
            [
                {"metric": metric, "value": value}
                for metric, value in metric_values.items()
            ]
        ),
        pattern=pattern,
        warnings=[],
        recommendations=[],
        disagreement_score=0.0,
    )


def test_explain_caveat():
    df = make_relationship("linear_positive", n=100)
    res = profile_pair(df, "x", "y")

    exp_with = res.explain(include_caveat=True)
    assert "causally without proper design" in exp_with

    exp_without = res.explain(include_caveat=False)
    assert "causally without proper design" not in exp_without


def test_explain_near_linear_references_metric_agreement():
    res = _result_for_explanation(
        "near_linear",
        {"pearson": 0.91, "spearman": 0.89, "kendall_tau_b": 0.74},
    )

    explanation = res.explain(include_caveat=False)

    assert "Pearson (0.910) and Spearman (0.890)" in explanation
    assert "closely aligned" in explanation


def test_explain_monotonic_nonlinear_references_rank_linear_gap():
    res = _result_for_explanation(
        "monotonic_nonlinear",
        {"pearson": 0.31, "spearman": 0.78, "kendall_tau_b": 0.59},
    )

    explanation = res.explain(include_caveat=False)

    assert (
        "Spearman (0.780) is meaningfully stronger than Pearson (0.310)" in explanation
    )
    assert "straight line" in explanation


def test_explain_nonmonotonic_dependence_references_standard_metric_gap():
    res = _result_for_explanation(
        "nonmonotonic_dependence",
        {
            "pearson": 0.04,
            "spearman": -0.03,
            "kendall_tau_b": -0.02,
            "distance_correlation": 0.47,
        },
    )

    explanation = res.explain(include_caveat=False)

    assert "Pearson (0.040) and Spearman (-0.030) are weak" in explanation
    assert "distance correlation (0.470) is higher" in explanation


def _noisy_circle(seed: int = 0, n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, size=n)
    radius = 5.0 * (1 + rng.normal(0, 0.05, size=n))
    return pd.DataFrame({"x": radius * np.cos(theta), "y": radius * np.sin(theta)})


def test_explain_lite_circle_credits_squared_value_not_missing_metrics():
    """A circle labeled nonmonotonic via sq_corr in lite mode must explain the
    magnitude/squared-value evidence — the old text fell back to 'standard
    metrics unavailable', giving no positive evidence for a real label (FU-E)."""
    res = profile_pair(_noisy_circle(), "x", "y", mode="lite")
    assert res.pattern == "nonmonotonic_dependence"
    explanation = res.explain(include_caveat=False)
    assert "squared" in explanation.lower()
    assert "unavailable" not in explanation.lower()


def test_explain_standard_circle_does_not_credit_sub_threshold_dcor():
    """A circle's distance correlation sits ~0.2 (below its floor) and does NOT
    drive the label — sq_corr does. The explanation must not credit that
    sub-threshold dcor as the evidence (the old code always did)."""
    pytest.importorskip("dcor")
    pytest.importorskip("sklearn")
    res = profile_pair(_noisy_circle(), "x", "y", mode="standard")
    assert res.pattern == "nonmonotonic_dependence"
    explanation = res.explain(include_caveat=False)
    assert "distance correlation" not in explanation.lower()
    assert "squared" in explanation.lower()


def test_explain_sinusoid_describes_oscillation():
    """A sinusoid labeled nonmonotonic via the oscillation gate must explain the
    reversing bin means, not distance correlation or magnitude."""
    rng = np.random.default_rng(0)
    n = 500
    x = rng.uniform(0, 5 * np.pi, size=n)
    df = pd.DataFrame({"x": x, "y": np.sin(x) + rng.normal(0, 0.1, size=n)})
    res = profile_pair(df, "x", "y", mode="lite")
    assert res.pattern == "nonmonotonic_dependence"
    explanation = res.explain(include_caveat=False)
    assert "reverses direction" in explanation


def test_explain_possible_outlier_or_leverage_references_rank_disagreement():
    res = _result_for_explanation(
        "possible_outlier_or_leverage",
        {"pearson": 0.82, "spearman": 0.31, "kendall_tau_b": 0.21},
    )

    explanation = res.explain(include_caveat=False)

    assert "Pearson (0.820) is much stronger than the rank-based metrics" in explanation
    assert "Spearman 0.310" in explanation
    assert "extreme values" in explanation


def test_explain_possible_outlier_or_leverage_sign_conflict_describes_direction():
    """When the leverage label comes from a Pearson/Spearman sign conflict, the
    explanation must describe a direction conflict — not 'Pearson is much
    stronger', which is false when the rank metrics are equally strong but
    opposite in sign."""
    res = _result_for_explanation(
        "possible_outlier_or_leverage",
        {"pearson": 0.96, "spearman": -0.94, "kendall_tau_b": -0.96},
    )

    explanation = res.explain(include_caveat=False)

    assert "opposite directions" in explanation
    assert "much stronger" not in explanation
    # Honesty (C1 #5): this label is also reachable when the trim/robust check
    # could not run, so the explanation must not assert that a check produced a
    # verdict — it states the sign conflict, which is the leverage signature.
    assert "trim/robust check indicates" not in explanation
    assert "signature of high-leverage points" in explanation


def test_explain_weak_or_no_relationship_references_lite_metric_limits():
    res = _result_for_explanation(
        "weak_or_no_relationship",
        {"pearson": 0.02, "spearman": -0.04, "kendall_tau_b": -0.02},
    )

    explanation = res.explain(include_caveat=False)

    assert "Pearson (0.020) and Spearman (-0.040) are weak" in explanation
    assert "without standard-mode nonlinear metrics" in explanation


def test_summary():
    df = make_relationship("linear_positive", n=100)
    res = profile_pair(df, "x", "y")

    summary_text = res.summary()
    assert "Relationship Profile: x vs y" in summary_text
    assert "Primary pattern:" in summary_text
    assert "Metrics:" in summary_text
    assert "Diagnostics:" in summary_text
    assert "disagreement_score" in summary_text
    assert "rank_linear_gap" in summary_text
    assert "nonmonotonic_gap" in summary_text
    # pearson_trimmed (the level Pearson moved to) is shown beside its delta on the
    # human-readable surfaces, not only in to_dict()/to_frame() (C5 #4).
    assert "pearson_trimmed" in summary_text
    assert "pearson\\_trimmed" in res.to_markdown()  # underscore escaped in markdown
    # Diagnostics rows are colon-aligned to a shared width even for the longest
    # label (pearson_spearman_signed_gap), so the ": " column lines up (C5 #5).
    lines = summary_text.splitlines()
    start = lines.index("Diagnostics:") + 1
    end = lines.index("", start)
    diag_lines = lines[start:end]
    colon_cols = {ln.index(":") for ln in diag_lines}
    assert len(colon_cols) == 1, diag_lines
    assert any("pearson_spearman_signed_gap" in ln for ln in diag_lines)
    assert "Recommendations:" in summary_text
    assert "Caveat:" in summary_text


def test_signed_pearson_spearman_gap_surfaced_in_text_output():
    """The signed Pearson-Spearman gap reveals sign disagreement that the
    absolute rank_linear_gap hides, so it must appear in both summary() and
    to_markdown(), not only in to_dict()/to_frame() (CR-7)."""
    import numpy as np

    x = np.arange(50, dtype=float)
    y = -np.arange(50, dtype=float)
    x[-1] = (
        10000  # one huge outlier flips Pearson positive while Spearman stays negative
    )
    y[-1] = 10000
    res = profile_pair(pd.DataFrame({"x": x, "y": y}), "x", "y")

    signed = res.diagnostics.pearson_spearman_signed_gap
    assert signed is not None
    # The signed gap carries strictly more (directional) information than its abs.
    assert abs(signed) > res.diagnostics.rank_linear_gap

    summary_text = res.summary()
    assert "pearson_spearman_signed_gap" in summary_text
    assert f"{signed:.3f}" in summary_text

    markdown = res.to_markdown()
    assert "pearson\\_spearman\\_signed\\_gap" in markdown


def test_summary_can_hide_caveat():
    df = make_relationship("linear_positive", n=100)
    res = profile_pair(df, "x", "y")

    summary_text = res.summary(include_caveat=False)

    assert "Caveat:" not in summary_text


def test_pair_result_to_markdown_includes_core_sections():
    df = make_relationship("linear_positive", n=100, random_state=42)
    res = profile_pair(df, "x", "y")

    markdown = res.to_markdown()

    assert markdown.startswith("# CorrSleuth Pair Report: `x` vs `y`")
    assert "**Primary pattern:** `near_linear`" in markdown
    assert "## Metrics" in markdown
    assert "| Metric | Value |" in markdown
    assert "| pearson |" in markdown
    assert "## Diagnostics" in markdown
    assert "| disagreement\\_score |" in markdown
    assert "## Warnings" in markdown
    assert "## Recommendations" in markdown
    assert "## Caveat" in markdown
    assert "causally without proper design" in markdown


def test_pair_result_to_markdown_title_neutralizes_hostile_column_names():
    """A column name with a backtick or newline must not break out of the title's
    code span or inject heading structure (C5 #3)."""
    df = pd.DataFrame({"a`b\nc": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.0, 5.0, 4.0]})
    res = profile_pair(df, "a`b\nc", "y")

    markdown = res.to_markdown()
    title = markdown.splitlines()[0]

    # The whole title is still a single heading line, and the backtick that would
    # have closed the code span (and the newline that would have split the line)
    # are gone, so nothing after the name is interpreted as markdown structure.
    assert title == "# CorrSleuth Pair Report: `ab c` vs `y`"
    assert "\n" not in title


def test_pair_result_to_markdown_can_hide_caveat():
    df = make_relationship("linear_positive", n=100, random_state=42)
    res = profile_pair(df, "x", "y")

    markdown = res.to_markdown(include_caveat=False)

    assert "## Caveat" not in markdown
    assert "causally without proper design" not in markdown


def test_pair_result_to_markdown_includes_bootstrap_sections():
    df = make_relationship("linear_positive", n=80, random_state=42)
    res = profile_pair(df, "x", "y", bootstrap=10, random_state=123)

    markdown = res.to_markdown(include_caveat=False)

    assert "## Bootstrap Intervals" in markdown
    assert "| Metric | CI low | CI high | Successful samples | Metric set |" in markdown
    assert "## Pattern Stability" in markdown
    assert "| Stability | Label | Metric set | Samples | Label counts |" in markdown
    assert "{" not in markdown
    assert "near\\_linear:" in markdown


def test_markdown_helpers_escape_cells_and_handle_arrays():
    assert escape_markdown_cell("a | b\nnear_linear *x* [y] `z`") == (
        "a \\| b near\\_linear \\*x\\* \\[y\\] \\`z\\`"
    )
    rendered = markdown_table(["Value"], [[np.array([1, 2])]])

    assert "\\[1 2\\]" in rendered


def test_plot_returns_figure():
    df = make_relationship("linear_positive", n=100)
    res = profile_pair(df, "x", "y")

    fig = res.plot(show=False)
    assert isinstance(fig, plt.Figure)


def test_plot_text_panel_includes_pattern_metrics_and_diagnostics():
    df = make_relationship("linear_positive", n=100, random_state=42)
    res = profile_pair(df, "x", "y")

    fig = res.plot(show=False)
    text_panel = fig.axes[2]
    rendered_text = "\n".join(text.get_text() for text in text_panel.texts)

    assert "Primary Pattern" in rendered_text
    assert res.pattern in rendered_text
    assert "n_used: 100" in rendered_text
    assert "Pearson:" in rendered_text
    assert "Spearman:" in rendered_text
    assert "Kendall Tau B:" in rendered_text
    assert "Diagnostics" in rendered_text
    assert "Disagreement:" in rendered_text
    assert "Rank-linear gap:" in rendered_text
    assert "Warnings" in rendered_text


def test_plot_text_panel_stays_on_panel_for_deep_mode_many_metrics():
    """The text panel must scale to fit its rows: deep mode's larger metrics
    block previously walked y_pos below the axes, rendering the warnings (the
    load-bearing caveats) off-panel (C7 #2)."""
    pytest.importorskip("dcor")
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    x = rng.normal(size=120)
    df = pd.DataFrame({"x": x, "y": x**2 + rng.normal(0, 0.3, size=120)})
    res = profile_pair(df, "x", "y", mode="deep")

    fig = res.plot(show=False)
    text_panel = fig.axes[2]
    y_positions = [text.get_position()[1] for text in text_panel.texts]

    # Deep mode has many metric rows, and every line still lands inside the axes.
    assert len(res.metrics) > 5
    assert all(0.0 <= y <= 1.0 for y in y_positions)
    # The warnings section header is present and on-panel (not clipped below).
    assert "Warnings" in "\n".join(t.get_text() for t in text_panel.texts)


def test_serialization():
    df = make_relationship("linear_positive", n=100)
    res = profile_pair(df, "x", "y")

    d = res.to_dict()
    assert d["x"] == "x"
    assert d["y"] == "y"
    assert "pattern" in d
    assert isinstance(d["metrics"], list)

    frame = res.to_frame()
    assert "pattern" in frame.columns
    assert "value" in frame.columns
    assert len(frame) >= 3  # at least core metrics
    assert res.bootstrap_intervals is None
    assert d["bootstrap_intervals"] is None


def test_to_dict_normalizes_unavailable_metrics_to_none_not_nan():
    """An unavailable metric value in the metrics table is float-NaN inside the
    frame (mixed float column), but to_dict() must emit None, not a bare
    float('nan') that is invalid under strict JSON."""
    import json
    import math

    pytest.importorskip("dcor")
    pytest.importorskip("sklearn")
    # n=3 standard mode: mutual information is not computable and lands as NaN.
    res = profile_pair(
        pd.DataFrame({"x": [1, 2, 3], "y": [1, 3, 2]}), "x", "y", mode="standard"
    )
    d = res.to_dict()

    mi = next(r for r in d["metrics"] if r["metric"] == "mutual_information")
    assert mi["value"] is None
    assert not any(
        isinstance(r["value"], float) and math.isnan(r["value"]) for r in d["metrics"]
    )
    # The whole public dict is strict-JSON serializable (rejects NaN/Infinity).
    json.dumps(d, allow_nan=False)


def test_result_exposes_structured_diagnostics():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [1, 2, 3, 4, 5]})
    res = profile_pair(df, "x", "y")

    assert res.diagnostics.rank_linear_gap == pytest.approx(0.0)
    assert res.diagnostics.pearson_spearman_signed_gap == pytest.approx(0.0)
    assert res.diagnostics.pearson_kendall_gap == pytest.approx(0.0)
    assert res.diagnostics.nonmonotonic_gap is None
    assert res.diagnostics.disagreement_score == pytest.approx(res.disagreement_score)


def test_segment_stepness_surfaced_on_diagnostics_and_frames():
    """segment_stepness is the number behind the step-vs-smooth mean_shape call;
    it must be exposed on result.diagnostics and in the serialized surfaces, not
    just consumed internally (FU-I / Chunk 5 #1). A clean step reads ~1, a smooth
    curve <= 0."""
    step = make_relationship("threshold_step", n=500, noise=0.1, random_state=42)
    res = profile_pair(step, "x", "y", mode="lite")

    assert res.diagnostics.mean_shape == "step_or_threshold"
    assert res.diagnostics.segment_stepness is not None
    assert res.diagnostics.segment_stepness > 0.8
    assert "segment_stepness" in res.to_dict()["diagnostics"]
    assert "diagnostic_segment_stepness" in res.to_frame().columns
    assert "segment_stepness" in res.summary()
    assert "segment\\_stepness" in res.to_markdown()  # underscore escaped

    smooth = make_relationship(
        "exponential_monotonic", n=500, noise=0.1, random_state=42
    )
    smooth_stepness = profile_pair(
        smooth, "x", "y", mode="lite"
    ).diagnostics.segment_stepness
    assert smooth_stepness is not None and smooth_stepness < 0.5


def test_disagreement_score_zero_when_correlations_unavailable():
    """Constant x makes pearson/spearman unavailable (None). The disagreement
    score must treat them as absent (contributing 0), not as a value of 0.0
    pulled in by an `or 0.0` fallback."""
    df = pd.DataFrame({"x": [3.0] * 50, "y": list(range(50))})
    res = profile_pair(df, "x", "y")

    # Both correlations are unavailable, so no metric is in disagreement.
    metric_values = dict(zip(res.metrics["metric"], res.metrics["value"], strict=True))
    assert metric_values["pearson"] is None
    assert metric_values["spearman"] is None
    assert res.disagreement_score == pytest.approx(0.0)


def test_serialization_includes_nested_and_flattened_diagnostics():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [1, 2, 3, 4, 5]})
    res = profile_pair(df, "x", "y")

    as_dict = res.to_dict()
    assert "diagnostics" in as_dict
    assert as_dict["diagnostics"]["rank_linear_gap"] == pytest.approx(0.0)
    assert as_dict["diagnostics"]["nonmonotonic_gap"] is None

    frame = res.to_frame()
    assert "diagnostic_rank_linear_gap" in frame.columns
    assert "diagnostic_pearson_spearman_signed_gap" in frame.columns
    assert "diagnostic_nonmonotonic_gap" in frame.columns
    assert "diagnostic_pearson_kendall_gap" in frame.columns
    assert "diagnostic_disagreement_score" in frame.columns
    assert "diagnostic_pearson_trimmed" in frame.columns
    assert "diagnostic_pearson_trim_delta" in frame.columns
    assert frame["diagnostic_rank_linear_gap"].iloc[0] == pytest.approx(0.0)


def test_secondary_axes_surfaced_in_every_output_form():
    """The five secondary diagnostic axes appear on result.diagnostics and in
    every rendered/serialized surface (summary, markdown, to_dict, to_frame)."""
    pytest.importorskip("dcor")
    pytest.importorskip("sklearn")
    df = make_relationship("circular", n=500, noise=0.1, random_state=42)
    res = profile_pair(df, "x", "y", mode="deep")

    # The circle is the showcase: closed loop, neither variable a function of
    # the other -- a story the single primary label cannot carry.
    d = res.diagnostics
    assert d.dependence_type == "closed_loop_or_multivalued"
    assert d.functional_direction == "neither_direction"
    assert d.variance_shape in (
        None,
        "constant",
        "increasing_spread",
        "decreasing_spread",
        "edge_high_spread",
        "center_high_spread",
    )

    nested = res.to_dict()["diagnostics"]
    assert nested["dependence_type"] == "closed_loop_or_multivalued"
    assert nested["mean_shape"] == d.mean_shape

    frame = res.to_frame()
    for col in (
        "diagnostic_mean_shape",
        "diagnostic_variance_shape",
        "diagnostic_dependence_type",
        "diagnostic_outlier_sensitivity",
        "diagnostic_functional_direction",
    ):
        assert col in frame.columns
    assert frame["diagnostic_dependence_type"].iloc[0] == "closed_loop_or_multivalued"

    summary = res.summary()
    assert "Relationship axes:" in summary
    assert "closed_loop_or_multivalued" in summary

    md = res.to_markdown()
    assert "## Relationship Axes" in md
    # Markdown escapes underscores in table cells (so they don't render italic).
    assert escape_markdown_cell("closed_loop_or_multivalued") in md


def test_sq_corr_robust_surfaced_on_diagnostics_and_every_output_form():
    """The robust squared-correlation companion is stored on
    result.diagnostics.sq_corr_robust and appears in summary, markdown, to_dict,
    and to_frame — not just consumed inside the cascade. A U-shape's sq_corr is
    robust (spread over many points), so the two values track closely."""
    rng = np.random.default_rng(0)
    x = rng.uniform(-3, 3, size=400)
    df = pd.DataFrame({"x": x, "y": x**2 + rng.normal(0, 0.3, size=400)})
    res = profile_pair(df, "x", "y")  # lite mode: sq_corr_robust needs no extras

    robust = res.diagnostics.sq_corr_robust
    assert robust is not None
    # Genuine radial link: robust stays well above the floor, and barely below
    # the raw value (the signal is spread over many points, not a few extremes).
    assert robust > 0.35
    assert abs(res.diagnostics.sq_corr) - robust < 0.15

    assert res.to_dict()["diagnostics"]["sq_corr_robust"] == pytest.approx(robust)
    assert "diagnostic_sq_corr_robust" in res.to_frame().columns
    assert "sq_corr_robust" in res.summary()
    # Markdown escapes underscores in table cells so they don't render italic.
    assert escape_markdown_cell("sq_corr_robust") in res.to_markdown()


def test_cluster_split_diagnostics_surfaced_on_every_output_form():
    """The two-group split diagnostics are stored on result.diagnostics and
    appear in summary, markdown, to_dict, and to_frame."""
    rng = np.random.default_rng(0)
    n = 400
    n1 = n // 2
    x = np.concatenate([rng.normal(0, 1, n1), rng.normal(5, 1, n - n1)])
    y = np.concatenate([rng.normal(0, 1, n1), rng.normal(5, 1, n - n1)])
    res = profile_pair(pd.DataFrame({"x": x, "y": y}), "x", "y")  # lite mode

    d = res.diagnostics
    assert d.cluster_split_r2 > 0.85
    assert d.cluster_valley_share < 0.03
    assert d.cluster_min_share == pytest.approx(0.5, abs=0.05)
    assert d.pearson_within_cluster < 0.15

    nested = res.to_dict()["diagnostics"]
    assert nested["cluster_split_r2"] == pytest.approx(d.cluster_split_r2)
    assert nested["pearson_within_cluster"] == pytest.approx(d.pearson_within_cluster)

    frame = res.to_frame()
    for col in (
        "diagnostic_cluster_split_r2",
        "diagnostic_cluster_valley_share",
        "diagnostic_cluster_min_share",
        "diagnostic_pearson_within_cluster",
    ):
        assert col in frame.columns

    assert "cluster_split_r2" in res.summary()
    assert escape_markdown_cell("pearson_within_cluster") in res.to_markdown()


def test_secondary_axes_default_to_na_when_not_assessable():
    """A constant input can populate no axes; they render as NA rather than
    raising, and serialize as None."""
    df = pd.DataFrame({"x": [3.0] * 50, "y": list(range(50))})
    res = profile_pair(df, "x", "y")

    assert res.diagnostics.mean_shape is None
    assert res.to_dict()["diagnostics"]["dependence_type"] is None
    assert "mean_shape           : NA" in res.summary()


def test_bootstrap_intervals_are_deterministic_and_serialized():
    df = make_relationship("linear_positive", n=80, random_state=42)

    res1 = profile_pair(df, "x", "y", bootstrap=25, random_state=123)
    res2 = profile_pair(df, "x", "y", bootstrap=25, random_state=123)

    assert res1.bootstrap_intervals is not None
    assert res1.bootstrap_stability is not None
    assert list(res1.bootstrap_intervals["metric"]) == [
        "pearson",
        "spearman",
        "kendall_tau_b",
    ]
    pd.testing.assert_frame_equal(res1.bootstrap_intervals, res2.bootstrap_intervals)
    assert res1.bootstrap_stability.to_dict() == res2.bootstrap_stability.to_dict()
    assert (
        res1.bootstrap_intervals["ci_low"] <= res1.bootstrap_intervals["ci_high"]
    ).all()

    summary = res1.summary(include_caveat=False)
    assert "Bootstrap intervals:" in summary
    assert "Pattern stability:" in summary
    assert "pearson" in summary
    # label_counts render as sorted "label: n" pairs (via _format_label_counts),
    # not a raw Python dict repr, matching to_markdown() (C5 #5).
    label_counts_line = next(ln for ln in summary.splitlines() if "label_counts:" in ln)
    assert "{" not in label_counts_line and "}" not in label_counts_line
    assert "'" not in label_counts_line

    as_dict = res1.to_dict()
    assert as_dict["bootstrap_intervals"] is not None
    assert as_dict["bootstrap_stability"] is not None
    assert as_dict["pattern_stability"] == pytest.approx(res1.pattern_stability)
    assert as_dict["bootstrap_label_counts"] == res1.bootstrap_label_counts
    assert as_dict["stability_label"] == res1.stability_label

    frame = res1.to_frame()
    assert "bootstrap_ci_low" in frame.columns
    assert "bootstrap_ci_high" in frame.columns
    assert "bootstrap_sample_size" in frame.columns
    assert "pattern_stability" in frame.columns
    assert "stability_label" in frame.columns
    assert "stability_metric_set" in frame.columns
    assert "bootstrap_pearson_ci_low" not in frame.columns
    assert "bootstrap_spearman_ci_high" not in frame.columns

    explanation = res1.explain(include_caveat=False)
    assert "Bootstrap resampling assigned the same diagnostic label" in explanation


def test_standard_mode_bootstrap_defaults_to_lite_metrics():
    pytest.importorskip("dcor")
    pytest.importorskip("sklearn")
    df = make_relationship("u_shape", n=80, random_state=42)

    res = profile_pair(df, "x", "y", mode="standard", bootstrap=10, random_state=123)

    assert res.bootstrap_intervals is not None
    assert set(res.bootstrap_intervals["metric"]) == {
        "pearson",
        "spearman",
        "kendall_tau_b",
    }
    assert res.bootstrap_stability is not None
    assert res.bootstrap_stability.metric_set == "lite"


def test_standard_bootstrap_metrics_require_explicit_opt_in():
    pytest.importorskip("dcor")
    pytest.importorskip("sklearn")
    df = make_relationship("u_shape", n=80, random_state=42)

    res = profile_pair(
        df,
        "x",
        "y",
        mode="standard",
        bootstrap=5,
        bootstrap_metrics="standard",
        random_state=123,
    )

    assert res.bootstrap_intervals is not None
    assert "distance_correlation" in set(res.bootstrap_intervals["metric"])
    assert "mutual_information" in set(res.bootstrap_intervals["metric"])
    assert res.bootstrap_stability is not None
    assert res.bootstrap_stability.metric_set == "standard"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"bootstrap": -1}, "positive integer"),
        ({"bootstrap": 0}, "positive integer"),
        ({"bootstrap": False}, "positive integer"),
        ({"bootstrap": 5, "bootstrap_metrics": "bogus"}, "bootstrap_metrics"),
        ({"bootstrap": 5, "bootstrap_metrics": []}, "at least one metric"),
        ({"bootstrap": 5, "bootstrap_metrics": ["pearson", "bogus"]}, "Unsupported"),
        ({"bootstrap": 5, "max_n_for_bootstrap": 0}, "positive integer"),
        ({"max_n_for_dcor": -1}, "max_n_for_dcor must be a positive integer"),
        ({"max_n_for_dcor": 0}, "max_n_for_dcor must be a positive integer"),
        ({"max_n_for_dcor": True}, "max_n_for_dcor must be a positive integer"),
        ({"max_n_for_dcor": 2.5}, "max_n_for_dcor must be a positive integer"),
    ],
)
def test_bootstrap_invalid_inputs_raise(kwargs, message):
    df = make_relationship("linear_positive", n=80, random_state=42)

    with pytest.raises(InputError, match=message):
        profile_pair(df, "x", "y", **kwargs)


def test_bootstrap_metrics_explicit_duplicates_are_deduped():
    """An explicit ``bootstrap_metrics`` sequence with a repeated name must not
    produce duplicate interval rows (C4 #4)."""
    df = make_relationship("linear_positive", n=80, random_state=42)

    res = profile_pair(
        df, "x", "y", bootstrap=5, bootstrap_metrics=["pearson", "pearson"]
    )

    assert res.bootstrap_intervals is not None
    pearson_rows = res.bootstrap_intervals[
        res.bootstrap_intervals["metric"] == "pearson"
    ]
    assert len(pearson_rows) == 1


def test_bootstrap_sequence_metric_set_is_preserved():
    df = make_relationship("linear_positive", n=80, random_state=42)

    res = profile_pair(
        df,
        "x",
        "y",
        bootstrap=10,
        bootstrap_metrics=["pearson"],
        random_state=123,
    )

    assert res.bootstrap_intervals is not None
    assert list(res.bootstrap_intervals["metric"]) == ["pearson"]
    assert res.bootstrap_intervals["metric_set"].iloc[0] == "pearson"
    assert res.bootstrap_stability is not None
    assert res.bootstrap_stability.metric_set == "pearson"


def test_pattern_stability_is_meaningful_for_custom_bootstrap_metric_subset():
    """A custom bootstrap_metrics subset must not break pattern stability.

    The stability cascade always needs the lite triple; if only the requested
    metric were computed per replicate (e.g. ["pearson"]), the cascade would
    short-circuit to not_computable and report 0.0 stability for an obviously
    stable relationship. Intervals still follow the requested subset."""
    df = make_relationship("linear_positive", n=120, random_state=42)

    res = profile_pair(
        df, "x", "y", bootstrap=25, bootstrap_metrics=["pearson"], random_state=7
    )

    assert res.pattern == "near_linear"
    # Intervals only for the requested metric...
    assert list(res.bootstrap_intervals["metric"]) == ["pearson"]
    # ...but stability is computed on the full cascade, so the strong linear
    # relationship is recovered rather than collapsing to not_computable.
    assert "not_computable" not in res.bootstrap_label_counts
    assert res.bootstrap_label_counts.get("near_linear", 0) >= 1
    assert res.pattern_stability > 0.5


def test_bootstrap_cap_warning_reaches_result_and_records_sample_size():
    df = make_relationship("linear_positive", n=80, random_state=42)

    res = profile_pair(
        df,
        "x",
        "y",
        bootstrap=10,
        random_state=123,
        max_n_for_bootstrap=40,
    )

    assert res.bootstrap_intervals is not None
    assert set(res.bootstrap_intervals["sample_size"]) == {40}
    assert any("Bootstrap samples are capped at 40 rows" in w for w in res.warnings)
    # The cap warning must disclose the m-out-of-n widening, not read as a pure
    # performance cap (CR-3).
    assert any("m-out-of-n" in w and "conservative" in w for w in res.warnings)


def test_bootstrap_incomplete_warning_attributes_degenerate_resamples():
    """When some resamples draw a near-constant column the metric is legitimately
    undefined there. The reliability warning must describe that, not claim the
    resamples were 'non-computable' (which reads as a failure) (CR-8)."""
    import numpy as np

    # A single differing value among many: many resamples draw an all-equal x.
    x = np.array([5.0] * 40 + [6.0])
    y = np.arange(41, dtype=float)
    res = profile_pair(
        pd.DataFrame({"x": x, "y": y}),
        "x",
        "y",
        bootstrap=200,
        random_state=0,
    )

    incomplete = [w for w in res.warnings if "Bootstrap intervals for" in w]
    assert incomplete, "expected an incomplete-resamples warning"
    warning = incomplete[0]
    assert "undefined on some resamples" in warning
    assert "near-constant" in warning
    assert "non-computable" not in warning
    # The interval really is based on fewer than all requested resamples.
    n_success = int(res.bootstrap_intervals["n_success"].iloc[0])
    assert n_success < 200


def test_lite_pattern_stability_caveat_for_standard_nonmonotonic_label():
    pytest.importorskip("dcor")
    pytest.importorskip("sklearn")
    df = make_relationship("u_shape", n=120, random_state=42)

    res = profile_pair(df, "x", "y", mode="standard", bootstrap=10, random_state=123)

    assert res.pattern == "nonmonotonic_dependence"
    assert res.bootstrap_stability is not None
    assert res.bootstrap_stability.metric_set == "lite"
    assert res.bootstrap_stability.dcor_in_cascade is False
    assert any("may not fully test" in w for w in res.warnings)
    assert "may not fully test" in res.explain(include_caveat=False)


def test_standard_only_caveat_agrees_across_surfaces_for_explicit_subset():
    """C5 #2: with an explicit bootstrap subset the metric_set label is not
    'lite' (it names the subset), but dcor is still absent from the replicate
    cascade. The warnings list and explain() must agree — both gate on
    ``dcor_in_cascade``, not the metric_set string — so the caveat fires on both,
    not just the warnings list."""
    pytest.importorskip("dcor")
    pytest.importorskip("sklearn")
    df = make_relationship("u_shape", n=120, random_state=42)

    res = profile_pair(
        df,
        "x",
        "y",
        mode="standard",
        bootstrap=10,
        bootstrap_metrics=["pearson"],
        random_state=123,
    )

    assert res.pattern == "nonmonotonic_dependence"
    assert res.bootstrap_stability is not None
    # The metric_set label is NOT "lite" here — the old gate would have missed it.
    assert res.bootstrap_stability.metric_set == "pearson"
    assert res.bootstrap_stability.dcor_in_cascade is False
    # Both surfaces now carry the caveat, in agreement.
    assert any("may not fully test" in w for w in res.warnings)
    assert "may not fully test" in res.explain(include_caveat=False)


def test_bootstrap_stability_recomputes_trim_sensitivity_per_replicate():
    """A trim-stable near_linear relationship with a large Pearson-Kendall gap
    must not be labeled possible_outlier_or_leverage in the bootstrap.

    The leverage rule gates on trim sensitivity. Replicates previously inherited
    a blanket ``outlier_sensitivity_unavailable`` flag, so any resample with
    ``|pearson| - |kendall| > 0.25`` was labeled leverage even though the
    original profile proved Pearson trim-stable — collapsing pattern stability
    to ~0 against the real near_linear label. Trim sensitivity is now recomputed
    per replicate."""
    pytest.importorskip("dcor")
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(3)
    n = 300
    x = rng.standard_t(2.0, size=n)
    y = x + rng.standard_t(2.0, size=n) * 0.45
    res = profile_pair(
        pd.DataFrame({"x": x, "y": y}),
        "x",
        "y",
        mode="deep",
        bootstrap=50,
        random_state=0,
    )
    metrics = {row["metric"]: row["value"] for _, row in res.metrics.iterrows()}

    # Preconditions: this is the leverage-flag trigger regime.
    assert res.pattern == "near_linear"
    assert abs(metrics["pearson"]) - abs(metrics["kendall_tau_b"]) > 0.25
    assert res.diagnostics.pearson_trim_delta < 0.20  # trim-stable, not leverage

    # The fix: replicates of a trim-stable pair are not labeled leverage, so
    # stability reflects the real label instead of collapsing to ~0.
    assert res.bootstrap_label_counts.get("possible_outlier_or_leverage", 0) == 0
    assert res.pattern_stability >= 0.5


def test_bootstrap_stability_recomputes_sq_corr_robust_per_replicate():
    """A lite ``nonmonotonic_dependence`` label assigned via the sq_corr route
    must be reproducible by the bootstrap. The classifier gates that route on
    *both* sq_corr and sq_corr_robust, so the per-replicate cascade has to compute
    sq_corr_robust too — omitting it made every replicate fail the gate and
    collapse pattern_stability to 0.0 with a `weak_or_no_relationship` count."""
    rng = np.random.default_rng(1)
    x = rng.uniform(-3, 3, size=400)
    df = pd.DataFrame({"x": x, "y": x**2 + rng.normal(0, 0.4, size=400)})

    res = profile_pair(df, "x", "y", mode="lite", bootstrap=30, random_state=7)

    # A U-shape reaches nonmonotonic_dependence via sq_corr alone in lite mode.
    assert res.pattern == "nonmonotonic_dependence"
    assert res.diagnostics.dependence_type == "magnitude_linked"
    # The label is stable — replicates reproduce it, not collapse to weak.
    assert res.bootstrap_stability.pattern_stability >= 0.8
    assert (
        res.bootstrap_stability.bootstrap_label_counts.get("weak_or_no_relationship", 0)
        <= 3
    )


def test_bootstrap_intervals_skipped_below_min_n_floor():
    """Below n=20 a percentile bootstrap is too unreliable, so intervals are
    returned as None with a warning; pattern stability is still reported
    (intervals-only floor)."""
    rng = np.random.default_rng(0)
    n = 12
    x = rng.normal(size=n)
    y = 2 * x + rng.normal(0, 0.1, size=n)
    res = profile_pair(
        pd.DataFrame({"x": x, "y": y}), "x", "y", bootstrap=50, random_state=1
    )

    assert res.bootstrap_intervals is None
    assert any("bootstrap intervals are not computed" in w for w in res.warnings)
    # Stability is intentionally still computed below the interval floor.
    assert res.bootstrap_stability is not None
    assert res.pattern_stability is not None
    # Rendering must tolerate stability-without-intervals.
    res.summary()
    res.to_markdown()


def test_bootstrap_intervals_present_at_floor():
    """At n >= 20 intervals are still produced (the floor is exclusive)."""
    rng = np.random.default_rng(0)
    n = 22
    x = rng.normal(size=n)
    y = 2 * x + rng.normal(0, 0.1, size=n)
    res = profile_pair(
        pd.DataFrame({"x": x, "y": y}), "x", "y", bootstrap=50, random_state=1
    )

    assert res.bootstrap_intervals is not None
    assert not any("bootstrap intervals are not computed" in w for w in res.warnings)


def test_bootstrap_cap_below_interval_floor_suppresses_intervals():
    """max_n_for_bootstrap must not bypass the interval floor: the floor keys off
    the effective per-replicate size, not the original n_used."""
    rng = np.random.default_rng(0)
    n = 100
    x = rng.normal(size=n)
    y = 2 * x + rng.normal(0, 0.1, size=n)
    res = profile_pair(
        pd.DataFrame({"x": x, "y": y}),
        "x",
        "y",
        bootstrap=50,
        max_n_for_bootstrap=10,  # caps replicates to 10 rows (< 20)
        random_state=1,
    )

    assert res.bootstrap_intervals is None
    assert any("bootstrap intervals are not computed" in w for w in res.warnings)


def test_bootstrap_cap_below_low_n_suppresses_meaningless_stability():
    """When the cap pushes replicates below the low-power threshold while the
    original sample is well above it, pattern stability would collapse to 0.0
    (every replicate judged low_power_or_uncertain) against a clean near_linear
    label. It is suppressed (None) with a warning instead."""
    rng = np.random.default_rng(0)
    n = 100
    x = rng.normal(size=n)
    y = 2 * x + rng.normal(0, 0.1, size=n)
    res = profile_pair(
        pd.DataFrame({"x": x, "y": y}),
        "x",
        "y",
        bootstrap=50,
        max_n_for_bootstrap=25,  # 20 <= 25 < 30: intervals OK, stability not
        random_state=1,
    )

    assert res.pattern == "near_linear"
    assert res.bootstrap_intervals is not None  # 25 >= interval floor
    assert res.bootstrap_stability is None
    assert res.pattern_stability is None
    assert any(
        "cannot meaningfully test the full-sample label" in w for w in res.warnings
    )


def test_bootstrap_stability_is_none_when_disabled():
    df = make_relationship("linear_positive", n=80, random_state=42)
    res = profile_pair(df, "x", "y")

    assert res.bootstrap_intervals is None
    assert res.bootstrap_stability is None
    assert res.pattern_stability is None
    assert res.bootstrap_label_counts is None
    assert res.stability_label is None


def test_bootstrap_label_counts_sum_matches_iterations():
    df = make_relationship("linear_positive", n=80, random_state=42)
    res = profile_pair(df, "x", "y", bootstrap=15, random_state=123)

    stability = res.bootstrap_stability
    assert stability is not None
    assert sum(stability.bootstrap_label_counts.values()) == stability.n_iterations
    assert stability.n_iterations == stability.n_bootstrap == 15
    assert res.pattern in stability.bootstrap_label_counts


def test_stability_label_thresholds():
    from corrsleuth.metrics.bootstrap import _stability_label

    assert _stability_label(0.0) == "low"
    assert _stability_label(0.49) == "low"
    assert _stability_label(0.50) == "medium"
    assert _stability_label(0.51) == "medium"
    assert _stability_label(0.79) == "medium"
    assert _stability_label(0.80) == "high"
    assert _stability_label(0.81) == "high"
    assert _stability_label(1.0) == "high"


def test_compute_bootstrap_intervals_wrapper_returns_only_intervals():
    from corrsleuth.metrics import compute_bootstrap_intervals
    from corrsleuth.validation.input import validate_pair

    df = make_relationship("linear_positive", n=80, random_state=42)
    pair = validate_pair(df, "x", "y")

    intervals = compute_bootstrap_intervals(
        pair=pair,
        bootstrap=10,
        bootstrap_metrics="lite",
        random_state=123,
        max_n_for_bootstrap=5000,
    )

    assert isinstance(intervals, pd.DataFrame)
    assert set(intervals["metric"]) == {"pearson", "spearman", "kendall_tau_b"}


def test_high_tie_rate_warning_reaches_result():
    df = pd.DataFrame(
        {
            "category": [0, 1, 2] * 33 + [0],
            "score": np.linspace(0, 1, 100),
        }
    )
    res = profile_pair(df, "category", "score")

    matching = [w for w in res.warnings if "category" in w and "tie rate" in w]
    assert matching, (
        f"expected tie-rate warning for 'category' in result, got {res.warnings}"
    )


def test_constant_input_safe_rendering():
    df = pd.DataFrame({"x": [1, 1, 1, 1], "y": [1, 2, 3, 4]})
    res = profile_pair(df, "x", "y")
    assert res.pattern == "not_computable"
    summary_text = res.summary()
    assert "NA" in summary_text
    fig = res.plot()
    assert isinstance(fig, plt.Figure)


def test_plotting_uses_clean_data():
    df = make_relationship("linear_positive", n=100)
    df.loc[0, "x"] = np.nan
    res = profile_pair(df, "x", "y")

    # Mutate df after profiling
    df["x"] = 0

    fig = res.plot()
    # It shouldn't crash or plot zeroes if it stored clean data properly
    assert isinstance(fig, plt.Figure)


@pytest.fixture
def fake_statsmodels(monkeypatch):
    """Make the mock statsmodels in tests/_mocks importable, scoped to one test.

    The mock lives outside tests/ proper so pytest's sys.path handling never
    shadows a real statsmodels install for the rest of the session. Any
    statsmodels modules imported while the mock is active are evicted again on
    teardown (monkeypatch then restores whatever was loaded before).
    """
    import sys
    from pathlib import Path

    for name in list(sys.modules):
        if name == "statsmodels" or name.startswith("statsmodels."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.syspath_prepend(str(Path(__file__).parent / "_mocks"))
    yield
    for name in list(sys.modules):
        if name == "statsmodels" or name.startswith("statsmodels."):
            del sys.modules[name]


def test_plot_lowess_optional(fake_statsmodels):
    df = make_relationship("linear_positive", n=100)
    res = profile_pair(df, "x", "y")

    fig = res.plot(show=False)
    assert isinstance(fig, plt.Figure)


def test_plot_lowess_subsample_is_deterministic(fake_statsmodels):
    """When n exceeds the LOWESS subsample cap, repeated plot() calls must
    produce the same smoother (seeded RNG, not the global numpy state)."""
    # n=2000 > 1000 LOWESS cap, so the subsample path is exercised
    df = make_relationship("linear_positive", n=2000, random_state=42)
    res = profile_pair(df, "x", "y")

    fig1 = res.plot(show=False)
    fig2 = res.plot(show=False)

    lines1 = fig1.axes[0].get_lines()
    lines2 = fig2.axes[0].get_lines()
    assert lines1 and lines2, "expected a LOWESS line on the scatter axis"
    for line1, line2 in zip(lines1, lines2, strict=True):
        x1, y1 = line1.get_data()
        x2, y2 = line2.get_data()
        assert np.array_equal(x1, x2)
        assert np.array_equal(y1, y2)


def test_plot_lowess_real_statsmodels():
    """Exercise the LOWESS path against real statsmodels when it is installed.

    pairplot deliberately swallows LOWESS failures, so without this assertion a
    crash in the real-statsmodels call would pass silently. Skipped unless
    statsmodels is available (one CI cell installs it).
    """
    pytest.importorskip("statsmodels.api")

    df = make_relationship("linear_positive", n=100, random_state=42)
    res = profile_pair(df, "x", "y")

    fig = res.plot(show=False)
    assert fig.axes[0].get_lines(), (
        "expected a LOWESS line from real statsmodels on the scatter axis"
    )
