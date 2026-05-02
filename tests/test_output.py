import pytest
import numpy as np
from corrsleuth.datasets import make_relationship
from corrsleuth.api import profile_pair
import matplotlib.pyplot as plt
import pandas as pd
from corrsleuth.result import CorrSleuthResult


def _result_for_explanation(pattern, metric_values):
    return CorrSleuthResult(
        x_name="x",
        y_name="y",
        metrics=pd.DataFrame(
            [{"metric": metric, "value": value} for metric, value in metric_values.items()]
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

    assert "Spearman (0.780) is meaningfully stronger than Pearson (0.310)" in explanation
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


def test_explain_possible_outlier_or_leverage_references_rank_disagreement():
    res = _result_for_explanation(
        "possible_outlier_or_leverage",
        {"pearson": 0.82, "spearman": 0.31, "kendall_tau_b": 0.21},
    )

    explanation = res.explain(include_caveat=False)

    assert "Pearson (0.820) is much stronger than Spearman (0.310)" in explanation
    assert "extreme values" in explanation


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
    assert "Recommendations:" in summary_text
    assert "Caveat:" in summary_text


def test_summary_can_hide_caveat():
    df = make_relationship("linear_positive", n=100)
    res = profile_pair(df, "x", "y")

    summary_text = res.summary(include_caveat=False)

    assert "Caveat:" not in summary_text
    
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
    assert len(frame) >= 3 # at least core metrics


def test_result_exposes_structured_diagnostics():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [1, 2, 3, 4, 5]})
    res = profile_pair(df, "x", "y")

    assert res.diagnostics.rank_linear_gap == pytest.approx(0.0)
    assert res.diagnostics.pearson_spearman_signed_gap == pytest.approx(0.0)
    assert res.diagnostics.pearson_kendall_gap == pytest.approx(0.0)
    assert res.diagnostics.nonmonotonic_gap is None
    assert res.diagnostics.disagreement_score == pytest.approx(res.disagreement_score)


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

def test_plot_lowess_optional(monkeypatch):
    import sys
    from pathlib import Path

    # Add tests directory to sys.path so our fake statsmodels is importable
    tests_dir = str(Path(__file__).parent)
    monkeypatch.syspath_prepend(tests_dir)

    df = make_relationship("linear_positive", n=100)
    res = profile_pair(df, "x", "y")

    fig = res.plot(show=False)
    assert isinstance(fig, plt.Figure)


def test_plot_lowess_subsample_is_deterministic(monkeypatch):
    """When n exceeds the LOWESS subsample cap, repeated plot() calls must
    produce the same smoother (seeded RNG, not the global numpy state)."""
    import sys
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).parent))

    # n=2000 > 1000 LOWESS cap, so the subsample path is exercised
    df = make_relationship("linear_positive", n=2000, random_state=42)
    res = profile_pair(df, "x", "y")

    fig1 = res.plot(show=False)
    fig2 = res.plot(show=False)

    lines1 = fig1.axes[0].get_lines()
    lines2 = fig2.axes[0].get_lines()
    assert lines1 and lines2, "expected a LOWESS line on the scatter axis"
    for line1, line2 in zip(lines1, lines2):
        x1, y1 = line1.get_data()
        x2, y2 = line2.get_data()
        assert np.array_equal(x1, x2)
        assert np.array_equal(y1, y2)
