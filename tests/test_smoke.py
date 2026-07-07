"""End-to-end smoke tests for the documented user workflow.

These exercise the full happy path a reader follows from the README — build a
DataFrame, profile a pair, scan a target, and render every output surface
(text, Markdown, frame, plot) — so an integration-level break is caught even if
every unit test still passes. They assert shape and non-emptiness, not exact
numbers (the unit tests own precise behavior).
"""

import matplotlib.pyplot as plt
import pandas as pd
import pytest

import corrsleuth as cs
from corrsleuth.datasets import make_relationship

VALID_LABELS = {
    "near_linear",
    "monotonic_nonlinear",
    "nonmonotonic_dependence",
    "possible_outlier_or_leverage",
    "weak_or_no_relationship",
    "low_power_or_uncertain",
    "not_computable",
    "mixed_or_ambiguous",
}


def _demo_frame(n=200, random_state=0):
    """A target plus predictors spanning the relationship shapes CorrSleuth labels."""
    linear = make_relationship("linear_positive", n=n, random_state=random_state)
    return pd.DataFrame(
        {
            "target": linear["y"],
            "linear": linear["x"],
            "logarithmic": make_relationship("monotonic_log", n=n, random_state=1)["x"],
            "u_shaped": make_relationship("u_shape", n=n, random_state=2)["x"],
            "noise": make_relationship("independent", n=n, random_state=3)["x"],
        }
    )


def test_profile_pair_end_to_end_renders_every_surface():
    df = _demo_frame()
    result = cs.profile_pair(df, "linear", "target")

    assert result.pattern in VALID_LABELS

    summary = result.summary()
    assert "Relationship Profile" in summary
    assert result.pattern in summary

    markdown = result.to_markdown()
    assert markdown.startswith("# CorrSleuth Pair Report")

    explanation = result.explain()
    assert isinstance(explanation, str) and explanation

    fig = result.plot(show=False)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_profile_pair_deep_mode_adds_chatterjee_xi():
    pytest.importorskip("dcor")
    pytest.importorskip("sklearn")
    df = _demo_frame()
    metrics = set(
        cs.profile_pair(df, "u_shaped", "target", mode="deep").metrics["metric"]
    )
    assert "chatterjee_xi" in metrics
    assert "chatterjee_xi_reverse" in metrics


def test_profile_pair_standard_mode_adds_distance_correlation():
    pytest.importorskip("dcor")
    df = _demo_frame()
    metrics = set(
        cs.profile_pair(df, "u_shaped", "target", mode="standard").metrics["metric"]
    )
    assert "distance_correlation" in metrics


def test_scan_target_end_to_end_renders_every_surface():
    df = _demo_frame()
    report = cs.scan_target(df, target="target")

    # One row per non-target column, all profiled successfully.
    frame = report.to_frame()
    assert set(frame["variable"]) == {"linear", "logarithmic", "u_shaped", "noise"}
    assert (frame["status"] == "ok").all()
    assert len(report.successes) == 4
    assert report.failures == []

    summary = report.summary()
    assert "Target scan: target" in summary

    markdown = report.to_markdown()
    assert markdown.startswith("# CorrSleuth Target Report")

    underrated = report.pearson_underrated()
    assert "pearson_underrate_score" in underrated.columns

    fig = report.plot_top(n=4)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_scan_target_records_skips_without_aborting():
    df = _demo_frame()
    df["category"] = ["a", "b"] * (len(df) // 2)  # non-numeric → skipped, not fatal

    report = cs.scan_target(df, target="target", columns=["linear", "category"])

    statuses = {e.column: e.status for e in report.entries}
    assert statuses["linear"] == "ok"
    assert statuses["category"] == "skipped"
