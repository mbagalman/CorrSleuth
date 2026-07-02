import matplotlib

matplotlib.use("Agg")  # noqa: E402  (must be set before pyplot import)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import pytest

from corrsleuth import scan_target
from corrsleuth.exceptions import InputError
from corrsleuth.result import CorrSleuthResult, MetricDiagnostics
from corrsleuth.scan import CorrSleuthTargetReport, TargetScanEntry


def _build_clean_df(n: int = 60, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    target = rng.normal(size=n)
    return pd.DataFrame(
        {
            "target": target,
            "linear": target + rng.normal(scale=0.2, size=n),
            "noise": rng.normal(size=n),
            "label": ["a", "b"] * (n // 2),
        }
    )


def test_scan_target_profiles_numeric_columns_excluding_target():
    df = _build_clean_df()
    report = scan_target(df, "target")

    assert isinstance(report, CorrSleuthTargetReport)
    profiled = {e.column for e in report.successes}
    assert profiled == {"linear", "noise"}
    assert "target" not in profiled
    assert "label" not in profiled


def test_scan_target_to_frame_has_one_row_per_entry_with_required_fields():
    df = _build_clean_df()
    report = scan_target(df, "target")
    frame = report.to_frame()

    assert len(frame) == len(report.entries)
    for col in (
        "variable",
        "target",
        "status",
        "pattern",
        "disagreement_score",
        "warnings",
        "recommendations",
        "metric_pearson",
        "metric_spearman",
        "metric_kendall_tau_b",
    ):
        assert col in frame.columns, f"missing column: {col}"
    assert (frame["target"] == "target").all()


def test_scan_target_skipped_entries_for_non_numeric_in_explicit_columns():
    df = _build_clean_df()
    report = scan_target(df, "target", columns=["linear", "label", "missing_col"])

    statuses = {e.column: e.status for e in report.entries}
    assert statuses["linear"] == "ok"
    assert statuses["label"] == "skipped"
    assert statuses["missing_col"] == "skipped"

    label_entry = next(e for e in report.entries if e.column == "label")
    assert label_entry.error_type == "NonNumeric"

    missing_entry = next(e for e in report.entries if e.column == "missing_col")
    assert missing_entry.error_type == "ColumnNotFound"


def test_scan_target_columns_can_include_target_and_records_skip():
    df = _build_clean_df()
    report = scan_target(df, "target", columns=["target", "linear"])

    target_entry = next(e for e in report.entries if e.column == "target")
    assert target_entry.status == "skipped"
    assert target_entry.error_type == "TargetExcluded"


def test_scan_target_errors_warn_captures_per_column_failures():
    df = _build_clean_df()
    df["bad"] = np.nan  # all-NaN column makes profile_pair raise InputError
    report = scan_target(df, "target", errors="warn")

    bad_entry = next(e for e in report.entries if e.column == "bad")
    assert bad_entry.status == "error"
    assert bad_entry.error_type == "InputError"
    # Other numeric columns still profiled
    assert any(e.status == "ok" and e.column == "linear" for e in report.entries)


def test_scan_target_errors_raise_propagates_first_failure():
    df = _build_clean_df()
    df["bad"] = np.nan

    with pytest.raises(InputError):
        scan_target(df, "target", errors="raise")


def test_scan_target_max_pairs_caps_candidate_count():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"target": rng.normal(size=40)})
    for i in range(5):
        df[f"col_{i}"] = rng.normal(size=40)

    report = scan_target(df, "target", max_pairs=2)
    profiled_columns = [e.column for e in report.successes]
    assert len(profiled_columns) == 2
    # Order preserved from DataFrame column order
    assert profiled_columns == ["col_0", "col_1"]


def test_scan_target_rejects_invalid_max_pairs():
    df = _build_clean_df()

    # A negative cap must raise rather than silently slicing candidates
    # from the end of the list.
    with pytest.raises(InputError, match="max_pairs must be a positive integer"):
        scan_target(df, "target", max_pairs=-1)
    with pytest.raises(InputError, match="max_pairs must be a positive integer"):
        scan_target(df, "target", max_pairs=0)
    with pytest.raises(InputError, match="max_pairs must be a positive integer"):
        scan_target(df, "target", max_pairs=True)
    with pytest.raises(InputError, match="max_pairs must be a positive integer"):
        scan_target(df, "target", max_pairs=2.5)


def test_scan_target_rejects_invalid_sample_size():
    df = _build_clean_df()

    with pytest.raises(InputError, match="sample_size must be a positive integer"):
        scan_target(df, "target", sample_size=-5)
    with pytest.raises(InputError, match="sample_size must be a positive integer"):
        scan_target(df, "target", sample_size=0)
    with pytest.raises(InputError, match="sample_size must be a positive integer"):
        scan_target(df, "target", sample_size=True)
    with pytest.raises(InputError, match="sample_size must be a positive integer"):
        scan_target(df, "target", sample_size=2.5)


def test_scan_target_rejects_duplicate_target_columns():
    df = _build_clean_df()
    df = pd.concat([df, df["target"]], axis=1)  # two 'target' columns

    with pytest.raises(InputError, match="matches multiple columns"):
        scan_target(df, "target")


def test_scan_target_surfaces_duplicate_candidate_columns_in_auto_scan():
    """A duplicated numeric predictor must not silently vanish from an automatic
    scan; it is reported once as a DuplicateColumn skip, and other columns are
    still profiled."""
    df = _build_clean_df()
    df = pd.concat([df, df["linear"]], axis=1)  # two 'linear' columns

    report = scan_target(df, "target")
    statuses = {e.column: e.status for e in report.entries}

    assert statuses.get("linear") == "skipped"
    linear_entry = next(e for e in report.entries if e.column == "linear")
    assert linear_entry.error_type == "DuplicateColumn"
    assert "matches multiple columns" in linear_entry.error_message
    # Exactly one skip entry for the duplicated name, not one per occurrence.
    assert sum(e.column == "linear" for e in report.entries) == 1
    # A non-duplicated numeric predictor is still profiled.
    assert statuses.get("noise") == "ok"


def test_scan_target_duplicate_candidate_in_explicit_columns_reports_duplicate():
    """An explicitly requested duplicated column is reported as DuplicateColumn,
    not mislabeled NonNumeric."""
    df = _build_clean_df()
    df = pd.concat([df, df["linear"]], axis=1)  # two 'linear' columns

    report = scan_target(df, "target", columns=["linear"])
    entry = next(e for e in report.entries if e.column == "linear")

    assert entry.status == "skipped"
    assert entry.error_type == "DuplicateColumn"
    assert "matches multiple columns" in entry.error_message


def test_scan_target_sample_size_is_deterministic():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "target": rng.normal(size=400),
            "x": rng.normal(size=400),
        }
    )
    r1 = scan_target(df, "target", sample_size=100, random_state=7)
    r2 = scan_target(df, "target", sample_size=100, random_state=7)

    pd.testing.assert_frame_equal(r1.to_frame(), r2.to_frame())


def test_scan_target_progress_does_not_change_results():
    df = _build_clean_df()
    no_progress = scan_target(df, "target", progress=False).to_frame()
    with_progress = scan_target(df, "target", progress=True).to_frame()

    pd.testing.assert_frame_equal(no_progress, with_progress)


def test_scan_target_rejects_unknown_errors_policy():
    df = _build_clean_df()
    with pytest.raises(InputError, match="errors policy"):
        scan_target(df, "target", errors="bogus")


def test_scan_target_rejects_missing_target_column():
    df = _build_clean_df().drop(columns=["target"])
    with pytest.raises(InputError, match="not found"):
        scan_target(df, "target")


def test_scan_target_rejects_non_numeric_target():
    df = _build_clean_df()
    with pytest.raises(InputError, match="not numeric"):
        scan_target(df, "label")


def test_scan_target_rejects_complex_target():
    df = _build_clean_df()
    df["cplx"] = df["target"] + 1j
    with pytest.raises(InputError, match="complex dtype"):
        scan_target(df, "cplx")


def test_scan_target_skips_complex_candidate_columns():
    # A complex candidate is excluded from auto-selection (columns=None) rather
    # than silently profiled after a lossy cast to the real axis.
    df = _build_clean_df()
    df["cplx"] = df["linear"] + 1j
    report = scan_target(df, "target")
    profiled = {e.column for e in report.successes}
    assert "cplx" not in profiled

    # Explicitly requesting it yields a ComplexDtype skip that names the dtype
    # and tells the caller how to cast, distinguishable from NonNumeric.
    report = scan_target(df, "target", columns=["linear", "cplx"])
    entry = next(e for e in report.entries if e.column == "cplx")
    assert entry.status == "skipped"
    assert entry.error_type == "ComplexDtype"
    assert "complex" in entry.error_message.lower()
    assert "real part or magnitude" in entry.error_message


def test_scan_target_forwards_profile_pair_kwargs():
    df = _build_clean_df()
    report = scan_target(df, "target", bootstrap=10, random_state=123)

    linear_entry = next(e for e in report.successes if e.column == "linear")
    assert linear_entry.result.bootstrap_intervals is not None
    assert linear_entry.result.bootstrap_stability is not None


def test_scan_target_summary_header_and_caveat():
    df = _build_clean_df()
    report = scan_target(df, "target")
    summary = report.summary()

    assert summary.startswith("Target scan: target")
    assert "profiled : 2" in summary
    assert "Caveat:" in summary


def test_scan_target_summary_can_suppress_caveat():
    df = _build_clean_df()
    report = scan_target(df, "target")
    assert "Caveat:" not in report.summary(include_caveat=False)


def test_target_report_to_markdown_includes_grouped_sections():
    rng = np.random.default_rng(0)
    n = 200
    target = np.exp(rng.uniform(0.1, 10, size=n))
    df = pd.DataFrame(
        {
            "target": target,
            "log_shape": np.log(target) + rng.normal(0, 0.1, size=n),
            "linear_match": target + rng.normal(0, 0.1, size=n),
            "noise": rng.normal(0, 1, size=n),
            "label": ["a", "b"] * (n // 2),
        }
    )

    report = scan_target(
        df, "target", columns=["log_shape", "linear_match", "noise", "label"]
    )
    markdown = report.to_markdown(top_n=2)

    assert markdown.startswith("# CorrSleuth Target Report: `target`")
    assert "## Overview" in markdown
    assert "| Profiled | Errored | Skipped |" in markdown
    assert "## Strongest near-linear relationships" in markdown
    assert "## Potential monotonic nonlinear relationships" in markdown
    assert "## Weak or no pairwise relationships" in markdown
    assert "## Variables Pearson may underrate" in markdown
    assert "## Skipped or failed" in markdown
    assert (
        "| Variable | Pattern | Pearson | Spearman | Disagreement | Warnings |"
        in markdown
    )
    assert "log\\_shape" in markdown
    assert "label" in markdown
    assert "## Caveat" in markdown
    assert "Pairwise association does not imply causation" in markdown


def test_target_report_to_markdown_is_deterministic_with_snapshot():
    df = _build_clean_df()
    report = scan_target(df, "target")

    first = report.to_markdown(include_caveat=False)
    second = report.to_markdown(include_caveat=False)

    assert first == second
    assert first == "\n".join(
        [
            "# CorrSleuth Target Report: `target`",
            "",
            "## Overview",
            "| Profiled | Errored | Skipped |",
            "| --- | --- | --- |",
            "| 2 | 0 | 0 |",
            "",
            "## Strongest near-linear relationships",
            "| Variable | Pattern | Pearson | Spearman | Disagreement | Warnings |",
            "| --- | --- | --- | --- | --- | --- |",
            "| linear | near\\_linear | 0.981 | 0.976 | 0.005 |  |",
            "",
            "## Weak or no pairwise relationships",
            "| Variable | Pattern | Pearson | Spearman | Disagreement | Warnings |",
            "| --- | --- | --- | --- | --- | --- |",
            "| noise | weak\\_or\\_no\\_relationship | 0.008 | 0.039 | 0.031 |  |",
        ]
    )


def test_target_report_to_markdown_lists_reliability_warning_section():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "target": rng.normal(size=n),
            "discrete": (rng.integers(0, 12, size=n)).astype(float),
            "clean": rng.normal(size=n),
        }
    )
    report = scan_target(df, "target")

    markdown = report.to_markdown(include_caveat=False)

    assert "## Variables with missingness or tie warnings" in markdown
    reliability_idx = markdown.index("## Variables with missingness or tie warnings")
    section = markdown[reliability_idx:]
    assert "discrete" in section
    assert "tie rate" in section
    assert "clean" not in section


def test_target_report_to_markdown_omits_empty_cross_cutting_sections():
    rng = np.random.default_rng(0)
    n = 200
    target = rng.uniform(-3, 3, size=n)
    df = pd.DataFrame(
        {
            "target": target,
            "linear": target + rng.normal(0, 0.1, size=n),
        }
    )
    report = scan_target(df, "target")

    markdown = report.to_markdown(include_caveat=False)

    assert "## Variables Pearson may underrate" not in markdown
    assert "## Variables with missingness or tie warnings" not in markdown
    assert "## Skipped or failed" not in markdown


def test_target_report_to_markdown_includes_other_or_inconclusive():
    rng = np.random.default_rng(0)
    n = 25
    df = pd.DataFrame(
        {
            "target": rng.normal(size=n),
            "candidate": rng.normal(size=n),
        }
    )
    report = scan_target(df, "target")

    markdown = report.to_markdown(include_caveat=False)

    assert "## Other or inconclusive" in markdown
    assert "low\\_power\\_or\\_uncertain" in markdown


def test_target_report_to_markdown_can_suppress_caveat():
    df = _build_clean_df()
    report = scan_target(df, "target")

    markdown = report.to_markdown(include_caveat=False)

    assert "## Caveat" not in markdown


def test_target_report_to_markdown_rejects_invalid_top_n():
    df = _build_clean_df()
    report = scan_target(df, "target")

    with pytest.raises(InputError, match="positive integer"):
        report.to_markdown(top_n=0)


def test_scan_summary_includes_pattern_sections_when_present():
    # Use a target derived from x to make monotonic_log appear as monotonic_nonlinear
    rng = np.random.default_rng(0)
    n = 200
    x = np.exp(rng.uniform(0.1, 10, size=n))
    df = pd.DataFrame(
        {
            "target": x,  # heavily skewed target
            "log_shape": np.log(x)
            + rng.normal(0, 0.1, size=n),  # monotonic_nonlinear (rank >> linear)
            "linear_match": x + rng.normal(0, 0.1, size=n),  # near_linear with x itself
            "noise": rng.normal(0, 1, size=n),  # weak_or_no_relationship
        }
    )
    report = scan_target(df, "target")
    summary = report.summary(include_caveat=False)

    patterns = {e.column: e.result.pattern for e in report.successes}
    assert patterns["log_shape"] == "monotonic_nonlinear"
    assert patterns["linear_match"] == "near_linear"
    assert patterns["noise"] == "weak_or_no_relationship"

    assert "Strongest near-linear relationships:" in summary
    assert "Potential monotonic nonlinear relationships:" in summary
    assert "Weak or no pairwise relationships:" in summary
    # Empty sections are omitted
    assert "Potential nonmonotonic relationships:" not in summary
    assert "Possible outlier-driven relationships:" not in summary

    # Each variable appears in exactly its expected section
    near_idx = summary.index("Strongest near-linear relationships:")
    mono_idx = summary.index("Potential monotonic nonlinear relationships:")
    weak_idx = summary.index("Weak or no pairwise relationships:")
    assert "linear_match" in summary[near_idx:mono_idx]
    assert "log_shape" in summary[mono_idx:weak_idx]
    assert "noise" in summary[weak_idx:]


def test_scan_summary_top_n_caps_section_entries():
    rng = np.random.default_rng(0)
    n = 200
    target = rng.uniform(-3, 3, size=n)
    df = pd.DataFrame({"target": target})
    for i in range(6):
        df[f"linear_{i}"] = target + rng.normal(0, 0.1 + 0.05 * i, size=n)
    report = scan_target(df, "target")
    summary = report.summary(top_n=2, include_caveat=False)

    near_block_start = summary.index("Strongest near-linear relationships:")
    next_blank = summary.find("\n\n", near_block_start)
    near_block = summary[
        near_block_start : next_blank if next_blank != -1 else len(summary)
    ]
    listed = [line for line in near_block.splitlines() if line.startswith("  linear_")]
    assert len(listed) == 2


def test_scan_summary_includes_pearson_underrate_section():
    rng = np.random.default_rng(0)
    n = 200
    x = np.exp(rng.uniform(0.1, 10, size=n))
    df = pd.DataFrame(
        {
            "target": x,
            "log_shape": np.log(x) + rng.normal(0, 0.1, size=n),
        }
    )
    report = scan_target(df, "target")
    summary = report.summary(include_caveat=False)

    log_entry = next(e for e in report.successes if e.column == "log_shape")
    assert log_entry.result.diagnostics.rank_linear_gap > 0.20
    assert "Variables Pearson may underrate:" in summary
    underrate_idx = summary.index("Variables Pearson may underrate:")
    assert "log_shape" in summary[underrate_idx:]


def test_scan_summary_lists_reliability_warning_columns():
    rng = np.random.default_rng(0)
    n = 200
    # 12 levels keeps unique_ratio above 0.05 (so low_unique_ratio does not
    # fire) while still producing a high tie rate per the 30% threshold.
    df = pd.DataFrame(
        {
            "target": rng.normal(size=n),
            "discrete": (rng.integers(0, 12, size=n)).astype(float),
            "clean": rng.normal(size=n),
        }
    )
    report = scan_target(df, "target")
    summary = report.summary(include_caveat=False)

    assert "Variables with missingness or tie warnings:" in summary
    section_idx = summary.index("Variables with missingness or tie warnings:")
    section = summary[section_idx:]
    assert "discrete" in section
    assert "tie rate" in section
    # The clean column should not appear in the reliability section
    clean_section_lines = [
        line for line in section.splitlines() if line.startswith("  clean")
    ]
    assert not clean_section_lines


def test_scan_summary_omits_cross_cutting_sections_when_empty():
    rng = np.random.default_rng(0)
    n = 200
    target = rng.uniform(-3, 3, size=n)
    df = pd.DataFrame(
        {
            "target": target,
            "linear": target + rng.normal(0, 0.1, size=n),
        }
    )
    report = scan_target(df, "target")
    summary = report.summary(include_caveat=False)

    assert "Variables Pearson may underrate:" not in summary
    assert "Variables with missingness or tie warnings:" not in summary
    assert "Skipped or failed:" not in summary


def test_scan_summary_lists_skipped_and_failed_entries():
    df = _build_clean_df()
    df["bad"] = np.nan
    report = scan_target(df, "target", columns=["linear", "label", "bad"])
    summary = report.summary(include_caveat=False)

    assert "Skipped or failed:" in summary
    failed_idx = summary.index("Skipped or failed:")
    failed_section = summary[failed_idx:]
    assert "bad" in failed_section
    assert "label" in failed_section


def test_scan_summary_is_deterministic():
    df = _build_clean_df()
    r1 = scan_target(df, "target").summary()
    r2 = scan_target(df, "target").summary()
    assert r1 == r2


def test_scan_summary_pearson_underrate_excludes_leverage_pattern():
    """Outlier-driven variables (Pearson >> Spearman) must not surface in the
    Pearson-may-underrate cross-cut, since that section is meant for the
    opposite story."""
    rng = np.random.default_rng(0)
    n = 300
    target = rng.normal(0, 0.1, size=n)
    leverage = rng.normal(0, 0.1, size=n)
    num_outliers = max(1, int(n * 0.02))
    target[-num_outliers:] = rng.uniform(8, 10, size=num_outliers)
    leverage[-num_outliers:] = rng.uniform(8, 10, size=num_outliers)
    df = pd.DataFrame({"target": target, "leverage": leverage})

    report = scan_target(df, "target")
    summary = report.summary(include_caveat=False)

    leverage_entry = next(e for e in report.successes if e.column == "leverage")
    metrics = {
        row["metric"]: row["value"]
        for _, row in leverage_entry.result.metrics.iterrows()
    }
    # Sanity-check the fixture: Pearson really is much stronger than Spearman.
    assert abs(metrics["pearson"]) - abs(metrics["spearman"]) > 0.20

    if "Variables Pearson may underrate:" in summary:
        underrate_idx = summary.index("Variables Pearson may underrate:")
        # Find the next section break or end-of-string
        next_break = summary.find("\n\n", underrate_idx)
        section = summary[
            underrate_idx : next_break if next_break != -1 else len(summary)
        ]
        assert "leverage" not in section


def test_pearson_underrated_ranks_nonlinear_above_noise():
    rng = np.random.default_rng(0)
    n = 300
    target = np.exp(rng.uniform(0.1, 10, size=n))
    df = pd.DataFrame(
        {
            "target": target,
            "log_shape": np.log(target) + rng.normal(0, 0.1, size=n),
            "noise": rng.normal(0, 1, size=n),
        }
    )
    report = scan_target(df, "target")

    ranked = report.pearson_underrated()

    assert list(ranked["variable"]) == ["log_shape"]
    assert ranked["pearson_underrate_score"].iloc[0] > 0.20
    assert ranked["spearman_excess_over_pearson"].iloc[0] > 0.20
    assert abs(ranked["metric_pearson"].iloc[0]) < abs(
        ranked["metric_spearman"].iloc[0]
    )
    assert "noise" not in set(ranked["variable"])


def test_pearson_underrated_includes_metric_and_gap_columns():
    rng = np.random.default_rng(0)
    n = 300
    target = np.exp(rng.uniform(0.1, 10, size=n))
    df = pd.DataFrame(
        {
            "target": target,
            "log_shape": np.log(target) + rng.normal(0, 0.1, size=n),
        }
    )
    ranked = scan_target(df, "target").pearson_underrated()

    for col in (
        "variable",
        "target",
        "pattern",
        "pearson_underrate_score",
        "spearman_excess_over_pearson",
        "kendall_excess_over_pearson",
        "nonmonotonic_gap",
        "disagreement_score",
        "metric_pearson",
        "metric_spearman",
        "metric_kendall_tau_b",
        "metric_distance_correlation",
        "metric_mutual_information",
        "warnings",
    ):
        assert col in ranked.columns


def test_pearson_underrated_threshold_controls_inclusion():
    rng = np.random.default_rng(0)
    n = 300
    target = np.exp(rng.uniform(0.1, 10, size=n))
    df = pd.DataFrame(
        {
            "target": target,
            "log_shape": np.log(target) + rng.normal(0, 0.1, size=n),
        }
    )
    report = scan_target(df, "target")

    default_ranked = report.pearson_underrated()
    high_threshold = report.pearson_underrated(threshold=1.0)

    assert not default_ranked.empty
    assert high_threshold.empty
    assert list(high_threshold.columns) == list(default_ranked.columns)


def test_pearson_underrated_excludes_leverage_pattern():
    rng = np.random.default_rng(0)
    n = 300
    target = rng.normal(0, 0.1, size=n)
    leverage = rng.normal(0, 0.1, size=n)
    num_outliers = max(1, int(n * 0.02))
    target[-num_outliers:] = rng.uniform(8, 10, size=num_outliers)
    leverage[-num_outliers:] = rng.uniform(8, 10, size=num_outliers)
    df = pd.DataFrame({"target": target, "leverage": leverage})

    ranked = scan_target(df, "target").pearson_underrated()

    assert ranked.empty


def test_pearson_underrated_rejects_invalid_threshold():
    df = _build_clean_df()
    report = scan_target(df, "target")

    with pytest.raises(InputError, match="non-negative number"):
        report.pearson_underrated(threshold=-0.1)
    with pytest.raises(InputError, match="non-negative number"):
        report.pearson_underrated(threshold=True)
    with pytest.raises(InputError, match="non-negative number"):
        report.pearson_underrated(threshold=np.nan)


def test_pearson_underrated_empty_report_keeps_documented_schema():
    df = _build_clean_df()
    report = scan_target(df, "target", columns=["label"])

    ranked = report.pearson_underrated()

    assert ranked.empty
    assert list(ranked.columns) == [
        "variable",
        "target",
        "pattern",
        "pearson_underrate_score",
        "spearman_excess_over_pearson",
        "kendall_excess_over_pearson",
        "nonmonotonic_gap",
        "disagreement_score",
        "metric_pearson",
        "metric_spearman",
        "metric_kendall_tau_b",
        "metric_distance_correlation",
        "metric_mutual_information",
        "warnings",
    ]


def _underrated_entry(column: str, pearson: float, spearman: float) -> TargetScanEntry:
    metrics = pd.DataFrame(
        {
            "metric": ["pearson", "spearman", "kendall_tau_b"],
            "value": [pearson, spearman, 0.0],
            "available": [True, True, True],
        }
    )
    result = CorrSleuthResult(
        x_name="target",
        y_name=column,
        metrics=metrics,
        pattern="monotonic_nonlinear",
        warnings=[],
        recommendations=[],
        disagreement_score=abs(abs(spearman) - abs(pearson)),
        diagnostics=MetricDiagnostics(
            rank_linear_gap=abs(abs(spearman) - abs(pearson)),
            pearson_spearman_signed_gap=pearson - spearman,
            nonmonotonic_gap=None,
            pearson_kendall_gap=abs(abs(pearson) - 0.0),
            disagreement_score=abs(abs(spearman) - abs(pearson)),
        ),
    )
    return TargetScanEntry(column=column, status="ok", result=result)


def test_pearson_underrated_sort_is_deterministic_with_name_tiebreaker():
    report = CorrSleuthTargetReport(
        target="target",
        entries=[
            _underrated_entry("tie_b", pearson=0.10, spearman=0.45),
            _underrated_entry("high", pearson=0.10, spearman=0.60),
            _underrated_entry("tie_a", pearson=0.10, spearman=0.45),
        ],
    )

    first = report.pearson_underrated()
    second = report.pearson_underrated()

    assert list(first["variable"]) == ["high", "tie_a", "tie_b"]
    assert list(second["variable"]) == list(first["variable"])


def _ranked_pearson_df(n: int = 100, random_state: int = 0) -> pd.DataFrame:
    """DataFrame with three columns of intentionally different abs(pearson)."""
    rng = np.random.default_rng(random_state)
    target = rng.uniform(-3, 3, size=n)
    return pd.DataFrame(
        {
            "target": target,
            "strong_lin": target + rng.normal(0, 0.05, size=n),  # near-perfect linear
            "medium_lin": target + rng.normal(0, 0.50, size=n),  # moderate
            "weak_lin": target + rng.normal(0, 2.0, size=n),  # weak
        }
    )


def test_plot_top_returns_figure_with_scatter_panels():
    df = _ranked_pearson_df()
    report = scan_target(df, "target")
    fig = report.plot_top(n=3, ncols=3)
    try:
        assert isinstance(fig, plt.Figure)
        # Three panels expected (nrows=1, ncols=3)
        assert len(fig.axes) == 3
        non_empty_titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
        assert len(non_empty_titles) == 3
        # Suptitle should reference the target name and the sort key
        assert "target" in fig._suptitle.get_text()
    finally:
        plt.close(fig)


def test_plot_top_filters_by_patterns():
    df = _ranked_pearson_df()
    report = scan_target(df, "target")
    # All three columns are designed to land near_linear, so the filter
    # should yield panels equal to the count of near_linear successes
    near_linear_columns = [
        e.column for e in report.successes if e.result.pattern == "near_linear"
    ]
    fig = report.plot_top(patterns=["near_linear"], ncols=3)
    try:
        non_empty_titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
        assert len(non_empty_titles) == len(near_linear_columns)
        # And no panel for an excluded pattern
        weak_fig = report.plot_top(patterns=["weak_or_no_relationship"], ncols=3)
        try:
            visible_text = [t.get_text() for ax in weak_fig.axes for t in ax.texts]
            assert any("No variables to plot" in t for t in visible_text)
        finally:
            plt.close(weak_fig)
    finally:
        plt.close(fig)


def test_plot_top_normalizes_string_patterns():
    df = _ranked_pearson_df()
    report = scan_target(df, "target")
    fig_str = report.plot_top(patterns="near_linear", ncols=3)
    fig_list = report.plot_top(patterns=["near_linear"], ncols=3)
    try:
        titles_str = [ax.get_title() for ax in fig_str.axes if ax.get_title()]
        titles_list = [ax.get_title() for ax in fig_list.axes if ax.get_title()]
        assert titles_str == titles_list
    finally:
        plt.close(fig_str)
        plt.close(fig_list)


def test_plot_top_handles_fewer_than_n_variables():
    df = _build_clean_df()  # has 2 numeric non-target columns
    report = scan_target(df, "target")
    fig = report.plot_top(n=10, ncols=3)
    try:
        # 1 row × 3 cols of axes; 2 populated, 1 hidden
        assert len(fig.axes) == 3
        non_empty_titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
        assert len(non_empty_titles) == 2
    finally:
        plt.close(fig)


def test_plot_top_returns_placeholder_figure_when_no_matches():
    df = _build_clean_df()
    report = scan_target(df, "target")
    fig = report.plot_top(patterns=["nonmonotonic_dependence"])
    try:
        assert isinstance(fig, plt.Figure)
        all_text = [t.get_text() for ax in fig.axes for t in ax.texts]
        assert any("No variables to plot" in t for t in all_text)
    finally:
        plt.close(fig)


def test_plot_top_sort_by_metric_ranks_panels_by_absolute_value():
    df = _ranked_pearson_df()
    report = scan_target(df, "target")
    fig = report.plot_top(n=3, sort_by="pearson", ncols=3)
    try:
        # First panel should have the highest |pearson| -> "strong_lin"
        first_title = fig.axes[0].get_title()
        assert first_title.startswith("strong_lin")
        # Last visible panel should be "weak_lin"
        non_empty = [ax for ax in fig.axes if ax.get_title()]
        assert non_empty[-1].get_title().startswith("weak_lin")
    finally:
        plt.close(fig)


def test_plot_top_uses_candidate_on_x_and_target_on_y():
    rng = np.random.default_rng(0)
    n = 50
    df = pd.DataFrame(
        {
            "y_target": rng.uniform(-3, 3, size=n),
            "predictor_a": rng.normal(size=n),
            "predictor_b": rng.normal(size=n),
        }
    )
    df["predictor_a"] = df["y_target"] + rng.normal(0, 0.1, size=n)

    report = scan_target(df, "y_target")
    fig = report.plot_top(n=2, ncols=2)
    try:
        for ax in fig.axes:
            if not ax.get_title():
                continue
            xlabel = ax.get_xlabel()
            ylabel = ax.get_ylabel()
            assert ylabel == "y_target", (
                f"target should be on y-axis; got xlabel={xlabel!r}, ylabel={ylabel!r}"
            )
            assert xlabel in {"predictor_a", "predictor_b"}, (
                f"candidate column expected on x-axis; got xlabel={xlabel!r}"
            )
    finally:
        plt.close(fig)


def test_plot_top_rejects_invalid_n_or_ncols():
    df = _build_clean_df()
    report = scan_target(df, "target")
    with pytest.raises(InputError, match="positive integer"):
        report.plot_top(n=0)
    with pytest.raises(InputError, match="positive integer"):
        report.plot_top(n=-1)
    with pytest.raises(InputError, match="positive integer"):
        report.plot_top(ncols=0)
    with pytest.raises(InputError, match="positive integer"):
        report.plot_top(n=True)


def test_plot_top_rejects_invalid_sort_by():
    df = _build_clean_df()
    report = scan_target(df, "target")
    with pytest.raises(InputError, match="Unknown sort_by"):
        report.plot_top(sort_by="bogus_metric")


def test_scan_summary_rejects_invalid_top_n():
    df = _build_clean_df()
    report = scan_target(df, "target")

    with pytest.raises(InputError, match="positive integer"):
        report.summary(top_n=0)
    with pytest.raises(InputError, match="positive integer"):
        report.summary(top_n=-1)
    with pytest.raises(InputError, match="positive integer"):
        report.summary(top_n=True)
    with pytest.raises(InputError, match="positive integer"):
        report.summary(top_n=2.5)


def test_scan_target_to_frame_keeps_documented_columns_when_all_skipped():
    df = _build_clean_df()
    report = scan_target(df, "target", columns=["label"])
    frame = report.to_frame()

    for col in (
        "variable",
        "target",
        "status",
        "error_type",
        "error_message",
        "pattern",
        "disagreement_score",
        "warnings",
        "recommendations",
        "metric_pearson",
        "metric_spearman",
        "metric_kendall_tau_b",
    ):
        assert col in frame.columns, f"missing column: {col}"
    assert frame["status"].iloc[0] == "skipped"
    assert pd.isna(frame["pattern"].iloc[0])
    assert pd.isna(frame["metric_pearson"].iloc[0])


def test_scan_target_to_frame_keeps_documented_columns_when_all_errored():
    df = _build_clean_df()
    df["bad"] = np.nan
    report = scan_target(df, "target", columns=["bad"])
    frame = report.to_frame()

    assert frame["status"].iloc[0] == "error"
    for col in (
        "pattern",
        "disagreement_score",
        "warnings",
        "recommendations",
        "metric_pearson",
        "metric_spearman",
        "metric_kendall_tau_b",
    ):
        assert col in frame.columns
        assert pd.isna(frame[col].iloc[0])


def test_scan_target_columns_string_is_normalized_to_single_element_list():
    df = _build_clean_df()
    report = scan_target(df, "target", columns="linear")

    assert [e.column for e in report.entries] == ["linear"]
    assert report.entries[0].status == "ok"
    # And we don't accidentally iterate the characters of the string
    assert all(e.column != "l" for e in report.entries)


def test_scan_target_entries_capture_both_skipped_and_profiled():
    df = _build_clean_df()
    report = scan_target(df, "target", columns=["noise", "label", "linear"])

    statuses = {e.column: e.status for e in report.entries}
    assert statuses == {"noise": "ok", "label": "skipped", "linear": "ok"}

    frame = report.to_frame()
    assert set(frame["variable"]) == {"noise", "label", "linear"}
    label_row = frame[frame["variable"] == "label"].iloc[0]
    assert label_row["status"] == "skipped"
    assert pd.isna(label_row["pattern"])
