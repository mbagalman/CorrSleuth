import numpy as np
import pandas as pd
import pytest

from corrsleuth import scan_target
from corrsleuth.exceptions import InputError
from corrsleuth.scan import CorrSleuthTargetReport, TargetScanEntry


def _build_clean_df(n: int = 60, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    target = rng.normal(size=n)
    return pd.DataFrame({
        "target": target,
        "linear": target + rng.normal(scale=0.2, size=n),
        "noise": rng.normal(size=n),
        "label": ["a", "b"] * (n // 2),
    })


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


def test_scan_target_sample_size_is_deterministic():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "target": rng.normal(size=400),
        "x": rng.normal(size=400),
    })
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


def test_scan_target_forwards_profile_pair_kwargs():
    df = _build_clean_df()
    report = scan_target(df, "target", bootstrap=10, random_state=123)

    linear_entry = next(e for e in report.successes if e.column == "linear")
    assert linear_entry.result.bootstrap_intervals is not None
    assert linear_entry.result.bootstrap_stability is not None


def test_scan_target_summary_lists_target_and_pattern_counts():
    df = _build_clean_df()
    report = scan_target(df, "target")
    summary = report.summary()

    assert "Target scan: target" in summary
    assert "profiled" in summary
    assert "Pattern counts:" in summary


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
