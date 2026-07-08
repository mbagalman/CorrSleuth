import sys

import matplotlib

matplotlib.use("Agg")  # noqa: E402  (must be set before pyplot import)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import pytest

from corrsleuth import profile_pair, scan_target
from corrsleuth.exceptions import InputError, OptionalDependencyError
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


@pytest.mark.parametrize("shape", ["sinusoid", "step"])
def test_scan_labels_match_candidate_to_target_orientation(shape):
    """Regression lock for the scan orientation BLOCKER (FU-C).

    ``scan_target`` must profile each pair as ``profile_pair(candidate, target)``
    so the direction-sensitive diagnostics condition on the *candidate* — the
    feature that drives the target. Before the flip, a target that is a
    non-invertible function of the candidate (a sinusoid or a step) had its bin
    means averaged over the candidate's branches and was mislabeled
    ``weak_or_no_relationship``. The scan entry's label must now match a direct
    ``profile_pair(candidate, target)`` call, and a step's ``breakpoint_x`` must
    land near the true cut in *candidate* units (it was in target units before)."""
    rng = np.random.default_rng(0)
    n = 400
    cand = rng.uniform(0, 3 * np.pi, size=n)
    if shape == "sinusoid":
        target = np.sin(cand) + rng.normal(0, 0.1, size=n)
    else:  # step at candidate ≈ π
        target = np.where(cand > np.pi, 1.0, -1.0) + rng.normal(0, 0.1, size=n)
    df = pd.DataFrame({"target": target, "cand": cand})

    entry = next(e for e in scan_target(df, "target").successes if e.column == "cand")
    direct = profile_pair(df, "cand", "target", mode="lite")
    assert entry.result_data.pattern == direct.pattern

    if shape == "sinusoid":
        assert entry.result_data.pattern == "nonmonotonic_dependence"
    else:
        assert entry.result_data.pattern == "monotonic_nonlinear"
        bp = entry.result_data.diagnostics.breakpoint_x
        assert bp is not None and abs(bp - np.pi) < 1.0, f"breakpoint_x={bp}"


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


def test_scan_to_frame_surfaces_diagnostics_and_stability_columns():
    """The scan frame exposes every MetricDiagnostics field as a diagnostic_*
    column (including the five secondary axes) and, when bootstrap is requested,
    the stability columns — mirroring CorrSleuthResult.to_frame (FU-F / Chunk 6
    #2, which found the scan frame dropped the entire diagnostics layer)."""
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "target": x,
            "curve": x**3 + rng.normal(0, 0.1, size=n),
            "noise": rng.normal(size=n),
        }
    )
    frame = scan_target(df, "target", bootstrap=20, random_state=0).to_frame()

    for axis in (
        "mean_shape",
        "variance_shape",
        "dependence_type",
        "outlier_sensitivity",
        "functional_direction",
    ):
        assert f"diagnostic_{axis}" in frame.columns
    for col in (
        "diagnostic_bin_lof_r2_gain",
        "diagnostic_n_influential_points",
        "diagnostic_disagreement_score",
    ):
        assert col in frame.columns
    for col in ("pattern_stability", "stability_label", "stability_metric_set"):
        assert col in frame.columns

    curve = frame[frame["variable"] == "curve"].iloc[0]
    assert 0.0 <= float(curve["pattern_stability"]) <= 1.0


def test_scan_to_frame_omits_stability_columns_without_bootstrap():
    """Stability columns are added only when bootstrapping was requested; the
    diagnostic_* columns are always present."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"target": rng.normal(size=80), "a": rng.normal(size=80)})
    frame = scan_target(df, "target").to_frame()

    assert "pattern_stability" not in frame.columns
    assert any(c.startswith("diagnostic_") for c in frame.columns)


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


def test_scan_target_reraises_missing_optional_dependency_under_warn(monkeypatch):
    """A missing optional dependency is systemic — it fails identically for every
    column — so ``errors="warn"`` must surface it once, not bury N identical
    ``OptionalDependencyError`` entries under a zero-success scan (C6 #4)."""
    # Simulate dcor not installed so standard mode cannot compute distance corr.
    for mod in list(sys.modules):
        if mod == "dcor" or mod.startswith("dcor."):
            monkeypatch.setitem(sys.modules, mod, None)
    monkeypatch.setitem(sys.modules, "dcor", None)

    df = _build_clean_df()
    with pytest.raises(OptionalDependencyError):
        scan_target(df, "target", mode="standard", errors="warn")


def test_scan_target_reraises_unknown_kwarg_typeerror_under_warn():
    """A misspelled ``profile_pair`` keyword is a config mistake, not per-column
    data — it must propagate even under ``errors="warn"`` (C6 #4)."""
    df = _build_clean_df()
    with pytest.raises(TypeError):
        scan_target(df, "target", errors="warn", not_a_real_kwarg=123)


def test_scan_target_warn_still_captures_per_column_data_errors():
    """The narrowed config-class set must not swallow genuine per-column data
    failures: an all-NaN column (``InputError`` from validate_pair) is still
    captured under ``errors="warn"`` while its neighbors profile fine."""
    df = _build_clean_df()
    df["bad"] = np.nan
    report = scan_target(df, "target", errors="warn")

    bad_entry = next(e for e in report.entries if e.column == "bad")
    assert bad_entry.status == "error"
    assert bad_entry.error_type == "InputError"
    assert any(e.status == "ok" for e in report.entries)


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


def test_scan_target_max_pairs_records_dropped_columns_as_skips():
    """Columns beyond the cap must not silently vanish (C6 #5) — they are
    recorded as ``MaxPairsExceeded`` skips so coverage reads honestly."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"target": rng.normal(size=40)})
    for i in range(5):
        df[f"col_{i}"] = rng.normal(size=40)

    report = scan_target(df, "target", max_pairs=2)

    dropped = [e for e in report.entries if e.error_type == "MaxPairsExceeded"]
    assert [e.column for e in dropped] == ["col_2", "col_3", "col_4"]
    assert all(e.status == "skipped" for e in dropped)
    # The coverage counters no longer read as if the scan were complete.
    assert "skipped  : 3" in report.summary()
    # And the dropped columns each get a row in the frame.
    frame = report.to_frame()
    assert set(frame.loc[frame["error_type"] == "MaxPairsExceeded", "variable"]) == {
        "col_2",
        "col_3",
        "col_4",
    }


def test_scan_target_max_pairs_at_or_above_candidate_count_adds_no_skips():
    """No spurious skip entries when the cap is not actually binding."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"target": rng.normal(size=40)})
    for i in range(3):
        df[f"col_{i}"] = rng.normal(size=40)

    report = scan_target(df, "target", max_pairs=3)
    assert not any(e.error_type == "MaxPairsExceeded" for e in report.entries)


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


def test_scan_target_rejects_invalid_direction():
    df = _build_clean_df()
    with pytest.raises(InputError, match="Unknown direction"):
        scan_target(df, "target", direction="sideways")


def _directional_df(n: int = 800, seed: int = 0) -> pd.DataFrame:
    """Y drives a step-shaped candidate (Xi = step(Y)) plus a genuine linear one.
    The step's shape only shows in the reverse (Y -> Xi) orientation."""
    rng = np.random.default_rng(seed)
    y = rng.normal(size=n)
    return pd.DataFrame(
        {
            "Y": y,
            "step": np.where(y > 0, 2.0, -2.0) + rng.normal(0, 0.3, size=n),
            "lin": y + rng.normal(0, 0.3, size=n),
        }
    )


def test_scan_direction_reverse_profiles_target_to_candidate():
    """direction='reverse' profiles profile_pair(target, candidate), so a
    candidate engineered as step(Y) reads as monotonic_nonlinear (its true shape),
    where the forward orientation only saw near_linear."""
    df = _directional_df()

    forward = {
        e.column: e.result_data.pattern
        for e in scan_target(df, "Y", direction="forward").successes
    }
    reverse = {
        e.column: e.result_data.pattern
        for e in scan_target(df, "Y", direction="reverse").successes
    }

    assert forward["step"] == "near_linear"  # predictive view hides the step
    assert reverse["step"] == "monotonic_nonlinear"  # truth orientation reveals it
    assert forward["lin"] == reverse["lin"] == "near_linear"  # genuine linear both ways


def test_scan_direction_both_flags_and_frames_reverse_shape():
    """direction='both' keeps the forward profile as primary, attaches the reverse
    shape, flags candidates whose reverse shape is structured while the forward is
    not, and exposes reverse_* columns in to_frame."""
    report = scan_target(_directional_df(), "Y", direction="both")

    # Primary result stays the forward (predictive) orientation.
    labels = {e.column: e.result_data.pattern for e in report.successes}
    assert labels["step"] == "near_linear"

    # The asymmetry section flags the step (reverse structured), not the linear.
    summary = report.summary()
    assert "Shape differs by direction" in summary
    section = summary[summary.index("Shape differs by direction") :]
    assert "step" in section
    assert "lin:" not in section  # genuine linear is not flagged

    # to_frame carries the reverse shape.
    frame = report.to_frame()
    assert "reverse_pattern" in frame.columns
    step_row = frame[frame["variable"] == "step"].iloc[0]
    assert step_row["pattern"] == "near_linear"
    assert step_row["reverse_pattern"] == "monotonic_nonlinear"
    assert step_row["reverse_mean_shape"] == "step_or_threshold"
    lin_row = frame[frame["variable"] == "lin"].iloc[0]
    assert lin_row["reverse_pattern"] == "near_linear"


def test_scan_direction_forward_default_has_no_reverse_columns():
    """The default (forward) scan is unchanged: no reverse profile is computed and
    to_frame carries no reverse_* columns."""
    report = scan_target(_directional_df(), "Y")

    assert all(e.reverse_result is None for e in report.successes)
    assert "reverse_pattern" not in report.to_frame().columns


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


def test_scan_target_caveat_warns_about_multiple_testing():
    """The scan applies no multiple-testing correction; both rendered caveats and
    the docstring must say so (C6 #3) — otherwise a wide noise scan reads as if
    its by-chance hits were findings."""
    df = _build_clean_df()
    report = scan_target(df, "target")

    summary = report.summary()
    markdown = report.to_markdown()
    for surface in (summary, markdown):
        assert "multiple-testing" in surface
        assert "hypothesis-generating" in surface

    assert "multiple-testing" in scan_target.__doc__


def test_target_report_to_markdown_includes_grouped_sections():
    rng = np.random.default_rng(0)
    n = 200
    # Light-tailed target keeps an independent predictor ('noise') correctly weak:
    # a heavy-tailed target can trip the bin-LoF oscillation gate on independent
    # noise via a few extreme Y values (tracked as the heavy-tail robustness
    # finding, FU-U). 'steep_curve' is a strong monotonic nonlinearity, so it
    # lands in both the monotonic-nonlinear and the Pearson-underrate sections.
    target = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "target": target,
            "steep_curve": np.exp(2.0 * target) + rng.normal(0, 0.1, size=n),
            "linear_match": target + rng.normal(0, 0.1, size=n),
            "noise": rng.normal(0, 1, size=n),
            "label": ["a", "b"] * (n // 2),
        }
    )

    report = scan_target(
        df, "target", columns=["steep_curve", "linear_match", "noise", "label"]
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
    assert "steep\\_curve" in markdown
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


def test_target_report_to_markdown_escapes_warnings_once():
    """Warnings in the pattern-section table are escaped exactly once by
    markdown_table. They must not be pre-escaped as well, or metacharacters like
    the underscores in metric names or the `<` in 'ratio (< 0.05)' render with
    doubled backslashes."""
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame(
        {"target": rng.normal(size=n), "lowcard": np.array([1.0, 2.0] * (n // 2))}
    )

    markdown = scan_target(df, "target").to_markdown()

    # A low-cardinality column emits a 'unique value ratio (< 0.05)' warning; the
    # '<' is escaped once as '\<', never doubled to '\\<'.
    assert "\\<" in markdown
    assert "\\\\" not in markdown  # no doubled backslash anywhere


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
    # Moderately skewed target (~5x), not the extreme exp(uniform(0.1, 10)) ~75x
    # tail that can trip the bin-LoF oscillation gate on independent noise
    # (tracked as the heavy-tail robustness finding, FU-U). Still makes log_shape
    # monotonic_nonlinear while noise stays weak.
    x = np.exp(rng.uniform(0.1, 4, size=n))
    df = pd.DataFrame(
        {
            "target": x,  # skewed target
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


def test_pearson_underrated_surfaces_lite_magnitude_linked_via_sq_corr():
    """A lite scan (no dcor) must still surface a magnitude-linked column — one
    the target depends on nonmonotonically — under 'Pearson may underrate', on the
    strength of sq_corr alone (C6 #6). Before the fix only rank/dcor gaps counted,
    so a U-shape (weak Pearson AND weak Spearman) never surfaced."""
    rng = np.random.default_rng(0)
    n = 400
    mag = rng.uniform(-3, 3, size=n)
    df = pd.DataFrame(
        {
            "target": mag**2 + rng.normal(0, 0.3, size=n),  # U-shape in mag
            "mag": mag,
            "linear": None,  # placeholder replaced below
        }
    )
    df["linear"] = df["target"] + rng.normal(0, 0.2, size=n)

    ranked = scan_target(df, "target").pearson_underrated()  # lite mode: no dcor

    assert "mag" in set(ranked["variable"])
    mag_row = ranked[ranked["variable"] == "mag"].iloc[0]
    # It surfaced on sq_corr, not the rank gaps (Spearman is ~0 for a U-shape).
    assert mag_row["sq_corr_excess_over_pearson"] > 0.20
    assert mag_row["spearman_excess_over_pearson"] < 0.20
    # The clean linear column is not falsely surfaced (|sq_corr| ~= |Pearson|).
    assert "linear" not in set(ranked["variable"])


def test_pearson_underrated_excludes_heavy_tail_sq_corr_artifact():
    """The underrate ranking must use the same robust evidence as the cascade: a
    heavy-tailed target vs an independent predictor whose *raw* sq_corr clears the
    0.35 bar but whose *robust* sq_corr collapses (so the cascade correctly calls
    it weak_or_no_relationship, dependence_type None) must NOT be promoted under
    'Pearson may underrate' — the exact artifact the FU-V robust gate suppresses."""
    rng = np.random.default_rng(574)
    noise = rng.normal(size=100)  # candidate drawn first (the seed-574 pair)
    target = np.exp(rng.uniform(0.1, 10, size=100))
    df = pd.DataFrame({"noise": noise, "target": target})

    report = scan_target(df, "target", mode="lite")
    entry = next(e for e in report.successes if e.column == "noise")

    # Preconditions: the raw artifact is present, but the cascade already rejects it.
    assert abs(entry.result_data.diagnostics.sq_corr) > 0.35
    assert entry.result_data.pattern == "weak_or_no_relationship"
    assert entry.result_data.diagnostics.dependence_type is None

    # The fix: it is not surfaced in the ranking (before, it appeared at ~0.32).
    ranked = report.pearson_underrated()
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
        "sq_corr_excess_over_pearson",
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
        "sq_corr_excess_over_pearson",
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
        x_name=column,
        y_name="target",
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


def _skewed_radial_df(n: int = 400, random_state: int = 7) -> pd.DataFrame:
    """A one-sided heavy-tailed target with (a) a tail-leveraged radial column
    whose symmetric shape degrades to mixed_or_ambiguous, (b) a clean linear
    column, and (c) pure noise. Mirrors the exponential blind-test case that
    motivated the "dependence may be understated" section."""
    rng = np.random.default_rng(random_state)
    y = rng.exponential(scale=1.0, size=n)
    return pd.DataFrame(
        {
            "target": y,
            "radial": (y - y.mean()) ** 2 + rng.normal(0, 0.05, size=n),
            "linear": 2 * y + rng.normal(0, 0.3, size=n),
            "noise": rng.normal(size=n),
        }
    )


def test_dependence_understated_surfaces_lite_radial_via_sq_corr_robust():
    """A lite scan (no dcor/MI) must surface a weak/ambiguous column that still
    carries robust radial dependence under "Dependence may be understated" — on
    sq_corr_robust alone — while leaving a clean-linear column (confident label)
    and pure noise (no evidence) out of the section."""
    report = scan_target(_skewed_radial_df(), "target", mode="lite")

    radial = next(e for e in report.successes if e.column == "radial")
    assert radial.result_data.pattern == "mixed_or_ambiguous"
    evidence = report._dependence_understatement(radial)
    assert evidence is not None
    signal, display_value, strength = evidence
    assert signal == "sq_corr_robust"
    assert strength > 0.20

    # Neither a confident linear label nor pure noise qualifies.
    for col in ("linear", "noise"):
        entry = next(e for e in report.successes if e.column == col)
        assert report._dependence_understatement(entry) is None

    text = report.summary()
    assert "Dependence may be understated" in text
    assert "radial (mixed_or_ambiguous; sq_corr_robust=" in text
    # The section is a strict subset of the flagged column.
    assert "linear (" not in text.split("Dependence may be understated")[1]


def test_dependence_understated_markdown_lists_signal_and_value():
    report = scan_target(_skewed_radial_df(), "target", mode="lite")
    md = report.to_markdown()

    assert "## Dependence may be understated" in md
    section = md.split("## Dependence may be understated")[1]
    # Header row plus the radial row, with its lite-computable signal.
    assert "Signal" in section and "sq_corr_robust" in section
    assert "radial" in section


def test_dependence_understated_absent_without_evidence():
    """A scan with only a clean-linear and a noise column emits no section."""
    df = _build_clean_df()
    report = scan_target(df, "target", columns=["linear", "noise"])
    assert "Dependence may be understated" not in report.summary()
    assert "## Dependence may be understated" not in report.to_markdown()


def test_dependence_understated_excludes_heavy_tail_sq_corr_artifact():
    """The section must use the same robust evidence as the cascade: the seed-574
    heavy-tailed pair whose *raw* sq_corr clears 0.35 but whose *robust* sq_corr
    collapses (cascade → weak_or_no_relationship) must NOT be flagged — the exact
    leverage artifact the FU-V robust gate suppresses."""
    rng = np.random.default_rng(574)
    noise = rng.normal(size=100)
    target = np.exp(rng.uniform(0.1, 10, size=100))
    df = pd.DataFrame({"noise": noise, "target": target})

    report = scan_target(df, "target", mode="lite")
    entry = next(e for e in report.successes if e.column == "noise")

    # Raw artifact present, robust value collapsed below the floor.
    assert abs(entry.result_data.diagnostics.sq_corr) > 0.35
    assert entry.result_data.diagnostics.sq_corr_robust <= 0.20
    assert report._dependence_understatement(entry) is None
    assert "Dependence may be understated" not in report.summary()
