# Changelog

All notable changes to CorrSleuth are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-06-28

### Fixed
- `biweight_midcorrelation` now uses the raw median absolute deviation
  (`scale=1.0`) in its Tukey weights, matching the canonical biweight
  midcorrelation (Wilcox; Langfelder & Horvath 2012). It previously used the
  normal-consistent MAD, which pushed the outlier-rejection cutoff out to ~13
  MADs and made the estimator less outlier-resistant than its name implies.
  This changes the metric's numeric output for non-degenerate data.

### Changed
- Unified the trimmed-Pearson logic: the outlier/leverage-sensitivity check in
  `api.py` now delegates to `metrics/robust.py` via a new
  `assess_outlier_sensitivity()` (which reuses `compute_trimmed_pearson`),
  removing a duplicate 1%-trim implementation, the previously dead-in-pipeline
  `compute_trimmed_pearson`, and three redundant `api.py` constants. The
  deep-mode `pearson_trimmed_1pct` metric and the leverage flag are now computed
  from the same trimmed value. Behavior is unchanged.
- Extracted the heuristic cascade's label-driving cut points into public,
  documented module-level constants in `corrsleuth/heuristics/classifier.py`
  (`STRONG_MAGNITUDE_THRESHOLD`, `WEAK_MAGNITUDE_THRESHOLD`,
  `RANK_LINEAR_GAP_THRESHOLD`, `PEARSON_KENDALL_GAP_THRESHOLD`,
  `NONMONOTONIC_MONOTONE_CEILING`, `NONMONOTONIC_DC_THRESHOLD`,
  `NEAR_LINEAR_GAP_THRESHOLD`, `WEAK_DC_THRESHOLD`) so they can be inspected and
  overridden. Behavior is unchanged.
- Replaced remaining inline threshold literals (low-n, missingness, unique-ratio
  in validation; bootstrap min-n) with named constants and added rationale
  docstrings to the existing robustness, bootstrap, and Chatterjee's-ξ
  thresholds.
- Split the `corrsleuth/scan.py` monolith into a `corrsleuth/scan/` package:
  `core` (the `scan_target` orchestration and `TargetScanEntry`), `report`
  (`CorrSleuthTargetReport` text/frame/Markdown rendering), and `plot`
  (`plot_top`). The public API (`scan_target`, `CorrSleuthTargetReport`,
  `TargetScanEntry` from `corrsleuth` or `corrsleuth.scan`) is unchanged.
- Documented that `scan_target` runs sequentially, with guidance on bounding
  cost for very wide DataFrames, in the scan docstring and interpretation guide.
- Expanded the ruff ruleset (added isort, pep8-naming, pyupgrade, simplify, and
  bugbear) and applied `ruff format` across the codebase; CI now enforces both
  `ruff check` and `ruff format --check`.
- Centralized the repeated "computed but no value" metric guard behind
  `MetricResult.no_value(...)`, and documented the intentional in-place
  enrichment of the internal `CleanPair` in `profile_pair`.

### Added
- `docs/thresholds-and-rationale.md` cataloguing every threshold in the package
  — its value, location, what it gates, the justification, and how to override
  the label-driving ones. Linked from the README and the interpretation guide.
- Static type checking: a `[tool.mypy]` configuration and a `mypy` CI job
  (non-strict) that the package now passes cleanly, plus `mypy` in the `dev`
  extra. Added docstrings to the internal bootstrap helpers.
- Property-based tests (`tests/test_property.py`, Hypothesis) asserting metric
  invariants over generated inputs: joint row-permutation invariance,
  constant-input → `None`, magnitude bounds, and symmetric-metric symmetry /
  Chatterjee ξ forward-reverse consistency.
- End-to-end smoke tests (`tests/test_smoke.py`) exercising the documented
  `profile_pair` / `scan_target` workflow across every render surface.
- Coverage reporting: `[tool.coverage]` config (branch coverage, `fail_under`
  floor), a `coverage` CI job, and `pytest-cov` / `hypothesis` in the `dev`
  extra.

## [0.1.0] - 2026-06-27

Initial release.

### Added
- `profile_pair()` for diagnosing a single numeric pairwise relationship,
  with `lite`, `standard`, and `deep` metric modes.
- `scan_target()` for profiling every eligible numeric predictor against a
  single numeric target, returning a `CorrSleuthTargetReport`.
- Core metrics (Pearson, Spearman, Kendall tau-b), standard-mode nonlinear
  metrics (Distance Correlation, Mutual Information), and deep-mode robust
  diagnostics (trimmed/winsorized/median-clipped Pearson, biweight
  midcorrelation, Chatterjee's ξ). Chatterjee's ξ uses the tie-corrected
  estimator from Chatterjee (2020), so it stays well-calibrated for discrete
  or low-cardinality responses.
- Heuristic diagnostic labels, validation warnings, recommendations, optional
  bootstrap intervals, and pattern-stability diagnostics.
- Deep-mode warning when Chatterjee's ξ is high but the assigned label is
  `weak_or_no_relationship` or `mixed_or_ambiguous`, so strong nonmonotonic
  dependence is never silently contradicted by the label.
- Outlier-sensitivity diagnostic that flags Pearson as leverage-sensitive when
  trimming the extreme 1% of x/y materially changes it, including the case where
  trimming flips the sign of the correlation.
- Missing-value policy on `profile_pair()` / `scan_target()`: `"pairwise"` drops
  rows missing in x or y; `"listwise"` drops rows missing in *any* column of the
  data (complete-case deletion); `"raise"` errors on any missing value in the
  pair.
- Matplotlib diagnostic plots (`result.plot()`, `report.plot_top()`).
- Markdown / dict / DataFrame exports for results and target reports. The
  summary and Markdown reports include the signed Pearson–Spearman gap, which
  reveals sign disagreement that the absolute rank/linear gap hides.
- `make_relationship()` deterministic relationship simulator, which validates
  its inputs (`n` must be an integer ≥ 2, `noise` must be non-negative) and
  raises `InputError` on bad arguments.
- Strict input validation with clear `InputError` messages: profiling a column
  against itself, duplicate column names, and non-positive `max_pairs` /
  `sample_size` values are rejected explicitly. Infinite values are rejected
  only when they appear in rows actually used after missing-value handling.
- Top-level exports for `CorrSleuthResult`, `MetricDiagnostics`, and the
  exception types (`CorrSleuthError`, `InputError`, `MetricComputationError`,
  `OptionalDependencyError`).

### Notes
- When a bootstrap cap (`max_n_for_bootstrap`) is smaller than the sample size,
  each replicate resamples that many rows (an m-out-of-n bootstrap); the
  resulting intervals are conservative (wider) and the emitted warning says so.

[0.2.0]: https://github.com/mbagalman/CorrSleuth/releases/tag/v0.2.0
[0.1.0]: https://github.com/mbagalman/CorrSleuth/releases/tag/v0.1.0
