# Changelog

All notable changes to CorrSleuth are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.0]: https://github.com/mbagalman/CorrSleuth/releases/tag/v0.1.0
