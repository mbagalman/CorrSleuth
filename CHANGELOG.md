# Changelog

All notable changes to CorrSleuth are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-06-10

Initial release.

### Added
- `profile_pair()` for diagnosing a single numeric pairwise relationship,
  with `lite`, `standard`, and `deep` metric modes.
- `scan_target()` for profiling every eligible numeric predictor against a
  single numeric target, returning a `CorrSleuthTargetReport`.
- Core metrics (Pearson, Spearman, Kendall tau-b), standard-mode nonlinear
  metrics (Distance Correlation, Mutual Information), and deep-mode robust
  diagnostics (trimmed/winsorized/median-clipped Pearson, biweight
  midcorrelation, Chatterjee's ξ).
- Heuristic diagnostic labels, validation warnings, recommendations, optional
  bootstrap intervals, and pattern-stability diagnostics.
- Deep-mode warning when Chatterjee's ξ is high but the assigned label is
  `weak_or_no_relationship` or `mixed_or_ambiguous`, so strong nonmonotonic
  dependence is never silently contradicted by the label.
- Matplotlib diagnostic plots (`result.plot()`, `report.plot_top()`).
- Markdown / dict / DataFrame exports for results and target reports.
- `make_relationship()` deterministic relationship simulator.
- Strict input validation with clear `InputError` messages: profiling a column
  against itself, duplicate column names, and non-positive `max_pairs` /
  `sample_size` values are rejected explicitly. Infinite values are rejected
  only when they appear in rows actually used after missing-value handling.
- Top-level exports for `CorrSleuthResult`, `MetricDiagnostics`, and the
  exception types (`CorrSleuthError`, `InputError`, `MetricComputationError`,
  `OptionalDependencyError`).

[0.1.0]: https://github.com/mbagalman/CorrSleuth/releases/tag/v0.1.0
