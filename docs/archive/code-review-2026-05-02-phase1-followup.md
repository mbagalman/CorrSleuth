# Code Review Followup - Phase 1 Pre-Release - 2026-05-02

Resolution log for [code-review-2026-05-02-phase1.md](code-review-2026-05-02-phase1.md). Fixes the P1 release blockers and the top P2 correctness items. Test count: 33 → 43, all passing.

```text
python -m pytest -q
43 passed in 13.06s
```

## Resolved

### P1 — Release blockers

- **1.1 LICENSE copyright holder.** [LICENSE](../../LICENSE) now reads `Copyright (c) 2026 Michael Bagalman`.
- **1.2 GitHub Actions CI.** Added [.github/workflows/ci.yml](../../.github/workflows/ci.yml). Matrix: Python 3.10–3.13 × `lite` / `standard` install legs, runs `pytest -q` on push and PR to `main`.
- **1.3 OptionalDependencyError tests.** Added four tests in [tests/test_metrics.py](../../tests/test_metrics.py) covering both standard-mode (raises `OptionalDependencyError` with the install hint) and lite-mode (returns unavailable) paths for each of `dcor` and `sklearn`. They use `monkeypatch.setitem(sys.modules, name, None)` to simulate missing extras at runtime, so they exercise the failure mode regardless of the actual environment. The pre-existing standard-mode tests now use `pytest.importorskip("dcor")` / `pytest.importorskip("sklearn")` so the lite CI leg skips them cleanly instead of failing.
- **1.4 Downsampling determinism.** `profile_pair()` now accepts `random_state: int = 42`, threaded into both `compute_distance_correlation` (downsample seed) and `compute_mutual_information` (passed to sklearn). Default is `42` for backwards compatibility. The downsampling warning now includes the seed value so the user can see what was used.

### P2 — Correctness/quality

- **2.1 LOWESS subsample seed.** [corrsleuth/plotting/pairplot.py](../../corrsleuth/plotting/pairplot.py) now uses a `numpy.random.default_rng(42)`-seeded subsample when `n > 1000`. Repeated `result.plot()` calls on the same data render the same smoother. Locked in by `test_plot_lowess_subsample_is_deterministic` in [tests/test_output.py](../../tests/test_output.py). Top-level `numpy` import moved out of the function body.
- **2.2 `disagreement_score` consistency.** Updated formula to `abs(|p| - |s|) + max(0, dc - max(|p|, |s|))` so the second term uses the same denominator as `MetricDiagnostics.nonmonotonic_gap`. This is a deviation from the PRD's original `dc - s` (FR-001.1), but aligns the score with the more recent diagnostics ticket and produces a single coherent definition of "extra dependence captured by dc beyond what either rank or linear caught."
- **2.3 `n_used >= 2` validation.** [corrsleuth/validation/input.py](../../corrsleuth/validation/input.py) now raises `InputError` for `n_used < 2`. The previous "all-NaN raises but n_used=1 passes" gap is closed; users with a single observation get a precise error message instead of a misleading `not_computable` label that recommends checking for constant variables.
- **2.4 Conflicting-direction warning relocated.** Moved the `sign(p) != sign(s)` check out of [corrsleuth/api.py](../../corrsleuth/api.py) into a new `detect_metric_warnings()` helper in [corrsleuth/heuristics/classifier.py](../../corrsleuth/heuristics/classifier.py), with the threshold extracted as a `CONFLICTING_SIGN_THRESHOLD = 0.3` module constant. `profile_pair` now calls `pair.warnings.extend(detect_metric_warnings(metrics_map))`. Added unit tests for the helper directly so future agreement-based warnings can be added in one place.

### Bonus cleanups picked up along the way

- **Mode validation now consistently raises `InputError`** (was `ValueError`) — matches the rest of the validation taxonomy. `mode='deep'` continues to raise `NotImplementedError` per AGENTS.md.
- **`profile_pair` now has a real docstring** describing every parameter, including the new `random_state`.
- **Variable shadowing in `profile_pair` removed.** The signed Pearson/Spearman previously named `p`/`s` in the conflicting-direction block (since deleted) and the absolute Pearson/Spearman in the disagreement-score block (now `abs_p`/`abs_s`) no longer share names.
- **Redundant `n_used < 30` check** dropped from the heuristic classifier; the `low_n` flag set in validation is the single source of truth.
- **README updated** to document `random_state` and to mention the `OptionalDependencyError` behavior up front.

## Second pass — remaining P2 polish

- **2.6 Type hints fixed.** `summary()` and `explain()` now declare `include_caveat: Optional[bool] = None`.
- **2.7 `disagreement_components` removed.** The dead field is gone from `HeuristicResult`; `MetricDiagnostics` is the single source of truth for gap values.
- **2.8 Dead module deleted.** `corrsleuth/utils/` (containing only an empty `types.py`) removed entirely.
- **2.9 `MetricComputationError` wired in.** Each `compute_*` function now wraps its underlying scipy/dcor/sklearn call in a narrow `try/except` that re-raises as `MetricComputationError` with the metric name in the message. Locked in by `test_compute_pearson_wraps_unexpected_failures_as_metric_error`. The exception is now a real part of the public API rather than a dead taxonomy entry.

Test count after the second pass: 44.

## Still open

Deferred — none are release blockers:

- 2.5 `MetricResult.available` overloads two concepts (dependency-available vs. value-was-computable). Schema change worth thinking through with v0.2 in mind.
- 2.10 Resolved as part of 1.4/2.4 — mode validation now uses `InputError`.
- 2.11 Metric/sensitivity functions still mutate `pair.warnings` and `pair.flags`. Acceptable for v0.1; pair this refactor with Phase 2 when more agreement-based warnings land.

All P3 polish items (`_format_value` duplication, `_CAVEAT` cross-module private import, `make_relationship` shape branch tidiness, README quickstart `show=True`, etc.) are still open and tracked in the original review.
