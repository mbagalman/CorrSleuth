# CorrSleuth Code Review - 2026-05-02

## Summary

The current implementation is a promising v0.1 skeleton and the existing test suite passes. The main issues are contract mismatches with `AGENTS.md` / the PRD and a few robustness gaps around edge cases that CorrSleuth explicitly wants to handle gracefully.

Current verification:

```text
C:\Users\micha\AppData\Local\Python\pythoncore-3.14-64\python.exe -m pytest -q
15 passed in 14.32s
```

## Findings

### P1 - `missing="listwise"` is documented but not implemented

File: `corrsleuth/validation/input.py`

The v0.1 contract documents `missing="pairwise"`, `"listwise"`, and `"raise"`, but validation only handles `"pairwise"` and `"raise"`. Passing `missing="listwise"` leaves NaNs in place, so SciPy metric calls can raise lower-level errors instead of producing a clean `CleanPair`.

Unknown `missing` values are also silently accepted, which makes typos dangerous.

Suggested fix:

- Treat `"listwise"` the same as `"pairwise"` for the current pairwise-only v0.1 workflow, or explicitly document a distinction if one is intended later.
- Raise `InputError` for unsupported `missing` values.
- Add tests for `"listwise"` and an invalid mode such as `"pairwse"`.

### P1 - `not_computable` results can crash `summary()` and `plot()`

Files: `corrsleuth/result.py`, `corrsleuth/plotting/pairplot.py`

When an input is constant, metric values are `None`, but `profile_pair()` still includes those rows in `metrics_df`. `summary()` and the plot text panel format every metric with `:.3f`, so a valid `not_computable` result can fail when the user calls the methods the result contract promises.

Suggested fix:

- Render `None` metric values as `NA`, `not computable`, or similar.
- Apply the same formatting helper in both `summary()` and the plot text panel.
- Add tests that call `summary()` and `plot()` after profiling a constant column.

### P2 - Standard-mode distance-correlation downsampling cannot be overridden

Files: `corrsleuth/api.py`, `corrsleuth/metrics/optional.py`

The spec says users may override the `n_used > 20_000` distance-correlation cap with `max_n_for_dcor=...`, but `profile_pair()` has no such parameter and `compute_distance_correlation()` hard-codes `20000`.

Suggested fix:

- Add `max_n_for_dcor: int | None = 20_000` to `profile_pair()`.
- Pass it through to `compute_distance_correlation()`.
- Interpret `None` as no cap.
- Keep the clear warning when downsampling occurs.

### P2 - Conflicting directional evidence warning is missing

File: `corrsleuth/heuristics/classifier.py`

The PRD calls for a warning when Pearson and Spearman have opposite signs and both are materially strong. The current heuristic cascade only assigns labels and recommendations; `profile_pair()` returns only validation and downsampling warnings.

Suggested fix:

- Detect `sign(pearson) != sign(spearman)` when both absolute values exceed the chosen threshold, currently documented around `0.3`.
- Append a cautious warning such as: `Pearson and Spearman have conflicting directions; inspect the scatter plot and check for nonlinearity, segments, or leverage points.`
- Add a targeted unit test.

### P2 - Plotting re-cleans the original DataFrame instead of using the validated pair

Files: `corrsleuth/api.py`, `corrsleuth/result.py`, `corrsleuth/plotting/pairplot.py`

The result stores the original `data` object and `plot_pair()` independently does `dropna()` on it. That means plotting can diverge from the data actually profiled, especially if additional missing policies are added, or if the caller mutates the DataFrame after profiling.

Suggested fix:

- Preserve the validated `CleanPair` data, or a small cleaned DataFrame copy, inside `CorrSleuthResult`.
- Have `.plot()` render that cleaned data rather than re-reading the original frame.
- Add a test that mutates the original DataFrame after profiling and confirms plotting still uses the profiled data.

### P3 - `include_caveat` on `profile_pair()` is unused

File: `corrsleuth/api.py`

`profile_pair()` exposes `include_caveat`, matching the documented API, but the value is never stored or used. Since `summary()` and `explain()` each take their own `include_caveat` argument, this parameter is currently misleading.

Suggested fix:

- Either remove `include_caveat` from `profile_pair()` or store it as a result-level default.
- If preserved, make `result.summary()` and `result.explain()` default to that stored value when their argument is omitted.

### P3 - Root import does not expose the simulator used by the core demo story

File: `corrsleuth/__init__.py`

The package root exports only `profile_pair`. The docs preserve `from corrsleuth.datasets import make_relationship`, so this is not strictly broken. Still, the v0.1 demo emphasis would be smoother if `make_relationship` were also easy to discover from the package namespace or clearly documented as a submodule-only API.

Suggested fix:

- Either export `make_relationship` at the package root, or keep the current API and make the README consistently show `from corrsleuth.datasets import make_relationship`.

## Suggested Fix Order

1. Implement documented missing-data modes and invalid-mode validation.
2. Make `None` metric values render safely in `summary()` and `plot()`.
3. Add `max_n_for_dcor` and pass it through standard mode.
4. Add conflicting directional evidence warnings.
5. Store and plot the validated pair data.
6. Decide whether `include_caveat` belongs on `profile_pair()`.
7. Decide whether to export `make_relationship` at package root.

## Recommended New Tests

- `missing="listwise"` produces a clean pair and does not leak NaNs into metric computation.
- Invalid `missing` values raise `InputError`.
- Constant input returns `not_computable`, and `summary()` / `plot()` do not crash.
- `max_n_for_dcor=None` disables downsampling.
- `max_n_for_dcor=...` downsampling appends a warning.
- Conflicting Pearson/Spearman signs append a warning.
- Plotting uses the profiled clean data, not a later-mutated original DataFrame.
