# CorrSleuth Code Review — Phase 1 Pre-Release - 2026-05-02

Fresh-eyes review of the entire package in preparation for public release. Goal: correctness, code style/efficiency, and documentation quality.

## Summary

The Phase 1 work landed cleanly. All 33 tests pass and the package implements the contract described in `AGENTS.md` end-to-end. The product principles (cautious language, lightweight base install, lazy optional dependencies, useful warnings) are clearly preserved in both code and outputs. Lite mode profiles 100k rows in ~0.12 s on this machine, well under the 2 s requirement.

Verification:

```text
python -m pytest -q
33 passed in 22.47s
```

The findings below are mostly polish — release blockers are P1, the rest are quality issues worth fixing before going public on PyPI.

---

## P1 — Release blockers

### 1.1 LICENSE has no copyright holder

File: [LICENSE](../../LICENSE)

```text
Copyright (c) 2026
```

The MIT license needs a holder. PyPI does not require it, but most license scanners and downstream integrators expect a non-empty holder line. Either add a name (e.g., `Michael Bagalman` or `CorrSleuth Contributors` to match `pyproject.toml`'s `authors` field) or both.

### 1.2 No GitHub Actions CI configured

The PRD ([development/PRD.md](../development/PRD.md), §8.1) and [`development/AGENTS.md`](../development/AGENTS.md) ("Documentation & Build Requirements") both call for GitHub Actions CI. There is no `.github/workflows/` directory. For a public release, CI is the user-facing trust signal that the test suite is real and runs on every push.

Suggested minimum:

- A workflow that runs `pytest` on push and PR for at least Python 3.10, 3.11, 3.12.
- Run twice per Python version: once with `pip install .` (lite-mode coverage) and once with `pip install .[standard]` (full coverage). Today, every test runs against an environment that already has `dcor` and `sklearn` installed, so the `OptionalDependencyError` path is **completely uncovered** — see 1.3.
- Cache pip to keep CI under a minute.
- Optional: a single performance-benchmark step asserting the 100k-row lite-mode budget.

### 1.3 `OptionalDependencyError` paths are never tested

Files: [corrsleuth/metrics/optional.py](../../corrsleuth/metrics/optional.py)

Both `compute_distance_correlation` and `compute_mutual_information` raise `OptionalDependencyError` when `mode="standard"` is requested without `dcor` / `sklearn`. The current tests run in an environment where both packages are installed, so this branch never executes. A user installing the lite version and calling `mode="standard"` is the most likely first failure mode for new adopters; it must be tested.

Suggested fix:

- Add tests that monkeypatch `sys.modules` (e.g., `monkeypatch.setitem(sys.modules, "dcor", None)`) and assert `OptionalDependencyError` is raised with an install-instruction message.
- Pair this with the CI matrix in 1.2 so the lite leg of CI also exercises the error path naturally.

### 1.4 Distance-correlation downsampling is non-deterministic across users

File: [corrsleuth/metrics/optional.py:25](../../corrsleuth/metrics/optional.py#L25)

```python
rng = np.random.default_rng(42) # Deterministic downsampling for tests
idx = rng.choice(pair.n_used, max_n_for_dcor, replace=False)
```

Hard-coding seed `42` makes the output deterministic for a given input, which is good for tests, but:

1. There is no documented way for a caller to control it. `profile_pair()` exposes `max_n_for_dcor` but no `random_state`. Two users running the same code on the same data will get the same number — fine — but the user has no escape hatch if they want a *different* sample.
2. The function silently mutates `pair.warnings` (and the validation object). A metric function reaching into the validation contract to append warnings is a tight coupling. Consider returning the warning alongside the `MetricResult` (e.g., add an optional `warnings: list[str]` field) and accumulating in `profile_pair()`.

Either way, the comment "Deterministic downsampling for tests" reveals the intent leaked into production. Recommend either:

- Plumbing a `random_state` parameter through `profile_pair()` → `compute_distance_correlation()` and defaulting to `42` (or a documented default), or
- Adding a docstring note on `compute_distance_correlation` and `profile_pair` that downsampling is deterministic with a fixed internal seed.

The same applies to [corrsleuth/metrics/optional.py:47](../../corrsleuth/metrics/optional.py#L47) where `mutual_info_regression(..., random_state=42)` is hard-coded — at least there sklearn does the seeding so it's not as hidden, but it still deserves a docstring mention.

---

## P2 — Quality and correctness

### 2.1 LOWESS smoother is non-deterministic on the same data

File: [corrsleuth/plotting/pairplot.py:51-52](../../corrsleuth/plotting/pairplot.py#L51-L52)

```python
import numpy as np
idx = np.random.choice(n, n_lowess, replace=False) if n > n_lowess else np.arange(n)
```

This uses the legacy global numpy RNG, so calling `result.plot()` twice on the same data renders different smoother curves whenever `n > 1000`. Users will treat this as a bug.

Suggested fix:

```python
rng = np.random.default_rng(42)
idx = rng.choice(n, n_lowess, replace=False) if n > n_lowess else np.arange(n)
```

Also, `import numpy as np` belongs at module top — it's already used implicitly (matplotlib pulls it) and it's odd that `numpy` is imported inside both plotting and metric functions. Move imports to the top of each module.

### 2.2 `disagreement_score` formula is inconsistent with `nonmonotonic_gap`

File: [corrsleuth/api.py:153](../../corrsleuth/api.py#L153)

```python
disagreement_score = abs(p - s) + max(0.0, dc - s)
```

The PRD's `disagreement_score` (FR-001.1) defines `abs(p - s) + max(0, dc - s)` — that's what is implemented, so the code matches the spec.

But the new `MetricDiagnostics.nonmonotonic_gap` (Ticket 1.2) defines `dc - max(|p|, |s|)`. The two related "how much does dc exceed the rank/linear measures" quantities use different denominators (`s` vs `max(p, s)`), and a user reading `summary()` will see `disagreement_score` and `nonmonotonic_gap` that don't visibly relate. Either:

- Document explicitly that they are computed from different bases, or
- Update `disagreement_score` to use `max(p, s)` so the two are consistent.

### 2.3 Heuristic `not_computable` recommendations dominate when the real issue is small `n`

File: [corrsleuth/heuristics/classifier.py:22](../../corrsleuth/heuristics/classifier.py#L22)

When a single non-NA row remains (e.g., `n_used == 1`), `pair.x_is_constant` is true (one unique value), so `not_computable` is assigned with priority 1 — *before* `low_power_or_uncertain` is checked. The user sees recommendations that say "Check for constant variables (zero variance)" when the true issue is "one row left after dropna". I confirmed this with:

```python
df = pd.DataFrame({'x': [1.0, np.nan, np.nan], 'y': [1.0, np.nan, np.nan]})
res = cs.profile_pair(df, 'x', 'y')
# pattern: not_computable
# recommendations: "Check for constant variables (zero variance)."
```

Two reasonable options:

- Detect this case in validation: if `n_used == 1` (or `< 2`), raise `InputError` outright — you cannot meaningfully profile a single point.
- Or, in the classifier, swap priority 1 and 2 *only when the constant-input flag is on a single observation* — but that's brittle. The validation-time fix is cleaner.

### 2.4 Conflicting-direction warning lives in `api.py`, not in the heuristic engine

File: [corrsleuth/api.py:133-138](../../corrsleuth/api.py#L133-L138)

```python
p = _metric_value(metrics_map, "pearson")
s = _metric_value(metrics_map, "spearman")

if p is not None and s is not None and abs(p) > 0.3 and abs(s) > 0.3:
    if (p > 0 and s < 0) or (p < 0 and s > 0):
        pair.warnings.append("Pearson and Spearman have conflicting directions; ...")
```

This is a heuristic about metric agreement and naturally belongs in `corrsleuth/heuristics/`. As written, `apply_heuristics` returns labels and recommendations but warnings are attached ad hoc in `profile_pair`. As more agreement-based warnings are added (Phase 2 will), the API module will grow into a second classifier. Suggested fix:

- Move the conflicting-direction check into `apply_heuristics` (or a sibling `apply_warnings`) and return warnings alongside labels and recommendations.
- Threshold (`0.3`) should be a named constant, not a magic number duplicated across PRD and code.

The condition `(p > 0 and s < 0) or (p < 0 and s > 0)` simplifies to `p * s < 0`, which is shorter and more obviously a sign-mismatch test.

### 2.5 Constant-input metric rows are emitted with `value=None` in `metrics_df`

File: [corrsleuth/api.py:142-146](../../corrsleuth/api.py#L142-L146)

```python
records = []
for k, v in metrics_map.items():
    if v.available:
        records.append({"metric": k, "value": v.value})
metrics_df = pd.DataFrame(records)
```

Pearson/Spearman/Kendall on a constant column return `MetricResult(value=None, available=True)`, so they appear in `metrics_df` with `value=None`. The `summary()` and `plot()` paths now handle this (rendered as `NA` after the prior code-review fixes), and `to_frame()` carries `None` through.

This is OK, but the intent in [corrsleuth/result.py:8-13](../../corrsleuth/result.py#L8-L13) was that `available` indicates "the dependency for this metric exists / the metric ran." Today `available=True` means both "import worked" and "metric makes sense for this input." Two distinct concepts share one flag. If you ever want to distinguish "skipped because constant input" vs "skipped because dependency missing" downstream, the current shape doesn't allow it. Consider renaming/adding:

```python
@dataclass
class MetricResult:
    name: str
    value: Optional[float]
    available: bool   # dependency available
    computed: bool    # actually produced a value
```

Not a release blocker. Worth a note for v0.2.

### 2.6 `summary()` and `explain()` type hints are wrong

Files: [corrsleuth/result.py:85](../../corrsleuth/result.py#L85), [corrsleuth/result.py:128](../../corrsleuth/result.py#L128)

```python
def summary(self, include_caveat: bool = None) -> str:
def explain(self, include_caveat: bool = None) -> str:
```

`bool = None` will fail strict type checking in mypy/pyright. Fix to `Optional[bool] = None`.

### 2.7 `HeuristicResult.disagreement_components` is dead

File: [corrsleuth/result.py:17-22](../../corrsleuth/result.py#L17-L22), [corrsleuth/heuristics/classifier.py:55](../../corrsleuth/heuristics/classifier.py#L55)

`apply_heuristics` populates `disagreement_components`, but no caller reads it — `_build_diagnostics` in `api.py` recomputes the gaps from `metrics_map`. Two implementations of the same idea is a maintenance hazard; pick one. Either:

- Have `_build_diagnostics` consume the heuristic's gaps (single source), or
- Drop `disagreement_components` from `HeuristicResult` (it's an internal contract; safe to change).

### 2.8 `corrsleuth/utils/types.py` is a dead module

Files: [corrsleuth/utils/types.py](../../corrsleuth/utils/types.py)

Empty file (just `from typing import …` re-exports). Nothing in the package imports it. There is no `corrsleuth/utils/__init__.py` either, so it works as an implicit namespace package only by accident. Delete the directory.

### 2.9 `MetricComputationError` is never raised

File: [corrsleuth/exceptions.py:17](../../corrsleuth/exceptions.py#L17)

Defined and exported, never raised. Either wire it into `compute_*` functions (catch upstream `ValueError`/`RuntimeError` and re-raise as `MetricComputationError` with the metric name), or remove it. Defining unused exceptions is a small public-API contract you'll have to honor later.

### 2.10 Mode validation uses `ValueError`, missing-mode validation uses `InputError`

File: [corrsleuth/api.py:100-103](../../corrsleuth/api.py#L100-L103) vs [corrsleuth/validation/input.py:48](../../corrsleuth/validation/input.py#L48)

Both should consistently raise `InputError` (a `CorrSleuthError` subclass per the taxonomy in `AGENTS.md`). Today an unknown `mode` gives `ValueError`, an unknown `missing` gives `InputError`. Use `InputError` everywhere a user passed a bad string argument.

### 2.11 Metric functions silently mutate the validation object

Files: [corrsleuth/metrics/optional.py:23](../../corrsleuth/metrics/optional.py#L23), [corrsleuth/api.py:120-128](../../corrsleuth/api.py#L120-L128)

`compute_distance_correlation()` appends to `pair.warnings`. `_compute_outlier_sensitivity()` is called from `api.py` and *also* mutates `pair.flags` and `pair.warnings`. The classifier then reads those flags to decide the leverage label.

This works, but the `CleanPair` is being treated as a mutable accumulator. Bugs from this pattern are easy to introduce later (e.g., calling validation twice, or running heuristics on a pair from a different code path). Cleaner pattern:

- Each step returns its own warnings/flags.
- `profile_pair` accumulates them into a final list.
- `apply_heuristics` receives a frozen set of flags.

Not urgent — flag for a refactor before Phase 2 if the heuristic engine grows.

---

## P3 — Style and small polish

### 3.1 Variable shadowing in `profile_pair`

File: [corrsleuth/api.py:133-150](../../corrsleuth/api.py#L133-L150)

```python
p = _metric_value(metrics_map, "pearson")  # signed
s = _metric_value(metrics_map, "spearman")
if p is not None and s is not None and abs(p) > 0.3 and abs(s) > 0.3:
    ...
# ...several lines later...
p = abs(_metric_value(metrics_map, "pearson") or 0.0)  # absolute
s = abs(_metric_value(metrics_map, "spearman") or 0.0)
```

`p` and `s` carry different meanings in two adjacent blocks. Rename the second pair (`abs_p`, `abs_s`) so a reader can't confuse signed vs. absolute.

### 3.2 Redundant `n_used < 30` check after `low_n` flag

File: [corrsleuth/heuristics/classifier.py:25](../../corrsleuth/heuristics/classifier.py#L25)

```python
elif "low_n" in flags or n_used < 30:
```

`low_n` is set in validation iff `n_used < 30`, so the second condition is always implied. Drop the `or n_used < 30` clause.

### 3.3 `_format_value` is defined twice

[corrsleuth/result.py:82](../../corrsleuth/result.py#L82) and [corrsleuth/plotting/pairplot.py:6](../../corrsleuth/plotting/pairplot.py#L6) define the same helper. Extract one copy (e.g., into `corrsleuth/_formatting.py` or as a module-level function in `result.py`) and import it from the plot module.

### 3.4 `_CAVEAT` is imported via a private name across modules

[corrsleuth/result.py:123](../../corrsleuth/result.py#L123) does `from corrsleuth.heuristics.explanations import _CAVEAT`. Importing a single-underscore "private" symbol from a sibling module is a code smell. Promote it to `CAVEAT` (no underscore) or move the constant into `result.py` and import from there in `explanations.py`.

### 3.5 `make_relationship` overwrites `x` for some shapes

File: [corrsleuth/datasets/simulations.py:33](../../corrsleuth/datasets/simulations.py#L33)

```python
x = rng.uniform(-3, 3, size=n)
y = np.zeros(n)

if shape_type == "linear_positive":
    ...
elif shape_type == "monotonic_log":
    x = np.exp(rng.uniform(0.1, 10, size=n))   # overwrites the line above
```

`monotonic_log` and `outlier_driven` overwrite/mutate `x` after generating it. Wasted draws shift the RNG state for downstream noise terms — which is intentional in `monotonic_log` (it gives the heavy-skew Pearson-vs-Spearman gap the test relies on), but unobvious. Pull each shape into its own branch where `x` and `y` are constructed together, or document the trade-off in a comment.

While there: `independent` uses `y = rng.normal(0, 1 + noise, size=n)` — quirky semantics for `noise` in this branch. Other branches add noise on top of a signal; this one inflates the marginal scale. Worth a one-line comment or a normalization to `noise=0` ⇒ standard normal.

### 3.6 README quickstart calls `result.plot(show=True)`

File: [README.md:42](../../README.md#L42)

```python
fig = result.plot(show=True)
```

`show=True` blocks the process in script (non-Jupyter) contexts. Most users running the quickstart in a terminal will be confused. Default the README example to `result.plot()` (returns the figure; matches the docstring contract) and add a sentence: "Pass `show=True` to display the figure in script contexts."

### 3.7 `to_frame()` repeats scalar diagnostic columns N times

File: [corrsleuth/result.py:163-173](../../corrsleuth/result.py#L163-L173)

`to_frame()` returns one row per metric, with `pattern`, `x`, `y`, and every `diagnostic_*` column duplicated. Functionally correct, but wasteful and a little awkward when downstream code does `df.groupby(...)`. The ticket pack (1.2) does say flatten "with a consistent prefix" — that's done — but consider whether a long-format metrics frame plus a separate scalar diagnostics frame would serve users better. Optional refinement; current shape is acceptable for v0.1.

### 3.8 Plot smoother subsamples without seed *and* the line-sort assumes monotone subsample

File: [corrsleuth/plotting/pairplot.py:53-57](../../corrsleuth/plotting/pairplot.py#L53-L57)

```python
z = lowess(y[idx], x[idx], frac=0.3)
order = np.argsort(z[:, 0])
ax_scatter.plot(z[order, 0], z[order, 1], ...)
```

The `argsort` step is correct, but `statsmodels.lowess` already returns rows sorted by x by default, so the `order` is the identity permutation in normal use. The mock `statsmodels` in `tests/statsmodels/api.py` returns unsorted data, so this is what makes the test pass. Either:

- Drop the `np.argsort` and trust `lowess` (then update the mock to also return sorted data), or
- Keep it and add a one-line comment explaining why.

### 3.9 `compute_*` functions don't have docstrings

Quick win for documentation. Each `compute_pearson/spearman/kendall/distance_correlation/mutual_information` should have a one-liner explaining what it returns, when it returns `available=False`, and what flags it requires from `CleanPair`.

### 3.10 README does not mention what to do without `[standard]` extras when using `mode="standard"`

The README discusses `mode="standard"` and notes the `OptionalDependencyError`. Add a single sentence explicitly: "If you call `mode='standard'` without installing `corrsleuth[standard]`, you will see `OptionalDependencyError`." Helps the most common first-time confusion.

---

## What's working well

- Data contract pipeline (`CleanPair` → `MetricResult` → `HeuristicResult` → `CorrSleuthResult`) matches the architecture spec.
- Lazy imports for `dcor` and `sklearn` are correct; base install really is lightweight.
- `OptionalDependencyError` messages tell the user how to fix the install.
- Cautious-language requirement is honored throughout `explanations.py` ("evidence consistent with…", "may suggest…").
- Constant-input handling cascades cleanly: `not_computable` label → `NA` rendered in summary and plot → no crashes in `to_dict` / `to_frame`.
- The non-causal caveat appears in both `summary()` and `explain()` and is suppressible.
- Test suite covers the canonical shape contracts, the conflicting-direction warning, the trim-sensitivity classifier path, and the LOWESS optional path via the in-tree mock.

---

## Suggested fix order

1. P1 items 1.1–1.4 (LICENSE holder, CI, lite-mode `OptionalDependencyError` test, downsampling determinism doc/seed).
2. P2 items 2.1–2.4 (LOWESS seed, `disagreement_score` consistency, single-row validation, conflicting-direction logic location).
3. P2 items 2.6–2.10 (type hints, dead code removal, exception consistency).
4. P3 polish.

After P1 + P2.1–2.4 the package is ready to release publicly.
