```markdown
# AGENTS.md – CorrSleuth Development Bible

**This is the canonical reference for all agents (human or AI) working on CorrSleuth v0.1.**  
Read the **Product Principles** first.  
Everything else in this file is subordinate to those principles.  
When in doubt, refer back to the core demo at the bottom.

---

## Project: CorrSleuth

CorrSleuth is a lightweight, local-first Python package for diagnosing pairwise relationships between numeric variables in pandas DataFrames. It sits above standard metric calculators and provides an interpretive workflow: compute multiple association measures, compare their agreement/disagreement, assign a cautious diagnostic label, and return explanations, warnings, recommendations, and plots.

The core v0.1 product is one excellent workflow:

```python
import corrsleuth as cs

result = cs.profile_pair(df, "x", "y", mode="lite")

result.pattern
result.summary()
result.explain()
result.plot()
```

CorrSleuth is not “another correlation matrix.” It is a relationship diagnosis engine.

---

## Product Principles

All agents working on this repository must preserve these principles:

1. **Diagnostic, not causal**  
   CorrSleuth identifies evidence consistent with relationship patterns.  
   It must never imply causation, treatment effects, causal mechanisms, or model specification certainty.

2. **Cautious language**  
   Use phrases like “evidence consistent with,” “may suggest,” and “appears consistent with.”  
   Avoid definitive language such as “this is U-shaped,” “this proves,” or “X causes Y.”

3. **Explain the disagreement**  
   The value of CorrSleuth is not the individual coefficients.  
   The value is explaining why Pearson, Spearman, Kendall, distance correlation, and mutual information agree or disagree.

4. **Useful warnings over silent failure**  
   Missingness, ties, constant inputs, small samples, nonnumeric inputs, infinities, unavailable optional dependencies, and computational shortcuts must be explicit.

5. **Lightweight first**  
   v0.1 should keep the base installation small.  
   Optional heavier metrics should be behind extras and lazy imports.

6. **One great pairwise profiler before broad automation**  
   Do not build target scanning, full correlation reports, sklearn transformers, categorical support, or HTML reports until the v0.1 `profile_pair()` workflow is solid.

---

## v0.1 Scope

### In Scope
- Numeric-vs-numeric pairwise profiling.
- `profile_pair()`.
- `CorrSleuthResult`.
- Lite metrics: Pearson, Spearman, Kendall tau-b.
- Optional standard metrics: Distance correlation, Mutual information.
- Heuristic relationship labeling.
- Warnings and recommendations.
- Deterministic relationship simulator.
- Basic diagnostic plot (scatter + rank-rank + metric summary).
- Local-only execution.
- pandas-native API.

### Out of Scope for v0.1
- Categorical or mixed-type variables.
- Target scanning across all variables.
- Full pairwise matrix scanning.
- HTML reports.
- sklearn transformer/selector.
- Variable clustering.
- Partial correlations.
- Causal inference.
- Automatic model fitting.
- Feature transformation recommendations that sound definitive.
- Network calls, telemetry, or cloud execution.

---

## Package Naming

Preferred user-facing package name:  
`pip install corrsleuth`

Preferred import:  
`import corrsleuth as cs`

---

## Expected Public API

### `profile_pair`

```python
result = cs.profile_pair(
    data=df,
    x="column_x",
    y="column_y",
    mode="lite",          # "lite" or "standard"
    missing="pairwise",   # "pairwise", "listwise", or "raise"
    include_caveat=True,
)
```

### `make_relationship`

```python
from corrsleuth.datasets import make_relationship

df = make_relationship(
    shape_type="u_shape",
    n=500,
    noise=0.1,
    random_state=42,
)
```

The first six shapes are required for v0.1 tests and must be deterministic when `random_state` is provided.

---

## Execution Modes

### `mode="lite"` (default)
Base mode. Only core dependencies required.  
Metrics: Pearson, Spearman, Kendall tau-b.

### `mode="standard"`
Adds Distance correlation and Mutual information.  
Dependencies (`corrsleuth[standard]`): `dcor`, `scikit-learn`.

**Important:** For large datasets (`n_used > 20_000`), automatically downsample to 20,000 rows when computing distance correlation (with a clear warning in `CorrSleuthResult.warnings`). Users may override with `max_n_for_dcor=...`.

If required optional dependencies are missing, raise `OptionalDependencyError` with install instructions. Do not silently skip metrics.

### `mode="deep"`
Reserved for future functionality. Raise `NotImplementedError` if called in v0.1.

---

## Internal Architecture

```text
profile_pair()
  → validate_pair()
  → CleanPair
  → compute_metrics()
  → MetricResult objects
  → classify_relationship()
  → HeuristicResult
  → CorrSleuthResult
```

---

## Suggested Module Layout

```text
corrsleuth/
  __init__.py
  api.py
  result.py
  exceptions.py

  validation/
    __init__.py
    input.py
    missing.py
    ties.py

  metrics/
    __init__.py
    core.py
    optional.py
    registry.py

  heuristics/
    __init__.py
    classifier.py
    rules.py
    explanations.py

  plotting/
    __init__.py
    pairplot.py

  datasets/
    __init__.py
    simulations.py

  utils/
    __init__.py
    dependencies.py
    types.py
```

---

## Internal Data Contracts

### `CleanPair`
```python
@dataclass
class CleanPair:
    x: pd.Series
    y: pd.Series
    x_name: str
    y_name: str
    n_original: int
    n_used: int
    missing_count: int
    missing_ratio: float
    x_unique_ratio: float
    y_unique_ratio: float
    x_is_constant: bool
    y_is_constant: bool
    flags: list[str]          # machine-readable
    warnings: list[str]       # raw validation warnings only
```

### `MetricResult`, `HeuristicResult`, and `CorrSleuthResult`
(See full definitions in `result.py` once implemented.)

`CorrSleuthResult` must expose:
- `.pattern`
- `.summary(include_caveat=True)`
- `.explain(include_caveat=True)`
- `.plot(show=False)` → **must return** a `matplotlib.figure.Figure` (never call `plt.show()` unless `show=True`)
- `.to_dict()`
- `.to_frame()`

---

## Exception Taxonomy

- `CorrSleuthError`
- `InputError`
- `OptionalDependencyError`
- `MetricComputationError`

---

## Heuristic Classification Rules

Use absolute values: `p = abs(pearson)`, `s = abs(spearman)`, `k = abs(kendall)`, `dc = distance_correlation`.

**Priority order (apply in this sequence):**
1. `not_computable`
2. `low_power_or_uncertain` (`n_used < 30`)
3. `possible_outlier_or_leverage`
4. `nonmonotonic_dependence`
5. `monotonic_nonlinear`
6. `near_linear`
7. `weak_or_no_relationship`
8. `mixed_or_ambiguous`

High ties, missingness, etc. produce **warnings**, not primary label overrides (unless they make computation invalid).

---

## Plotting Requirements

`.plot(show=False)` must return a `matplotlib.figure.Figure`.  
Keep it simple: scatterplot + rank-rank plot + metric summary.

---

## Explanation Requirements

Use cautious, disagreement-focused language with a non-causal caveat by default.

---

## Testing Strategy

- Core API + metric fidelity tests
- Validation edge cases
- Optional dependency tests
- Simulator determinism
- Canonical shape classification tests (using the 6 required shapes)
- Plot contract tests

---

## Performance Guidance

`mode="lite"` on 100,000 rows should complete in under 2 seconds.  
Add a timing benchmark to CI.

---

## Documentation & Build Requirements

- Modern `pyproject.toml`, Python 3.10+, MIT License
- GitHub Actions CI
- Strong README with quickstart and five canonical examples

**Suggested README opening:**
```markdown
# CorrSleuth

Correlation is not one number.

CorrSleuth profiles numeric pairwise relationships in pandas DataFrames by comparing multiple association measures and translating their agreement or disagreement into practical diagnostics.
```

---

## Core Demo to Preserve

```python
import corrsleuth as cs
from corrsleuth.datasets import make_relationship

df = make_relationship("u_shape", n=500, noise=0.1, random_state=42)

result = cs.profile_pair(df, "x", "y", mode="standard")

print(result.pattern)
print(result.explain())
fig = result.plot()
```

This demo **is** the product. Everything else supports it.

---

## Anti-Goals for Agents

*(Same strong list as before — omitted here for brevity; keep unchanged)*

---

**Suggested GitHub Issues for v0.1** (in rough dependency order):
1. Initialize package skeleton and `pyproject.toml`
2. Add custom exceptions + dataclasses
3. Implement pair validation
4. Implement core metrics
5. Implement `CorrSleuthResult`
6. Implement heuristic classifier
7. Implement `summary()` / `explain()`
8. Add optional dependency support + standard mode
9. Implement simulator
10. Implement diagnostic plotting
11. Add tests + CI
12. Write README + examples

---

When uncertain, choose the simpler implementation that preserves trust and the Product Principles.