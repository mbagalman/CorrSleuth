# CorrSleuth

Correlation is not one number.

CorrSleuth profiles numeric pairwise relationships in pandas DataFrames by comparing multiple association measures and translating their agreement or disagreement into practical diagnostics.

Most tools give you a correlation matrix. CorrSleuth tells you where the correlation matrix may be misleading.

CorrSleuth is diagnostic, not causal. It identifies evidence consistent with relationship patterns, but it does not prove causation, treatment effects, or model specification certainty.

## Installation

Install the base package for core metrics (Pearson, Spearman, Kendall tau-b):
```bash
pip install corrsleuth
```

Install with `standard` mode to include Distance Correlation and Mutual Information:
```bash
pip install corrsleuth[standard]
```

## Quickstart

```python
import corrsleuth as cs
from corrsleuth.datasets import make_relationship

# Generate a simulated dataset (e.g., a U-shape relationship)
df = make_relationship("u_shape", n=500, noise=0.1, random_state=42)

# Profile the relationship
result = cs.profile_pair(df, "x", "y", mode="standard")

# Print the diagnostic label
print(result.pattern)

# Get a plain-English explanation
print(result.explain())

# Generate a multi-panel diagnostic plot
fig = result.plot()
```

Pass `show=True` to `result.plot()` when working interactively and you want
Matplotlib to display the figure immediately.

Example explanation:

```text
Evidence consistent with a relationship that is not simply increasing or decreasing (e.g., U-shaped or cyclical). Standard linear and rank metrics may understate this relationship. Do not interpret this association causally without proper design or controls.
```

## Canonical Examples

CorrSleuth includes a relationship simulator that generates common patterns.

### 1. Near Linear
```python
df = make_relationship("linear_positive")
result = cs.profile_pair(df, "x", "y")
# Pattern: near_linear
```

### 2. Monotonic Nonlinear
```python
df = make_relationship("monotonic_log")
result = cs.profile_pair(df, "x", "y")
# Pattern: monotonic_nonlinear
```

### 3. Nonmonotonic Dependence
```python
df = make_relationship("u_shape")
result = cs.profile_pair(df, "x", "y", mode="standard")
# Pattern: nonmonotonic_dependence
```

### 4. Outlier Driven
```python
df = make_relationship("outlier_driven")
result = cs.profile_pair(df, "x", "y")
# Pattern: possible_outlier_or_leverage
```

### 5. Independent
```python
df = make_relationship("independent")
result = cs.profile_pair(df, "x", "y")
# Pattern: weak_or_no_relationship
```

## How It Works

CorrSleuth takes a fundamentally different approach to bivariate analysis. Instead of relying on a single metric, it computes multiple complementary association measures and compares them:

1. **Pearson**: Captures linear correlation.
2. **Spearman**: Captures monotonic (rank-based) correlation.
3. **Kendall tau-b**: A robust rank correlation that handles ties well.
4. **Distance Correlation** *(Standard mode)*: Captures non-linear dependencies.
5. **Mutual Information** *(Standard mode)*: Captures arbitrary statistical dependence.

By examining where these metrics **agree or disagree**, CorrSleuth assigns a heuristic diagnostic label (e.g., `monotonic_nonlinear` if Spearman is high but Pearson is low, or `nonmonotonic_dependence` if Distance Correlation is high but Spearman is low).

## Scope

CorrSleuth v0.1 focuses on one workflow: numeric-vs-numeric pairwise profiling.

In scope:
- Profiling one numeric pair with `profile_pair()`.
- Lite metrics: Pearson, Spearman, and Kendall tau-b.
- Standard metrics: Distance Correlation and Mutual Information.
- Heuristic diagnostic labels, warnings, recommendations, and diagnostic plots.
- Deterministic simulated relationships through `make_relationship()`.

Out of scope for v0.1:
- Categorical or mixed-type variables.
- Full correlation matrices or target scans.
- HTML reports.
- Scikit-learn transformers or automated model fitting.
- Causal inference.

## Missing Data and Warnings

`profile_pair()` supports three missing-data modes:

- `missing="pairwise"` drops rows missing either selected variable.
- `missing="listwise"` currently behaves the same as `pairwise` for the selected pair.
- `missing="raise"` raises an error if either selected variable contains missing values.

Validation warnings are exposed through `result.warnings`. CorrSleuth warns about small samples, high missingness, low unique-value ratios, constant inputs, downsampling, and conflicting directional evidence when applicable.

## Standard Mode

`mode="standard"` adds Distance Correlation and Mutual Information. It requires the optional dependencies installed by:

```bash
pip install corrsleuth[standard]
```

If those dependencies are not available, CorrSleuth raises `OptionalDependencyError` instead of silently skipping metrics.

For Distance Correlation, CorrSleuth downsamples to 20,000 rows by default when `n_used` is larger than that cap and records a warning. Use `max_n_for_dcor=None` to disable this cap. The downsample is seeded by `random_state` (default `42`), so repeated runs on the same input return the same number.

If you call `mode="standard"` without installing the extras, CorrSleuth raises `OptionalDependencyError` with install instructions rather than silently skipping metrics.

## API Reference

### `profile_pair()`
The main entry point for profiling a numeric pair.

```python
def profile_pair(
    data: pd.DataFrame,
    x: str,
    y: str,
    mode: str = "lite",                 # "lite" or "standard"
    missing: str = "pairwise",          # "pairwise", "listwise", or "raise"
    include_caveat: bool = True,        # Includes causal caveats in explanations
    max_n_for_dcor: int | None = 20000, # Downsampling cap for Distance Correlation
    random_state: int = 42,             # Seed for downsampling and MI estimator
    bootstrap: int | None = None,       # Optional bootstrap interval count
    bootstrap_metrics: str = "lite",    # "lite", "standard", or metric names
    max_n_for_bootstrap: int | None = 5000,
) -> CorrSleuthResult
```

Set `bootstrap=500` to compute approximate 95% bootstrap intervals for
Pearson, Spearman, and Kendall tau-b. Bootstrap intervals are disabled by
default. Even in `mode="standard"`, bootstrap uses lite metrics unless you
explicitly pass `bootstrap_metrics="standard"`, because distance correlation
and mutual information can be expensive to resample.

### `CorrSleuthResult`
The object returned by `profile_pair()`.
- `.pattern`: The assigned heuristic label (e.g., `"near_linear"`).
- `.summary()`: Returns a string summary of the metrics, label, warnings, recommendations, and caveat.
- `.explain()`: Returns a plain-English narrative interpreting the results.
- `.plot(show=False)`: Generates a 1x3 Matplotlib diagnostic figure.
- `.bootstrap_intervals`: Optional bootstrap interval table when requested.
- `.to_dict()` / `.to_frame()`: Serializes the output for downstream pipelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
