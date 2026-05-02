# CorrSleuth

Correlation is not one number.

CorrSleuth profiles numeric pairwise relationships in pandas DataFrames by comparing multiple association measures and translating their agreement or disagreement into practical diagnostics.

Most tools give you a correlation matrix. CorrSleuth tells you where the correlation matrix may be misleading.

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
fig = result.plot(show=True)
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

## API Reference

### `profile_pair()`
The main entry point for profiling a numeric pair.

```python
def profile_pair(
    data: pd.DataFrame,
    x: str,
    y: str,
    mode: str = "lite",           # "lite" or "standard"
    missing: str = "pairwise",    # "pairwise", "listwise", or "raise"
    include_caveat: bool = True,  # Includes causal caveats in explanations
    max_n_for_dcor: int = 20000   # Downsampling cap for Distance Correlation
) -> CorrSleuthResult
```

### `CorrSleuthResult`
The object returned by `profile_pair()`.
- `.pattern`: The assigned heuristic label (e.g., `"near_linear"`).
- `.summary()`: Returns a printed tabular summary of the metrics.
- `.explain()`: Returns a plain-English narrative interpreting the results.
- `.plot(show=False)`: Generates a 1x3 Matplotlib diagnostic figure.
- `.to_dict()` / `.to_frame()`: Serializes the output for downstream pipelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
