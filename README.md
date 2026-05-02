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
