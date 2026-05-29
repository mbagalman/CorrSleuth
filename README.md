# CorrSleuth

Correlation is not one number.

CorrSleuth profiles numeric pairwise relationships in pandas DataFrames by comparing multiple association measures and translating their agreement or disagreement into practical diagnostics.

Most tools give you a correlation matrix. CorrSleuth tells you where the correlation matrix may be misleading.

CorrSleuth is diagnostic, not causal. It identifies evidence consistent with relationship patterns, but it does not prove causation, treatment effects, or model specification certainty.

For a guide to what each diagnostic label means, when it can mislead, and what to do next, see the [interpretation guide](https://github.com/mbagalman/CorrSleuth/blob/main/docs/interpretation-guide.md).

## Installation

Install the base package for core metrics (Pearson, Spearman, Kendall tau-b):
```bash
pip install corrsleuth
```

Install with `standard` mode to include Distance Correlation and Mutual Information:
```bash
pip install corrsleuth[standard]
```

`mode="deep"` uses the base installation and adds lightweight robust
correlation diagnostics; it does not require extra dependencies.

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
6. **Robust Pearson-family diagnostics** *(Deep mode)*: Trimmed Pearson, winsorized Pearson, biweight midcorrelation, and median-clipped Pearson help check whether Pearson appears sensitive to extreme values.
7. **Chatterjee's ξ** *(Deep mode)*: An asymmetric coefficient of association — `ξ(X → Y)` measures whether `Y` is a (noisy) function of `X`. Catches U-shape and other functional dependencies that Pearson and Spearman miss.

By examining where these metrics **agree or disagree**, CorrSleuth assigns a heuristic diagnostic label (e.g., `monotonic_nonlinear` if Spearman is high but Pearson is low, or `nonmonotonic_dependence` if Distance Correlation is high but Spearman is low).

## Scope

CorrSleuth focuses on numeric pairwise profiling and target-oriented scans.

In scope:
- Profiling one numeric pair with `profile_pair()`.
- Scanning every numeric column against a single target with `scan_target()`.
- Lite metrics: Pearson, Spearman, and Kendall tau-b.
- Standard metrics: Distance Correlation and Mutual Information.
- Deep metrics: lightweight robust correlation diagnostics.
- Heuristic diagnostic labels, warnings, recommendations, and diagnostic plots.
- Deterministic simulated relationships through `make_relationship()`.

Out of scope for now:
- Categorical or mixed-type variables.
- Full correlation matrices.
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

## Deep Mode

`mode="deep"` adds robust correlation diagnostics while keeping the base
installation lightweight:

- `pearson_trimmed_1pct`: Pearson after dropping rows outside the 1st/99th percentile range of either variable.
- `pearson_winsorized_1pct`: Pearson after clipping both variables at their 1st/99th percentiles.
- `biweight_midcorrelation`: A median/MAD-based robust correlation.
- `pearson_median_clipped_20pct`: Pearson after clipping deviations around each median at the 80th percentile.
- `chatterjee_xi`: Chatterjee's coefficient of correlation, an *asymmetric* measure that captures whether `Y` is a (noisy) function of `X`. Range is approximately `[0, 1]`; values near 1 indicate a strong functional relationship. Detects U-shape and other dependencies that Pearson and Spearman miss.
- `chatterjee_xi_reverse`: Same statistic in the opposite direction (`ξ(Y → X)`). For target scans this is the candidate→target direction, which is usually the one feature-engineering users want.

These are robustness and dependence diagnostics, not definitive replacements
for Pearson or visual inspection. The robust correlations are most useful when
Pearson is strong but rank metrics or plots suggest leverage-sensitive
behavior. Chatterjee's ξ is most useful for surfacing functional dependence
that the linear/rank pair miss. Deep mode does not compute Distance
Correlation or Mutual Information; use `mode="standard"` for those nonlinear
dependence metrics. `metric_pearson_trimmed_1pct` and
`result.diagnostics.pearson_trimmed` use the same 1% trimmed-Pearson
sensitivity calculation so users see one consistent trim value.

`chatterjee_xi` is reported as `ξ(pair.x → pair.y)`, so for a profile
called as `profile_pair(df, "x", "y")` the value answers "is `y` a function
of `x`?". `chatterjee_xi_reverse` reports the same statistic with the
arguments swapped (`ξ(pair.y → pair.x)`), so target scans get both the
target→candidate direction (`chatterjee_xi`) and the candidate→target
direction (`chatterjee_xi_reverse`) without an extra call. The metric
converges slowly on small samples and returns `None` with a warning when
`n_used < 20`. The value is invariant to the row order of the underlying
DataFrame: ties on `X` are broken lexicographically by `Y` so the result
depends only on the multiset of `(x, y)` pairs. See
[docs/phase4-nonlinear-metrics-design-note.md](https://github.com/mbagalman/CorrSleuth/blob/main/docs/phase4-nonlinear-metrics-design-note.md)
for the rationale and the candidates that were considered and deferred.

## API Reference

### `profile_pair()`
The main entry point for profiling a numeric pair.

```python
def profile_pair(
    data: pd.DataFrame,
    x: str,
    y: str,
    mode: str = "lite",                 # "lite", "standard", or "deep"
    missing: str = "pairwise",          # "pairwise", "listwise", or "raise"
    include_caveat: bool = True,        # Includes causal caveats in explanations
    max_n_for_dcor: int | None = 20000, # Downsampling cap for Distance Correlation
    random_state: int = 42,             # Seed for downsampling and MI estimator
    bootstrap: int | None = None,       # Optional bootstrap interval count
    bootstrap_metrics: str = "lite",    # "lite", "standard", or metric names
    max_n_for_bootstrap: int | None = 5000,
) -> CorrSleuthResult
```

Set `bootstrap=200` to compute approximate 95% bootstrap intervals and pattern
stability for Pearson, Spearman, and Kendall tau-b. Bootstrap diagnostics are
disabled by default. Even in `mode="standard"`, bootstrap uses lite metrics
unless you explicitly pass `bootstrap_metrics="standard"`, because distance
correlation and mutual information can be expensive to resample. Standard
bootstrap metrics require the `[standard]` extras even when the main
`profile_pair()` call uses `mode="lite"`. Higher bootstrap counts and standard
metrics can take many seconds on larger datasets, especially with distance
correlation.

### `CorrSleuthResult`
The object returned by `profile_pair()`.
- `.pattern`: The assigned heuristic label (e.g., `"near_linear"`).
- `.summary()`: Returns a string summary of the metrics, label, warnings, recommendations, and caveat.
- `.explain()`: Returns a plain-English narrative interpreting the results.
- `.plot(show=False)`: Generates a 1x3 Matplotlib diagnostic figure.
- `.bootstrap_intervals`: Optional bootstrap interval table when requested.
- `.pattern_stability`: Optional share of bootstrap samples with the same label.
- `.bootstrap_label_counts`: Optional diagnostic label counts from bootstrap samples.
- `.stability_label`: Optional `"low"`, `"medium"`, or `"high"` stability label.
- `.to_markdown(include_caveat=None)`: Exports a compact Markdown report with metrics, diagnostics, warnings, recommendations, optional bootstrap sections, and the non-causal caveat by default.
- `.to_dict()` / `.to_frame()`: Serializes the output for downstream pipelines.

### `scan_target()`
Profile every eligible numeric predictor against a single numeric target column.

```python
def scan_target(
    data: pd.DataFrame,
    target: str,
    *,
    columns: Sequence[str] | None = None, # Restrict scan to these columns
    mode: str = "lite",                   # Forwarded to profile_pair
    missing: str = "pairwise",            # Forwarded to profile_pair
    errors: str = "warn",                 # "warn" captures per-column failures, "raise" propagates
    max_pairs: int | None = None,         # Cap on columns profiled
    sample_size: int | None = None,       # Optional one-time row downsample
    progress: bool = False,               # Use tqdm if installed; documented no-op otherwise
    random_state: int = 42,
    **profile_pair_kwargs,                # e.g. bootstrap=, include_caveat=
) -> CorrSleuthTargetReport
```

Quick example:

```python
report = cs.scan_target(df, target="sales")
print(report.summary())
report.to_frame()  # one row per profiled or skipped column
```

Non-numeric or missing columns listed in `columns=` are recorded as `skipped` entries with `error_type` and `error_message` rather than aborting the scan. With `errors="warn"` (default), exceptions raised by `profile_pair()` are captured as `error` entries. Use `errors="raise"` to fail fast.

### `CorrSleuthTargetReport`
The object returned by `scan_target()`.
- `.target`: Name of the target column.
- `.entries`: List of `TargetScanEntry` objects, one per inspected column.
- `.successes` / `.failures`: Convenience splits.
- `.summary(top_n=5, include_caveat=True)`: Section-structured text overview. Pattern sections (`Strongest near-linear relationships`, `Potential monotonic nonlinear relationships`, `Potential nonmonotonic relationships`, `Possible outlier-driven relationships`, `Weak or no pairwise relationships`) are emitted only when populated, each capped at `top_n`. A cross-cutting `Variables Pearson may underrate` section lists variables whose `rank_linear_gap` or `nonmonotonic_gap` exceeds 0.20. `Variables with missingness or tie warnings` surfaces columns whose validation warnings mention ties, missingness, low unique ratios, small samples, or constant inputs. `Skipped or failed` lists non-numeric / errored columns. The output is deterministic; entries within each section are sorted by `disagreement_score` descending.
- `.to_frame()`: One row per inspected column with variable, target, status, pattern, disagreement score, warnings, recommendations, and per-metric value columns. Skipped or errored rows leave metric columns NaN and populate `error_type` / `error_message`.
- `.to_markdown(top_n=5, include_caveat=True)`: Exports the grouped target report as Markdown with overview counts, populated pattern sections, cross-cutting Pearson-underrated and reliability-warning sections, skipped/failed rows, and the non-causal caveat by default.
- `.pearson_underrated(threshold=0.20)`: Returns a DataFrame of variables where Pearson may understate the relationship. The ranking is directional: Spearman, Kendall, or standard-mode nonmonotonic evidence must exceed Pearson by more than `threshold`, so outlier/leverage cases where Pearson is stronger than rank metrics are excluded. Rows include metric values, explicit excess-over-Pearson gap values, raw `nonmonotonic_gap`, pattern, disagreement score, and warnings, sorted by strongest evidence.
- `.plot_top(n=12, sort_by="disagreement_score", patterns=None, ncols=3, figsize=None, show=False)`: Compact gallery of the top-`n` target relationships as a Matplotlib `Figure`. `sort_by` accepts `"disagreement_score"` (raw value descending) or a metric name (`pearson`, `spearman`, `kendall_tau_b`, `distance_correlation`, `mutual_information`, or a deep-mode robust metric, `chatterjee_xi`, or `chatterjee_xi_reverse`; sorted by absolute value descending). `patterns=` filters to specific diagnostic labels and accepts either a single string or an iterable. Each panel is a scatter of the candidate column versus the target, titled with the column name, pattern, and key metric values. When fewer than `n` variables match, the unused grid slots are hidden; an empty filter yields a placeholder figure rather than raising.

## Documentation

- [Interpretation Guide](https://github.com/mbagalman/CorrSleuth/blob/main/docs/interpretation-guide.md) — meaning, typical metric pattern, common examples, recommended next steps, and caveats for every diagnostic label, plus topic notes on Pearson misleads, monotonicity, leverage, ties, and performance modes.
- [Phase 4 Nonlinear Metrics Design Note](https://github.com/mbagalman/CorrSleuth/blob/main/docs/phase4-nonlinear-metrics-design-note.md) — why Chatterjee's ξ was chosen over HSIC, MGC, MIC, and Hoeffding's D for `mode="deep"`.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/mbagalman/CorrSleuth/blob/main/LICENSE) file for details.
