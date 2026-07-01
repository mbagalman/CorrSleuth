# CorrSleuth

> Correlation is not one number.

**CorrSleuth is a relationship diagnosis engine for pandas.** Point it at two
numeric columns and it measures their relationship several different ways,
compares the results, and tells you in plain English what kind of relationship
it is — and where a plain correlation could mislead you.

## Why CorrSleuth?

A single correlation number (like Pearson's *r*) squeezes a rich relationship
into one value, and that value can lie. Two columns can have *r* ≈ 0 while being
tightly linked in a U-shape, or show a high *r* that is really driven by a
couple of outliers. The number alone never tells you which situation you are in.

CorrSleuth runs several complementary measures, looks at where they **agree or
disagree**, and turns that into an actionable diagnosis:

- **See what a correlation matrix hides.** Catches nonlinear, non-monotonic, and
  outlier-driven relationships that a single coefficient misses.
- **Get answers in plain English.** Every result comes with a diagnostic label,
  a written explanation, and a recommended next step — not just numbers.
- **Spot leverage and outlier traps.** Flags when a strong-looking correlation is
  actually driven by a few extreme points.
- **Scan a whole dataset against a target.** Rank every numeric predictor against
  an outcome column in one call.
- **Stay honest about uncertainty.** Built-in warnings for small samples, ties,
  missingness, and conflicting evidence, plus optional bootstrap stability.
- **Keep installs light.** The base install needs only pandas, numpy, scipy,
  and matplotlib (for the diagnostic plots); the heavier nonlinear metrics
  (Distance Correlation, Mutual Information) are opt-in extras.

CorrSleuth is **diagnostic, not causal**. It identifies evidence consistent with
relationship patterns, but it does not prove causation, treatment effects, or
model specification certainty.

For a guide to what each diagnostic label means, when it can mislead, and what to
do next, see the [interpretation guide](https://github.com/mbagalman/CorrSleuth/blob/main/docs/interpretation-guide.md).
For how it works under the hood — the measures, how they're compared, and the
label logic — see the [methodology doc](https://github.com/mbagalman/CorrSleuth/blob/main/docs/methodology.md).

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

For a progress bar during `scan_target(progress=True)`, install the optional
`progress` extra:
```bash
pip install corrsleuth[progress]
```

For the optional LOWESS trend overlay on the diagnostic scatter plots, install
the `plot` extra (the plot still renders without it, just without the smoother):
```bash
pip install corrsleuth[plot]
```

## Quick Start

Point CorrSleuth at two numeric columns of your own DataFrame and read the
diagnosis:

```python
import corrsleuth as cs

# `df` is your pandas DataFrame; replace the column names with your own.
result = cs.profile_pair(df, "ad_spend", "revenue")

print(result.pattern)    # the diagnostic label, e.g. "near_linear"
print(result.explain())  # a plain-English interpretation
print(result.summary())  # metrics, diagnostics, warnings, and recommendations

result.plot(show=True)    # a 1x3 diagnostic figure (scatter, ranks, summary)
```

`profile_pair()` works on the base install. Pass `mode="standard"` to add
nonlinear metrics (Distance Correlation, Mutual Information; requires the
`[standard]` extra) or `mode="deep"` for robust diagnostics that need no extra
dependencies.

**No dataset handy?** CorrSleuth ships a simulator so you can try it immediately:

```python
from corrsleuth.datasets import make_relationship

# A U-shaped relationship: Pearson is near zero, but the variables are related.
df = make_relationship("u_shape", n=500, noise=0.1, random_state=42)

result = cs.profile_pair(df, "x", "y", mode="standard")
print(result.pattern)    # nonmonotonic_dependence
print(result.explain())
```

```text
Evidence consistent with a relationship that is not simply increasing or decreasing (e.g., U-shaped or cyclical). Standard linear and rank metrics may understate this relationship. Do not interpret this association causally without proper design or controls.
```

**Screening many predictors against one outcome?** Use `scan_target()`:

```python
# Build a frame with a few predictors and one target column.
data = make_relationship("linear_positive", n=300, random_state=0).rename(
    columns={"x": "ad_spend", "y": "revenue"}
)
data["noise"] = make_relationship("independent", n=300, random_state=1)["y"]

report = cs.scan_target(data, target="revenue")
print(report.summary())          # grouped, ranked overview of every predictor
report.to_frame()                # one tidy row per profiled column
```

That is enough to get productive. The sections below explain the metrics, modes,
diagnostic labels, and the full API.

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
5. **Mutual Information** *(Standard mode)*: Captures arbitrary statistical dependence. Reported as raw, unnormalized MI (in nats) — it is `>= 0` and unbounded, **not** on a 0–1 scale, so read it relatively rather than like a correlation coefficient.
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
- `missing="listwise"` drops rows missing a value in *any* column of `data`
  (complete-case deletion) before selecting the pair, so it coincides with
  `pairwise` only when `data` contains just the two profiled columns. In a
  `scan_target()` run this profiles every column on the same complete-case rows.
- `missing="raise"` raises an error if either selected variable contains missing values.

Validation warnings are exposed through `result.warnings`. CorrSleuth warns about small samples, high missingness, low unique-value ratios, constant inputs, downsampling, conflicting directional evidence, and (in deep mode) high Chatterjee's ξ alongside a weak or ambiguous label, when applicable.

## Standard Mode

`mode="standard"` adds Distance Correlation and Mutual Information. It requires the optional dependencies installed by:

```bash
pip install corrsleuth[standard]
```

If those dependencies are not available, CorrSleuth raises `OptionalDependencyError` (importable as `from corrsleuth import OptionalDependencyError`) instead of silently skipping metrics.

For Distance Correlation, CorrSleuth downsamples to 20,000 rows by default when `n_used` is larger than that cap and records a warning. Use `max_n_for_dcor=None` to disable this cap. The downsample is seeded by `random_state` (default `42`), so repeated runs on the same input return the same number.

## Deep Mode

`mode="deep"` adds robust correlation diagnostics while keeping the base
installation lightweight:

- `pearson_trimmed_1pct`: Pearson after dropping rows outside the 1st/99th percentile range of either variable.
- `pearson_winsorized_1pct`: Pearson after clipping both variables at their 1st/99th percentiles.
- `biweight_midcorrelation`: A median/MAD-based robust correlation.
- `pearson_median_clipped_20pct`: Pearson after clipping deviations around each median at the 80th percentile.
- `chatterjee_xi`: Chatterjee's coefficient of correlation, an *asymmetric* measure that captures whether `Y` is a (noisy) function of `X`. Values sit near 0 under independence and approach 1 for strong functional dependence; finite-sample estimates can be slightly negative. Detects U-shape and other dependencies that Pearson and Spearman miss.
- `chatterjee_xi_reverse`: Same statistic in the opposite direction (`ξ(Y → X)`). For target scans this is the candidate→target direction, which is usually the one feature-engineering users want.

These are robustness and dependence diagnostics, not definitive replacements
for Pearson or visual inspection. The robust correlations are most useful when
Pearson is strong but rank metrics or plots suggest leverage-sensitive
behavior. Chatterjee's ξ is most useful for surfacing functional dependence
that the linear/rank pair miss. Deep mode does not compute Distance
Correlation or Mutual Information; use `mode="standard"` for those nonlinear
dependence metrics. The label cascade does not consult ξ, so a strongly
nonmonotonic pair keeps its lite-style label in deep mode — but when ξ
exceeds 0.35 and the label is `weak_or_no_relationship` or
`mixed_or_ambiguous`, CorrSleuth emits a warning so the dependence is not
silently contradicted by the label. `metric_pearson_trimmed_1pct` and
`result.diagnostics.pearson_trimmed` use the same 1% trimmed-Pearson
sensitivity calculation so users see one consistent trim value.

`chatterjee_xi` is reported as `ξ(pair.x → pair.y)`, so for a profile
called as `profile_pair(df, "x", "y")` the value answers "is `y` a function
of `x`?". `chatterjee_xi_reverse` reports the same statistic with the
arguments swapped (`ξ(pair.y → pair.x)`), so target scans get both the
target→candidate direction (`chatterjee_xi`) and the candidate→target
direction (`chatterjee_xi_reverse`) without an extra call. The metric
converges slowly on small samples and returns `None` with a warning when
`n_used < 20`. It uses the tie-corrected estimator from Chatterjee (2020), so it
stays well-calibrated when `Y` is discrete or low-cardinality. Ties in the sort
variable are broken with a seeded random permutation (Chatterjee's prescription;
breaking them by `Y` would leak the response and inflate ξ), so values are
reproducible for a given input and `random_state` but, when the sort variable
has ties, depend on that tie-break rather than being invariant to row order. See
[docs/nonlinear-metrics-design.md](https://github.com/mbagalman/CorrSleuth/blob/main/docs/nonlinear-metrics-design.md)
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
    random_state: int = 42,             # Seed: dcor downsampling, MI, bootstrap resampling, xi tie-breaking
    bootstrap: int | None = None,       # Optional bootstrap interval count
    bootstrap_metrics: str | Sequence[str] = "lite",  # "lite", "standard", or metric names
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

Two conservative guards mean these fields can be omitted (left `None`) on small
or heavily capped samples: **intervals** are not reported when the effective
per-replicate size is below 20 (too few rows for a reliable percentile
bootstrap), and **pattern stability** is not reported when `max_n_for_bootstrap`
caps replicates below 30 on a larger original sample (every replicate would be
judged low-power, making stability meaningless against the full-sample label). A
warning explains each case.

### `CorrSleuthResult`
The object returned by `profile_pair()`.
- `.pattern`: The assigned heuristic label (e.g., `"near_linear"`).
- `.summary()`: Returns a string summary of the metrics, label, warnings, recommendations, and caveat.
- `.explain()`: Returns a plain-English narrative interpreting the results.
- `.plot(show=False)`: Generates a 1x3 Matplotlib diagnostic figure.
- `.bootstrap_intervals`: Optional bootstrap interval table when requested (`None` when intervals are suppressed for too-small effective replicates; see above).
- `.pattern_stability`: Optional share of bootstrap samples with the same label (`None` when stability is suppressed for cap-induced low-power replicates; see above).
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
- `.summary(top_n=5, include_caveat=True)`: Section-structured text overview. Pattern sections (`Strongest near-linear relationships`, `Potential monotonic nonlinear relationships`, `Potential nonmonotonic relationships`, `Possible outlier-driven relationships`, `Weak or no pairwise relationships`) are emitted only when populated, each capped at `top_n`. Variables whose pattern falls outside that set (e.g. `low_power_or_uncertain`, `mixed_or_ambiguous`) appear in an `Other or inconclusive` section so no profiled variable disappears from the summary. A cross-cutting `Variables Pearson may underrate` section lists variables where rank (Spearman/Kendall) or standard-mode nonmonotonic evidence exceeds Pearson by more than 0.20; the gap is directional, so leverage cases where Pearson is stronger than the rank metrics are excluded. `Variables with missingness or tie warnings` surfaces columns whose validation warnings mention ties, missingness, low unique ratios, small samples, or constant inputs. `Skipped or failed` lists non-numeric / errored columns. The output is deterministic; entries within each section are sorted by `disagreement_score` descending.
- `.to_frame()`: One row per inspected column with variable, target, status, pattern, disagreement score, warnings, recommendations, and per-metric value columns. Skipped or errored rows leave metric columns NaN and populate `error_type` / `error_message`.
- `.to_markdown(top_n=5, include_caveat=True)`: Exports the grouped target report as Markdown with overview counts, populated pattern sections, cross-cutting Pearson-underrated and reliability-warning sections, skipped/failed rows, and the non-causal caveat by default.
- `.pearson_underrated(threshold=0.20)`: Returns a DataFrame of variables where Pearson may understate the relationship. The ranking is directional: Spearman, Kendall, or standard-mode nonmonotonic evidence must exceed Pearson by more than `threshold`, so outlier/leverage cases where Pearson is stronger than rank metrics are excluded. Rows include metric values, explicit excess-over-Pearson gap values, raw `nonmonotonic_gap`, pattern, disagreement score, and warnings, sorted by strongest evidence.
- `.plot_top(n=12, sort_by="disagreement_score", patterns=None, ncols=3, figsize=None, show=False)`: Compact gallery of the top-`n` target relationships as a Matplotlib `Figure`. `sort_by` accepts `"disagreement_score"` (raw value descending) or a metric name (`pearson`, `spearman`, `kendall_tau_b`, `distance_correlation`, `mutual_information`, or a deep-mode robust metric, `chatterjee_xi`, or `chatterjee_xi_reverse`; sorted by absolute value descending). `patterns=` filters to specific diagnostic labels and accepts either a single string or an iterable. Each panel is a scatter of the candidate column versus the target, titled with the column name, pattern, and key metric values. When fewer than `n` variables match, the unused grid slots are hidden; an empty filter yields a placeholder figure rather than raising.

## Documentation

- [Methodology](https://github.com/mbagalman/CorrSleuth/blob/main/docs/methodology.md) — for statisticians and data scientists: the pipeline, what each association measure detects and assumes, how the measures are compared (the `disagreement_score`), the label cascade, the bootstrap-stability approach, reproducibility, and limitations.
- [Interpretation Guide](https://github.com/mbagalman/CorrSleuth/blob/main/docs/interpretation-guide.md) — meaning, typical metric pattern, common examples, recommended next steps, and caveats for every diagnostic label, plus topic notes on Pearson misleads, monotonicity, leverage, ties, and performance modes.
- [Thresholds and Rationale](https://github.com/mbagalman/CorrSleuth/blob/main/docs/thresholds-and-rationale.md) — every cut point that drives a label or warning, with its value, location, justification, and how to override the label-driving ones.
- [Nonlinear Metrics Design Note](https://github.com/mbagalman/CorrSleuth/blob/main/docs/nonlinear-metrics-design.md) — why Chatterjee's ξ was chosen over HSIC, MGC, MIC, and Hoeffding's D for `mode="deep"`.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/mbagalman/CorrSleuth/blob/main/LICENSE) file for details.
