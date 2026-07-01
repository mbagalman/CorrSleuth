# CorrSleuth Interpretation Guide

This guide is for analysts using CorrSleuth in everyday EDA. It explains what
each diagnostic label means, what typical metric pattern produces it, what to
do next, and where the labels can mislead you. CorrSleuth's labels are
**heuristic diagnostics**, not statistical truth claims — read this guide as a
field manual, not as theory.

For the package's API surface, see the [README](../README.md). For the
research that motivates particular metric choices, see
[nonlinear-metrics-design.md](nonlinear-metrics-design.md).

## Contents

- [What CorrSleuth Does and Does Not Do](#what-corrsleuth-does-and-does-not-do)
- [How a Label Is Assigned](#how-a-label-is-assigned)
- [Diagnostic Labels](#diagnostic-labels)
  - [`near_linear`](#near_linear)
  - [`monotonic_nonlinear`](#monotonic_nonlinear)
  - [`nonmonotonic_dependence`](#nonmonotonic_dependence)
  - [`possible_outlier_or_leverage`](#possible_outlier_or_leverage)
  - [`weak_or_no_relationship`](#weak_or_no_relationship)
  - [`low_power_or_uncertain`](#low_power_or_uncertain)
  - [`not_computable`](#not_computable)
  - [`mixed_or_ambiguous`](#mixed_or_ambiguous)
- [Topics](#topics)
  - [When Pearson Can Be Misleading](#when-pearson-can-be-misleading)
  - [Monotonic vs Nonmonotonic Relationships](#monotonic-vs-nonmonotonic-relationships)
  - [Outlier-Sensitive Correlations](#outlier-sensitive-correlations)
  - [Missing Data and Ties](#missing-data-and-ties)
  - [Performance Modes](#performance-modes)
- [Further Reading](#further-reading)

## What CorrSleuth Does and Does Not Do

**CorrSleuth does:**

- Profile a single numeric pair (`profile_pair`) or every numeric column
  against a target (`scan_target`).
- Compute several complementary association measures — Pearson, Spearman,
  Kendall tau-b, and (in deeper modes) distance correlation, mutual
  information, robust Pearson variants, and Chatterjee's ξ.
- Compare those measures against one another to detect agreement,
  disagreement, and patterns that a single coefficient would hide.
- Surface a heuristic diagnostic label, plain-English explanation,
  warnings, recommendations, and an optional sampling-uncertainty signal
  (`bootstrap=…`).

**CorrSleuth does not:**

- Make causal claims. Every label is paired with a non-causal caveat.
- Replace visual inspection. Every label suggests inspecting the diagnostic
  plot before acting.
- Handle categorical or mixed-type data. The current scope is
  numeric-vs-numeric only.
- Build full models. CorrSleuth is a diagnostic step *before* model
  selection, not a substitute for it.
- Guarantee that a strong association implies predictive usefulness in a
  multivariate model.

The most useful frame: CorrSleuth tells you where a default
correlation matrix is most likely to mislead you, so you can decide which
pairs deserve closer inspection.

## How a Label Is Assigned

CorrSleuth uses a fixed-priority cascade. Higher-priority rules short-circuit
later ones, so each pair gets exactly one label. Comparisons mostly use the
**absolute** Pearson, Spearman, and Kendall values, so the overall direction of
the association (positive vs negative) does not by itself change the label. The
one exception is a **sign conflict** between Pearson and Spearman (opposite
signs, both at least moderate): that is a leverage signature, so it is routed to
`possible_outlier_or_leverage` (or `mixed_or_ambiguous`) and kept out of
`near_linear`/`monotonic_nonlinear`. The current order is:

1. `not_computable` — when one variable is constant or core metrics fail to
   compute.
2. `low_power_or_uncertain` — when `n_used < 30`.
3. `possible_outlier_or_leverage` — when Pearson is strong, materially
   stronger than the rank metrics, and a trimmed-Pearson sensitivity check
   says Pearson is leverage-driven (or sensitivity could not be computed).
4. `nonmonotonic_dependence` — when distance correlation is high while
   Pearson and Spearman are weak (requires `mode="standard"`).
5. `monotonic_nonlinear` — when Spearman is meaningfully stronger than
   Pearson.
6. `near_linear` — when Pearson and Spearman are both strong and close.
7. `weak_or_no_relationship` — when all available metrics are small.
8. `mixed_or_ambiguous` — fallback when none of the above applies.

The exact thresholds live in
[`corrsleuth/heuristics/classifier.py`](../corrsleuth/heuristics/classifier.py)
as documented module-level constants, and every cut point in the package —
what it gates, its value, and why — is catalogued in
[thresholds-and-rationale.md](thresholds-and-rationale.md). They are
intentionally conservative — a borderline case usually drops to
`mixed_or_ambiguous` rather than overclaiming a pattern.

## Diagnostic Labels

Each label below documents:

- **Meaning** — what the label asserts about the relationship.
- **Typical metric pattern** — the metric values that usually produce it.
- **Common examples** — relationship shapes in the wild that fit.
- **Recommended next steps** — what to do when you see it.
- **Caveats** — known ways the label can be wrong or misleading.

### `near_linear`

**Meaning.** Evidence consistent with an approximately linear association.
Both variables scale together smoothly, and Pearson is a defensible summary.

**Typical metric pattern.** `|pearson| > 0.50`, `|spearman| > 0.50`, and the
absolute gap between Pearson and Spearman is below `0.15`. If
`mode="standard"`, distance correlation is usually similar in magnitude.

**Common examples.**

- Two columns measured in the same units (height in inches and centimeters).
- Sales versus advertising spend in a regime where the response is
  approximately additive.
- Output of two well-calibrated sensors measuring the same physical quantity.

**Recommended next steps.**

- A standard linear model or Pearson correlation is likely appropriate.
- Inspect the scatter plot once to confirm there are no clusters or
  segmentation that linearity would average over.

**Caveats.**

- "Approximately linear" doesn't rule out heteroskedasticity (variance that
  changes with `x`). Pearson's value is fine; standard-error-based
  inferences may still need care.
- A near-linear pair can still be **spurious** — confounders, selection
  effects, or shared trends can produce strong linear association without
  any direct relationship. The non-causal caveat applies.
- Strong near-linear associations between predictors hint at
  multicollinearity if you plan to put both into a regression.

### `monotonic_nonlinear`

**Meaning.** Evidence consistent with a directional relationship that is
not well summarized by a straight line. Spearman captures the monotonic
trend that Pearson is missing.

**Typical metric pattern.** `|spearman| > 0.50` and `|spearman| - |pearson|
> 0.20`. Kendall tau-b usually agrees with Spearman.

**Common examples.**

- Logarithmic responses (e.g., `y = log(x) + noise` for positive `x`).
- Saturating responses (e.g., diminishing returns, dose-response curves).
- Power laws (`y = x^k` for `k != 1`).
- Income vs spending — often monotonic but not linear.

**Recommended next steps.**

- Inspect the scatter plot for curvature.
- Consider monotonic transformations (log, sqrt, rank) before linear
  modeling, or use models that fit nonlinearity directly (splines,
  isotonic regression, gradient-boosted trees).
- For feature engineering, this is one of the patterns where Pearson
  underrates the variable; see `report.pearson_underrated()`.

**Caveats.**

- Pearson can underrate the *strength* of the relationship but the
  *direction* is still informative. A positive Spearman with a near-zero
  Pearson is a strong signal, not a contradiction.
- Heavy ties or coarse discretization can inflate the Spearman-vs-Pearson
  gap without an underlying nonlinear relationship — check the
  `high_tie_rate` warning.

### `nonmonotonic_dependence`

**Meaning.** Evidence consistent with a relationship that is not simply
increasing or decreasing — U-shapes, V-shapes, oscillations. Pearson and
Spearman both miss it, but distance correlation flags it.

**Typical metric pattern.** Available only in `mode="standard"`:
`|pearson| < 0.25`, `|spearman| < 0.25`, `distance_correlation > 0.35`.
In `mode="deep"`, Chatterjee's ξ usually shows the same story
asymmetrically (`ξ(X → Y)` is high; `ξ(Y → X)` may be lower).

**Common examples.**

- `y = x²` on data centered around zero — quadratic relationships.
- Dose-response curves with a peak (e.g., medication response that drops at
  both low and high doses).
- Cyclical patterns (seasonality vs an outcome).

**Recommended next steps.**

- Inspect the scatter plot. The shape is usually obvious by eye.
- Consider polynomial features, splines, or tree-based models that handle
  nonmonotonic shapes natively.
- If you're doing feature selection by `|pearson|`, this variable would be
  silently discarded. `scan_target()` surfaces it explicitly.

**Caveats.**

- Requires `mode="standard"`. In `lite` mode, distance correlation is
  unavailable and the cascade falls through to
  `weak_or_no_relationship` or `mixed_or_ambiguous`.
- Distance correlation is sensitive to extreme values. A handful of
  outliers can produce a high `dcor` for an otherwise weak relationship —
  always check the plot.
- Bootstrap pattern stability for this label uses *lite* metrics by
  default, so the stability number is a lower bound on what
  standard-metric stability would show. If you need a tighter check,
  pass `bootstrap_metrics="standard"`.

### `possible_outlier_or_leverage`

**Meaning.** Pearson appears strong, but the strength may be driven by a
small number of extreme observations rather than the bulk of the data.
Trimmed Pearson confirms (or could not refute) leverage.

**Typical metric pattern.** `|pearson| > 0.50`, `|pearson| - |spearman| >
0.20` or `|pearson| - |kendall| > 0.25`, and the
`pearson_trim_sensitive` (or `outlier_sensitivity_unavailable`) flag is
set on the result.

**Common examples.**

- A near-zero relationship dominated by 1–5 extreme points that pull
  Pearson up.
- Heavy-tailed financial returns where rare events dominate the linear fit.
- Bug-report counts where one viral incident inflates the correlation.

**Recommended next steps.**

- Inspect the scatter plot for isolated extreme points.
- Look at `result.diagnostics.pearson_trim_delta`. A large delta confirms
  the trimmed estimate disagrees with the raw Pearson.
- Consider robust estimators (`mode="deep"` exposes
  `pearson_winsorized_1pct`, `biweight_midcorrelation`, and
  `pearson_median_clipped_20pct`).
- For predictive modeling, the leverage points may be **outliers** worth
  removing or **legitimate signal** worth keeping; CorrSleuth can't tell
  you which without context.

**Caveats.**

- "Outlier" in CorrSleuth means *statistical leverage*, not "wrong data".
  The point may be real and important.
- The label requires the trim-sensitivity check to fire. Below ~50 rows,
  the trim check is skipped (`outlier_sensitivity_unavailable`), and the
  label is assigned more liberally — read it as a hypothesis to verify
  visually.

### `weak_or_no_relationship`

**Meaning.** Little to no evidence of pairwise association under the
metrics that were available.

**Typical metric pattern.** `|pearson| < 0.20`, `|spearman| < 0.20`, and
either `distance_correlation < 0.20` or distance correlation was not
computed.

**Common examples.**

- Truly independent variables.
- A nonlinear relationship that the available metrics couldn't see (most
  often: a nonmonotonic relationship in `lite` mode where distance
  correlation isn't available).
- A relationship that is conditional on a third variable — e.g., income
  and spending are weakly correlated overall but tightly correlated
  *within* age groups (Simpson's paradox).

**Recommended next steps.**

- Treat the variable as a low-priority predictor on its own.
- If the domain suggests a nonlinear or conditional story, re-run with
  `mode="standard"` (or `mode="deep"` for Chatterjee's ξ) before
  discarding.
- Consider whether a third variable is masking the relationship.

**Caveats.**

- "Weak under these metrics" ≠ "no relationship." A nonmonotonic
  relationship in `lite` mode can land here.
- The label cascade does not consult Chatterjee's ξ, so even in
  `mode="deep"` a strongly nonmonotonic pair can land here. When that
  happens (ξ above 0.35 with a weak or ambiguous label), CorrSleuth adds a
  warning to `result.warnings` pointing you to the scatter plot and
  `mode="standard"`.
- A weak pairwise relationship can still be a useful predictor in a
  multivariate model that captures interactions — pairwise scans like
  `scan_target` are not a substitute for cross-validated feature
  importance.

### `low_power_or_uncertain`

**Meaning.** Sample size is too small to confidently describe the
relationship shape — the metric values may not stabilize until more data
arrives.

**Typical metric pattern.** `n_used < 30` after missing-value handling.
Any label downstream of this in the cascade is suppressed.

**Common examples.**

- Pilot studies, small experiments, or analysis on a heavily-filtered
  segment.
- Joins where the overlap between two tables is small.
- Pairwise analysis on a column with high missingness.

**Recommended next steps.**

- Collect more data if possible.
- Lean on domain knowledge instead of the metric value.
- Check the `pattern_stability` from a bootstrap (`bootstrap=200`) — it
  will usually be low, which is itself the headline.

**Caveats.**

- The 30-row threshold is conservative. For some metrics (e.g., Pearson
  on near-Gaussian data) `n=20` may already be informative; for others
  (Kendall under heavy ties) `n=100` may not be enough. The threshold is
  a heuristic floor, not a guarantee of power above it.
- A small sample with a *very* clean signal (e.g., `r = 0.99`) can still
  be informative; CorrSleuth will tell you the size is small but the
  metrics will still be reported.

### `not_computable`

**Meaning.** One or both variables are constant, or the core metrics
otherwise failed to compute.

**Typical metric pattern.** Pearson, Spearman, or Kendall returned `None`
or `NaN`. Common cause: a column with zero variance after missing-value
handling.

**Common examples.**

- A column that is identically zero or identically `NaN` for the
  in-sample rows.
- A heavily-filtered slice where the predictor became constant.
- Categorical-coded columns accidentally passed in as numeric.

**Recommended next steps.**

- Check for constant variables (`x.std() == 0` or `x.nunique() <= 1`).
- Check for misalignment between two columns — different missingness
  patterns can leave you with very few overlapping rows.
- Drop the column or revisit the filter that produced it.

**Caveats.**

- This is a hard signal: the relationship cannot be assessed at all.
  Anything downstream that consumes this result needs to handle the
  unavailable metrics explicitly.

### `mixed_or_ambiguous`

**Meaning.** The metrics disagree in a way that doesn't strongly match a
canonical pattern. This is the cascade's fallback when no other rule
fires.

**Typical metric pattern.** Several common shapes land here:

- Moderate Pearson and Spearman (both in the 0.20–0.50 range) with a
  small gap.
- Strong Pearson with rank metrics that don't disagree enough to trigger
  the leverage rule.
- Distance correlation is moderately positive but the Pearson/Spearman
  pair don't fit either `weak` or `nonmonotonic`.

**Common examples.**

- A relationship that is partly linear, partly noise, partly conditional
  on a third variable.
- A heavily-discrete variable where rank metrics behave erratically.
- A mixture of two subpopulations with different relationship shapes.

**Recommended next steps.**

- Inspect the scatter plot — this is the label most likely to need
  visual interpretation.
- Check whether the pattern holds within important segments. Mixture
  effects often resolve themselves once you condition on the right
  grouping.
- Look at the diagnostics: `disagreement_score`, `rank_linear_gap`, and
  `nonmonotonic_gap` together can hint at which canonical pattern is
  closest.

**Caveats.**

- This label is the residual of the cascade. Future rule additions may
  pull some of these cases into more specific labels — don't read it as
  a permanent classification.

## Topics

### When Pearson Can Be Misleading

Pearson correlation answers a narrow question: how strongly does a
straight line fit? It is the right tool when the relationship is
genuinely linear, the data is roughly Gaussian, and there are no
high-leverage points. CorrSleuth surfaces the cases where one of those
assumptions breaks:

- **Curvature.** A monotonic but nonlinear relationship (`y = log(x)`,
  `y = x^2.5`) lowers Pearson while leaving Spearman strong — that's the
  `monotonic_nonlinear` story.
- **Nonmonotonicity.** A U-shape or cyclical pattern can leave Pearson
  near zero even though the variables are tightly related — that's
  `nonmonotonic_dependence` (visible only with distance correlation
  available).
- **Leverage.** A handful of extreme points can pull Pearson up
  dramatically while the rank metrics stay flat — that's
  `possible_outlier_or_leverage`.
- **Heteroskedasticity.** Pearson is still computable, but its standard
  error becomes unreliable. CorrSleuth doesn't flag this directly, but
  the diagnostic plot's scatter usually makes it obvious.

The `report.pearson_underrated()` method on a target scan ranks
variables where rank metrics or distance correlation exceed Pearson by
more than a configurable threshold — those are the variables most likely
to be discarded too early by a default correlation matrix.

### Monotonic vs Nonmonotonic Relationships

A relationship is **monotonic** if `y` consistently increases (or
consistently decreases) with `x`, even if the rate changes. It is
**nonmonotonic** if `y` reverses direction at some point.

| Property | Monotonic | Nonmonotonic |
|---|---|---|
| Spearman magnitude | High | Low |
| Pearson magnitude | High to low (depends on curvature) | Low |
| Distance correlation | High | High |
| Common shapes | Logarithmic, exponential, power, threshold | U-shape, V-shape, parabolic, cyclical |
| Best fit family | Monotonic transforms, splines | Polynomials, splines, kernels, trees |

In `lite` mode CorrSleuth can detect monotonic-nonlinear relationships
(via the rank vs linear gap) but **cannot** distinguish nonmonotonic
dependence from independence. If you suspect a U-shape, run with
`mode="standard"` or `mode="deep"` so distance correlation or
Chatterjee's ξ is available.

### Outlier-Sensitive Correlations

The `possible_outlier_or_leverage` label is gated by a sensitivity check:
CorrSleuth recomputes Pearson after dropping the outer 1% of `x` and `y`
and checks whether the value moves by more than 0.20. Only when it does
(or when the check could not run because `n_used` is too small) does the
label fire. The trimmed value lives at
`result.diagnostics.pearson_trimmed`; the size of the move lives at
`result.diagnostics.pearson_trim_delta`.

**Limitation: the gate trims only 1% per tail.** A leverage cluster larger
than roughly 1% of the rows (say, 2% of the data sitting at an extreme) is
only partly removed by the trim, so the trimmed Pearson can stay close to the
raw value and the `possible_outlier_or_leverage` label may not fire — even
though the relationship really is leverage-influenced. This is a deliberate,
conservative choice: a cluster that large is often closer to a subpopulation
or mixture than to a few stray outliers. When you suspect a wider leverage
cluster, do not rely on the label alone — inspect the deep-mode robust metrics
below (especially `biweight_midcorrelation` and `pearson_median_clipped_20pct`,
which down-weight far more than 1%); if they collapse toward zero while Pearson
stays high, leverage is doing the work regardless of the label.

`mode="deep"` adds four robust correlation diagnostics that you can
inspect alongside Pearson:

- **`pearson_trimmed_1pct`** — Pearson after dropping rows outside the
  1st/99th percentiles of either variable. The same value as
  `result.diagnostics.pearson_trimmed` (they're computed once and
  reused).
- **`pearson_winsorized_1pct`** — Pearson after clipping both variables
  at their 1st/99th percentiles. Less aggressive than trimming because
  the clipped values still contribute.
- **`biweight_midcorrelation`** — A median-based robust correlation
  using Tukey biweights.
- **`pearson_median_clipped_20pct`** — Pearson after clipping deviations
  from each median at the 80th percentile. More aggressive
  median-centered alternative to winsorizing.

If all four robust metrics agree with Pearson, the relationship is
probably not leverage-driven. If they collapse toward zero, leverage is
likely doing most of the work.

### Missing Data and Ties

CorrSleuth's three missing-data modes (`pairwise`, `listwise`, `raise`)
all currently behave the same way for a single pair: drop rows where
either variable is missing. The result records `n_used`, `missing_count`,
and `missing_ratio` in validation; high missingness emits a warning.

For ties, CorrSleuth tracks two related signals:

- **`x_unique_ratio` / `y_unique_ratio`** — fraction of distinct values
  in each column. The legacy `low_unique_ratio` warning fires when
  either falls below 5%.
- **`x_tie_rate` / `y_tie_rate`** — fraction of observations whose value
  is shared with at least one other row. The `high_tie_rate` warning
  fires per-variable when its rate exceeds 30%.

Heavy ties affect the rank-based metrics most: Spearman uses tie
correction but loses resolution; Kendall tau-b is the most robust of the
three rank measures under ties. Pearson is unaffected by ties as long as
the variance is non-zero. CorrSleuth's tie-rate warnings name the
specific column so you can decide whether the rank story for that pair
is reliable.

### Performance Modes

| Mode | Metrics added | Optional dependency | Typical use |
|------|---------------|---------------------|-------------|
| `lite` (default) | Pearson, Spearman, Kendall tau-b | None | Fast pairwise screening; safe to run on wide DataFrames. |
| `standard` | + Distance correlation, mutual information | `corrsleuth[standard]` (`dcor`, `scikit-learn`) | When you need to detect nonmonotonic dependence. |
| `deep` | + Robust Pearson variants, Chatterjee's ξ (both directions) | None | When you need leverage diagnostics or asymmetric functional dependence. |

Notes:

- `mode="deep"` does **not** include the `standard` metrics. It is the
  no-new-dependency tier for analysts who want richer diagnostics
  without the extras.
- Deep mode's robust diagnostics need `n_used >= 50`; below that they
  return `None` with a single consolidated warning. Chatterjee's ξ has
  a lower floor of `n_used >= 20`.
- `bootstrap=…` works in any mode but defaults to *lite* metrics for
  the resampling pass, even when the main profile uses `standard`. Pass
  `bootstrap_metrics="standard"` to opt into resampling distance
  correlation and mutual information (slower).
- Distance correlation downsamples to 20 000 rows by default
  (`max_n_for_dcor=20000`). The downsample is seeded for reproducibility.
- `scan_target()` profiles columns **sequentially** — one `profile_pair`
  call at a time, with no parallelism. For typical EDA this is fine, but a
  very wide DataFrame (hundreds to thousands of columns) combined with
  `mode="deep"` or `bootstrap=…` can take a while. Bound the cost by
  narrowing the candidate set (`columns=`, `max_pairs=`) or downsampling
  rows (`sample_size=`). There is no built-in `joblib`/multiprocessing
  path today; if you need one, parallelize across columns at the call site
  (each `profile_pair` is independent) or open an issue.

If you are not sure which mode to use: start with `lite`. Move to
`standard` when an analyst suspects nonmonotonic structure that the lite
metrics aren't catching. Use `deep` when leverage or asymmetric
dependence is the question, and you don't want to install extras.

## Further Reading

- [README](../README.md) — installation, quickstart, full API surface.
- [Methodology](methodology.md) — the statistical "how it works": the pipeline,
  each measure's assumptions, the `disagreement_score`, the label cascade, and
  the bootstrap-stability approach. This guide is the field manual; that doc is
  the theory.
- [Nonlinear metrics design note](nonlinear-metrics-design.md) — why
  Chatterjee's ξ was chosen over HSIC, MGC, and MIC, and which other
  nonlinear measures were deferred.
- [thresholds-and-rationale.md](thresholds-and-rationale.md) — every threshold
  in the package: what it gates, its value, the rationale, and how to override
  the label-driving ones.
- [`corrsleuth/heuristics/classifier.py`](../corrsleuth/heuristics/classifier.py)
  — exact thresholds for each label in the cascade.
- [`corrsleuth/heuristics/explanations.py`](../corrsleuth/heuristics/explanations.py)
  — the short narrative each label gets via `result.explain()`.
