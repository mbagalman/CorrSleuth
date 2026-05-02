# CorrSleuth Adoption Ticket Pack

This ticket pack translates a data-scientist adoption review into actionable work. The guiding question is:

> What would make CorrSleuth useful enough for a data scientist at a mid-size company to adopt as a regular EDA tool?

The answer is not simply "add more correlation metrics." CorrSleuth's strongest product wedge is interpretation: helping analysts understand when common association metrics agree, disagree, or warn that Pearson correlation is incomplete.

The current pairwise profiler is the right foundation. Regular adoption will come from three additions:

1. Richer explanation of pairwise results.
2. Stability and sensitivity checks that make labels more trustworthy.
3. Target-oriented scanning that helps users find which pairs deserve inspection.

## Phase 1 - Make `profile_pair()` Release-Strong

Phase 1 keeps the v0.1 promise narrow: one excellent pairwise profiler. These tickets improve trust, explanation quality, and result usefulness without adding broad automation.

### Ticket 1.1 - Add a "Why This Label" Explanation Layer

Priority: Phase 1 / High

Status: Complete

Completed in: `b374fd8 Add adoption roadmap and metric-aware explanations`

Completion notes:

- `result.explain()` now passes result metrics into the explanation generator.
- Explanations include metric-aware context for `near_linear`, `monotonic_nonlinear`, `nonmonotonic_dependence`, `possible_outlier_or_leverage`, and `weak_or_no_relationship`.
- Tests cover the required label-specific explanation paths.

Goal:

Make `result.explain()` describe the metric pattern that led to the diagnostic label, not just the label's generic meaning.

Why:

A data scientist does not only want to know that a pair was labeled `monotonic_nonlinear`; they want to know why. CorrSleuth's differentiation is explaining disagreement between Pearson, Spearman, Kendall, distance correlation, and mutual information. Generic label text is useful, but the product becomes much more valuable when the explanation references the actual observed metrics.

User story:

As an analyst, I want `result.explain()` to tell me which metrics agreed or disagreed, so I can decide whether the diagnostic label is credible and what to inspect next.

Implementation notes:

- Add a helper that receives the result's metrics and pattern.
- Extract key metric values by name.
- Generate short, cautious sentences based on observed gaps:
  - Pearson weak but Spearman/Kendall strong.
  - Pearson and Spearman both strong and close.
  - Pearson/Spearman weak but distance correlation strong.
  - Pearson much stronger than rank metrics.
  - Standard metrics unavailable in lite mode.
- Keep language diagnostic and non-causal.
- Preserve `include_caveat=True` behavior.

Example output:

```text
Evidence appears consistent with monotonic nonlinear association. Spearman is much stronger than Pearson, suggesting the relationship may be directional but not well summarized by a straight line. This is diagnostic evidence only; do not interpret the association causally without proper design or controls.
```

Acceptance criteria:

- `result.explain()` includes at least one sentence referencing metric agreement or disagreement.
- Explanations remain cautious and non-causal.
- Existing labels still have fallback text if metric values are unavailable.
- Unit tests cover at least:
  - `near_linear`
  - `monotonic_nonlinear`
  - `nonmonotonic_dependence`
  - `possible_outlier_or_leverage`
  - `weak_or_no_relationship`

### Ticket 1.2 - Add Structured Metric Agreement Diagnostics to the Result

Priority: Phase 1 / High

Status: Complete

Completed in: `a065ba0 Add structured metric diagnostics`

Completion notes:

- Added public `result.diagnostics` using a `MetricDiagnostics` dataclass.
- `to_dict()` now includes nested diagnostics.
- `to_frame()` now includes flattened `diagnostic_*` columns for pandas-friendly downstream use.
- Lite-mode results represent standard-only diagnostic components, such as `nonmonotonic_gap`, explicitly as unavailable.

Goal:

Expose the main disagreement components used by CorrSleuth in a stable structured object so users can inspect why a label was assigned and use those values in downstream pandas workflows.

Why:

The current `disagreement_score` is helpful, but too compressed. A data scientist will trust the package more if they can inspect the components behind the score. This also prepares the package for future ranking workflows such as "relationships Pearson may underrate."

User story:

As an analyst, I want to inspect metric gaps directly, so I can understand whether a relationship was labeled because of rank-vs-linear disagreement, nonlinear dependence, or outlier sensitivity.

Implementation notes:

- Add a public attribute named `result.diagnostics`.
- Prefer a lightweight dataclass or dictionary with a stable schema.
- Include:
  - `rank_linear_gap = abs(abs(pearson) - abs(spearman))`
  - `pearson_spearman_signed_gap = pearson - spearman`
  - `nonmonotonic_gap = distance_correlation - max(abs(pearson), abs(spearman))` when distance correlation is available
  - `pearson_kendall_gap = abs(abs(pearson) - abs(kendall_tau_b))`
  - `disagreement_score`
- Include these fields in `to_dict()` under a nested `diagnostics` key.
- Flatten diagnostic fields in `to_frame()` with a consistent prefix such as:
  - `diagnostic_rank_linear_gap`
  - `diagnostic_nonmonotonic_gap`
  - `diagnostic_pearson_kendall_gap`
  - `diagnostic_disagreement_score`
- Use the same flattened names in future scan reports so users can sort and filter with ordinary pandas operations.

Acceptance criteria:

- Result objects expose metric-gap diagnostics in a stable public shape.
- `to_dict()` includes the diagnostic components.
- `to_frame()` includes flattened diagnostic columns prefixed with `diagnostic_`.
- Tests verify component values on simple synthetic data.
- Missing optional metrics are represented explicitly, not silently omitted.

### Ticket 1.3 - Improve `summary()` for Analyst Use

Priority: Phase 1 / Medium

Status: Complete

Completed in: `a12f11d Improve result summary diagnostics`

Completion notes:

- `summary()` now includes a dedicated diagnostics section with disagreement score and key metric gaps.
- Summary rendering continues to handle unavailable or non-computable values as `NA`.
- Tests assert the summary includes relationship identity, pattern, metrics, diagnostics, recommendations, and caveat behavior.

Goal:

Make `result.summary()` a compact but complete analyst-readable report.

Why:

For adoption, users need a single text block they can paste into a notebook, PR, Slack thread, or analysis note. The summary should show the label, metrics, warnings, recommendations, and caveat in a readable order.

User story:

As an analyst, I want `result.summary()` to give me enough context to understand the result without separately calling every attribute.

Implementation notes:

- Keep returning a string.
- Include:
  - relationship name: `x vs y`
  - primary pattern
  - metric table
  - disagreement score
  - key diagnostic gaps, if Ticket 1.2 is complete
  - warnings
  - recommendations
  - non-causal caveat by default
- Render unavailable or non-computable values as `NA`.
- Avoid definitive language.

Acceptance criteria:

- `summary()` includes pattern, metric values, warnings, recommendations, and caveat.
- Summary includes either `disagreement_score` or the new diagnostic components.
- Constant-input results do not crash.
- Unit tests assert key sections appear.

### Ticket 1.4 - Integrate Outlier Sensitivity Into the Leverage Diagnostic

Priority: Phase 1 / High

Status: Complete

Completed in: `ce6cc6e Integrate outlier sensitivity diagnostic`

Completion notes:

- `profile_pair()` now computes a lightweight 1% trimmed-Pearson sensitivity check when the sample is large enough.
- The `possible_outlier_or_leverage` heuristic now requires trim-sensitive evidence, or explicitly unavailable sensitivity evidence, before assigning the leverage label.
- Stable trimmed-Pearson cases avoid the outlier/leverage label when the only evidence is Pearson-vs-rank disagreement.
- Sensitivity outputs are stored on `result.diagnostics` as `pearson_trimmed` and `pearson_trim_delta`, and are serialized/flattened with the other diagnostics.
- Tests cover the `outlier_driven` integration path and the stable-trim classifier path.

Goal:

Make the `possible_outlier_or_leverage` label more trustworthy by adding a lightweight outlier sensitivity check that feeds into the heuristic classifier.

Why:

Outlier-driven correlation is one of the most common practical EDA failure modes. A label based only on Pearson being stronger than rank metrics is useful, but it can create confusing results if a later sensitivity check says Pearson is stable under trimming. To avoid contradictory diagnostics, the sensitivity check should inform the label rather than sit only as an independent warning.

User story:

As an analyst, I want CorrSleuth to warn me when Pearson correlation appears sensitive to extreme observations, so I do not overstate a linear relationship that may be leverage-driven.

Implementation notes:

- Add an internal sensitivity calculation for lite mode:
  - baseline Pearson
  - Pearson after trimming the outer 1 percent of x or y, if sample size permits
  - optionally Pearson after winsorizing x and y
- Keep it lightweight and deterministic.
- Do not add heavy dependencies.
- Feed sensitivity results into the `possible_outlier_or_leverage` heuristic.
- Recommended heuristic behavior:
  - If Pearson is much stronger than rank metrics and trimmed Pearson changes materially, assign `possible_outlier_or_leverage`.
  - If Pearson is much stronger than rank metrics but trimmed Pearson is stable, avoid the outlier label unless another leverage signal is present; use `mixed_or_ambiguous` or another better-fitting label plus a cautionary note.
  - If sensitivity cannot be computed because `n_used` is too small, preserve the existing Pearson-vs-rank warning logic but mark the evidence as lower confidence.
- Add warning text when Pearson changes materially, for example by more than 0.2.
- Avoid emitting contradictory messages such as `possible_outlier_or_leverage` together with "Pearson is stable under trimming."
- Store sensitivity outputs in diagnostics.

Acceptance criteria:

- `outlier_driven` simulations produce an outlier/leverage warning.
- The `possible_outlier_or_leverage` label is assigned only when sensitivity evidence supports it or sensitivity could not be computed and Pearson-vs-rank disagreement is strong.
- Stable trimmed-Pearson cases do not receive contradictory outlier/leverage messaging.
- Small samples skip sensitivity checks with no crash.
- Constant inputs skip sensitivity checks safely.
- `result.to_dict()` includes sensitivity diagnostics when computed.

### Ticket 1.5 - Strengthen the Diagnostic Plot

Priority: Phase 1 / Medium

Status: Complete

Completed in: `de4e7cc Strengthen diagnostic plot panel`

Completion notes:

- The 1x3 plot structure remains: raw scatter, rank-rank scatter, and a text diagnostic panel.
- The text panel now includes `n_used`, primary pattern, metric values, diagnostic gaps, trim delta, and warnings status.
- Optional LOWESS remains guarded so smoother availability does not break the required plot contract.
- Tests now assert the text panel includes the primary pattern, key metrics, diagnostics, and warning section.

Goal:

Make `result.plot()` visually explain why the label was assigned.

Why:

The first plot is the visual proof-of-value. Data scientists are more likely to trust a diagnostic label if the plot shows raw data, rank behavior, metrics, and warnings together.

User story:

As an analyst, I want the diagnostic plot to help me visually validate the label, so I can decide whether to trust the pattern or investigate further.

Implementation notes:

- Preserve the existing 1x3 structure:
  - raw scatter
  - rank-rank scatter
  - metrics and diagnostic text panel
- Ensure optional LOWESS/smoother cannot break plotting.
- If a smoother is unavailable, the plot should still work without warning.
- Show:
  - pattern
  - Pearson, Spearman, Kendall, and optional metrics
  - warnings, capped to avoid overflow
  - `n_used`
- For large samples, continue using alpha blending or hexbin.

Acceptance criteria:

- `.plot(show=False)` always returns a Matplotlib `Figure`.
- Plotting works when optional smoother dependencies are missing.
- Plotting works when optional smoother dependencies are installed.
- Plot text panel includes the primary pattern and key metric values.
- Regression test covers the optional smoother path.

## Phase 2 - Add Stability and Confidence Signals

Phase 2 makes CorrSleuth more trustworthy. These features help users understand whether a label is stable or brittle.

### Ticket 2.1 - Add Bootstrap Metric Intervals

Priority: Phase 2 / High

Status: In review

Implementation notes from current branch:

- `profile_pair()` now accepts `bootstrap`, `bootstrap_metrics`, and `max_n_for_bootstrap`.
- Bootstrap intervals are opt-in and deterministic via `random_state`.
- The default bootstrap metric set is lite metrics only, including when `mode="standard"` is used.
- Standard metric bootstrapping requires explicit `bootstrap_metrics="standard"`.
- Results expose `result.bootstrap_intervals`; `summary()`, `to_dict()`, and `to_frame()` include bootstrap interval output.

Goal:

Allow users to estimate uncertainty around core metrics using bootstrap resampling.

Why:

Rule-based labels can feel brittle. Bootstrap intervals help users understand whether metric differences are stable or likely sampling noise. This is especially useful for small-to-medium datasets where a single coefficient can be misleading.

User story:

As an analyst, I want approximate bootstrap intervals for key metrics, so I can tell whether a diagnostic pattern is stable enough to trust.

Proposed API:

```python
result = cs.profile_pair(
    df,
    "x",
    "y",
    bootstrap=500,
    random_state=42,
)
```

Implementation notes:

- Start with Pearson, Spearman, and Kendall in lite mode.
- Bootstrap lite metrics by default even when `profile_pair(mode="standard")` is used.
- Standard-mode bootstrap for distance correlation and mutual information must be explicit opt-in because those metrics can be expensive at the 20,000-row cap.
- Add a parameter such as `bootstrap_metrics="lite"` by default, with possible future values:
  - `"lite"`: Pearson, Spearman, Kendall only
  - `"standard"`: include distance correlation and mutual information with caps
  - explicit list of metric names
- Cap bootstrap sample size separately from the main standard-mode distance-correlation cap if standard bootstrapping is enabled.
- Return intervals such as:
  - `pearson_ci_low`
  - `pearson_ci_high`
  - `spearman_ci_low`
  - `spearman_ci_high`
- Add warnings for very small samples.
- Keep random state deterministic.

Acceptance criteria:

- Bootstrap output is deterministic with `random_state`.
- Bootstrap can be disabled by default.
- Bootstrap uses lite metrics by default, including when the main profile is in standard mode.
- Bootstrapping standard metrics requires explicit opt-in.
- `summary()` displays intervals when available.
- Tests verify interval columns exist and are ordered correctly.

### Ticket 2.2 - Add Pattern Stability

Priority: Phase 2 / High

Goal:

Estimate how often the same diagnostic label appears across bootstrap samples.

Why:

A relationship labeled `monotonic_nonlinear` in 92 percent of bootstrap samples is more compelling than one labeled that way in 43 percent. Pattern stability turns CorrSleuth labels from brittle heuristics into more credible diagnostics.

User story:

As an analyst, I want to know how stable the assigned pattern is under resampling, so I can decide whether to act on it or treat it as tentative.

Implementation notes:

- Requires Ticket 2.1.
- Pattern stability should use the same bootstrap metric policy as Ticket 2.1.
- By default, stability is based on lite metrics only, even if the original profile used standard mode.
- If the original label depends on standard-only evidence, such as `nonmonotonic_dependence` from distance correlation, explain that lite-only stability may not fully test that standard-mode label.
- For each bootstrap sample:
  - compute metrics
  - apply the same heuristic classifier
  - record label
- Add:
  - `pattern_stability`
  - `bootstrap_label_counts`
  - `stability_label`, such as `low`, `medium`, `high`
- Explain stability cautiously.

Acceptance criteria:

- Result includes the share of bootstrap samples matching the original label.
- `summary()` and `explain()` mention stability when available.
- Stability output states which metric set was bootstrapped, such as `lite` or `standard`.
- Tests use deterministic simulated data.
- Computation is opt-in and does not slow default `profile_pair()`.

### Ticket 2.3 - Add Sample-Size and Ties Reliability Notes

Priority: Phase 2 / Medium

Goal:

Improve reliability warnings for small samples, low unique-value ratios, and ties.

Why:

Rank-based metrics can behave unexpectedly with many ties or ordinal/compressed variables. CorrSleuth already warns about low unique ratios, but the explanation could be more actionable and more clearly tied to rank-metric reliability.

User story:

As an analyst, I want CorrSleuth to tell me when rank metrics may be unstable, so I do not over-interpret Spearman or Kendall results on discrete or tied data.

Implementation notes:

- Add tie-rate estimates for x and y.
- Report:
  - unique ratios
  - tie rates
  - whether rank metrics may be affected
- Keep high ties as warnings, not primary label overrides, unless computation is invalid.

Acceptance criteria:

- `CleanPair` includes tie-related metadata.
- Warnings mention the affected variable when possible.
- Tests cover discrete variables with many ties.

## Phase 3 - Build the Daily EDA Workflow

Phase 3 is where CorrSleuth becomes a regular tool rather than a pairwise helper. The core feature is target scanning.

### Ticket 3.1 - Implement `scan_target()`

Priority: Phase 3 / Highest

Goal:

Add a target-oriented scanning workflow that profiles every eligible numeric predictor against one target column.

Why:

In real EDA, analysts rarely start by knowing the one pair to inspect. They usually ask: "Which variables relate to my target, and how?" `scan_target()` is the workflow that makes CorrSleuth useful in day-to-day feature exploration.

User story:

As a data scientist, I want to scan all numeric variables against a target, so I can find relationships Pearson may underrate, nonlinear signals, weak/noisy variables, and variables needing caution.

Proposed API:

```python
report = cs.scan_target(
    df,
    target="sales",
    mode="lite",
    missing="pairwise",
    errors="warn",
    progress=False,
)

report.summary()
report.to_frame()
report.plot_top(n=12)
```

Implementation notes:

- Numeric-vs-numeric only.
- Exclude the target column itself.
- Use `profile_pair()` internally.
- Return a new result object such as `CorrSleuthTargetReport`.
- Include progress-safe behavior for wide data:
  - `columns=None`
  - `max_pairs=None`
  - `sample_size=None`
  - `errors="warn"` or `errors="raise"`
- `errors="warn"` should catch exceptions from `profile_pair()` per column and store them in the report instead of failing the entire scan.
- A malformed, constant, all-null, nonnumeric, or otherwise invalid column should not crash a long scan unless `errors="raise"`.
- Store per-column errors in report output with fields such as:
  - `status`
  - `error_type`
  - `error_message`
- Add optional progress reporting:
  - `progress=False` by default to avoid adding output in scripts and tests.
  - If `progress=True` and `tqdm` is installed, use it.
  - If `progress=True` and `tqdm` is unavailable, use a simple lightweight fallback or documented no-op.
  - Keep `tqdm` optional; do not add it to base dependencies unless there is a strong reason.
- Keep mode behavior consistent with `profile_pair()`.

Acceptance criteria:

- `scan_target()` profiles all numeric predictor columns against a numeric target.
- Non-numeric columns are skipped with clear warnings or errors, depending on the chosen policy.
- With `errors="warn"`, one failing column does not stop the scan.
- Report output includes status/error fields for skipped or failed columns.
- `progress=True` does not change results and remains optional.
- `report.to_frame()` returns one row per profiled variable.
- Each row includes:
  - variable name
  - target name
  - pattern
  - metrics
  - disagreement score
  - warnings
  - recommendations
- Tests cover a mixed DataFrame with numeric and non-numeric columns.

### Ticket 3.2 - Add Target Scan Summary Sections

Priority: Phase 3 / High

Goal:

Make `scan_target().summary()` organize findings into practical EDA sections.

Why:

A raw table is useful, but the product value is interpretation and prioritization. A data scientist wants to know which variables to inspect first.

User story:

As an analyst, I want the scan summary to group variables by diagnostic pattern, so I can quickly identify promising features and caution areas.

Suggested summary sections:

```text
Strongest near-linear relationships
Potential monotonic nonlinear relationships
Potential nonmonotonic relationships
Variables Pearson may underrate
Possible outlier-driven relationships
Variables with missingness or tie warnings
Weak or no pairwise relationships
```

Implementation notes:

- Sort by pattern and disagreement score.
- Include top N per section.
- Keep language cautious.
- Include a caveat that pairwise association does not imply causation or predictive usefulness by itself.

Acceptance criteria:

- Summary sections are deterministic.
- Empty sections are omitted or shown as "none found."
- Tests verify variables land in expected sections for simulated data.

### Ticket 3.3 - Add `plot_top()` for Target Reports

Priority: Phase 3 / Medium

Goal:

Create a compact visual gallery of the most interesting target relationships.

Why:

After scanning a target, analysts need to inspect the top candidates visually. A gallery saves repeated manual calls to `profile_pair().plot()`.

User story:

As an analyst, I want to plot the top relationships from a target scan, so I can visually validate the relationships CorrSleuth flagged.

Proposed API:

```python
fig = report.plot_top(
    n=12,
    sort_by="disagreement_score",
    patterns=["monotonic_nonlinear", "nonmonotonic_dependence"],
)
```

Implementation notes:

- Start simple:
  - scatter plots only
  - pattern label in title
  - metric snippets
- Avoid huge dashboard complexity.
- Return a Matplotlib `Figure`.

Acceptance criteria:

- `plot_top()` returns a figure.
- Supports sorting by disagreement score or absolute metric value.
- Supports filtering by pattern.
- Handles fewer than N available variables gracefully.

### Ticket 3.4 - Add Ranking for "Pearson May Underrate"

Priority: Phase 3 / High

Goal:

Create a ranking that identifies variables where Pearson correlation may understate the relationship.

Why:

This is one of CorrSleuth's most valuable practical promises. In feature engineering, users want to find variables that a standard correlation matrix might cause them to discard too early.

User story:

As a feature engineer, I want CorrSleuth to show me variables Pearson may underrate, so I can inspect nonlinear or monotonic relationships before excluding useful predictors.

Implementation notes:

- Use existing disagreement components:
  - high Spearman/Kendall relative to Pearson
  - high distance correlation relative to Pearson/Spearman
  - high disagreement score
- Add a report method:

```python
report.pearson_underrated()
```

- Return a DataFrame sorted by strongest evidence.

Acceptance criteria:

- Method returns only variables meeting a documented threshold.
- Method includes the metric values and gap values that justify inclusion.
- Tests verify known simulated nonlinear variables rank above independent variables.

## Phase 4 - Expand Metrics Carefully

Phase 4 adds more statistics only when they improve diagnosis. Avoid turning CorrSleuth into a bag of coefficients.

### Ticket 4.1 - Add Robust Correlation Metrics in `deep` Mode

Priority: Phase 4 / Medium

Goal:

Add robust correlation estimates that help diagnose outlier-sensitive relationships.

Why:

Pearson can be dominated by extreme observations. Robust correlation metrics make the outlier/leverage diagnostic stronger and more evidence-based.

Candidate metrics:

- Winsorized Pearson.
- Trimmed Pearson.
- Biweight midcorrelation.
- Percentage bend correlation.

Implementation notes:

- Put these behind `mode="deep"` or a separate option.
- Avoid mandatory heavy dependencies.
- Document them as robustness diagnostics, not replacements for visual inspection.

Acceptance criteria:

- Robust metrics are not computed in default lite mode.
- Robust metrics appear in result metrics when enabled.
- Outlier-driven simulations show meaningful differences between Pearson and robust metrics.

### Ticket 4.2 - Evaluate Additional Nonlinear Dependence Measures

Priority: Phase 4 / Medium

Goal:

Research and optionally implement additional nonlinear dependence measures for `deep` mode.

Why:

Distance correlation and mutual information are strong starting points, but some relationship shapes may benefit from other dependence measures. These should be added only if they improve interpretation and are maintainable.

Candidate metrics:

- HSIC.
- Hoeffding's D.
- Chatterjee's xi.
- MGC via `hyppo`.
- Maximal Information Coefficient, if dependency and licensing tradeoffs are acceptable.

Implementation notes:

- Create a short design note before implementation.
- Evaluate:
  - dependency weight
  - install friction
  - performance
  - interpretability
  - stability
  - license compatibility
- Prefer optional extras and lazy imports.

Acceptance criteria:

- A design note compares candidate metrics.
- Any added metric has tests, docs, and clear interpretation guidance.
- No new heavy dependency is added to base install.

## Phase 5 - Reporting and Team Adoption

Phase 5 helps CorrSleuth fit into shared analysis workflows.

### Ticket 5.1 - Add Markdown Export for Results

Priority: Phase 5 / Medium

Goal:

Allow pairwise and target reports to export compact Markdown summaries.

Why:

Data scientists often need to share findings in notebooks, docs, GitHub issues, PRs, or Slack. Markdown export makes CorrSleuth easier to use in team workflows without building full HTML reports yet.

Proposed API:

```python
result.to_markdown()
report.to_markdown()
```

Acceptance criteria:

- Markdown output includes metrics, pattern, warnings, recommendations, and caveat.
- Target reports include grouped sections.
- Output is deterministic and snapshot-testable.

### Ticket 5.2 - Add Method Notes and Interpretation Guide

Priority: Phase 5 / Medium

Goal:

Document what each metric means, when it can mislead, and how CorrSleuth uses it.

Why:

Adoption depends on trust. Analysts need to understand that these labels are heuristic diagnostics, not truth claims.

Suggested docs pages:

- What CorrSleuth does and does not do.
- Understanding diagnostic labels.
- When Pearson can be misleading.
- Monotonic vs nonmonotonic relationships.
- Outlier-sensitive correlations.
- Missing data and ties.
- Performance modes.

Acceptance criteria:

- README links to the interpretation guide.
- Every diagnostic label has:
  - meaning
  - typical metric pattern
  - common examples
  - recommended next steps
  - caveats

## Recommended Roadmap

Suggested implementation order:

1. Phase 1: finish pairwise trust and interpretation.
2. Phase 2: add stability and sensitivity signals.
3. Phase 3: implement `scan_target()` and target-report prioritization.
4. Phase 4: expand metrics carefully through optional/deep modes.
5. Phase 5: add export and interpretation docs.

The main adoption milestone is Phase 3. Once users can run:

```python
report = cs.scan_target(df, target="sales", mode="standard")
report.summary()
report.to_frame()
```

CorrSleuth becomes a regular EDA workflow rather than a useful pairwise diagnostic helper.
