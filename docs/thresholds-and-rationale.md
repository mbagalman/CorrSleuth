# Thresholds and Rationale

CorrSleuth turns continuous correlation coefficients into discrete labels and
warnings. Every one of those conversions happens at a **threshold** — a cut
point such as "Pearson above 0.50 counts as strong" or "fewer than 30 rows is
low power." This document lists every threshold, where it lives, what it gates,
and why it has the value it does.

It exists because hard-coded cut points are the place a rule-based system feels
most arbitrary. The honest summary is:

- **These are conventions, not parameters fit to your data.** They are chosen
  from established effect-size and sample-size guidance, then sanity-checked so
  the bundled synthetic scenarios
  ([`corrsleuth/datasets/simulations.py`](../corrsleuth/datasets/simulations.py))
  land on their intended labels with margin. They were **not** produced by
  optimizing against a labeled benchmark, and CorrSleuth does not claim they are
  optimal for any particular domain.
- **The cascade is deliberately conservative.** Borderline pairs fall through to
  `mixed_or_ambiguous` rather than overclaiming a pattern. A label is an
  invitation to look at the scatter plot, not a verdict.
- **They are readable and overridable.** The label-driving thresholds are
  public, module-level constants (see [Overriding thresholds](#overriding-thresholds)).

If you only read one thing: the labels are a **field guide, not a hypothesis
test.** Treat every threshold as a documented convention you are free to adjust.

## Contents

- [How a value gets chosen](#how-a-value-gets-chosen)
- [Label cascade thresholds](#label-cascade-thresholds)
- [Sample-size and data-quality gates](#sample-size-and-data-quality-gates)
- [Outlier / robustness thresholds](#outlier--robustness-thresholds)
- [Bootstrap stability bands](#bootstrap-stability-bands)
- [Supplementary warning thresholds](#supplementary-warning-thresholds)
- [Scan-level thresholds](#scan-level-thresholds)
- [Why 0.20 shows up so often](#why-020-shows-up-so-often)
- [Overriding thresholds](#overriding-thresholds)
- [Known limitations](#known-limitations)

## How a value gets chosen

Two kinds of reasoning sit behind the numbers:

1. **Effect-size convention.** Magnitude bands follow the widely used guidance
   for interpreting a correlation coefficient (Cohen, *Statistical Power
   Analysis for the Behavioral Sciences*, 1988): `|r| ≈ 0.1` is a "small"
   effect, `≈ 0.3` "medium", `≈ 0.5` "large". CorrSleuth's "strong" floor sits
   on the large boundary (0.50) and its "weak" ceiling sits below the medium
   boundary (0.20).

2. **Separation on known shapes.** Gap and closeness thresholds (how much
   Spearman must exceed Pearson to call a relationship nonlinear, how close they
   must be to call it linear) are set so the synthetic relationships in the
   bundled scenarios separate cleanly, with the borderline cases intentionally
   left to fall through to `mixed_or_ambiguous`.

Neither kind is an empirical estimate with a confidence interval. They are
defaults that behave sensibly across common cases; tune them when your domain
disagrees.

## Label cascade thresholds

Defined in
[`corrsleuth/heuristics/classifier.py`](../corrsleuth/heuristics/classifier.py).
These are the most consequential thresholds — they assign the primary label.
The cascade evaluates rules in priority order and returns the first match;
`p`, `s`, `k` are `|pearson|`, `|spearman|`, `|kendall_tau_b|`, and `dc` is
distance correlation (standard mode only).

| Constant | Value | Gates | Rationale |
|---|---|---|---|
| `STRONG_MAGNITUDE_THRESHOLD` | 0.50 | "strong" floor for the leverage, `monotonic_nonlinear`, and `near_linear` rules | Cohen's "large effect" boundary for a correlation. |
| `WEAK_MAGNITUDE_THRESHOLD` | 0.20 | `weak_or_no_relationship` (Pearson **and** Spearman must fall under it) | Below Cohen's "medium" boundary, so genuinely moderate relationships are not called weak. |
| `RANK_LINEAR_GAP_THRESHOLD` | 0.20 | how far a rank coefficient must exceed Pearson (or vice versa) to signal nonlinearity / leverage | One effect-size band; comfortably above the bootstrap spread of these coefficients at n ≥ 30. Shared by the leverage rule (`p − s`) and `monotonic_nonlinear` (`s − p`). |
| `PEARSON_KENDALL_GAP_THRESHOLD` | 0.25 | Pearson-vs-Kendall gap in the leverage rule | Kendall's τ is numerically smaller than Spearman's ρ for the same monotone signal (`τ ≈ (2/π)·arcsin ρ`), so a wider gap carries the same evidence. |
| `NONMONOTONIC_MONOTONE_CEILING` | 0.25 | ceiling on `p` and `s` for `nonmonotonic_dependence` | Both monotone measures must be weak before a high distance correlation is read as nonmonotonic (e.g. U-shaped) rather than a monotone trend the rank metrics already saw. |
| `NONMONOTONIC_DC_THRESHOLD` | 0.35 | distance-correlation floor for `nonmonotonic_dependence` | "Real dependence" cut point; set equal to `XI_DEPENDENCE_WARN_THRESHOLD` so the cascade and the deep-mode ξ warning agree. |
| `NEAR_LINEAR_GAP_THRESHOLD` | 0.15 | max `|p − s|` for `near_linear` (both must already be strong) | Tight closeness test that keeps monotone-but-curved relationships out of the "approximately linear" bucket. |
| `WEAK_DC_THRESHOLD` | 0.20 | distance-correlation ceiling for `weak_or_no_relationship` | When distance correlation is available it must also be small, so a hidden nonmonotonic signal is not mislabeled "no relationship." |

The plain-language version of each rule and its failure modes is in
[interpretation-guide.md](interpretation-guide.md#how-a-label-is-assigned).

## Sample-size and data-quality gates

| Constant | Value | Location | Gates | Rationale |
|---|---|---|---|---|
| `LOW_N_THRESHOLD` | 30 | `validation/input.py` | sets the `low_n` flag → `low_power_or_uncertain`; also gates bootstrap stability warnings | Conventional "small sample" rule of thumb (where the t-distribution approaches normal). A floor, not a guarantee — see [the caveat](interpretation-guide.md#low_power_or_uncertain). |
| `_TIE_RATE_WARN_THRESHOLD` | 0.30 | `validation/input.py` | high-tie-rate warning | Above ~30% tied values, Spearman/Kendall tie-correction is working hard enough that effective rank resolution drops. |
| `_HIGH_MISSINGNESS_THRESHOLD` | 0.50 | `validation/input.py` | high-missingness warning | Past half missing, deletion has removed most data and any coefficient is on an unrepresentative remainder. |
| `_LOW_UNIQUE_RATIO_THRESHOLD` | 0.05 | `validation/input.py` | low-unique-ratio warning | < 5% distinct values means each value is shared by ~20 rows on average — effectively discrete, so rank metrics are tie-unstable. |
| `_MIN_N_FOR_CHATTERJEE_XI` | 20 | `metrics/nonlinear.py` | minimum n before ξ is computed | ξ converges slowly and is biased on tiny samples; 20 is a conservative floor below the labeling cutoff, since ξ is only ever a supplementary diagnostic. |
| `_MIN_N_FOR_ROBUST` / `ROBUST_METRIC_MIN_N` | 50 | `metrics/robust.py` | minimum n before robust deep-mode metrics run | A 1% trim removes too few rows to mean anything below ~50; 50 keeps ≥ 1 row in each trimmed tail. |
| `_MIN_N_AFTER_TRIM` | 30 | `metrics/robust.py` | minimum n that must survive trimming | Mirrors `LOW_N_THRESHOLD` so a trimmed correlation is never reported on a sample CorrSleuth would otherwise call low-power. |

## Outlier / robustness thresholds

Defined in [`corrsleuth/api.py`](../corrsleuth/api.py) (deep-mode outlier
sensitivity check) and [`corrsleuth/metrics/robust.py`](../corrsleuth/metrics/robust.py).

| Constant | Value | Location | Gates | Rationale |
|---|---|---|---|---|
| `_OUTLIER_TRIM_QUANTILE` / `_TAIL_FRACTION` | 0.01 | `api.py`, `robust.py` | tail fraction trimmed per side (1% per side, 2% per variable) before recomputing Pearson | Gentle trim: neutralizes a few extreme leverage points without reshaping a clean distribution, so a large gap implicates a handful of rows. See the [1%-trim limitation](../README.md) for what it misses. |
| `_OUTLIER_SENSITIVE_DELTA` | 0.20 | `api.py` | change in Pearson after trimming above which the pair is flagged leverage-sensitive | One effect-size band of movement is "material." Computed from the **signed** difference so a sign flip counts in full. |
| `_OUTLIER_MIN_N_FOR_TRIM` | 50 | `api.py` | minimum n before the trim check runs | Same reasoning as `_MIN_N_FOR_ROBUST`. |
| `_OUTLIER_MIN_N_AFTER_TRIM` | 30 | `api.py` | minimum n remaining after trim | Same reasoning as `_MIN_N_AFTER_TRIM`. |
| `_bend` `beta` | 0.20 | `robust.py` | bending constant of the biweight midcorrelation | Standard default from Wilcox's robust-statistics work (matches scipy/astropy biweight), trading a little Gaussian efficiency for resistance to ~20% contamination. |

## Bootstrap stability bands

Defined in [`corrsleuth/metrics/bootstrap.py`](../corrsleuth/metrics/bootstrap.py).
Pattern stability is the fraction of bootstrap replicates whose label matches
the original. These two cut points are presentation bands for a continuous
score, **not** significance tests.

| Constant | Value | Band | Rationale |
|---|---|---|---|
| `_STABILITY_HIGH_THRESHOLD` | 0.80 | ≥ 0.80 → "high" | At most ~1 in 5 resamples disagreed. |
| `_STABILITY_MEDIUM_THRESHOLD` | 0.50 | ≥ 0.50 → "medium", else "low" | The point at which the modal label no longer holds a majority — the natural "treat this label as shaky" line. |

## Supplementary warning thresholds

These add cautionary text but never change the primary label
([`corrsleuth/heuristics/classifier.py`](../corrsleuth/heuristics/classifier.py)).

| Constant | Value | Gates | Rationale |
|---|---|---|---|
| `CONFLICTING_SIGN_THRESHOLD` | 0.30 | both `|pearson|` and `|spearman|` must exceed this before an opposite-sign disagreement is worth a warning | Below this both coefficients are near zero and a sign flip is just noise. |
| `XI_DEPENDENCE_WARN_THRESHOLD` | 0.35 | Chatterjee's ξ above which a weak/ambiguous label gets a "may understate dependence" warning | Matches `NONMONOTONIC_DC_THRESHOLD` so the two dependence signals share one cut point. |

## Scan-level thresholds

Defined in [`corrsleuth/scan/report.py`](../corrsleuth/scan/report.py).

| Constant | Value | Gates | Rationale |
|---|---|---|---|
| `_PEARSON_UNDERRATE_GAP` | 0.20 | how far rank/nonmonotonic evidence must exceed Pearson to surface a variable in the "Pearson may underrate" scan section | Set equal to the cascade's `RANK_LINEAR_GAP_THRESHOLD` so the scan callout and the per-pair `monotonic_nonlinear` label fire on the same gap. |

## Why 0.20 shows up so often

`0.20` is `RANK_LINEAR_GAP_THRESHOLD`, `WEAK_MAGNITUDE_THRESHOLD`,
`WEAK_DC_THRESHOLD`, `_OUTLIER_SENSITIVE_DELTA`, `_PEARSON_UNDERRATE_GAP`, and
the biweight `beta`. This is partly intentional and partly coincidental:

- As a **magnitude ceiling** (weak labels) it marks "below a medium effect."
- As a **gap / delta** (nonlinearity, leverage, underrate callout) it marks
  "one effect-size band of separation," which is large enough to exceed the
  sampling spread of these coefficients at n ≥ 30 while staying sensitive.
- The biweight `beta = 0.20` is unrelated — it is a robustness convention
  (resist ~20% contamination), not an effect-size band.

Sharing the value keeps related signals (e.g. the `monotonic_nonlinear` label
and the scan's "underrate" callout) firing in lockstep, which is intentional.

## Overriding thresholds

The label-driving cascade constants are **public module-level constants**, so
advanced users can inspect or override them before profiling:

```python
import corrsleuth.heuristics.classifier as clf

# e.g. require a larger Pearson/Spearman gap before calling a relationship
# nonlinear, for a noisier domain:
clf.RANK_LINEAR_GAP_THRESHOLD = 0.30

result = corrsleuth.profile_pair(df, "x", "y")
```

Notes and caveats:

- Overriding mutates module state for the process. Set values once at startup;
  do not toggle them mid-analysis.
- The constants prefixed with `_` (sample-size gates, robustness internals) are
  intentionally private and may change between releases — treat them as
  documented behavior, not API.
- A per-call configuration object is a deliberate non-goal for now: it would
  thread parameters through the whole pipeline for a feature few users need.
  Module-level override covers the realistic cases. If you have a use case that
  needs per-call thresholds, open an issue.

## Known limitations

- **No single value fits every domain.** A field where `r = 0.4` is a strong
  result and one where `r = 0.9` is unremarkable will both be served imperfectly
  by one fixed band. Adjust the constants, or read the raw coefficients in the
  result and apply your own judgment.
- **The bands have hard edges.** A pair at `|p − s| = 0.149` is `near_linear`
  and one at `0.151` is not, despite being statistically indistinguishable. The
  conservative fallthrough and the bootstrap stability signal
  (`bootstrap=…`) are the intended guards: a label that flips across a hairline
  boundary will usually show low stability.
- **They were tuned on synthetic scenarios, not a labeled real-world
  benchmark.** They behave sensibly on the bundled shapes and on common EDA
  cases; they are not validated against a gold-standard corpus.

## See also

- [interpretation-guide.md](interpretation-guide.md) — what each label means and
  how to act on it.
- [phase4-nonlinear-metrics-design-note.md](phase4-nonlinear-metrics-design-note.md)
  — why particular metrics were chosen.
- [`corrsleuth/heuristics/classifier.py`](../corrsleuth/heuristics/classifier.py)
  — the constants and the cascade, with inline docstrings.
