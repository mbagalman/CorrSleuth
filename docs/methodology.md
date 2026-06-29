# How CorrSleuth Works (Methodology)

This document explains, for statisticians and data scientists, what CorrSleuth
computes and how it turns those computations into a diagnosis. It is the
"why should I trust this" companion to the [interpretation
guide](interpretation-guide.md) (which is a field manual for *acting* on the
labels) and [thresholds-and-rationale.md](thresholds-and-rationale.md) (which
justifies every numeric cut point).

## 1. What problem it solves

A single correlation coefficient summarizes a bivariate relationship in one
number, and that compression is lossy in ways that routinely mislead:

- **Pearson's *r* assumes a linear relationship.** A strong nonlinear or
  non-monotonic dependence (e.g. a U-shape) can give *r* ≈ 0.
- **Pearson is not robust.** A handful of high-leverage points can inflate *r*,
  or even flip its sign, relative to the bulk of the data.
- **Rank coefficients (Spearman, Kendall) fix linearity but only see monotone
  structure** — they also miss U-shapes, and they don't tell you whether a
  strong *r* is leverage-driven.

CorrSleuth's premise is that **the disagreements between complementary
association measures are themselves informative**. It computes several measures,
quantifies where they agree and disagree, and maps the pattern to a single
diagnostic label plus the evidence behind it. It is a **screening / diagnostic**
step that tells you *which pairs deserve a closer look and why* — it is not an
inferential or causal procedure.

**Scope.** Numeric-vs-numeric pairs only. No causal claims, no model fitting, no
categorical/mixed-type handling. Every label is explicitly diagnostic and is
paired with a non-causal caveat and a recommendation to inspect the scatter.

## 2. The pipeline

`profile_pair(data, x, y, mode=...)` runs a fixed sequence:

1. **Validate & clean** (`validation/input.py`). Apply the missing-data policy,
   coerce to numeric, drop the unusable rows, and record data-quality flags
   (constant input, low *n*, high tie rate, low unique ratio, high
   missingness). The result is an internal `CleanPair` that downstream steps can
   assume is well-formed.
2. **Compute association measures** for the chosen mode (§3).
3. **Outlier/leverage sensitivity check** (§6): recompute Pearson on a 1%-trimmed
   sample and measure how much it moves.
4. **Compare the measures** into diagnostic gaps and a `disagreement_score` (§4).
5. **Assign one label** via a fixed-priority cascade (§5).
6. **(Optional) Quantify sampling uncertainty** with the bootstrap (§7).

The output (`CorrSleuthResult`) carries the raw metrics, the label, the
diagnostic gaps, warnings, recommendations, a plain-English explanation, and
(when requested) bootstrap intervals and pattern stability.

`scan_target(data, target)` applies `profile_pair` to every eligible numeric
column against a target and aggregates the results; the per-pair methodology is
identical.

## 3. The association measures

Measures are grouped into three **modes** of increasing cost. Each higher mode is
a superset of the information in the lower ones.

| Measure | Mode | Detects | Key assumptions / sensitivities |
|---|---|---|---|
| **Pearson *r*** | lite | Linear association | Assumes linearity; not robust to outliers/leverage |
| **Spearman ρ** | lite | Monotone association (rank) | Monotone only; robust to monotone outliers; degraded by heavy ties |
| **Kendall τ-b** | lite | Monotone association (rank, tie-corrected) | Monotone only; built from concordant−discordant *pair* counts (not rank variance like ρ), so it is numerically smaller than ρ for the same signal |
| **Distance correlation** | standard | *Any* statistical dependence | Population dCor = 0 **iff** independent; range [0, 1]; needs `dcor` |
| **Mutual information** | standard | *Any* statistical dependence | KSG estimator; **raw/unnormalized (nats, ≥ 0, unbounded)** — not a 0–1 scale; needs `scikit-learn` |
| **Trimmed / winsorized / median-clipped Pearson, biweight midcorrelation** | deep | Whether Pearson is leverage-driven | Robust variants of Pearson; computed only when *n* ≥ 50 |
| **Chatterjee's ξ** (both directions) | deep | *Functional* dependence, **asymmetric** | ξ(X→Y) → 1 when Y is a function of X, → 0 under independence; *n* ≥ 20 |

Notes that matter for interpretation:

- **Distance correlation** (Székely, Rizzo & Bakirov, 2007) is the workhorse for
  detecting non-monotonic dependence: unlike Pearson/Spearman it is zero only
  under genuine independence, so a high dCor with weak Pearson **and** weak
  Spearman is the signature of a U-shape or other non-monotone structure.
- **Mutual information** is reported as the raw KSG estimate in nats. It is `≥ 0`
  and unbounded, so read its magnitude *relatively* (larger = more shared
  information), never as if it were on Pearson's 0–1 scale.
- **Chatterjee's ξ** (2020) is **asymmetric**: ξ(X→Y) measures whether Y is a
  noisy function of X, which need not equal ξ(Y→X). Both directions are computed.
  Ties in the sort variable are broken by a *seeded random permutation* (its
  theory requires random tie-breaking; ordering ties by the response would leak
  it and inflate ξ), so ξ is reproducible for a fixed `random_state` but noisier
  for low-cardinality sort variables. See the [phase-4 design
  note](phase4-nonlinear-metrics-design-note.md) for why ξ was chosen over HSIC,
  MGC, MIC, and Hoeffding's D.

## 4. Comparing the measures

CorrSleuth derives a small set of **diagnostic gaps** (surfaced on
`result.diagnostics`) and a scalar `disagreement_score`. Let `p`, `s`, `k`, `dc`
be the (signed) Pearson, Spearman, Kendall, and distance-correlation values.

- `pearson_spearman_signed_gap = p − s`. The **signed** difference. This is the
  one that exposes a directional conflict: `+0.8` vs `−0.8` gives `1.6`, not `0`.
- `rank_linear_gap = | |p| − |s| |`. The magnitude discrepancy (how much stronger
  one is than the other, ignoring sign).
- `pearson_kendall_gap = | |p| − |k| |`.
- `nonmonotonic_gap = dc − max(|p|, |s|)`. Distance correlation in excess of the
  strongest monotone signal; the positive part is evidence of non-monotone
  dependence. Reported raw (can be negative).

The headline **`disagreement_score`** aggregates two orthogonal kinds of
disagreement:

```
disagreement_score = |p − s|  +  max(0, dc − max(|p|, |s|))
                     └ rank-vs-linear ┘   └ non-monotone excess ┘
```

- The first term uses the **signed** Pearson−Spearman difference, so a sign
  conflict (a leverage signature) registers as large disagreement rather than
  being hidden by equal magnitudes. For same-sign metrics it equals `||p|−|s||`.
- The second term is non-zero only when distance correlation exceeds the best
  monotone measure — i.e. when there is dependence the linear/rank measures
  cannot see. It is `0` in lite mode (no dCor).

A score near `0` means the measures tell the same story; a large score means they
disagree, which is exactly when a single coefficient is most likely to mislead.
Unavailable metrics (e.g. a constant column) contribute nothing rather than a
spurious `0`.

## 5. The label cascade

The diagnosis is a **fixed-priority cascade**: rules are tested in order and the
first match wins, so every pair receives exactly one label. The cut points are
documented module-level constants (see
[thresholds-and-rationale.md](thresholds-and-rationale.md)); the values below are
the defaults.

1. **`not_computable`** — a variable is constant or a core metric failed.
2. **`low_power_or_uncertain`** — `n_used < 30`.
3. **`possible_outlier_or_leverage`** — Pearson is strong (`|p| > 0.50`) **and**
   either materially exceeds the rank metrics in magnitude (`|p| − |s| > 0.20` or
   `|p| − |k| > 0.25`) **or conflicts in sign** with Spearman (opposite signs,
   both `≥ 0.30`), **and** the trimmed-Pearson check says Pearson is
   leverage-sensitive (or sensitivity could not be computed). This rule requires
   *independent* leverage evidence — a gap alone is not enough.
4. **`nonmonotonic_dependence`** — `|p|` and `|s|` are both weak (`< 0.25`) while
   distance correlation is high (`> 0.35`). Requires `mode="standard"`.
5. **`monotonic_nonlinear`** — Spearman is meaningfully stronger than Pearson
   (`|s| > 0.50` and `|s| − |p| > 0.20`), without a Pearson/Spearman sign
   conflict.
6. **`near_linear`** — Pearson and Spearman are both strong (`> 0.50`) and close
   (`||p|−|s|| < 0.15`), without a sign conflict.
7. **`weak_or_no_relationship`** — all available measures are small (`|p|, |s| <
   0.20`, and `dc < 0.20` when present).
8. **`mixed_or_ambiguous`** — fallback when none of the above matches.

Two design choices a reviewer should know:

- **Magnitude with a signed-conflict guard.** Most comparisons use absolute
  magnitudes (direction alone does not change the label). The exception is a
  **Pearson/Spearman sign conflict** (opposite signs, both `≥ 0.30`): it is a
  leverage signature, so it routes to `possible_outlier_or_leverage` (with trim
  evidence) or `mixed_or_ambiguous`, and is explicitly disqualified from
  `near_linear`/`monotonic_nonlinear`.
- **Deliberately conservative.** Borderline cases fall through to
  `mixed_or_ambiguous` rather than overclaiming. The thresholds are *conventions*
  (effect-size bands à la Cohen, sanity-checked against bundled synthetic
  scenarios), not parameters fit to a labeled benchmark — see the thresholds doc.

## 6. Leverage / outlier sensitivity

The `possible_outlier_or_leverage` label is gated on direct evidence, not just a
metric gap. CorrSleuth recomputes Pearson after trimming the outer 1% of each
variable and measures the **signed** change:

```
trim_delta = | pearson_full − pearson_trimmed |
```

A large `trim_delta` (default `> 0.20`) means a small fraction of extreme points
is driving the linear correlation, so the relationship is flagged
leverage-sensitive. The signed comparison is deliberate: a sign flip
(`+0.55 → −0.55`) is the most leverage-sensitive case there is, and comparing
magnitudes-of-magnitudes would score it `0` and call it "stable." The trimmed
Pearson and the robust deep-mode variants (winsorized, median-clipped, biweight
midcorrelation) require `n ≥ 50` to be meaningful.

## 7. Sampling uncertainty: the bootstrap

`bootstrap=B` adds a nonparametric assessment of how stable the result is under
resampling. Each of the `B` replicates draws rows with replacement and is
re-validated and re-profiled exactly like the original pair.

- **Percentile intervals.** For each requested metric, the 2.5th/97.5th
  percentiles of its bootstrap distribution form an approximate 95% interval.
- **Pattern stability.** Each replicate is re-labeled through the same cascade;
  `pattern_stability` is the fraction of replicates whose label matches the
  original. The cascade always evaluates at least the lite triple per replicate,
  so stability is meaningful even when intervals are requested for a custom
  metric subset.
- **m-out-of-n capping.** `max_n_for_bootstrap` caps the rows drawn per replicate
  for cost. Resampling fewer rows than the data contains widens the intervals
  (they become conservative by roughly `sqrt(n / m)`); a warning discloses this
  whenever the cap binds.

Two conservative guards keep the bootstrap honest at small effective sizes (both
key off the *effective per-replicate* size, `min(n_used, max_n_for_bootstrap)`):

- **Intervals** are not reported below 20 rows per replicate — a
  with-replacement resample of so few points cannot represent the tails, so the
  percentiles imply false precision. `bootstrap_intervals` is then `None` with a
  warning.
- **Pattern stability** is suppressed (`None`, with a warning) when the cap pushes
  replicates below 30 rows on a *larger* original sample: every replicate would
  be judged low-power, making stability meaningless against the full-sample
  label. Genuinely small (uncapped) samples keep their stability signal, which
  correctly reports a stably low-power label.

## 8. Reproducibility

`random_state` (default 42) seeds every stochastic step — distance-correlation
downsampling, the mutual-information estimator, bootstrap resampling, and
Chatterjee's ξ tie-break — so repeated runs on the same input return identical
numbers. The one non-determinism a user can hit deliberately: under ties in its
sort variable, ξ depends on the (seeded) random tie-break, so it is reproducible
for a fixed `random_state` but not invariant to input row order.

## 9. Limitations and honest caveats

- **Heuristic, not inferential.** The labels come from threshold rules, not
  hypothesis tests; thresholds are documented conventions, not fitted or
  optimal, and are overridable.
- **Diagnostic, not causal.** No causal, treatment-effect, or model-specification
  claims. A strong association need not be predictively useful in a multivariate
  model.
- **Pairwise and numeric only.** No multivariate adjustment (confounding,
  partial correlation) and no categorical/mixed-type support.
- **Mutual information is unnormalized** (nats) — interpret relatively.
- **Chatterjee's ξ is noisier for low-cardinality sort variables** (random
  tie-break) and is bounded below 1 for discrete predictors even under perfect
  dependence.
- **Always inspect the scatter.** Every label is a pointer to look, not a verdict.

## 10. References and further reading

- [interpretation-guide.md](interpretation-guide.md) — per-label meaning,
  typical metric patterns, and how to act on each label.
- [thresholds-and-rationale.md](thresholds-and-rationale.md) — every cut point,
  its value, and its justification.
- [phase4-nonlinear-metrics-design-note.md](phase4-nonlinear-metrics-design-note.md)
  — why Chatterjee's ξ was selected for deep mode.
- Chatterjee, S. (2020). *A new coefficient of correlation.* JASA.
- Székely, G., Rizzo, M., & Bakirov, N. (2007). *Measuring and testing
  dependence by correlation of distances.* Annals of Statistics.
- Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). *Estimating mutual
  information.* Physical Review E. (The KSG estimator used by scikit-learn.)
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences.*
  (Source of the effect-size magnitude bands.)
