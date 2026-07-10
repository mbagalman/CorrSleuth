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

- **Pearson's *r* summarizes *linear* association.** A strong nonlinear or
  non-monotonic dependence (e.g. a U-shape) can give *r* ≈ 0.
- **Pearson is not robust.** A handful of high-leverage points can inflate *r*,
  or even flip its sign, relative to the bulk of the data.
- **Rank coefficients (Spearman, Kendall) replace the linear target with a
  monotone/rank one, so they only see monotone structure** — they also miss
  U-shapes, and they don't tell you whether a
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

**Design philosophy.** CorrSleuth prioritizes **transparency and conservatism**
over statistical optimality or automation. The label cascade uses fixed,
documented, overridable thresholds rather than learned or optimized cutoffs, and
every label is returned alongside the raw metrics and diagnostics that produced
it — so you can always audit the reasoning, or override the automated judgment
after inspecting the scatter. When the evidence is ambiguous the tool prefers to
say so (`mixed_or_ambiguous`) rather than overclaim a pattern.

## 2. The pipeline

`profile_pair(data, x, y, mode=...)` runs a fixed sequence:

1. **Validate & clean** (`validation/input.py`). Apply the missing-data policy,
   verify each column is a real-valued numeric dtype (non-numeric input raises;
   complex dtypes are rejected rather than silently projected onto the real axis)
   and cast it
   to float, drop the unusable rows, and record data-quality flags
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

Measures are grouped into three **modes**. All modes compute the lite metrics.
`standard` adds distance correlation and mutual information (for non-monotone
dependence). `deep` is a **strict superset** of `standard` — it computes those
same metrics **and** adds the robust-Pearson family and Chatterjee's ξ (for
leverage and asymmetric functional dependence). Because `deep` includes the
standard metrics, both `standard` and `deep` require the `corrsleuth[standard]`
extras and raise `OptionalDependencyError` when they are missing.

| Measure | Mode | Detects | Key assumptions / sensitivities |
|---|---|---|---|
| **Pearson *r*** | lite | Linear association | Captures linear association only; not robust to outliers/leverage |
| **Spearman ρ** | lite | Monotone association (rank) | Monotone only; robust to monotone outliers; degraded by heavy ties |
| **Kendall τ-b** | lite | Monotone association (rank, tie-corrected) | Monotone only; built from concordant−discordant *pair* counts (not rank variance like ρ), so it is numerically smaller than ρ for the same signal |
| **Distance correlation** | standard | *Any* statistical dependence | Population dCor = 0 **iff** independent; range [0, 1]; needs `dcor` |
| **Mutual information** | standard | *Any* statistical dependence | KSG estimator; **raw/unnormalized (nats, unbounded above)** — not a 0–1 scale; the population value is ≥ 0 but the estimator can be slightly negative under near-independence; needs `scikit-learn` |
| **Trimmed / winsorized / median-clipped Pearson, biweight midcorrelation** | deep | Whether Pearson is leverage-driven | Robust variants of Pearson; computed only when *n* ≥ 50 |
| **Chatterjee's ξ** (both directions) | deep | *Functional* dependence, **asymmetric** | ξ(X→Y) → 0 under independence and → 1 for functional dependence with a rich/tie-free sort variable (discrete or heavily tied X can stay below 1 even under perfect dependence); *n* ≥ 20 |

Notes that matter for interpretation:

- **Distance correlation** (Székely, Rizzo & Bakirov, 2007) is the workhorse for
  detecting non-monotonic dependence: unlike Pearson/Spearman the *population*
  dCor is zero only under genuine independence, so a high dCor with weak Pearson
  **and** weak Spearman is the signature of a U-shape or other non-monotone
  structure. (The sample estimator is biased slightly positive even under
  independence, so read small values as "near zero," not exactly zero.)
- **Mutual information** is reported as the raw KSG estimate in nats. The
  population MI is `≥ 0` and unbounded above, but the KSG *estimator* can return
  a small **negative** value under (near-)independence — a known property of the
  estimator, not an error — so read a small or slightly negative value as "near
  zero." Read its magnitude *relatively* (larger = more shared information), never
  as if it were on Pearson's 0–1 scale. In practice it serves as a *detector* of
  arbitrary dependence alongside distance correlation, not as a standalone
  strength measure.
- **Chatterjee's ξ** (2020) is **asymmetric**: ξ(X→Y) measures whether Y is a
  noisy function of X, which need not equal ξ(Y→X). Because it is asymmetric —
  and sensitive to the cardinality of the conditioning (sort) variable — both
  directions (`chatterjee_xi` and `chatterjee_xi_reverse`) are always reported in
  deep mode. Ties in the sort variable are broken by a *seeded random
  permutation* (its theory requires random tie-breaking; ordering ties by the
  response would leak it and inflate ξ), so ξ is reproducible for a fixed
  `random_state` but noisier for low-cardinality sort variables. See the
  [nonlinear-metrics design note](nonlinear-metrics-design.md) for why ξ was
  chosen over HSIC, MGC, MIC, and Hoeffding's D.

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
- `bin_lof_r2_gain`, `bin_reversal_count`, and `sq_corr` are the core shape
  diagnostics (no mode gate; pure numpy/scipy) that feed the label cascade but
  are **not** included in the `disagreement_score` formula below. (Additional
  lite-computable diagnostics — the segmentation family `segment_gain` /
  `segment_stepness` / `segment_jump_ratio` / `breakpoint_x`, and the two-group
  `cluster_split_*` family — feed the secondary axes described in §5; see there.)
  Two of the core three carry **robust leave-one-out companions** that gate the
  cascade routes: `bin_lof_r2_gain_robust` (the smallest gain after dropping any
  single bin) and `sq_corr_robust` (the smallest `|sq_corr|` after dropping the
  few most extreme squared points), so a lone extreme bin or point cannot
  manufacture a curvature/oscillation/magnitude signal on a structureless
  predictor.
  `bin_lof_r2_gain` is the **degrees-of-freedom-adjusted** R² of an
  equal-frequency-bin model of Y|X minus the adjusted R² of a linear fit — a
  lack-of-fit diagnostic in the spirit of the classical grouped-X F-test (Neter,
  Kutner, Nachtsheim & Wasserman), adapted to a p-value-free adjusted-R² gain
  (equal-frequency bins of a continuous X stand in for the classical test's exact
  replicates, so it is a diagnostic rather than a calibrated F-test). It
  catches smooth monotonic curvature and step/threshold functions the
  rank-vs-linear gap misses. The df adjustment matters: a plain R² difference
  credits the many-parameter bin model for degrees of freedom the line lacks,
  giving a positive null bias that reads ordinary noisy-linear data as curved. `bin_reversal_count`, computed from
  the same bins, is how many times the sequence of bin means changes direction
  (counted with hysteresis so noise wiggle is not a turn: 0 for a monotone
  trend, 1 for a single bend, 2+ for an oscillation) — meaningful only jointly
  with a high `bin_lof_r2_gain`, since pure noise reverses constantly with
  near-zero gain. `sq_corr` is `corr((X−x̄)², (Y−ȳ)²)` (the correlation of the
  mean-centered squares) — it catches dependence carried
  in magnitude rather than sign (e.g. points scattered around a circle), which
  distance correlation itself under-reads for that shape. See
  [shape-diagnostics-design.md](shape-diagnostics-design.md).

The headline **`disagreement_score`** aggregates two orthogonal kinds of
disagreement:

```
disagreement_score = |p − s|  +  max(0, dc − max(|p|, |s|))
                     └ rank-vs-linear ┘   └ non-monotone excess ┘
```

- The first term, `|p − s|`, is the **absolute value of the *signed*
  difference** — not `||p| − |s||`. So a sign conflict (a leverage signature:
  `+0.8` vs `−0.8` → `1.6`) contributes fully rather than being hidden by equal
  magnitudes. For same-sign metrics the two forms coincide.
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
   both absolute magnitudes `≥ 0.30`), **and** the trimmed-Pearson check says
   Pearson is
   leverage-sensitive (or sensitivity could not be computed). The rule looks for
   *independent* leverage evidence beyond the gap — the trimmed-Pearson check
   flagging sensitivity — but when that check cannot run it still routes here
   (conservatively, since the strong-Pearson-plus-sign-conflict shape is itself a
   leverage signature) rather than reading the pair as clean.
4. **`nonmonotonic_dependence`** — any of **four** routes fires. The first
   three require both monotone metrics weak (`|p|, |s| < 0.25`): distance
   correlation is high (`> 0.35`, `mode="standard"` only); the squared-value
   correlation `|corr((X−x̄)², (Y−ȳ)²)|` is high (`> 0.35`, `sq_corr`, no mode
   gate — added because a true circular/radial relationship structurally caps
   distance correlation around ~0.2, even noiseless) **and** its leave-the-top-out
   robust value clears 0.20; or a **single bend** with massive bin structure
   (`bin_reversal_count ≥ 1` with `bin_lof_r2_gain > 0.30`, raw **and**
   leave-one-bin-out robust — the lite-mode route for a V or off-center U whose
   arm geometry defeats `sq_corr`). The **fourth**, the oscillation route, is
   allowed up to a *moderate* trend (`max(|p|, |s|) < 0.50`, the ceiling
   `OSCILLATION_MONOTONE_CEILING`): the bin means reverse direction repeatedly
   with substantial bin structure (`bin_reversal_count ≥ 2` **and**
   `bin_lof_r2_gain > 0.15`, raw **and** robust — added because an oscillating
   relationship like a sinusoid keeps distance correlation only marginally above
   its floor and has no magnitude signature for `sq_corr`). It gets the higher
   ceiling because it is the strictest joint gate of the four (two reversals plus
   raw and robust gain), so a skew-tilted sinusoid that picks up a spurious
   moderate trend still resolves rather than falling to `mixed_or_ambiguous`. The
   robust (leave-one-out) gates on the last two routes are essential: pure noise
   or a lone extreme bin/point produces reversals or a raw gain with near-zero
   surviving structure.
5. **`monotonic_nonlinear`** — Spearman is meaningfully stronger than Pearson
   (`|s| > 0.50` and `|s| − |p| > 0.20`), **or** the bin lack-of-fit diagnostic
   finds real curvature (`bin_lof_r2_gain > 0.05`, no mode gate — added because a
   smooth monotonic curve or step function can keep Pearson and Spearman close
   together despite genuine nonlinearity) on a strong monotone relationship.
   "Strong" is `|s| > 0.50`, **or** `max(|p|, |s|) > 0.50` with a moderate
   monotone trend (`|s| ≥ 0.20`) when the *robust* leave-one-bin-out gain also
   clears the floor — this catches a curve that is flat in the middle and steep
   in the tails (a cubic), whose compressed middle ranks pull Spearman below
   Pearson, while the robust gain keeps a single leverage bin from faking it.
   All routes require no Pearson/Spearman sign conflict.
6. **`near_linear`** — Pearson and Spearman are both strong (`|p| > 0.50` and
   `|s| > 0.50`) and close (`||p|−|s|| < 0.15`), without a sign conflict.
7. **`weak_or_no_relationship`** — all available measures are small (`|p|, |s| <
   0.20`, `dc < 0.20` when present, and the **robust** squared-value correlation
   `|sq_corr_robust| < 0.20` when computable — the robust value, not the raw one,
   so a heavy-tailed variable whose spurious `sq_corr` collapses once its few
   extreme squared points are dropped is correctly read as weak rather than held
   out of "no relationship" by an artifact).
8. **`mixed_or_ambiguous`** — fallback when none of the above matches.

Two design choices a reviewer should know:

- **Magnitude with a signed-conflict guard.** Most comparisons use absolute
  magnitudes (direction alone does not change the label). The exception is a
  **Pearson/Spearman sign conflict** (opposite signs, both absolute magnitudes
  `≥ 0.30`): it is a
  leverage signature, so it routes to `possible_outlier_or_leverage` (with trim
  evidence) or `mixed_or_ambiguous`, and is explicitly disqualified from
  `near_linear`/`monotonic_nonlinear`.
- **Deliberately conservative.** The cascade is tuned to avoid false "strong
  pattern" claims rather than to maximize sensitivity: borderline cases fall
  through to `mixed_or_ambiguous` rather than overclaiming. The thresholds are
  *conventions* (effect-size bands à la Cohen, then sanity-checked against the
  bundled synthetic generators in `datasets/simulations.py`), **not** parameters
  fit or optimized on a labeled benchmark — see the thresholds doc.
- **Exploratory, not confirmatory, in `scan_target`.** Scanning many predictors
  against a target applies **no multiple-testing correction** — by design, since
  this is a screening step, not confirmatory inference. Treat the ranked results
  as hypothesis-generating: a pattern worth acting on should be confirmed on held-out
  data or with a proper model, not taken as a tested finding because it topped a scan.

The single primary label is intentionally one-dimensional. A relationship has
several **orthogonal** properties one label cannot carry at once (its mean can
be linear while its variance grows and a few rows drive it), so alongside the
label CorrSleuth derives a small set of **secondary diagnostic axes** —
`mean_shape`, `variance_shape`, `dependence_type`, `outlier_sensitivity`,
`functional_direction` — as coarse categorical summaries of the numeric
diagnostics already computed (`derive_diagnostic_axes` in
`heuristics/classifier.py`). They are derived from the evidence, not read off
the label, so they stay orthogonal to it; each keeps its underlying number on
`result.diagnostics`. `variance_shape` in particular tests for
**heteroscedasticity** — a Koenker-studentized Breusch-Pagan test on the
linear-fit residuals for significance, with a Goldfeld-Quandt residual-variance
ratio for effect size and direction (`metrics/variance.py`), assessed only when
the mean is adequately linear so a curved mean's misspecification residuals are
not mistaken for changing noise variance. A third, independent check — the
edge-vs-middle ("bowtie") residual-variance ratio — catches a *symmetric*
variance pattern (spread high at both extremes of X and calm in the middle, or
the reverse) that Goldfeld-Quandt's low-vs-high split and Breusch-Pagan's
linear auxiliary regression are both blind to by construction (a bowtie's
low-x and high-x groups have similar variance, and its squared-residuals-vs-x
relationship is U/hill-shaped rather than linear); it reports
`edge_high_spread` / `center_high_spread`. `mean_shape` refines a curved
*monotone* mean into `smooth_curve` versus `step_or_threshold` (with a
`breakpoint_x`) using a single-breakpoint search (`metrics/shape.py`): a step's
segments are flat, so a two-level model fits as well as a two-line one, while a
smooth curve's segments are sloped. A strong monotone trend whose binned means
still reverse direction repeatedly (robustly) reads instead as
`oscillating_trend` — a trend with a superimposed wave (compound trend + periodic
residual), detected before the step/smooth split so the wave is not misread as a
step. A trend containing a genuine **level shift** reads `discontinuous_jump`:
the unconstrained two-line fit's boundary gap, in units of the noisier side's
residual sigma (`segment_jump_ratio`), survives localized refits around the
boundary — catching a jump that is huge against the noise but invisible on the
R²-scale diagnostics because the trend soaks up the variance. A continuous
kink, a smooth curve (whose chords displace only globally), a fold, and a
heavy tail's separation from the bulk are all excluded by construction and by
the joint gates. `dependence_type` can also read `two_group_shift`
(`metrics/mixture.py`): the pooled correlation is carried almost entirely by
the separation between two well-separated groups of rows — a high-variance
two-group split of the association-axis projection with an empty valley at the
boundary, a smaller group big enough to be a subpopulation rather than
leverage, and the within-group correlation collapsed. That is the
lurking-grouping-variable / mixture signature (the aggregation trap behind
Simpson-style reversals); since a flat threshold effect produces the same
joint distribution, the paired warning presents both readings.
`outlier_sensitivity` refines the
trim-sensitivity verdict with row-level Cook's distance (`metrics/influence.py`)
into `single_point_driven` versus `high_leverage_cluster`; because Cook's
distance has no 1%-trim blind spot, it can flag a leverage cluster the primary
label missed. See the [interpretation
guide](interpretation-guide.md#secondary-diagnostic-fields) for the full value
list.

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

`bootstrap=B` adds a **nonparametric percentile bootstrap** assessment of how
stable the result is under resampling. Each of the `B` replicates resamples the
already-cleaned paired rows with replacement and rebuilds a per-replicate
`CleanPair` — recomputing the constant/tie/unique-ratio/low-n state, the metrics,
the trim-sensitivity check, and the label. (It does **not** re-run raw-input
validation or the missing-data policy; those run once on the original data.)
Re-running the heuristic cascade — not just resampling the already-computed
metrics — on every replicate is deliberate: the stability estimate then reflects
the actual decision procedure (data-quality guards, leverage check, and the
label rule), not just the sampling variability of a single coefficient.

- **Percentile intervals.** For each requested metric, the 2.5th/97.5th
  percentiles of its bootstrap distribution form an approximate 95% interval.
- **Pattern stability.** Each replicate is re-labeled through the same cascade;
  `pattern_stability` is the fraction of replicates whose label matches the
  original. The cascade always evaluates at least the lite triple per replicate,
  so stability is meaningful for lite-expressible labels even when intervals are
  requested for a custom metric subset. **Caveat:** a *standard-only* label —
  currently `nonmonotonic_dependence`, whose distance-correlation route needs
  `mode="standard"` — can only be re-tested faithfully when `bootstrap_metrics`
  includes `distance_correlation` (or `"standard"`). With the default lite
  bootstrap the replicate cascade cannot reassess dCor, so stability is
  approximate for that label and CorrSleuth emits a warning saying so. The
  warning is conservative: the label's other three lite routes (`sq_corr`, the
  bin-reversal oscillation gate, and the single-bend gate) *are* recomputed per
  replicate in every mode, so a label driven by those routes is fully re-tested
  despite the warning.
- **m-out-of-n capping.** `max_n_for_bootstrap` caps the rows drawn per replicate
  for cost. Resampling fewer rows than the data contains widens the intervals
  (for a √n-rate statistic they become conservative by roughly `sqrt(n / m)`; for
  a bounded coefficient near ±1 this is a rough guide only, since the sampling
  distribution is non-normal there); a warning discloses this whenever the cap
  binds.

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

## 8. Validation and testing

The behavior described above is enforced by an automated test suite — run in CI
across Python 3.10–3.14 with a branch-coverage gate — not merely asserted here.

- **Property-based tests** (Hypothesis, `tests/test_property.py`) check invariants
  that must hold for *any* generated input, not just hand-picked cases: joint
  row-permutation invariance for the order-independent measures; constant input
  → `None`; Pearson/Spearman/Kendall staying within `[-1, 1]` and Chatterjee's ξ
  at or below 1; symmetry of the symmetric measures; and the forward/reverse
  consistency of ξ. (Distance correlation gets a single representative
  property check under an optional-dependency guard, not the full generated
  sweep.) ξ is *deliberately excluded* from the permutation-invariance check,
  because its seeded tie-break makes it non-invariant under ties (§3, §9) —
  encoding the
  limitation as a test rather than hiding it.
- **Synthetic data generators** (`make_relationship` in `datasets/simulations.py`)
  produce known data-generating processes — linear, monotone-log, U-shape,
  outlier-driven, and independent — used to confirm the *intended* label emerges
  for each shape, which is also how the cascade thresholds were sanity-checked.
- **End-to-end smoke tests** (`tests/test_smoke.py`) exercise the full public API
  (`profile_pair`, `scan_target`) and every render surface (text summary,
  Markdown, tidy frame, plot), so an integration-level break is caught even when
  the unit tests still pass.
- **Edge-case unit tests** across the suite cover constants, heavy ties, small /
  low-power *n*, Pearson sign flips, leverage, the missing-data policies, and the
  bootstrap interval/stability guards.
- **Numerical robustness.** The underlying scipy / scikit-learn / `dcor` calls are
  wrapped with explicit error handling, and degenerate inputs return a documented
  "not computable" result (`value=None`) rather than raising or propagating `NaN`.

## 9. Reproducibility

`random_state` (default 42) seeds every stochastic step — distance-correlation
downsampling, the mutual-information estimator, bootstrap resampling, and
Chatterjee's ξ tie-break — so repeated runs on the same input return identical
numbers. The one non-determinism a user can hit deliberately: under ties in its
sort variable, ξ depends on the (seeded) random tie-break, so it is reproducible
for a fixed `random_state` but not invariant to input row order.

## 10. Limitations and honest caveats

- **Heuristic, not inferential.** The primary labels come from threshold rules,
  not hypothesis tests; thresholds are documented conventions, not fitted or
  optimal, and are overridable. The one place a significance test enters is the
  `variance_shape` axis, which pairs a Breusch-Pagan p-value with an effect-size
  floor (§5) — deliberately, because at large *n* the test alone rejects for
  trivially small heteroscedasticity, so the effect-size floor is what gates the
  label.
- **Diagnostic, not causal.** No causal, treatment-effect, or model-specification
  claims. A strong association need not be predictively useful in a multivariate
  model.
- **Pairwise and numeric only.** No multivariate adjustment (confounding,
  partial correlation) and no categorical/mixed-type support.
- **`scan_target` is exploratory.** No multiple-testing correction is applied
  across the scanned columns; the ranked output is hypothesis-generating, and
  scanning many predictors will surface some strong-looking pairs by chance.
- **The 1% trim targets tail leverage.** The leverage check trims only the outer
  1% per variable, so it is most sensitive to a handful of extreme points; a
  mid-range leverage cluster larger than ~1% of the data can move Pearson
  without the *primary label* flagging it. The `outlier_sensitivity` axis
  partially covers this gap — its row-level Cook's distance has no 1%-trim blind
  spot and can flag a mid-range `high_leverage_cluster` the label missed (§5) —
  but the primary-label leverage rule itself remains trim-based. (See the
  trim-limitation note in the interpretation guide.)
- **Mutual information is unnormalized** (nats) — interpret relatively; the KSG
  estimator can also be slightly negative near independence (read as "near zero").
- **Chatterjee's ξ is noisier for low-cardinality sort variables** (random
  tie-break) and is bounded below 1 for discrete predictors even under perfect
  dependence.
- **The bootstrap assumes exchangeable / i.i.d. rows.** Like any row-resampling
  bootstrap, intervals and stability are not valid under strong serial
  dependence, clustering, or other non-exchangeable structure.
- **The shape diagnostics catch specific shapes, not all nonlinearity or
  nonmonotonicity.** `sq_corr` is tuned for magnitude/radial dependence
  (roughly, an even function of X and/or Y); oscillating shapes are caught by
  the separate bin-reversal route, which tolerates a *moderate* net trend (up to
  `max(|p|, |s|) < 0.50`) — so a sinusoid whose sampled range gives it a
  substantial net linear component still resolves, and only a **strong** net
  trend (`|s| ≥ 0.50`, e.g. an integer number of cycles riding a steep ramp)
  takes it out of this label (it then surfaces as `mean_shape =
  oscillating_trend` instead). At small `n` the ~10 bins cannot resolve many
  cycles (validated misses: 3–5 cycles at n=100 under heavy noise). See
  [shape-diagnostics-design.md](shape-diagnostics-design.md).
  `BIN_LOF_R2_GAIN_THRESHOLD`'s margin (0.05) is thinner than most cascade
  thresholds, so it leans more on the `simulations.py` regression coverage
  than on a single hand-picked value.
- **The symmetric-shape detectors assume roughly balanced support.** `sq_corr`
  (magnitude/radial) and the single-bend V route both center on the mean, so a
  strongly **skewed or one-sided heavy-tailed** variable degrades them: the
  centered squares fold and a few tail points inflate Pearson, so a U/V shape on
  such support can read `mixed_or_ambiguous` rather than nonmonotonic even though
  the dependence is real. When this happens the dependence is usually *not*
  silently dropped — the scan's "dependence may be understated" section surfaces
  it via the robust `sq_corr`, and deep-mode distance correlation / ξ / mutual
  information still fire — but the primary label is conservative. Inspect the
  scatter (and, for engineered `candidate = f(target)` data, the reverse
  direction) on skewed variables.
- **Mixture / two-group detection has a subpopulation-size floor.** The
  `two_group_shift` axis fires only for a clearly separated pair of groups
  (roughly ≥ 4 within-group standard deviations apart) where the smaller group is
  at least ~10% of the rows. A **sparse local subgroup** (say an 8% subpopulation
  carrying a relationship the other 92% lack) is below that floor and is *not*
  flagged — local/subgroup discovery is outside CorrSleuth's global-pairwise
  design, deliberately, to avoid a pattern-finding machine that overfits small
  subsets. Such a pair reads `mixed_or_ambiguous`/`weak`; inspect the scatter.
- **`variance_shape` (the value) can echo a leverage cluster rather than report
  an independent phenomenon.** It is gated against curvature artifacts (a
  curved mean suppresses it) but not against leverage artifacts: a
  high-leverage cluster can genuinely widen the Goldfeld-Quandt/bowtie ratio in
  the region it occupies, so `increasing_spread`/`decreasing_spread`/
  `edge_high_spread`/`center_high_spread` can co-occur with
  `outlier_sensitivity = single_point_driven`/`high_leverage_cluster` from the
  same rows. The *warning* corrects for this where it matters: when
  `n_influential_points >= 1`, the variance test is recomputed excluding the
  Cook's-flagged row(s), and if the signal vanishes on the remainder the
  warning attributes it to that same row rather than reporting independent
  evidence (if it survives exclusion, both warnings are kept, since a genuine
  cluster and genuine independent heteroscedasticity can coexist).
- **Always inspect the scatter.** Every label is a pointer to look, not a verdict.

## 11. References and further reading

- [interpretation-guide.md](interpretation-guide.md) — per-label meaning,
  typical metric patterns, and how to act on each label.
- [thresholds-and-rationale.md](thresholds-and-rationale.md) — every cut point,
  its value, and its justification.
- [shape-diagnostics-design.md](shape-diagnostics-design.md) — why
  `bin_lof_r2_gain` and `sq_corr` were added, and the misses they fix.
- [nonlinear-metrics-design.md](nonlinear-metrics-design.md)
  — why Chatterjee's ξ was selected for deep mode.
- Test suite (§8): `tests/test_property.py` (property-based invariants),
  `tests/test_smoke.py` (end-to-end API/render), and
  `corrsleuth/datasets/simulations.py` (`make_relationship` synthetic
  generators) — the executable evidence behind the claims above.
- Chatterjee, S. (2020). [*A new coefficient of correlation.*](https://par.nsf.gov/servlets/purl/10339804) JASA.
- Székely, G., Rizzo, M., & Bakirov, N. (2007). [*Measuring and testing
  dependence by correlation of distances.*](https://www.jstor.org/stable/25464608) Annals of Statistics.
- Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). [*Estimating mutual
  information.*](https://journals.aps.org/pre/pdf/10.1103/PhysRevE.69.066138) Physical Review E. (The KSG estimator used by scikit-learn.)
- Cohen, J. (1988). [*Statistical Power Analysis for the Behavioral Sciences.*](https://utstat.toronto.edu/~brunner/oldclass/378f16/readings/CohenPower.pdf)
  (Source of the effect-size magnitude bands.)
