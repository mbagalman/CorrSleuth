# Shape Diagnostics — Design Note

> This note evaluates candidate fixes for four heuristic-cascade misses found
> during user testing on synthetic data with known relationship shapes, picks
> the two worth implementing now (`bin_lof_r2_gain`, `sq_corr`), and documents
> why the fourth (periodic/cyclical dependence) was deliberately deferred. It is
> the "why these diagnostics" companion to the [methodology
> doc](methodology.md), in the same spirit as the [nonlinear metrics design
> note](nonlinear-metrics-design.md).
>
> **Update:** the deferred periodicity detector has since been implemented as
> `bin_reversal_count` — both of the original blockers were resolved by later
> blind-test evidence and a dedicated validation sweep. See the amendment at
> the end of section 4 for what shipped and how it differs from the sketch
> deferred here.

## Context

A user ran `profile_pair(mode="standard")` across a synthetic dataset with
known relationship shapes and found four mislabels:

| Shape | Labeled | Should be |
|---|---|---|
| Exponential, logarithmic (smooth monotonic curves) | `near_linear` | `monotonic_nonlinear` |
| Step / threshold function | `near_linear` | `monotonic_nonlinear` |
| Sinusoidal / periodic | `mixed_or_ambiguous` | *(no existing label fits well — see below)* |
| Circular (points scattered around a ring) | `weak_or_no_relationship` | `nonmonotonic_dependence` |

Root cause: the cascade in `corrsleuth/heuristics/classifier.py` has exactly
one nonlinearity test (the Spearman-vs-Pearson gap) and one nonmonotonicity
test (distance correlation exceeding a floor while Pearson/Spearman stay
below a ceiling). Both are too narrow for these four shapes — verified
directly against the data, not assumed:

- **Smooth monotonic curves and step functions** can keep Pearson and
  Spearman close together (measured gap 0.03–0.10 on realistic, non-rigged
  data) even though real curvature exists — confirmed via a quadratic-fit R²
  gain of +0.06 to +0.14 over a linear fit.
- **A true circular relationship** structurally caps distance correlation
  around dCor ≈ 0.19–0.20, even in the noiseless limit (confirmed by
  simulating a clean circle with the `dcor` package directly) — not a
  threshold-tuning problem.
- **A sinusoid sampled over a finite number of cycles** carries a real,
  non-zero linear/monotonic component (`corr(x, sin(x))` over ~2 cycles is
  intrinsically ≈ −0.39, confirmed by direct simulation, even with zero
  noise) — none of the existing metrics cleanly separate this from a
  moderate monotonic trend.

The evaluation criteria mirror the nonlinear-metrics note:

| Criterion | Treatment in this note |
|---|---|
| Dependency weight | Hard preference — no new dependency, or lite-computable |
| Install friction | Same — ideally no mode gate at all |
| Performance | Cheap enough to run on every pair, every mode |
| Interpretability | Must be explainable in one sentence |
| Stability | Must behave well at small `n` and show a wide margin between "real signal" and "null" on the bundled scenarios |
| Statistical grounding | Prefer an established technique over an invented heuristic |

## Candidates evaluated

| Diagnostic | New deps | Compute | Symmetry | Range | Verdict |
|---|---|---|---|---|---|
| Bin lack-of-fit R² gain (`bin_lof_r2_gain`) | None | O(n log n) | Directional (X→Y) | (−ε, ~1] | **Implement** |
| Squared correlation (`sq_corr`) | None | O(n) | Symmetric | [−1, 1] | **Implement** |
| Mutual information as a cascade signal | None (already computed) | — | Symmetric | unbounded (nats) | **Implement as warning only** |
| Bin-mean reversal count (periodicity detector) | None | O(n log n) | Directional | integer ≥ 0 | **Defer** *(since implemented — see §4 amendment)* |

### 1. Bin lack-of-fit R² gain — *primary candidate*

Sort by X, split into ~10 equal-frequency bins, and compare the R² of a
bin-mean model of Y to the R² of a straight-line fit. This is the classical
lack-of-fit F-test using grouped X as a stand-in for replicates (Neter,
Kutner, Nachtsheim & Wasserman, *Applied Linear Statistical Models*), not an
invented heuristic — a bin model can only match or beat a line, so a positive
gain is evidence of curvature a line doesn't capture.

**Pros**
- No new dependency, no mode gate — pure numpy/scipy, cheap enough for `lite`.
- Catches both target misses in one diagnostic: smooth monotonic curvature
  (exponential, logarithmic) and step/threshold functions, which have very
  different shapes but the same failure mode (a small Spearman-Pearson gap).
- Established statistical technique, not a novel invention — a plus for a
  library aimed at statisticians.
- Wide margin on the bundled test scenarios: genuinely linear data measures
  ≈ −0.01 to −0.006 (heteroscedastic-but-linear-in-mean case); real curvature
  measures ≥ 0.06. The floor of 0.05 sits in that gap.

**Cons**
- The margin (0.05) is thinner than most other cascade thresholds
  (0.15–0.20), because the smallest true positive found (a realistic
  logarithmic curve, ≈ 0.06) sits close to the largest true negative
  (heteroscedastic linear data, ≈ −0.006 to +0.003 for a few outliers). This
  threshold leans more heavily on the `simulations.py` regression coverage
  (linear data checked across many seeds) than most.
- The bin count is itself a small hyperparameter (adaptive: `clip(n // 10, 5,
  20)`); an unusual `n` at the edges of that range could behave slightly
  differently, though the floor (`n >= 50`, guaranteeing ≥ 5 bins of ≥ 10
  points) keeps bin means reasonably stable.
- Directional in effect (bins are formed on X, not symmetrized), matching how
  it's used: it only ever feeds `monotonic_nonlinear`, where X is already the
  designated predictor.

**Verdict: Implement**, as an additional route into `monotonic_nonlinear`
alongside the existing Spearman-Pearson gap rule.

### 2. Squared correlation (`sq_corr`)

`corr(X², Y²)` — catches dependence carried in magnitude rather than sign or
rank. For a circular relationship (`X² + Y² ≈ const`), knowing `X` constrains
`|Y|` but not `sign(Y)`, so Pearson/Spearman/distance correlation on the raw
values are all near zero while `sq_corr` is strongly (typically negatively)
correlated.

**Pros**
- No new dependency, no mode gate — a single `pearsonr` call on squared
  values.
- Fixes the circular-data miss that distance correlation itself cannot fix
  (dCor is structurally capped ≈ 0.2 for this shape, confirmed by direct
  simulation of a clean circle).
- Validated broadly, not just on the one flagged pair: across all 393
  column-pairs in the bundled test dataset where Pearson and Spearman are
  already weak (candidates for `weak_or_no_relationship`), exactly 4 show
  `|sq_corr| > 0.30` — the circular pair, the two already-correctly-labeled
  U-shape/inverted-U pairs, and one piecewise pair — and every other pair,
  including true null pairs, falls below 0.11. Wide, clean margin.
- Symmetric and simple to explain: "the squared values are correlated even
  though the raw values aren't."

**Cons**
- Tuned for a specific family of shapes — roughly, dependence that's an even
  function of X and/or Y (circles, U-shapes, some magnitude/variance-linked
  relationships). It is not a general nonmonotonic-dependence detector the
  way distance correlation is; a shape with no magnitude signature (e.g. some
  oscillating shapes) will not trigger it. See the periodicity discussion
  below.
- Also fires for classic U-shapes, which already reach
  `nonmonotonic_dependence` via distance correlation in `mode="standard"` —
  redundant there, but this is what makes U-shapes newly detectable in `lite`
  and `deep` mode too (a genuine, if incidental, improvement: those modes
  previously had no route to `nonmonotonic_dependence` at all).

**Verdict: Implement**, as an additional route into `nonmonotonic_dependence`
alongside the existing distance-correlation rule, and into the ceiling check
for `weak_or_no_relationship` (mirroring the existing distance-correlation
ceiling, so a moderate `sq_corr` also falls through to `mixed_or_ambiguous`
rather than "no relationship").

### 3. Mutual information as a cascade signal

`mutual_information` (KSG estimator) is already computed in `mode="standard"`
but was not read anywhere in the label cascade before this change. Measured
on the bundled test data: ≈ 0.000 for every true-null pair, ≥ 0.6 for every
real-dependence pair tested (including the circular case, 1.226 nats) — a
strong binary "real dependence exists" signal.

**Pros**
- Already computed — no new cost in `mode="standard"`.
- Very clean separation between null and real dependence in the cases
  checked.

**Cons**
- Raw KSG mutual information is unbounded and scale-sensitive (depends on
  `n`, the estimator's `k`, and dimensionality); a fixed threshold on the raw
  value isn't safely portable across sample sizes without more calibration
  than one dataset can support.
- Even after transforming to a bounded, correlation-like scale via
  `sqrt(1 - exp(-2*MI))` (the Gaussian-equivalent-correlation identity), the
  transformed value saturates similarly (≈ 0.85–0.99) across very different
  dependence *strengths* and *shapes* in the cases checked — it's a good
  detector of "is there dependence at all," not a precise discriminator of
  magnitude or shape. Not precise enough to drive a primary label decision.

**Verdict: Implement as a warning only**, generalizing the existing
Chatterjee's-ξ "this label may understate the relationship" warning
(`detect_metric_warnings`) to also consider mutual information on the shared
transformed scale. This is a defense-in-depth safety net, not the primary fix
mechanism — the primary fixes are `bin_lof_r2_gain` and `sq_corr` above.

### 4. Bin-mean reversal count (periodicity detector) — *deferred, then implemented*

> The section below is the original "why deferred" reasoning, preserved for the
> record. It was **superseded** by the amendment at the end of the section: both
> blockers were later resolved and the detector shipped as `bin_reversal_count`.

The natural extension of the bin-lack-of-fit idea: after confirming
`bin_lof_r2_gain` clears a floor (there's real structure beyond a line), count
sign changes in the sequence of bin means. Zero or one reversal is consistent
with a monotonic curve or a single-bend U-shape; multiple reversals suggest
oscillation.

**Why this would help.** The periodic/sinusoidal miss is not fixed by
`bin_lof_r2_gain` or `sq_corr` above (a sinusoid sampled over a finite range
carries a real, non-zero linear component that keeps it out of the
nonmonotonic-dependence ceiling, and its magnitude signature is weak).
Calling a 2-cycle sine wave `nonmonotonic_dependence` — the same label used
for a simple U-shape — would undersell the cyclical structure and could
mislead an analyst into looking for one inflection point rather than
periodicity, so this would need a **new** primary label
(e.g. `cyclical_or_oscillating_dependence`), not just another route into an
existing one.

**Why it's deferred rather than shipped alongside the other two:**

- **Reversal count alone is not trustworthy.** On the bundled null pair
  (`noise_a`/`noise_b`), bin means show 6 "reversals" — pure noise oscillates
  around a near-zero baseline just as much as a real 2-cycle signal does.
  Reversal count is only meaningful *conditional on* `bin_lof_r2_gain` also
  clearing a floor, which adds a second threshold to calibrate jointly rather
  than independently — more surface area than the two additions above.
- **No single-dataset validation of margins.** `bin_lof_r2_gain` and
  `sq_corr` above both showed clean, wide-margin separation between real
  signal and null on the one bundled dataset available. A periodicity
  threshold would need validation across varying cycle counts, sample sizes,
  and noise levels before it could be trusted the same way — one dataset
  isn't enough evidence.
- **A new primary label is cross-cutting.** Per the project's established
  process (this note's own precedent, and the nonlinear-metrics note before
  it), a new label needs entries in `_EXPLANATIONS`/`_RECOMMENDATIONS`
  (`heuristics/explanations.py`), a `STANDARD_ONLY_LABELS` decision, a
  cascade-priority placement decision, and updates across
  `interpretation-guide.md`, `methodology.md`, `thresholds-and-rationale.md`,
  and the README — a materially larger change than extending two existing
  labels.

**Verdict: Defer.** Worth a follow-up design note once the periodicity
threshold has been validated across more than one synthetic scenario.

**Amendment — implemented.** Both blockers above were later resolved, and the
detector shipped as `bin_reversal_count` (computed by `compute_bin_lof` in
`metrics/shape.py`, from the same bins as `bin_lof_r2_gain`):

- *"Reversal count alone is not trustworthy"* — confirmed, and resolved by the
  joint gate this section anticipated: the count only fires alongside
  `bin_lof_r2_gain > OSCILLATION_BIN_LOF_FLOOR` (0.3, deliberately far above
  the 0.05 curvature threshold). On genuinely blind test data, pure noise
  measured *more* raw reversals than a real 2.5-cycle sinusoid (16 vs. 4) with
  a bin-fit gain 15× smaller (0.057 vs. 0.826) — the floor, not the count, is
  what excludes noise. The count itself also got more robust than the sketch
  here: turning points are confirmed with hysteresis (a reversal counts only
  after the bin means move ≥ 15% of their range back from the last extreme),
  which a comparison sweep showed eliminates the false "oscillating" reads a
  per-step de-noising filter still allowed on noisy single-bend shapes.
- *"No single-dataset validation of margins"* — resolved by a 2,080-run sweep
  (13 shapes × 4 sample sizes × 4 noise levels × 10 seeds, plus a sinusoid
  grid over 6 cycle counts): zero false positives among the negative controls,
  detection 10/10 in every sinusoid cell except 3–5 cycles at n=100 under
  heavy noise (where ~10 bins genuinely cannot resolve the cycles), and a
  known-shape check that a U-shape reads exactly 1 reversal. See
  `OSCILLATION_MIN_REVERSALS` / `OSCILLATION_BIN_LOF_FLOOR` in
  `thresholds-and-rationale.md`.
- *"Would need a new primary label"* — resolved differently, and more cheaply,
  than this section costed out: the secondary-axis taxonomy (added after this
  note was written) carries the cyclical nuance as
  `dependence_type = "oscillating"`, so the primary label extends the existing
  `nonmonotonic_dependence` (a third, lite-computable route alongside `dc` and
  `sq_corr`) rather than introducing `cyclical_or_oscillating_dependence`. An
  analyst sees "real nonmonotonic dependence" at the label level and "it
  oscillates — look for periodicity, not a single inflection point" on the
  axis.

One known limitation carries over from the "Context" section: a sinusoid whose
sampled range gives it a substantial net linear component (integer cycle
counts measured |ρ| up to ~0.49) still fails the rule's monotone ceiling
(`p, s < 0.25`) and falls to `mixed_or_ambiguous` — the oscillation route
deliberately reuses the existing rule-4 gate rather than loosening it.

## Recommendation

**Implement `bin_lof_r2_gain` and `sq_corr`** as two new no-mode-gate shape
diagnostics in `corrsleuth/metrics/shape.py`, wired into the existing
`monotonic_nonlinear` and `nonmonotonic_dependence` cascade rules as
additional routes (not new labels). **Extend the existing Chatterjee's-ξ
warning mechanism** to also consider mutual information, as a coarse
defense-in-depth safety net. **Defer** a periodicity diagnostic and any new
`cyclical_or_oscillating_dependence` label to a follow-up design note.
*(Update: the periodicity diagnostic was subsequently implemented as
`bin_reversal_count` — see the §4 amendment. It did not need a new primary
label after all: it routes into `nonmonotonic_dependence` and is described by
the `dependence_type = "oscillating"` secondary axis.)*

Why extend existing labels rather than add new ones for the first two cases?
Both `bin_lof_r2_gain` and `sq_corr` showed clean, wide-margin separation
between real signal and null on the bundled test data, and the shapes they
catch (smooth monotonic curves, step functions, circular/radial dependence)
are semantically identical to what `monotonic_nonlinear` and
`nonmonotonic_dependence` already mean — "monotonic but not linear" and "real
dependence the monotone measures miss," respectively. No new concept needs
introducing to an analyst who already understands those two labels. The
periodic case was originally judged different in kind — but the later
`dependence_type = "oscillating"` axis (added by the secondary-diagnostic
taxonomy) carried that nuance without a new primary label; see the §4
amendment.

## Implementation summary

This change ships:

- `corrsleuth/metrics/shape.py` exposing `compute_bin_lof_r2_gain(pair)` and
  `compute_squared_correlation(pair)`. Both return a `MetricResult` but are
  kept out of `metrics_map` in `api.py` — they feed the cascade and
  `result.diagnostics`, but never appear in the public metrics table
  alongside primary association coefficients like Pearson or distance
  correlation. *(Update: `compute_bin_lof_r2_gain` was later renamed to
  `compute_bin_lof` and now returns a dict of `bin_lof_r2_gain` plus
  `bin_reversal_count` — see the §4 amendment.)*
- New cascade constants in `classifier.py`: `BIN_LOF_R2_GAIN_THRESHOLD = 0.05`
  and `SQ_CORR_THRESHOLD = 0.35` (the latter matches
  `NONMONOTONIC_DC_THRESHOLD` for consistency).
- `monotonic_nonlinear` and `nonmonotonic_dependence` cascade rules extended
  with OR-branches (see [thresholds-and-rationale.md](thresholds-and-rationale.md)
  and [methodology.md](methodology.md#5-the-label-cascade) for the exact
  conditions); `weak_or_no_relationship`'s ceiling extended to mirror the
  `sq_corr` floor, so a moderate magnitude-linked signal falls through to
  `mixed_or_ambiguous` rather than being called "no relationship" — the same
  conservative buffer distance correlation already gets.
- `detect_metric_warnings` generalized to consider mutual information
  (transformed via `sqrt(1 - exp(-2*MI))`) alongside Chatterjee's ξ.
- `corrsleuth/metrics/bootstrap.py`'s per-replicate loop computes both new
  diagnostics unconditionally (they have no mode gate) so bootstrap replicates
  can reproduce labels that only the new routes can assign — otherwise
  `pattern_stability` would read as artificially low for exactly the labels
  this change fixes.
- Four new realistic (non-rigged) `shape_type`s in
  `corrsleuth/datasets/simulations.py`: `exponential_monotonic`,
  `logarithmic_monotonic`, `threshold_step`, `circular`. (The existing
  `monotonic_log` scenario uses a deliberately skewed X distribution that
  guarantees the old gap rule fires — not representative of ordinary data,
  which is why this class of miss went unnoticed until real user testing; it
  is kept as-is for regression safety, alongside the new realistic variant.)
- Tests in `tests/test_metrics.py` (reference-implementation checks for both
  diagnostics, constant-input/min-n guards) and `tests/test_heuristics.py`
  (one assert-label test per new shape, plus a 20-seed regression check that
  genuinely linear data stays `near_linear` — guarding
  `BIN_LOF_R2_GAIN_THRESHOLD`'s thinner-than-usual margin).

## Open questions for the reviewer

1. **`STANDARD_ONLY_LABELS`.** `nonmonotonic_dependence` is no longer
   categorically standard-only (the `sq_corr` route is lite-computable), but
   it stays in `STANDARD_ONLY_LABELS` for now because distinguishing
   post-hoc which route produced the label (dc vs. `sq_corr`) is out of scope
   for this change — the bootstrap-stability and `.explain()` warnings are
   worded to note this can be conservative. Worth a follow-up if it proves
   confusing in practice.
2. **Should `threshold_step` get its own label?** A step/threshold function is
   technically monotonic-but-not-linear, so `monotonic_nonlinear` is
   defensible, but a business user might want to know specifically "this
   looks like a threshold effect" rather than "some general nonlinear
   monotonic curve." Deferred rather than adding a third label in the same
   change; open to revisiting.
3. **Periodicity. Resolved — since implemented.** The thresholds were
   subsequently validated across many datasets (a 2,080-run sweep) and the
   detector shipped as `bin_reversal_count`; see the §4 amendment. It reached
   `nonmonotonic_dependence` via a third cascade route plus a
   `dependence_type = "oscillating"` axis value, rather than the separate
   design note / new primary label this question anticipated.

## Out of scope for this change

- A reversal-count periodicity diagnostic. *(Subsequently implemented as
  `bin_reversal_count` — but as a route into the existing
  `nonmonotonic_dependence` label plus a `dependence_type = "oscillating"`
  secondary axis, not the standalone `cyclical_or_oscillating_dependence`
  label sketched here. See the §4 amendment.)*
- Distinguishing which cascade route (dc vs. `sq_corr`, or rank-gap vs.
  bin-lof) produced a given label, for bootstrap-gating or explanatory
  purposes.
- A dedicated `step_or_threshold` label.
- Bootstrap *interval* reporting for `bin_lof_r2_gain`/`sq_corr` themselves —
  they are diagnostic-only inputs to the cascade, not user-facing metrics, so
  they are recomputed per bootstrap replicate to keep the cascade correct but
  are not added to `bootstrap_intervals`.
