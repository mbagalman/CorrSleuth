# Changelog

All notable changes to CorrSleuth are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - Unreleased

### Fixed
- **Tied X values no longer make shape diagnostics depend on input row order.**
  `bin_lof_r2_gain` (and its robust/reversal companions) and the segmentation
  diagnostics binned sorted X by *position*, so tied X values could land in
  different bins depending on the order rows arrived in — a pure row permutation
  of identical data could flip the diagnostics and hence the label. Bin
  boundaries and segmentation breakpoints now never split a run of equal X
  (tie-safe binning; byte-identical to the old binning on continuous, tie-free
  X, so calibration is unchanged), and when X has too few distinct values to
  form the minimum bin count the diagnostics are withheld rather than reported
  on order-dependent bins. Guarded by a Hypothesis permutation-invariance
  property test.
- **Robust squared correlation (`sq_corr_robust`) is now re-estimated on each
  retained sample.** The leave-the-extremes-out deletion values previously
  reused the full-sample means when centering before squaring; each deletion now
  re-centers on the retained points, as a deletion estimate must. Re-running
  `validation/sq_corr_sweep.py` confirms the change is calibration-neutral, and
  a full blind-corpus diff shows no label changes.
- **Mutual information now respects discreteness — symmetrically and
  independent of encoding.** The continuous KSG estimator misestimates
  low-cardinality data, so each column is classified into three states (a
  cardinality/repetition test, deliberately not a whole-number test, so the same
  categories encoded `0…19` or `0.0…1.9` give the same MI). **Discrete** — at
  most 20 distinct levels, every observed level appearing at least twice —
  dispatches to the matching scikit-learn estimator:
  `mutual_info_regression` with `discrete_features` for a discrete feature,
  `mutual_info_classif` on integer-coded levels for a discrete target; both
  mixed cases run the same discrete–continuous estimator, so the reported MI is
  symmetric in X and Y (as MI must be). **Unestimable** — low-cardinality
  mixing repeated and singleton levels — withholds MI with a warning: the
  discrete estimator silently discards singleton-level observations (it could
  read a near-deterministic relationship as MI ≈ 0) and the continuous
  estimator misestimates tied data (its estimate there is not even
  relabeling-invariant). **Continuous** — everything else, including
  high-cardinality integers like ages and small all-distinct samples — is
  unaffected. In the bootstrap the
  classification is made once on the full original data and held fixed across
  replicates, so one bootstrap distribution never mixes two different
  estimators.
- **A binding bootstrap cap is now disclosed as affecting pattern stability,
  not only interval width.** With `max_n_for_bootstrap` capping replicates at m
  < n rows, each replicate's label is recomputed on the m-row resample, so
  `pattern_stability` measures the stability of the *m-sample* decision
  procedure — noisier than the full sample, which near a heuristic threshold can
  understate the full-sample label's stability. The cap warning now says so,
  `BootstrapStability.sample_size` records the per-replicate size actually
  used, and the README/methodology framing of capped intervals was tightened
  (m-out-of-n bands are conservative approximations; uncapped intervals are
  full-size but still approximate, never exact).
- **Complex-valued columns are now rejected instead of silently truncated.**
  `pandas.api.types.is_numeric_dtype` treats complex dtypes as numeric, so a
  complex column passed the validation gate and was then cast to `float`,
  discarding the imaginary part (with a `ComplexWarning`) and profiling only the
  real axis. `profile_pair` and `scan_target` now raise `InputError` for a
  complex `x`/`y`/target, and complex candidates are excluded from auto-selection
  (or skipped with a dedicated `ComplexDtype` error type when named explicitly,
  so report consumers can tell complex rejection apart from `NonNumeric`).
  Cast to a real dtype yourself (real part or magnitude) if that projection is
  what you intend.
- **Directional sign conflicts are no longer mislabeled as agreement.** When
  Pearson and Spearman point in opposite directions with comparable magnitude
  (e.g. +0.8 vs −0.8, a leverage signature), the cascade compared *absolute*
  magnitudes, so the gap read as 0 and the pair was labeled `near_linear` with a
  `disagreement_score` of ~0. The cascade now detects the signed conflict:
  such pairs are `possible_outlier_or_leverage` (with leverage evidence) or
  `mixed_or_ambiguous`, never `near_linear`/`monotonic_nonlinear`, and the
  `disagreement_score` uses the signed Pearson−Spearman difference so it
  reflects the conflict instead of hiding it. The `explain()` text for a
  sign-conflict leverage result now describes the *direction* conflict rather
  than claiming Pearson is "much stronger" than the rank metrics (which is false
  when they are equally strong but opposite in sign).
- **Bootstrap stability no longer over-assigns leverage labels.** Trim
  sensitivity is now recomputed per bootstrap replicate instead of assuming
  `outlier_sensitivity_unavailable`. Previously, a trim-stable `near_linear`
  relationship with a large Pearson–Kendall gap had every resample labeled
  `possible_outlier_or_leverage` — a label the original profile explicitly
  rejected — collapsing pattern stability to ~0.
- **Chatterjee's ξ calibration under ties.** Ties in the sort variable are now
  broken with a seeded random permutation (Chatterjee's prescription) instead of
  by the response. The old Y-based tie-break leaked the response into the
  ordering and inflated ξ — an independent binary `X` against normal `Y`
  produced ξ ≈ 0.99 (a false functional-dependence signal); it is now ≈ 0.
  `compute_chatterjee_xi[_reverse]` take a `random_state` (seeded from
  `profile_pair`), so values are reproducible for a given input; under ties they
  are no longer invariant to input row order (they were spuriously invariant
  before). ξ remains usable for discrete/low-cardinality variables.
- **Bootstrap pattern stability for custom metric sets.** Pattern stability now
  always evaluates the label cascade on at least the lite triple
  (Pearson/Spearman/Kendall) per replicate, decoupled from `bootstrap_metrics`
  (which still controls only which *intervals* are reported). Previously a
  custom subset like `bootstrap_metrics=["pearson"]` made every replicate label
  `not_computable`, reporting 0.0 stability for an obviously stable
  relationship.
- `biweight_midcorrelation` now uses the raw median absolute deviation
  (`scale=1.0`) in its Tukey weights, matching the canonical biweight
  midcorrelation (Wilcox; Langfelder & Horvath 2012). It previously used the
  normal-consistent MAD, which pushed the outlier-rejection cutoff out to ~13
  MADs and made the estimator less outlier-resistant than its name implies.
  It also applies the 9-MAD rejection indicator *per variable* (a point beyond
  the cutoff in x has only its x-weight zeroed, still contributing its y-weight),
  as the canonical estimator prescribes, rather than dropping the whole row
  whenever either variable rejects it. This changes the metric's numeric output
  for non-degenerate data.
- **Smooth monotonic curves and step functions were mislabeled `near_linear`.**
  The cascade's only nonlinearity test — the Spearman-vs-Pearson gap — can stay
  small for exponential/logarithmic curves and step/threshold functions over an
  ordinary (non-rigged) X range, even though real curvature exists. A new
  degrees-of-freedom-adjusted equal-frequency-bin lack-of-fit diagnostic
  (`bin_lof_r2_gain`, see `corrsleuth/metrics/shape.py`) now provides an
  alternate route into `monotonic_nonlinear` for these cases. The df adjustment
  keeps the gain's null expectation at ~0 regardless of `n`, so ordinary
  noisy-linear data — including a bivariate normal at moderate ρ — is not read as
  curved; the threshold-calibration sweep lives in `validation/bin_lof_sweep.py`.
  No mode gate — runs in `lite` too. This bin-LoF route also no longer requires a
  *strong* Spearman: a monotone curve that is flat in the middle and steep in the
  tails (e.g. a cubic) depresses Spearman below Pearson, so a moderate Spearman
  with strong Pearson and confirmed curvature previously fell through to
  `mixed_or_ambiguous`. It now promotes to `monotonic_nonlinear` when the *robust*
  (leave-one-bin-out) bin-LoF gain also clears the floor — so a single leverage
  bin cannot fake the promotion (`validation/curvature_promotion_sweep.py` shows
  zero incremental false positives on linear/leverage families).
- **Circular/radial dependence was mislabeled `weak_or_no_relationship`.** A
  true circular relationship (points scattered around a ring) structurally
  caps distance correlation around dCor ≈ 0.2, even noiseless, so it never
  cleared the `nonmonotonic_dependence` floor. A new squared-value correlation
  diagnostic (`sq_corr = corr((X−x̄)², (Y−ȳ)²)`, the correlation of the
  mean-centered squares, `corrsleuth/metrics/shape.py`) now provides an alternate
  route into `nonmonotonic_dependence`, and into the `weak_or_no_relationship`
  ceiling check, for this shape. Centering before squaring makes it
  translation-invariant, so it catches a ring wherever it is centered, not only
  at the origin. No mode gate — as a
  side effect, classic U-shapes are now also detectable via `sq_corr` in
  `lite` and `deep` mode, not only `mode="standard"`.
  The `sq_corr` routes additionally require a **leave-the-top-out robust value**
  (`sq_corr_robust`, the smallest `|sq_corr|` after dropping the few points most
  extreme in either squared variable) to clear an asymmetric lower floor
  (`SQ_CORR_ROBUST_FLOOR = 0.20`): a heavy-tailed variable — in a scan, the
  target — can manufacture a raw `|sq_corr|` over the threshold with a handful of
  extreme squared values, mislabeling an independent predictor magnitude-linked.
  The robust value collapses in that case; a genuine magnitude link is spread
  over many points and keeps it high. The floor is lower than the raw threshold
  because a genuine link, once firing, stays well above where an artifact
  collapses — chosen (see `validation/sq_corr_sweep.py`) to preserve essentially
  all circle/U-shape detections while removing the large majority of the
  artifacts; a rare artifact whose *bulk* correlation survives the drop is
  indistinguishable from a weak real link and remains.
  See `docs/shape-diagnostics-design.md` for the full investigation
  (including why a periodic/cyclical case is deliberately deferred).
  `result.explain()` now describes the mechanism that actually fired — magnitude
  (`sq_corr`), oscillation (the bin-reversal gate), a closed loop, or distance
  correlation — instead of always crediting distance correlation, which for a
  circle sits below its own floor and does not drive the label.
- **Oscillating/periodic dependence was mislabeled `weak_or_no_relationship`
  outside `mode="standard"`.** A sinusoid over a few cycles keeps Pearson,
  Spearman, *and* the magnitude diagnostic `sq_corr` all near zero, and even
  distance correlation reads only marginally above its floor — so in `lite`
  and `deep` mode a strong deterministic function was actively undersold as
  "no relationship" (deep mode's Chatterjee-ξ warning fired, but the primary
  label was still wrong). A new bin-mean reversal count
  (`bin_reversal_count`, computed by `compute_bin_lof` in
  `corrsleuth/metrics/shape.py` from the same bins as `bin_lof_r2_gain`, with
  range-scaled hysteresis so noise wiggle is not counted as a turn) now
  provides a third, lite-computable route into `nonmonotonic_dependence`:
  `bin_reversal_count >= OSCILLATION_MIN_REVERSALS` (2) jointly with
  `bin_lof_r2_gain > OSCILLATION_BIN_LOF_FLOOR` (0.15, calibrated for the
  df-adjusted gain via `validation/bin_lof_sweep.py`). The joint gate is
  essential — pure noise produces *more* raw reversals than a real sinusoid
  (16 vs. 4 measured on blind test data) but a bin-fit gain ~15× smaller —
  and was validated over a 2,080-run sweep (13 shapes × sample sizes × noise
  levels × 10 seeds: zero false positives) before the thresholds were locked.
  The oscillation gate additionally requires a **leave-one-bin-out robust gain**
  (`bin_lof_r2_gain_robust`, the smallest gain obtained by dropping any single
  bin) to clear the same floor: a genuine oscillation is spread across many bins
  and barely moves, but a lone extreme Y in one bin — most likely when the
  target is heavy-tailed, as in a scan — can push the raw gain over the floor
  and manufacture a false oscillation on an independent predictor. The robust
  gain collapses in that case, so it is not fooled; the raw gain still drives the
  (rank-trend-gated) curvature route, where curvature legitimately concentrates
  in the extreme bins, so its calibration is unchanged (verified by the
  heavy-tailed-Y section added to `validation/bin_lof_sweep.py`).
  A new `dependence_type = "oscillating"` axis value distinguishes the
  cyclical case from a single-bend U-shape (which reads exactly 1 reversal),
  so the label says "real nonmonotonic dependence" and the axis says "look
  for periodicity, not one inflection point". Adds a `sinusoidal` shape to
  `make_relationship()`. Bootstrap replicates recompute the reversal count,
  so `pattern_stability` fully re-tests oscillation-driven labels in every
  mode. This closes the periodic/cyclical case deliberately deferred in
  `docs/shape-diagnostics-design.md` (see the amendment there) and
  substantially resolves the deep-mode open question in
  `docs/nonlinear-metrics-design.md`. `compute_bin_lof_r2_gain` was renamed
  to `compute_bin_lof` and now returns a dict of both bin diagnostics
  (internal `corrsleuth.metrics` API; the public `profile_pair`/`scan_target`
  surface is unchanged).
- **The heteroscedasticity warning could read as a second, independent
  problem when it was really the same leverage row `outlier_sensitivity`
  already flagged.** A single influential row can simultaneously manufacture
  or mask a correlation *and* skew Goldfeld-Quandt/bowtie's group variances,
  producing two independent-sounding warnings about one root cause. When
  `n_influential_points >= 1`, `profile_pair` now recomputes heteroscedasticity
  excluding the Cook's-flagged row(s) (`compute_heteroscedasticity_excluding`,
  reusing the same arithmetic as `compute_heteroscedasticity` — no new
  statistic); if the variance signal vanishes on the remainder, the warning is
  reworded to attribute it to that same row instead of reporting it as
  independent evidence. If the signal survives exclusion (a genuine leverage
  cluster and genuinely independent heteroscedasticity can coexist), both
  warnings are still reported unchanged. No added computation when
  `n_influential_points` is 0 or unavailable (the common case). New
  `compute_influential_mask` (`corrsleuth/metrics/influence.py`) exposes the
  boolean Cook's-flagged-row mask for this purpose.

### Changed
- **`mode="deep"` is now a strict superset of `mode="standard"` and requires the
  `corrsleuth[standard]` extras** (breaking change). In 0.1.0, deep computed only
  the robust-Pearson family and Chatterjee's ξ and ran on the base install. It
  now *also* computes distance correlation and mutual information — so a deep
  profile carries the full metric set — and therefore raises
  `OptionalDependencyError` (naming *deep* mode) when `dcor`/`scikit-learn` are
  absent, exactly as standard mode does. This removes a confusing gap where the
  most thorough-sounding mode silently omitted the standard-mode metrics, and it
  makes two previously-unreachable deep-mode signals live: a circle now reports
  `dependence_type=closed_loop_or_multivalued` with
  `functional_direction=neither_direction` alongside its distance correlation. If
  you called `mode="deep"` on a base install, run
  `pip install corrsleuth[standard]`.
- Bootstrap **intervals** are now computed only when the *effective
  per-replicate size* is `>= 20` (i.e. `min(n_used, max_n_for_bootstrap)`), not
  just when `n_used >= 20`. Below that a percentile bootstrap is too unreliable
  to report (false precision), so `bootstrap_intervals` is `None` with a
  warning. This closes a path where a small `max_n_for_bootstrap` (e.g. 10) on a
  large sample bypassed the floor.
- Bootstrap **pattern stability** is suppressed (`None`, with a warning) when
  `max_n_for_bootstrap` caps replicates below the low-power threshold (30) while
  the original sample is above it. Previously every capped replicate was judged
  `low_power_or_uncertain`, collapsing stability to 0.0 against a clean
  full-sample label (e.g. `n=100, max_n_for_bootstrap=25` on strong linear data
  reported `pattern_stability=0.0`). Genuinely small samples (uncapped,
  `n_used < 30`) are unaffected and keep their stability signal.
- `scan_target(errors="warn")` no longer swallows **systemic** failures. A
  missing optional dependency (`OptionalDependencyError` from `mode="standard"`/
  `"deep"` without the extras) or a misspelled `profile_pair` keyword
  (`TypeError`) fails identically for every column, so it is now propagated even
  under the default `errors="warn"` — surfacing one actionable error instead of a
  scan that "completes" with N identical error entries and zero successes.
  Genuine per-column data failures (e.g. an all-NaN or constant column, which
  raise `InputError`) are still captured as `error` entries.
- `scan_target(max_pairs=...)` now records the columns dropped by the cap as
  `skipped` entries (`error_type="MaxPairsExceeded"`) instead of omitting them
  entirely. Previously `summary()` read "profiled: N, skipped: 0" on a wider
  DataFrame as if coverage were complete, hiding both that the scan was truncated
  and *which* (data-order-dependent) columns were left unprofiled.
- The target-scan caveat (`summary()`, `to_markdown()`, and the `scan_target`
  docstring) now states that the scan applies **no multiple-testing correction**,
  so across many candidates some patterns appear by chance and the rankings are
  hypothesis-generating. A wide scan of noise previously decorated variables
  across the pattern/underrate sections with no such warning at the point of use.
- Documented that mutual information is reported as **raw, unnormalized** MI (in
  nats, `>= 0`, unbounded — not a 0–1 scale) in the `compute_mutual_information`
  docstring and the README, so its magnitude isn't misread as a correlation.
- Noted in the Chatterjee's ξ docstring that a heavily tied sort variable adds
  sampling variability (the random tie-break would shift the value), so ξ is
  noisier for low-cardinality sort variables.
- Unified the trimmed-Pearson logic: the outlier/leverage-sensitivity check in
  `api.py` now delegates to `metrics/robust.py` via a new
  `assess_outlier_sensitivity()` (which reuses `compute_trimmed_pearson`),
  removing a duplicate 1%-trim implementation, the previously dead-in-pipeline
  `compute_trimmed_pearson`, and three redundant `api.py` constants. The
  deep-mode `pearson_trimmed_1pct` metric and the leverage flag are now computed
  from the same trimmed value. Behavior is unchanged.
- Extracted the heuristic cascade's label-driving cut points into public,
  documented module-level constants in `corrsleuth/heuristics/classifier.py`
  (`STRONG_MAGNITUDE_THRESHOLD`, `WEAK_MAGNITUDE_THRESHOLD`,
  `RANK_LINEAR_GAP_THRESHOLD`, `PEARSON_KENDALL_GAP_THRESHOLD`,
  `NONMONOTONIC_MONOTONE_CEILING`, `NONMONOTONIC_DC_THRESHOLD`,
  `NEAR_LINEAR_GAP_THRESHOLD`, `WEAK_DC_THRESHOLD`) so they can be inspected and
  overridden. Behavior is unchanged.
- Replaced remaining inline threshold literals (low-n, missingness, unique-ratio
  in validation; bootstrap min-n) with named constants and added rationale
  docstrings to the existing robustness, bootstrap, and Chatterjee's-ξ
  thresholds.
- Split the `corrsleuth/scan.py` monolith into a `corrsleuth/scan/` package:
  `core` (the `scan_target` orchestration and `TargetScanEntry`), `report`
  (`CorrSleuthTargetReport` text/frame/Markdown rendering), and `plot`
  (`plot_top`). The public API (`scan_target`, `CorrSleuthTargetReport`,
  `TargetScanEntry` from `corrsleuth` or `corrsleuth.scan`) is unchanged.
- Documented that `scan_target` runs sequentially, with guidance on bounding
  cost for very wide DataFrames, in the scan docstring and interpretation guide.
- Expanded the ruff ruleset (added isort, pep8-naming, pyupgrade, simplify, and
  bugbear) and applied `ruff format` across the codebase; CI now enforces both
  `ruff check` and `ruff format --check`.
- Centralized the repeated "computed but no value" metric guard behind
  `MetricResult.no_value(...)`, and documented the intentional in-place
  enrichment of the internal `CleanPair` in `profile_pair`.

### Added
- **Bidirectional scan** (`scan_target(..., direction=...)`, default `"forward"`,
  unchanged). CorrSleuth's shape diagnostics describe `E[y | x]`, so a scan of
  `profile_pair(candidate, target)` characterizes the *predictive* direction —
  but when data is engineered as `candidate = f(target)` (steps, saturation,
  sigmoids, sinusoids, heteroscedasticity), the shape only shows in the reverse
  orientation. `"reverse"` profiles `profile_pair(target, candidate)`
  (`E[candidate | target]`), and `"both"` profiles both — keeping the forward
  profile as primary, attaching the reverse shape (`reverse_pattern` /
  `reverse_mean_shape` / `reverse_dependence_type` in `to_frame()`), and adding a
  "Shape differs by direction" section that flags candidates whose reverse shape
  is a structured nonlinearity while their forward shape is not — the
  `candidate = f(target)` signature. The primary association metrics
  (Pearson/Spearman/Kendall/dCor/MI) are symmetric and identical either way; only
  the shape diagnostics are directional, so the reverse view re-describes shape,
  it does not find new relationships. `"both"` costs two `profile_pair` calls per
  candidate.
- **"Dependence may be understated" scan section.** `summary()` and
  `to_markdown()` now surface a cross-cutting section listing weak/ambiguous
  candidates (`weak_or_no_relationship` / `mixed_or_ambiguous`) that nonetheless
  carry strong nonmonotonic or radial dependence evidence — so a real
  relationship the headline label understates does not go unnoticed in a wide
  scan. It hoists the per-pair "may understate" warning
  (`detect_metric_warnings`) into a scan-level callout and, critically, also
  fires in **lite** mode on the leave-the-top-out `sq_corr_robust` (no optional
  dependency), where distance correlation / Chatterjee's ξ / mutual information
  are unavailable. Because the signal is the *robust* squared correlation, a
  heavy-tailed variable's leverage artifact (which inflates raw `sq_corr` but
  collapses under the drop) is excluded, matching the cascade. This is the
  primary remedy for symmetric shapes (U/V/circular) degrading to
  `mixed_or_ambiguous` under one-sided heavy-tailed support, where a
  tail-inflated Pearson keeps the pair out of the nonmonotonic route.
- **`sq_corr_robust` on `result.diagnostics`.** The leave-the-top-out companion
  to `sq_corr` (previously computed but consumed only inside the cascade) is now
  stored on `MetricDiagnostics` and rendered in every surface — `summary()`,
  `to_markdown()`, `to_dict()`, and as a `diagnostic_sq_corr_robust` column in
  both `to_frame()` methods — parallel to the raw `sq_corr` beside it, so callers
  can see how much of a magnitude signal survives dropping the few most extreme
  squared points.
- **Regression-influence diagnostics** (`compute_influence` in
  `corrsleuth/metrics/influence.py`, no new dependency): row-level Cook's
  distance on the `y ~ x` fit, exposed as `max_cook_distance` and
  `n_influential_points` on `result.diagnostics`, refining the
  `outlier_sensitivity` axis into `single_point_driven` (one dominant row) vs
  `high_leverage_cluster` (several) vs `low`. Because Cook's distance has no
  blind spot for a mid-range leverage cluster larger than the 1% trim fraction,
  this axis can flag `high_leverage_cluster` even when the trim check called
  Pearson stable. Uses the softer Cook & Weisberg `D > 0.5` cutoff (a masked
  outlier cluster deflates each point's Cook's distance below the classical
  `D > 1`).
- **Single-bend V detection in lite mode** (`SINGLE_BEND_BIN_LOF_FLOOR =
  0.30`, no new metric). A *linear-armed* V defeats both lite nonmonotonic
  detectors even when perfectly centered: on even point density the vertex
  value sits far below the mean of |Y|, so the centered square `(Y − ȳ)²` is
  minimized mid-arm rather than at the vertex, folding the squares and
  collapsing `sq_corr` to ~0.16 (vs the 0.35 route floor) — and its single
  bend reads exactly 1 reversal, below the oscillation route's 2. The
  blind-test uniform V therefore read `weak_or_no_relationship` in lite mode —
  a confident wrong answer, with `bin_lof_r2_gain = 0.90` sitting right there
  — while deep mode corrected it via distance correlation. Rule 4 gains a
  fourth, lite-computable arm for the weak-monotone zone (both metrics <
  0.25): at least one robust bin-mean reversal with the raw AND
  leave-one-bin-out `bin_lof_r2_gain` above 0.30 — deliberately double the
  oscillation route's floor, because the key adversary (the heavy-tailed-Y
  artifact, whose one extreme bin manufactures a large raw gain and exactly
  one reversal) is itself a one-turn shape that the robust gain collapses for.
  The `dependence_type` axis reads `nonmonotone` for these in every mode.
  Calibrated in `validation/single_bend_sweep.py`: worst robust gain across
  every weak-zone negative family was 0.078 (a 4× margin below the floor);
  rank-balanced V's fired 90–100% with gains 0.66–0.91. A strongly *tilted* V
  carries a genuine monotone trend, never enters the weak zone, and keeps its
  monotone-family label by design. Deep-mode behavior is unchanged (verified
  by a full label/axis diff: zero new diffs); in lite mode the only blind-test
  pair affected is the uniform V, which now resolves.
- **Moderate-trend oscillation detection** (`OSCILLATION_MONOTONE_CEILING =
  0.50`, no new metric). The oscillation route into `nonmonotonic_dependence`
  (and the `dependence_type = "oscillating"` axis value) used to share the
  dc / sq_corr routes' weak-trend ceiling (`max(|p|, |s|) < 0.25`), which
  stranded skew-tilted oscillations in a dead zone: a sinusoid on one-sided
  exponential/lognormal support picks up a spurious *moderate* monotone tilt
  (the blind-test exponential sinusoid measures |s| ≈ 0.34) — too much trend
  for the weak-trend routes, too little for the strong-trend
  `oscillating_trend` axis value (≥ 0.50), so it read `mixed_or_ambiguous` /
  `monotone`. The oscillation route now has its own ceiling of 0.50 (equal to
  `STRONG_MAGNITUDE_THRESHOLD`, so the label route and `oscillating_trend`
  tile the trend axis with no gap, and no rule-5/6 label ever competes). The
  route can afford the headroom because it is the strictest joint gate in the
  cascade — reversals ≥ 2 AND raw AND leave-one-bin-out `bin_lof_r2_gain` >
  0.15, which pure noise (≤ ~0.05) and smooth monotone shapes (0–1 reversals)
  cannot fake. Calibrated in `validation/oscillation_ceiling_sweep.py`: 0%
  in-zone fires for every moderate-trend negative family at every size, with
  one documented mild residual (a noisy *tilted-U* occasionally counts 2
  reversals — its `nonmonotonic_dependence` label is still correct; only the
  "oscillating" axis word over-specifies the single bend); tilted and
  skew-supported sinusoids fire 90–100%. On the blind-test data the only
  behavior change is the exponential sinusoid resolving correctly — verified
  by a full label/axis diff across all 25 columns x 4 distribution variants x
  both orientations.
- **Discontinuity / level-shift detection** (`mean_shape = "discontinuous_jump"`,
  extends `compute_segmentation` in `corrsleuth/metrics/shape.py`, no new
  dependency). A sharp jump embedded in an otherwise strong trend is invisible
  to the R²-scale diagnostics — the trend soaks up the variance, so a 6.5σ
  level shift in a |p| ≈ 0.97 relationship reads `bin_lof_r2_gain` ≈ 0.047 and
  `segment_gain` ≈ 0.05 and passed as `near_linear` with `mean_shape = linear`
  (blind-test X22, in every distribution variant). The new
  `segment_jump_ratio` diagnostic measures the fitted gap between the two
  lines of the best *unconstrained* two-line split, at the boundary x, in
  units of the **noisier side's** residual σ — taken as the min over the
  global fit and localized refits on windows adjacent to the boundary. The
  unconstrained fit makes a continuous kink read ~0 (its two lines meet at the
  boundary); the localization collapses a smooth curve's chord-displacement
  artifact (a moderate sigmoid drops from ~4.3 globally to ~0.5 locally); and
  the noisier-side σ keeps a heavy tail's separation from the bulk (a leverage
  artifact) from reading as a jump. When the ratio clears 3.0 jointly with
  sloped segments, monotone bin means, and a real rank trend
  (`_is_discontinuous_jump`), the pair reads `mean_shape =
  "discontinuous_jump"`, `breakpoint_x` reports where the jump sits, and a
  warning advises looking for a threshold, policy change, or regime switch at
  that point. The primary label is unchanged (the trend genuinely dominates).
  Calibrated in `validation/segment_jump_sweep.py`: one residual fire in 720
  negative trials (a single seed of an adversarial near-noiseless folded
  heavy-tail family, locally indistinguishable from a jump), all real-data
  checks clean; genuine jumps detected from ~3σ (partial) and 70–100% from 4σ
  up at every size. Withheld below n = 150, where a moderate smooth curve is
  not reliably separable from a jump.
- **Two-group / mixture detection** (`dependence_type = "two_group_shift"`,
  new `corrsleuth/metrics/mixture.py`, no new dependency). A pooled correlation
  can be carried almost entirely by a *between-group mean shift*: two
  well-separated clouds of rows with little or no association inside either —
  the lurking-grouping-variable / mixture situation (and the aggregation trap
  behind Simpson-style reversals). `compute_cluster_split` measures the
  ingredients on the association-axis projection (the first principal component
  of the z-scored pair): `cluster_split_r2` (variance share of the best
  two-group split — exact 1-D 2-means via a prefix-sum scan; a unimodal cloud is
  structurally capped near 0.64–0.75), `cluster_valley_share` (occupancy of the
  band around the split boundary — near zero when "almost no points bridge the
  gap"), `cluster_min_share` (subpopulation vs. leverage handful), and
  `pearson_within_cluster` (how much association survives inside the groups).
  All four are on `result.diagnostics` and every render surface. When the five
  jointly-calibrated gates hold (`validation/cluster_split_sweep.py`: 0 fires in
  680 negative trials across bivariate normals, skewed/heavy-tailed links,
  curves, sloped steps/changepoints, heteroscedastic fans, leverage clusters,
  and sparse subgroups; 90–100% detection of blob mixtures ≥ 4 within-group
  stds apart down to a 12% subpopulation), the `dependence_type` axis reads
  `two_group_shift` (in preference to the generic `monotone`) and a warning
  explains that the pooled correlation describes the group separation, not a
  continuous trend — advising to identify the grouping and analyze the groups
  separately. Statistical honesty note: from a single pair, a two-subpopulation
  mixture and a *flat threshold effect* are the same joint distribution, so the
  warning presents both readings; a step that keeps a within-segment slope keeps
  a high within-group Pearson and is not flagged. Lite-computable (numpy only);
  withheld for n < 100 or coarse discrete data (< 10 distinct values), where a
  lattice fakes an empty valley. The primary label is unchanged — this refines
  the secondary axis and adds the warning.
- **Compound trend + oscillation detection** (`mean_shape = "oscillating_trend"`,
  no new dependency or threshold). A strong monotone trend whose binned
  conditional means still reverse direction two or more times — robustly — is now
  labeled a *trend with a superimposed oscillation* (a linear ramp plus a wave;
  trend + periodic residual) and carries a companion "compound trend-plus-wave"
  warning pointing at the periodic residual structure a single line or monotone
  curve leaves behind. Previously such a shape was misread as `step_or_threshold`
  (the single-breakpoint search forces one break onto the wave). Both the primary
  cascade's oscillation route and the `dependence_type = oscillating` axis require
  a *weak* trend, so a wave riding on a strong trend reached neither; this reuses
  the same already-calibrated joint gate (`OSCILLATION_MIN_REVERSALS` +
  `OSCILLATION_BIN_LOF_FLOOR` on the leave-one-bin-out robust gain) in the
  strong-trend regime. The primary label is unchanged (the trend still dominates);
  this refines the secondary `mean_shape` axis and adds the warning. A pure
  sinusoid or U-shape (weak Spearman) is unaffected — it stays `curved` with
  `dependence_type = oscillating`. Across the blind-test datasets (uniform /
  exponential / lognormal, both orientations, every column) the only pair flagged
  is the genuine trend+wave case; zero false positives on steps, curves, clusters,
  linear, and heteroscedastic shapes.
- **Breakpoint localization** (`compute_segmentation` in
  `corrsleuth/metrics/shape.py`, no new dependency): a single-breakpoint search
  refines a curved *monotone* `mean_shape` into `smooth_curve` versus
  `step_or_threshold`, and reports `segment_gain`, `segment_stepness` (the
  fraction of the fit gain a flat-segment model captures — the number behind the
  step-vs-smooth call, `≈ 1` for a step and `≤ 0` for a smooth bend), and
  `breakpoint_x` (where a step sits) on `result.diagnostics`. The two are told
  apart by whether a two-*level*
  (flat-segment) model fits as well as a two-*line* model — a step's segments
  are flat, a smooth curve's are sloped — computed with an O(n) prefix-sum scan
  over mean-centered inputs (the centering keeps the closed-form residual
  identities numerically stable when x/y sit far from zero).
  A `threshold_step` pair keeps its `monotonic_nonlinear` label but now reports
  `mean_shape=step_or_threshold` with the jump location; exponential/logarithmic
  curves report `smooth_curve` with no spurious breakpoint. Monotone
  piecewise-linear kinks fold into `smooth_curve` (not reliably separable from a
  smooth bend over a finite range); non-monotone curves (U-shapes) stay the
  generic `curved`.
- **Heteroscedasticity diagnostics** (`corrsleuth/metrics/variance.py`,
  no new dependency): `compute_heteroscedasticity` runs a Koenker-studentized
  Breusch-Pagan test and a Goldfeld-Quandt residual-variance ratio on the
  linear-fit residuals, exposed as `bp_pvalue` and `gq_ratio` on
  `result.diagnostics` and populating the `variance_shape` axis
  (`constant` / `increasing_spread` / `decreasing_spread`). A `near_linear`
  pair with a growing funnel now keeps its label but reports
  `variance_shape=increasing_spread` and a warning that homoscedastic inference
  may be unreliable. Only assessed when the mean is adequately linear (a curved
  mean's misspecification residuals are not mistaken for changing variance).
  Adds a `heteroscedastic` shape to `make_relationship()`.
- **Symmetric ("bowtie") variance detection**, extending the heteroscedasticity
  diagnostics above: `compute_heteroscedasticity` now also returns
  `bowtie_ratio` — the combined low+high-thirds residual variance over the
  middle third's — exposed on `result.diagnostics` and populating two new
  `variance_shape` values, `edge_high_spread` (spread high at both extremes of
  x, calm in the middle) and `center_high_spread` (the reverse). Goldfeld-Quandt's
  low-vs-high split cannot see this shape by construction (both edges have
  similar variance, so the ratio reads ~1), and Breusch-Pagan's linear
  auxiliary regression can miss it too (the squared-residuals-vs-x relationship
  is U/hill-shaped, not linear) — so this is checked independently, additive to
  the existing funnel check rather than replacing it. New threshold
  `BOWTIE_RATIO_FLOOR` (2.5). Adds a `bowtie_variance` shape to
  `make_relationship()`.
- **Secondary diagnostic axes** on `result.diagnostics`: five coarse
  categorical summaries — `mean_shape`, `variance_shape`, `dependence_type`,
  `outlier_sensitivity`, and `functional_direction` — describing orthogonal
  properties of a relationship that the single primary `pattern` label cannot
  carry at once. Derived from the numeric diagnostics/metrics already computed
  (`derive_diagnostic_axes` in `heuristics/classifier.py`), so a `near_linear`
  pair can simultaneously report high `outlier_sensitivity`, and a circular
  pair reports `dependence_type=closed_loop_or_multivalued` with
  `functional_direction=neither_direction` in deep mode. Surfaced in
  `summary()`, `to_markdown()`, `to_dict()`, and `to_frame()`. Each axis keeps
  its underlying number alongside it. `scan_target` profiles each pair as
  `profile_pair(data, candidate, target)`, so these direction-sensitive axes and
  `breakpoint_x` describe how the candidate drives the target — the
  feature-screening question — with the panel plot drawn candidate-on-x to match,
  and the forward `chatterjee_xi` giving the candidate→target direction.
  `scan_target().to_frame()` exposes each pair's diagnostics as `diagnostic_*`
  columns (the numeric diagnostics and the five axes), plus `pattern_stability` /
  `stability_label` / `stability_metric_set` when bootstrapping was requested —
  matching the per-pair `CorrSleuthResult.to_frame`.
- `corrsleuth/metrics/shape.py`: two no-new-dependency shape diagnostics,
  `bin_lof_r2_gain` (the degrees-of-freedom-adjusted bin-mean-model R² minus
  linear-fit R², a classical lack-of-fit test) and `sq_corr`
  (`corr((X−x̄)², (Y−ȳ)²)`, the correlation of the mean-centered squares), wired
  into the `monotonic_nonlinear` and `nonmonotonic_dependence` cascade rules as
  additional constants `BIN_LOF_R2_GAIN_THRESHOLD` (0.05) and
  `SQ_CORR_THRESHOLD` (0.35). Diagnostic-only — surfaced on
  `result.diagnostics`, not in the metrics table.
- `detect_metric_warnings` now also considers mutual information (converted
  to a comparable scale via `sqrt(1 - exp(-2*MI))`) alongside Chatterjee's ξ
  when warning that a weak/ambiguous label may understate real dependence, so
  the warning can fire in `mode="standard"` too, not only deep mode.
- Four new realistic `shape_type`s for `make_relationship()`:
  `exponential_monotonic`, `logarithmic_monotonic`, `threshold_step`, and
  `circular`.
- `docs/shape-diagnostics-design.md`, the design note behind the diagnostics
  above.
- `docs/thresholds-and-rationale.md` cataloguing every threshold in the package
  — its value, location, what it gates, the justification, and how to override
  the label-driving ones. Linked from the README and the interpretation guide.
- Static type checking: a `[tool.mypy]` configuration and a `mypy` CI job
  (non-strict) that the package now passes cleanly, plus `mypy` in the `dev`
  extra. Added docstrings to the internal bootstrap helpers.
- Property-based tests (`tests/test_property.py`, Hypothesis) asserting metric
  invariants over generated inputs: joint row-permutation invariance,
  constant-input → `None`, magnitude bounds, and symmetric-metric symmetry /
  Chatterjee ξ forward-reverse consistency.
- End-to-end smoke tests (`tests/test_smoke.py`) exercising the documented
  `profile_pair` / `scan_target` workflow across every render surface.
- Coverage reporting: `[tool.coverage]` config (branch coverage, `fail_under`
  floor), a `coverage` CI job, and `pytest-cov` / `hypothesis` in the `dev`
  extra.

## [0.1.0] - 2026-06-27

Initial release.

### Added
- `profile_pair()` for diagnosing a single numeric pairwise relationship,
  with `lite`, `standard`, and `deep` metric modes.
- `scan_target()` for profiling every eligible numeric predictor against a
  single numeric target, returning a `CorrSleuthTargetReport`.
- Core metrics (Pearson, Spearman, Kendall tau-b), standard-mode nonlinear
  metrics (Distance Correlation, Mutual Information), and deep-mode robust
  diagnostics (trimmed/winsorized/median-clipped Pearson, biweight
  midcorrelation, Chatterjee's ξ). Chatterjee's ξ uses the tie-corrected
  estimator from Chatterjee (2020), so it stays well-calibrated for discrete
  or low-cardinality responses.
- Heuristic diagnostic labels, validation warnings, recommendations, optional
  bootstrap intervals, and pattern-stability diagnostics.
- Deep-mode warning when Chatterjee's ξ is high but the assigned label is
  `weak_or_no_relationship` or `mixed_or_ambiguous`, so strong nonmonotonic
  dependence is never silently contradicted by the label.
- Outlier-sensitivity diagnostic that flags Pearson as leverage-sensitive when
  trimming the extreme 1% of x/y materially changes it, including the case where
  trimming flips the sign of the correlation.
- Missing-value policy on `profile_pair()` / `scan_target()`: `"pairwise"` drops
  rows missing in x or y; `"listwise"` drops rows missing in *any* column of the
  data (complete-case deletion); `"raise"` errors on any missing value in the
  pair.
- Matplotlib diagnostic plots (`result.plot()`, `report.plot_top()`).
- Markdown / dict / DataFrame exports for results and target reports. The
  summary and Markdown reports include the signed Pearson–Spearman gap, which
  reveals sign disagreement that the absolute rank/linear gap hides.
- `make_relationship()` relationship simulator (reproducible when seeded with
  `random_state=`; nondeterministic by default), which validates its inputs
  (`n` must be an integer ≥ 2, `noise` must be non-negative) and raises
  `InputError` on bad arguments.
- Strict input validation with clear `InputError` messages: profiling a column
  against itself, duplicate column names, and non-positive `max_pairs` /
  `sample_size` values are rejected explicitly. Infinite values are rejected
  only when they appear in rows actually used after missing-value handling.
- Top-level exports for `CorrSleuthResult`, `MetricDiagnostics`, and the
  exception types (`CorrSleuthError`, `InputError`, `MetricComputationError`,
  `OptionalDependencyError`).

### Notes
- When a bootstrap cap (`max_n_for_bootstrap`) is smaller than the sample size,
  each replicate resamples that many rows (an m-out-of-n bootstrap); the
  resulting intervals are conservative (wider) and the emitted warning says so.

[0.2.0]: https://github.com/mbagalman/CorrSleuth/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/mbagalman/CorrSleuth/releases/tag/v0.1.0
