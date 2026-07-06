# Nonlinear Dependence Measures — Design Note

> This note evaluates candidate nonlinear dependence measures, picks the ones
> worth implementing (Chatterjee's ξ for `mode="deep"`), and documents why the
> rest are deferred or rejected. It is the "why these metrics" companion to the
> [methodology doc](methodology.md).

## Context

CorrSleuth already exposes two nonlinear dependence measures, both behind the
`[standard]` extras:

- **Distance correlation (`dcor`)** — symmetric, sensitive to any form of
  dependence, computed via the [`dcor`](https://pypi.org/project/dcor/)
  package.
- **Mutual information** — symmetric, estimated with scikit-learn's
  KSG-style `mutual_info_regression`.

This note asks whether additional nonlinear measures would meaningfully improve
diagnosis. The evaluation criteria are:

| Criterion | Treatment in this note |
|---|---|
| Dependency weight | Hard preference — anything new should be no-new-dep, or behind an existing extra |
| Install friction | Same |
| Performance | Must scale to ~20 K rows |
| Interpretability | Analyst must be able to read the value |
| Stability | Must behave well at small `n` and under ties |
| License compatibility | Hard gate — no GPL or patented |

The ticket explicitly says "create a short design note **before**
implementation" and lists implementation as optional. This note picks one
candidate to implement now (Chatterjee's ξ) and defers the rest with reasons.

## Candidates evaluated

| Metric | New deps | Compute | Symmetry | Range | License | Verdict |
|---|---|---|---|---|---|---|
| HSIC | None (kernel choice required) | O(n²) | Symmetric | [0, ∞) | OK | **Defer** |
| Hoeffding's D | None | O(n log n) | Symmetric | ~[0, 1] | OK | **Defer** |
| Chatterjee's ξ | None | O(n log n) | **Asymmetric** | ~[0, 1] | OK | **Implement** |
| MGC via `hyppo` | `hyppo` (heavy) | O(n²)–O(n³) | Symmetric | [0, 1] | BSD | **Defer** |
| MIC via `minepy` | `minepy` (patent-encumbered) | O(n^1.6) | Symmetric | [0, 1] | Unclear | **Reject** |

### 1. Hilbert-Schmidt Independence Criterion (HSIC)

Tests independence by embedding each variable into a reproducing kernel
Hilbert space and measuring the Hilbert-Schmidt norm of the covariance
operator between embeddings. The normalized variant (NHSIC) lies in `[0, 1]`.

**Pros**
- Detects any form of dependence under regularity assumptions.
- Pure numpy possible; no new dependency.

**Cons**
- Requires choosing a kernel (typically Gaussian/RBF) and a bandwidth
  (median heuristic is the usual default but is ad-hoc).
- The kernel/bandwidth become part of the public API surface and need
  documenting; analysts who don't know what a kernel is would have to
  trust the default.
- O(n²) memory for the kernel matrix — uncomfortable at the 20 K cap.
- Largely overlaps with what `dcor` already provides.

**Verdict: Defer.** Adds API complexity (kernel choice) without much
information beyond `dcor` for the typical pairwise EDA use case. Worth
revisiting only if a concrete user request shows up.

### 2. Hoeffding's D

Classical nonparametric statistic from Hoeffding (1948), sensitive to all
forms of dependence under continuity. Computable from order statistics in
O(n log n).

**Pros**
- No new dependency; pure numpy/scipy.
- Symmetric; behaves like `dcor` for many shapes.
- Long, stable literature; well-understood.

**Cons**
- Largely subsumed by what `dcor` provides — adding a second symmetric
  general-purpose dependence test inflates the result schema without
  surfacing a new story.
- The interpretation "departure from independence on a 0–1 scale" is the
  same story `dcor` already tells; users would need a comparison/threshold
  framework that we don't have.

**Verdict: Defer.** Reasonable second-tier candidate, but it duplicates
the gap that `dcor` already fills. Open to revisiting if a future heuristic
needs a no-new-dep symmetric alternative when `[standard]` isn't installed.

### 3. Chatterjee's ξ — *primary candidate*

From Chatterjee (2020), [*A new coefficient of correlation*](https://par.nsf.gov/servlets/purl/10339804) (JASA). Defined as

```
ξ_n(X → Y) = 1 - (3 · Σ |r_{i+1} - r_i|) / (n² - 1)
```

where the data is sorted by `X` and `r_i` is the rank of `Y_{(i)}`. The
statistic is **asymmetric**: `ξ(X → Y)` measures whether `Y` is a (noisy)
function of `X`.

**Pros**
- **No new dependency**, no hyperparameters — pure numpy + scipy ranks.
- O(n log n) — scales fine to 20 K and beyond.
- **Asymmetry is genuinely novel** for this codebase. `dcor` and MI are
  symmetric; they can't tell the analyst whether `Y = f(X)` or `X = f(Y)`.
  For target scans where the question is "is Y a function of these
  features?", that direction matters.
- Bounded approximately in `[0, 1]` (technically `[-0.5, 1]` for finite
  `n`); easy to read.
- Strong recent literature: ~1 K citations as of 2025, multiple follow-up
  papers establishing power and asymptotic properties.
- Detects shapes Pearson/Spearman miss — including U-shape / parabolic
  relationships where rank metrics also fail.

**Cons**
- The asymmetry is a feature but also a small extra concept to
  communicate. We need to make clear in the docstring/README that
  `ξ(X → Y) ≠ ξ(Y → X)` is intentional.
- Slow asymptotic convergence — for `n < ~20` the empirical value has
  high variance. We gate this with a `_MIN_N_FOR_CHATTERJEE_XI = 20` check
  + warning, matching the style used for the robust metrics.
- Tie-breaking in the sort variable matters; we break those ties with a seeded
  random permutation (Chatterjee's prescription), **not** by `Y`. Ordering tied
  sort-variable values by `Y` leaks the response and inflates ξ toward 1 even
  under independence; random tie-breaking keeps it calibrated. Ties in `Y` are
  handled by the tie-corrected denominator. A consequence is that, when the sort
  variable has ties, ξ depends on the seeded tie-break rather than being
  invariant to input row order. Heavy-tie datasets are also flagged by the
  existing `high_tie_rate` warning.

**Verdict: Implement.** Implemented in `corrsleuth/metrics/nonlinear.py`
in the same PR as this note. Surfaces as `chatterjee_xi` in
`mode="deep"` results.

### 4. Multiscale Graph Correlation (MGC) via `hyppo`

Constructs a series of nearest-neighbor graphs over both variables and
finds the optimal scale at which dependence is strongest.

**Pros**
- Powerful; competitive with `dcor` and HSIC across many shapes.
- Has theoretical guarantees and a published validation suite.

**Cons**
- `hyppo` itself isn't huge, but it has a non-trivial dependency tree
  (numba, sklearn, etc.) that we'd inherit. CorrSleuth's promise is a
  light base install plus thin extras; pulling `hyppo` in even as an
  extra grows the install footprint of `[standard]` materially.
- O(n²) at minimum; the multiscale variant pushes towards O(n³).
- Value interpretation is "MGC test statistic on a 0–1 scale" — same
  story as `dcor` but harder to explain.
- Useful primarily for higher-dimensional or structured data, which
  CorrSleuth doesn't handle yet (numeric pairwise only).

**Verdict: Defer.** Heavy install footprint and complex theory not
justified by marginal improvement over `dcor` for v0.x pairwise scope.

### 5. Maximal Information Coefficient (MIC) via `minepy`

Discretizes data into 2D grids and finds the binning that maximizes
normalized mutual information.

**Cons**
- `minepy` is the canonical implementation. Its license terms have been
  unclear at various points in its history, and the original MIC method
  is associated with patent claims (Reshef et al., 2011). License risk
  for a library that aims to be permissively redistributable.
- Statistical power has been contested in the literature (Simon and
  Tibshirani 2014; Gorfine et al. 2012) — multiple papers show MIC
  underperforming `dcor` and Hoeffding's D on common shapes.
- Heavy dependency.

**Verdict: Reject.** License risk + contested utility. Not worth the
maintenance burden.

## Recommendation

**Implement Chatterjee's ξ in `mode="deep"`** as the only new nonlinear
dependence measure for now. Defer Hoeffding's D as a second-tier candidate
that could be added if a future heuristic specifically needs a no-new-dep
*symmetric* dependence measure. Defer HSIC and MGC; reject MIC.

Why deep mode (rather than standard)? The ticket explicitly asked for "deep
mode," and ξ has no new dependency — that fits the deep-mode shape that
was established in Ticket 4.1 ("no-new-dep diagnostic add-ons"). Standard
mode remains the home for measures that pull in heavy optional
dependencies (`dcor`, scikit-learn).

## Implementation summary

This PR ships:

- `corrsleuth/metrics/nonlinear.py` exposing `compute_chatterjee_xi(pair)`.
- Wire-up in `corrsleuth/api.py`'s deep-mode dispatch.
- `chatterjee_xi` added to `_VALID_SORT_KEYS` in `corrsleuth/scan/plot.py` so
  target scans can sort `plot_top()` by ξ.
- Tests in `tests/test_metrics.py` covering: clean-linear → high ξ; U-shape
  detected when Pearson/Spearman miss it; asymmetry on many-to-one
  relationships; near-zero on independent data; constant-input → `None`;
  small-sample guard fires below `n = 20`; lite mode does not include ξ.

Conventions applied:

- `pair.x_is_constant or pair.y_is_constant → None`, matching every
  other metric.
- `pair.n_used < 20 → None` plus a single warning about the small-sample
  bias. Threshold is lower than the robust metrics' 50 because ξ
  converges faster.
- Sort-key ties are broken with a **seeded random permutation** (`random_state`),
  not by `Y`; ties in `Y` are handled by the tie-corrected denominator. Ordering
  sort-key ties by `Y` would leak the response and inflate ξ, so random
  tie-breaking is required for calibration. The value is reproducible for a given
  input and `random_state`, but — when the sort variable has ties — is not
  invariant to input row order. Heavy-tie datasets are also flagged by the
  `high_tie_rate` warning from Ticket 2.3. (Earlier drafts of this note used a
  stable sort + ordinal Y-ranking; that was corrected in 0.2.0 — see the updated
  "Tie-breaking" bullet above and the CHANGELOG.)

## Open questions for the reviewer

1. **Default direction. Resolved during review** — both directions are now
   computed in deep mode. `chatterjee_xi` is `ξ(pair.x → pair.y)` and
   `chatterjee_xi_reverse` is `ξ(pair.y → pair.x)`. For target scans, the
   reverse direction is `ξ(candidate → target)` — the one feature-engineering
   users typically want when prioritizing predictors. Both names are valid
   `sort_by` keys for `report.plot_top()`.
2. **Heuristic interaction. Substantially resolved, by a different route** —
   ξ itself still does not feed `apply_heuristics` (it remains
   diagnostic/warning-only), but the underlying gap this question pointed at
   (a strong nonmonotonic dependence, e.g. a sinusoid, reading
   `weak_or_no_relationship` in deep mode because `dcor` isn't available) was
   closed by the lite-computable shape diagnostics instead:
   `nonmonotonic_dependence` is now assignable in every mode via `sq_corr`
   (magnitude/radial dependence) or the bin-reversal oscillation gate
   (periodic dependence) — see
   [shape-diagnostics-design.md](shape-diagnostics-design.md). Those routes
   are also mode-independent, which consulting ξ (deep-only) would not have
   been — the same call would have produced different labels per mode, a
   reproducibility cost the shape-diagnostic routes avoid.
3. **Hoeffding's D.** Worth implementing as a second symmetric
   dependence measure for the deep-mode toolbox? Open question.

## Out of scope for this PR

- Implementing Hoeffding's D (deferred).
- Heuristic changes that consume ξ.
- Bootstrap / stability for ξ (could plug into Ticket 2.1's framework
  later if reviewer wants).
