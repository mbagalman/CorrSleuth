"""Threshold-calibration sweep for the discontinuity (level-shift) gate.

`compute_segmentation` (metrics/shape.py) reports `segment_jump_ratio`: the
fitted gap between the two lines of the best unconstrained two-line fit,
evaluated at the boundary x, in units of the **noisier side's** residual sigma
— taken as the MIN of the global fit and localized refits on the windows
adjacent to the boundary. The localization is what separates a genuine
discontinuity (the gap survives any window) from a smooth curve whose global
chords are displaced by tail curvature (a moderate sigmoid measures ~4.3
globally but ~0.5 locally); the noisier-side sigma is what keeps a heavy
tail's separation from the bulk (a leverage artifact with wildly asymmetric
per-segment scatter) from reading as a jump.

The classifier reads it jointly with three guards (see `_is_discontinuous_jump`
in heuristics/classifier.py): `segment_stepness` below the step threshold (a
flat-flat jump is already `step_or_threshold`), **zero** robust bin-mean
reversals (a co-directional level shift keeps the binned conditional mean
monotone; a fold/U reads 1, a wave 2+), and a real *rank* trend (`|spearman|`
at least weak — deliberately not `max(|p|, |s|)`, so a leverage-manufactured
Pearson on a folded heavy-tailed shape cannot supply the "trend"). When
everything holds, the pair gets `mean_shape = "discontinuous_jump"`, a
populated `breakpoint_x`, and a level-shift warning.

Residual: one fire in 720 negative trials survives — a single seed of the
adversarial *near-noiseless folded heavy tail* family, which at that sampling
is locally a noiseless smooth curve and genuinely indistinguishable from a
jump. All real-data checks (25 blind-test columns x 4 distribution variants x
both orientations) are clean.

This script calibrates `SEGMENT_JUMP_RATIO_FLOOR`:

- **Negatives** (must not fire the joint gate): linear across noise levels,
  continuous piecewise-linear kinks (mild and sharp — the closest honest
  negative), smooth curves (exponential, log, sigmoid, quadratic), sinusoids
  with and without trend, flat steps (owned by `step_or_threshold`), noise,
  heteroscedastic fans, heavy-tailed-x linear links, and a leverage cluster.
- **Positives** (should fire): a jump of {2..8}x sigma embedded in a linear
  trend, with the same or a different slope on each side, at center and
  off-center breakpoints, across sample sizes.

Run: ``python validation/segment_jump_sweep.py``  (base install only).
Excluded from the sdist via MANIFEST.in, like tests/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from corrsleuth.metrics.shape import compute_bin_lof, compute_segmentation
from corrsleuth.validation.input import validate_pair

# Candidate gate floors (the shipped values live in heuristics/classifier.py).
# The metric itself withholds segment_jump_ratio below n=150
# (_MIN_N_FOR_JUMP_RATIO in metrics/shape.py): at n=100 a moderate sigmoid's
# ratio tail crossed 3.0 (max 3.58 over 30 seeds), while from n=150 up the
# worst smooth-family value across 50 seeds x 6 families was 2.75 — so the
# n=100 rows below report NA and cannot fire.
JUMP_RATIO_FLOOR = 3.0
STEPNESS_CEILING = 0.5  # SEGMENT_STEPNESS_THRESHOLD
MAX_REVERSALS = 1  # OSCILLATION_MIN_REVERSALS (must stay below)
# The trend gate reads |spearman| only (not max(|p|,|s|)): a heavy-tailed
# folded shape can carry a leverage-manufactured Pearson ~0.75 with a weak,
# opposite-sign rank trend -- no coherent trend for a jump to be embedded in.
TREND_FLOOR = 0.20  # WEAK_MAGNITUDE_THRESHOLD

_SIZES = (100, 200, 500, 1500)
_SEEDS = range(10)


def _fires(x: np.ndarray, y: np.ndarray) -> tuple[bool, float | None]:
    pair = validate_pair(pd.DataFrame({"x": x, "y": y}), "x", "y")
    seg = compute_segmentation(pair)
    lof = compute_bin_lof(pair)
    ratio = seg["segment_jump_ratio"].value
    stepness = seg["segment_stepness"].value
    reversals = lof["bin_reversal_count"].value
    p = abs(float(np.corrcoef(x, y)[0, 1]))
    s = abs(float(pd.Series(x).corr(pd.Series(y), method="spearman")))
    del p  # the gate deliberately ignores Pearson (see TREND_FLOOR note)
    fired = (
        ratio is not None
        and ratio > JUMP_RATIO_FLOOR
        and stepness is not None
        and stepness < STEPNESS_CEILING
        and reversals is not None
        and reversals < MAX_REVERSALS
        and s >= TREND_FLOOR
    )
    return fired, ratio


def _negatives(rng: np.random.Generator, n: int) -> dict[str, tuple]:
    u = rng.uniform(-3, 3, size=n)
    z = rng.normal(size=n)
    ln = np.exp(rng.normal(size=n))
    cases = {
        "linear_lo_noise": (u, 0.8 * u + rng.normal(0, 0.2, n)),
        "linear_hi_noise": (u, 0.8 * u + rng.normal(0, 1.0, n)),
        "bvn_r0.7": (z, 0.7 * z + np.sqrt(1 - 0.49) * rng.normal(size=n)),
        "kink_mild": (u, np.where(u > 0, 1.5 * u, u) + rng.normal(0, 0.3, n)),
        "kink_sharp": (u, np.where(u > 0, 2.0 * u, 0.2 * u) + rng.normal(0, 0.3, n)),
        "kink_lo_noise": (u, np.where(u > 0, 2.0 * u, 0.2 * u) + rng.normal(0, 0.1, n)),
        "exp_curve": (u, np.exp(0.8 * u) + rng.normal(0, 0.2, n)),
        "log_curve": (None, None),
        "sigmoid_moderate": (u, 1 / (1 + np.exp(-2 * u)) + rng.normal(0, 0.05, n)),
        "sigmoid_sharp": (u, 1 / (1 + np.exp(-8 * u)) + rng.normal(0, 0.05, n)),
        "quadratic": (u, u**2 + rng.normal(0, 0.5, n)),
        "sinusoid": (u, np.sin(2 * np.pi * u / 3) + rng.normal(0, 0.2, n)),
        "sine_plus_trend": (
            u,
            0.9 * u + 1.5 * np.sin(2 * np.pi * u / 3) + rng.normal(0, 0.25, n),
        ),
        "flat_step": (u, (u > 0).astype(float) + rng.normal(0, 0.15, n)),
        "noise": (z, rng.normal(size=n)),
        "hetero_fan": (u, 0.8 * u + rng.normal(0, 0.2 + 0.4 * np.abs(u), n)),
        "lognormal_x_linear": (ln, 2 * ln + rng.normal(0, 0.5, n)),
        "leverage_2pct": (None, None),
        # A degenerate U against a heavy-tailed variable, viewed forward
        # (candidate -> target): a folded two-branch shape whose tail
        # manufactures |pearson| ~ 0.75 while the rank trend is weak and
        # opposite in sign. The two-line fit's boundary gap there is the
        # fold's branch separation, not a level shift — the rank-trend gate
        # (|s| >= TREND_FLOOR, not max(|p|, |s|)) is what excludes it.
        "folded_heavy_tail": ((ln - ln.mean()) ** 2 + rng.normal(0, 0.05, n), ln),
    }
    xl = rng.uniform(0.5, 10, n)
    cases["log_curve"] = (xl, np.log(xl) + rng.normal(0, 0.15, n))
    xb = rng.normal(size=n)
    yb = xb + rng.normal(0, 0.5, n)
    k = max(2, n // 50)
    xb[:k] += 12
    yb[:k] += 12
    cases["leverage_2pct"] = (xb, yb)
    return cases


def _positives(rng: np.random.Generator, n: int) -> dict[str, tuple]:
    sigma = 0.3
    cases: dict[str, tuple] = {}
    for js in (2, 3, 4, 6, 8):
        for where, cut in (("mid", 0.0), ("off", 1.5)):
            u = rng.uniform(-3, 3, size=n)
            same = u + js * sigma * (u > cut) + rng.normal(0, sigma, n)
            # Anchor both branches at the cut so the boundary jump is exactly
            # js*sigma regardless of the cut location and the new slope.
            diff = np.where(u > cut, cut + 0.5 * (u - cut) + js * sigma, u)
            diff = diff + rng.normal(0, sigma, n)
            cases[f"jump{js}s_{where}_same_slope"] = (u, same)
            cases[f"jump{js}s_{where}_new_slope"] = (u, diff)
    return cases


def _run_family(title: str, build) -> None:
    print(f"\n=== {title} ===")
    print(f"{'case':26} {'n':>5} {'fire%':>6} {'ratio(min-max)':>16}")
    for n in _SIZES:
        by_case: dict[str, list] = {}
        for seed in _SEEDS:
            rng = np.random.default_rng(1000 * n + seed)
            for name, (x, y) in build(rng, n).items():
                fired, ratio = _fires(np.asarray(x, float), np.asarray(y, float))
                by_case.setdefault(name, []).append((fired, ratio))
        for name, rows in sorted(by_case.items()):
            fire = 100.0 * sum(f for f, _ in rows) / len(rows)
            ratios = [r for _, r in rows if r is not None]
            rng_txt = f"{min(ratios):.2f}-{max(ratios):.2f}" if ratios else "NA"
            print(f"{name:26} {n:>5} {fire:>5.0f}% {rng_txt:>16}")


def main() -> None:
    print(
        "Gate under test: jump_ratio>"
        f"{JUMP_RATIO_FLOOR}, stepness<{STEPNESS_CEILING}, "
        f"reversals<{MAX_REVERSALS}, max(|p|,|s|)>={TREND_FLOOR}"
    )
    _run_family("NEGATIVES (target: 0% fire)", _negatives)
    _run_family("POSITIVES (target: fire from ~3-sigma jumps up)", _positives)


if __name__ == "__main__":
    main()
