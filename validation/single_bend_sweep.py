"""Calibration sweep for the single-bend (V / off-center U) route floor.

Lite mode detects nonmonotonic dependence through `sq_corr` (magnitude/radial
links -- parabolic U-shapes, circles) and the oscillation route (reversals >= 2
with substantial bin structure). A **linear-armed V** falls through both: on
even point density the vertex value sits far below the mean of |Y|, so the
centered square `(Y - ybar)^2` is minimized mid-arm rather than at the vertex,
folding the squares and collapsing `sq_corr` (~ 0.16 for the blind-test
uniform V, versus 0.35 required, despite a textbook V shape); and a single
bend reads exactly 1 reversal (versus 2 required). With both monotone metrics
~ 0 the pair lands in `weak_or_no_relationship` -- a confident wrong answer
that deep mode corrects via distance correlation but lite mode cannot.

The fix is a third rule-4 arm for the weak-monotone zone
(`max(|p|, |s|) < 0.25`): `bin_reversal_count >= 1` jointly with the raw AND
leave-one-bin-out `bin_lof_r2_gain` above `SINGLE_BEND_BIN_LOF_FLOOR`
(heuristics/classifier.py). A single confirmed turn carries less information
than the oscillation route's two, so the floor must sit well above the
oscillation route's 0.15 -- the key adversary being the **heavy-tailed-Y
artifact**: an independent predictor against a heavy-tailed target, where one
extreme-Y bin manufactures a large raw gain and *exactly one* reversal. The
leave-one-bin-out robust gain collapses for that artifact (the structure lives
in one bin) while a genuine V's gain is spread across bins and barely moves.

This script measures, in the weak-monotone zone only:

- **Negatives** (must not fire): pure noise, weak linear links, heavy-tailed-Y
  independent predictors (normal, t2, and exp-of-uniform tails), independent
  heavy-tail vs heavy-tail, weak heteroscedastic fans, and sparse subgroups.
- **Positives** (should fire): V / absolute-value shapes across vertex
  positions, arm-slope asymmetries, and noise levels. Only the
  **rank-balanced** combinations are in scope by design: a V whose vertex and
  arm slopes leave both monotone metrics weak (the blind-test uniform V
  measures |p| = 0.02, |s| = 0.01 despite its off-center vertex, because the
  arms cancel in rank terms). A strongly *tilted* V — vertex far off-center
  without compensating asymmetry — carries a genuine monotone trend
  (max(|p|, |s|) >= 0.25), never enters the weak zone, and keeps its
  monotone-family label: a partial description, not the confident wrong
  answer (`weak_or_no_relationship`) this gate exists to fix. Measured:
  rank-balanced V's fire 90-100% with robust gains 0.66-0.91 (2-3x the
  floor); out-of-zone V's report "-" below.

Run: ``python validation/single_bend_sweep.py``  (base install only).
Excluded from the sdist via MANIFEST.in, like tests/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from corrsleuth.metrics.shape import compute_bin_lof
from corrsleuth.validation.input import validate_pair

# The gate under test (shipped value in heuristics/classifier.py).
FLOOR = 0.30  # SINGLE_BEND_BIN_LOF_FLOOR (raw AND robust)
WEAK_CEILING = 0.25  # NONMONOTONIC_MONOTONE_CEILING
MIN_REVERSALS = 1

_SIZES = (100, 200, 500, 1500)
_SEEDS = range(10)


def _fires(x: np.ndarray, y: np.ndarray) -> tuple[bool, float | None]:
    """Return (fired, robust_gain_when_rev_ok) for the single-bend gate."""
    pair = validate_pair(pd.DataFrame({"x": x, "y": y}), "x", "y")
    lof = compute_bin_lof(pair)
    reversals = lof["bin_reversal_count"].value
    gain = lof["bin_lof_r2_gain"].value
    robust = lof["bin_lof_r2_gain_robust"].value
    p = abs(float(np.corrcoef(x, y)[0, 1]))
    s = abs(float(pd.Series(x).corr(pd.Series(y), method="spearman")))
    in_zone = max(p, s) < WEAK_CEILING
    rev_ok = reversals is not None and reversals >= MIN_REVERSALS
    fired = (
        in_zone
        and rev_ok
        and gain is not None
        and gain > FLOOR
        and robust is not None
        and robust > FLOOR
    )
    contender = robust if (in_zone and rev_ok and robust is not None) else None
    return fired, contender


def _negatives(rng: np.random.Generator, n: int) -> dict[str, tuple]:
    z = rng.normal(size=n)
    u = rng.uniform(-3, 3, size=n)
    heavy = np.exp(rng.uniform(0.1, 10, size=n))  # the FU-U artifact family
    t2 = rng.standard_t(2, size=n)
    cases = {
        "noise": (z, rng.normal(size=n)),
        "weak_linear": (u, 0.12 * u + rng.normal(0, 1.0, n)),
        "heavy_y_exp10": (rng.normal(size=n), heavy),
        "heavy_y_t2": (rng.normal(size=n), t2),
        "heavy_y_lognorm3": (rng.normal(size=n), np.exp(3 * rng.normal(size=n))),
        "heavy_vs_heavy": (
            np.exp(2 * rng.normal(size=n)),
            np.exp(2 * rng.normal(size=n)),
        ),
        "hetero_no_trend": (u, rng.normal(0, 0.3 + 0.5 * np.abs(u), n)),
        "subgroup_8pct": (None, None),
    }
    xg = rng.normal(size=n)
    yg = rng.normal(size=n)
    ks = max(8, int(0.08 * n))
    yg[:ks] = xg[:ks] * 0.9 + rng.normal(0, 0.1, ks)
    cases["subgroup_8pct"] = (xg, yg)
    return cases


def _positives(rng: np.random.Generator, n: int) -> dict[str, tuple]:
    cases: dict[str, tuple] = {}
    u = rng.uniform(0, 1, size=n)
    for vq in (0.25, 0.4, 0.5, 0.6, 0.75):
        for asym in (1.0, 2.0, 4.0):
            v = np.abs(u - vq) * np.where(u > vq, asym, 1.0)
            scale = float(np.std(v))
            cases[f"v_q{vq}_a{asym:.0f}"] = (u, v + rng.normal(0, 0.25 * scale, n))
    # Off-center smooth U (quadratic with shifted vertex) and a noisy V.
    cases["u_offcenter"] = (u, (u - 0.3) ** 2 + rng.normal(0, 0.02, n))
    vn = np.abs(u - 0.5)
    cases["v_noisy"] = (u, vn + rng.normal(0, 0.5 * float(np.std(vn)), n))
    return cases


def _run_family(title: str, build) -> None:
    print(f"\n=== {title} ===")
    print(f"{'case':20} {'n':>5} {'fire%':>6} {'robust(max, rev>=1 in-zone)':>28}")
    for n in _SIZES:
        by_case: dict[str, list] = {}
        for seed in _SEEDS:
            rng = np.random.default_rng(1000 * n + seed)
            for name, (x, y) in build(rng, n).items():
                by_case.setdefault(name, []).append(
                    _fires(np.asarray(x, float), np.asarray(y, float))
                )
        for name, rows in sorted(by_case.items()):
            fire = 100.0 * sum(f for f, _ in rows) / len(rows)
            contenders = [c for _, c in rows if c is not None]
            worst = f"{max(contenders):.3f}" if contenders else "-"
            print(f"{name:20} {n:>5} {fire:>5.0f}% {worst:>28}")


def main() -> None:
    print(
        f"Single-bend gate: rev>={MIN_REVERSALS}, raw&robust gain>{FLOOR}, "
        f"max(|p|,|s|)<{WEAK_CEILING}"
    )
    _run_family("NEGATIVES (target: 0% fire)", _negatives)
    _run_family("POSITIVES (V / off-center U; target: fire)", _positives)


if __name__ == "__main__":
    main()
