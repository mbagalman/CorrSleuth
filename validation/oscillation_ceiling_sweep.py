"""Calibration sweep for the oscillation route's moderate-trend ceiling.

The oscillation route into `nonmonotonic_dependence` (and the
`dependence_type = "oscillating"` axis value) fires on `bin_reversal_count >= 2`
jointly with the raw AND leave-one-bin-out `bin_lof_r2_gain` both above 0.15.
It used to share the dc / sq_corr routes' weak-trend ceiling
(`max(|p|, |s|) < 0.25`), which stranded skew-tilted oscillations in a dead
zone: a sinusoid on one-sided (exponential/lognormal) support picks up a
spurious *moderate* monotone tilt (|s| ~ 0.34 for the blind-test exponential
sinusoid) — too much trend for the weak-trend routes, too little for the
strong-trend `oscillating_trend` axis value (|s| >= 0.50).

`OSCILLATION_MONOTONE_CEILING = 0.50` (heuristics/classifier.py) gives the
oscillation route its own ceiling, tiling the trend axis with no gap. This
script validates the newly exposed zone (0.25 <= max(|p|, |s|) < 0.50):

- **Negatives** (must not fire the oscillation gate): moderate-correlation
  bivariate normals, monotone curves and heavy-tailed links noised into the
  moderate band, moderate heteroscedastic fans, leverage, sparse subgroups,
  and single-bend shapes with a moderate tilt (a tilted U reads 1 reversal,
  not 2). The gate's own floors do the work: pure noise never exceeds a
  df-adjusted gain of ~0.05, and smooth monotone shapes measure 0-1 reversals.
- **Positives** (should fire): sinusoids tilted into the moderate band by an
  additive trend or by skewed one-sided support (the blind-test X12 family).

Cases whose sampled max(|p|, |s|) lands outside [0.25, 0.50) are counted
separately (out-of-zone) so the fire rates below describe the contested zone
only.

Measured residual: every negative family is 0% in-zone at every size except
`tilted_u` (a noisy tilted quadratic), where noise wiggle occasionally pushes
the single bend past the reversal hysteresis (2 fires across all in-zone
trials, none at n = 1500). For that shape the *label* is still correct — a
tilted U is genuinely nonmonotonic dependence, previously stranded in
mixed_or_ambiguous — and only the `dependence_type = "oscillating"` axis word
over-specifies its single bend, so the residual is a mild imprecision, not a
false relationship claim.

Run: ``python validation/oscillation_ceiling_sweep.py``  (base install only).
Excluded from the sdist via MANIFEST.in, like tests/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from corrsleuth.metrics.shape import compute_bin_lof
from corrsleuth.validation.input import validate_pair

# The gate under test (shipped values in heuristics/classifier.py).
CEILING = 0.50  # OSCILLATION_MONOTONE_CEILING
MIN_REVERSALS = 2  # OSCILLATION_MIN_REVERSALS
GAIN_FLOOR = 0.15  # OSCILLATION_BIN_LOF_FLOOR (raw AND robust)

_SIZES = (100, 200, 500, 1500)
_SEEDS = range(10)


def _fires(x: np.ndarray, y: np.ndarray) -> tuple[bool, bool]:
    """Return (in_zone, fired) for the moderate-trend oscillation gate."""
    pair = validate_pair(pd.DataFrame({"x": x, "y": y}), "x", "y")
    lof = compute_bin_lof(pair)
    reversals = lof["bin_reversal_count"].value
    gain = lof["bin_lof_r2_gain"].value
    robust = lof["bin_lof_r2_gain_robust"].value
    p = abs(float(np.corrcoef(x, y)[0, 1]))
    s = abs(float(pd.Series(x).corr(pd.Series(y), method="spearman")))
    in_zone = 0.25 <= max(p, s) < CEILING
    fired = (
        max(p, s) < CEILING
        and reversals is not None
        and reversals >= MIN_REVERSALS
        and gain is not None
        and gain > GAIN_FLOOR
        and robust is not None
        and robust > GAIN_FLOOR
    )
    return in_zone, fired


def _negatives(rng: np.random.Generator, n: int) -> dict[str, tuple]:
    z = rng.normal(size=n)
    u = rng.uniform(-3, 3, size=n)
    ex = rng.exponential(1.0, size=n)
    ln = np.exp(rng.normal(size=n))
    cases = {
        # Moderate-correlation elliptical nulls.
        "bvn_r0.30": (z, 0.30 * z + np.sqrt(1 - 0.09) * rng.normal(size=n)),
        "bvn_r0.40": (z, 0.40 * z + np.sqrt(1 - 0.16) * rng.normal(size=n)),
        "bvn_r0.45": (z, 0.45 * z + np.sqrt(1 - 0.2025) * rng.normal(size=n)),
        # Monotone shapes noised into the moderate band.
        "linear_mod": (u, 0.35 * u + rng.normal(0, 1.0, n)),
        "log_curve_mod": (None, None),
        "exp_curve_mod": (u, np.exp(0.5 * u) + rng.normal(0, 2.0, n)),
        "sqrt_mod": (None, None),
        # Heavy-tailed links in the moderate band.
        "lognorm_link_mod": (ln, np.log1p(ln) + rng.normal(0, 1.0, n)),
        "exp_link_mod": (ex, ex + rng.normal(0, 2.5, n)),
        # Moderate heteroscedastic fan.
        "hetero_mod": (u, 0.3 * u + rng.normal(0, 0.3 + 0.5 * np.abs(u), n)),
        # A single-bend shape with a moderate tilt (1 reversal, not 2).
        "tilted_u": (u, 0.35 * u + 0.3 * u**2 + rng.normal(0, 0.8, n)),
        # Leverage cluster tuned toward the moderate band.
        "leverage_mod": (None, None),
        # Sparse subgroup (X20-style).
        "subgroup_mod": (None, None),
    }
    xl = rng.uniform(0.5, 10, n)
    cases["log_curve_mod"] = (xl, np.log(xl) + rng.normal(0, 1.2, n))
    xs = rng.uniform(0, 9, n)
    cases["sqrt_mod"] = (xs, np.sqrt(xs) + rng.normal(0, 1.2, n))
    xb = rng.normal(size=n)
    yb = 0.2 * xb + rng.normal(0, 1.0, n)
    k = max(2, n // 50)
    xb[:k] += 8
    yb[:k] += 8
    cases["leverage_mod"] = (xb, yb)
    xg = rng.normal(size=n)
    yg = rng.normal(size=n)
    ks = max(8, int(0.15 * n))
    yg[:ks] = xg[:ks] * 0.9 + rng.normal(0, 0.1, ks)
    cases["subgroup_mod"] = (xg, yg)
    return cases


def _positives(rng: np.random.Generator, n: int) -> dict[str, tuple]:
    u = rng.uniform(-3, 3, size=n)
    ex = rng.exponential(2.0, size=n)
    return {
        # Sinusoid tilted into the moderate band by an additive trend.
        "sine_tilt_0.25": (
            u,
            0.25 * u + np.sin(2 * np.pi * u / 3) + rng.normal(0, 0.2, n),
        ),
        "sine_tilt_0.40": (
            u,
            0.40 * u + np.sin(2 * np.pi * u / 3) + rng.normal(0, 0.2, n),
        ),
        # Sinusoid on one-sided exponential support (the X12 blind-test family):
        # the skew itself supplies the spurious moderate tilt.
        "sine_exp_support": (ex, np.sin(1.5 * ex) + rng.normal(0, 0.25, n)),
        # Damped oscillation with a mild trend.
        "damped_tilt": (
            u,
            0.3 * u
            + np.exp(-0.3 * np.abs(u)) * np.sin(2.5 * u)
            + rng.normal(0, 0.15, n),
        ),
    }


def _run_family(title: str, build) -> None:
    print(f"\n=== {title} ===")
    print(f"{'case':20} {'n':>5} {'in-zone':>8} {'fire%(zone)':>12} {'fire%(all)':>11}")
    for n in _SIZES:
        by_case: dict[str, list] = {}
        for seed in _SEEDS:
            rng = np.random.default_rng(1000 * n + seed)
            for name, (x, y) in build(rng, n).items():
                in_zone, fired = _fires(np.asarray(x, float), np.asarray(y, float))
                by_case.setdefault(name, []).append((in_zone, fired))
        for name, rows in sorted(by_case.items()):
            zone = [f for z, f in rows if z]
            zone_txt = f"{100.0 * sum(zone) / len(zone):.0f}%" if zone else "-"
            allp = 100.0 * sum(f for _, f in rows) / len(rows)
            print(f"{name:20} {n:>5} {len(zone):>5}/10 {zone_txt:>12} {allp:>10.0f}%")


def main() -> None:
    print(
        f"Oscillation gate: rev>={MIN_REVERSALS}, raw&robust gain>{GAIN_FLOOR}, "
        f"max(|p|,|s|)<{CEILING} (was 0.25 for all routes)"
    )
    _run_family("NEGATIVES (target: 0% fire, esp. in-zone)", _negatives)
    _run_family("POSITIVES (tilted oscillations; target: fire)", _positives)


if __name__ == "__main__":
    main()
