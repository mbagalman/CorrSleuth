"""Threshold-calibration sweep for the two-group / mixture split gate.

`compute_cluster_split` (metrics/mixture.py) measures whether a pair's pooled
correlation is carried by a between-group mean shift: the best two-group split
of the association-axis projection (`cluster_split_r2`), the emptiness of the
band around the split boundary (`cluster_valley_share` — "no points bridging
the gap"), the smaller group's share (`cluster_min_share`), and how much
association survives inside the groups (`pearson_within_cluster`). The
classifier fires `dependence_type = "two_group_shift"` plus a warning when all
gates hold jointly (see `_is_two_group_shift` in heuristics/classifier.py).

This script calibrates the gate floors:

- **Negatives** (must not fire): bivariate normals across correlations,
  linear/monotone links with uniform / exponential / lognormal / t3 marginals,
  smooth curves (quadratic, saturation, sigmoid, log), a step and a changepoint
  *with within-segment slopes*, heteroscedastic fans, a leverage cluster, a
  sparse 8% subgroup, and pure noise. A *flat* step of a continuous variable is
  reported separately: it is the same joint distribution as a two-group mixture
  (two separated groups, mean shift, no within-group trend), so the gate firing
  there is by design, not a false positive — the warning text presents both
  readings.
- **Positives** (should fire): two diagonal Gaussian blobs across separations
  (in within-group stds), group shares, and within-group correlations,
  including the mixture-with-mild-within-trend case.

For every case it reports each gate value and the joint firing rate under the
candidate floors, so a floor change can be justified against measured margins.

Run: ``python validation/cluster_split_sweep.py``  (base install only).
Excluded from the sdist via MANIFEST.in, like tests/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from corrsleuth.metrics.mixture import compute_cluster_split
from corrsleuth.validation.input import validate_pair

# Candidate gate floors (the shipped values live in heuristics/classifier.py).
SPLIT_R2_FLOOR = 0.70
VALLEY_SHARE_CEILING = 0.03
MIN_SHARE_FLOOR = 0.10
GLOBAL_PEARSON_FLOOR = 0.35
WITHIN_RATIO_CEILING = 0.40

_SIZES = (100, 200, 500, 1500)
_SEEDS = range(10)


def _fires(x: np.ndarray, y: np.ndarray) -> tuple[bool, dict]:
    df = pd.DataFrame({"x": x, "y": y})
    pair = validate_pair(df, "x", "y")
    m = compute_cluster_split(pair)
    p = abs(float(np.corrcoef(x, y)[0, 1]))
    vals = {name: m[name].value for name in m}
    vals["pearson"] = p
    fired = (
        vals["cluster_split_r2"] is not None
        and vals["cluster_split_r2"] > SPLIT_R2_FLOOR
        and vals["cluster_valley_share"] is not None
        and vals["cluster_valley_share"] < VALLEY_SHARE_CEILING
        and vals["cluster_min_share"] is not None
        and vals["cluster_min_share"] >= MIN_SHARE_FLOOR
        and vals["pearson_within_cluster"] is not None
        and p >= GLOBAL_PEARSON_FLOOR
        and vals["pearson_within_cluster"] <= WITHIN_RATIO_CEILING * p
    )
    return fired, vals


def _negatives(rng: np.random.Generator, n: int) -> dict[str, tuple]:
    z = rng.normal(size=n)
    e = rng.normal(size=n)
    u = rng.uniform(-3, 3, size=n)
    ex = rng.exponential(1.0, size=n)
    ln = np.exp(rng.normal(size=n))
    t3 = rng.standard_t(3, size=n)
    cases = {
        "noise": (z, rng.normal(size=n)),
        "bvn_r0.5": (z, 0.5 * z + np.sqrt(1 - 0.25) * e),
        "bvn_r0.7": (z, 0.7 * z + np.sqrt(1 - 0.49) * e),
        "bvn_r0.9": (z, 0.9 * z + np.sqrt(1 - 0.81) * e),
        "uniform_linear": (u, 0.8 * u + rng.normal(0, 0.5, n)),
        "exp_linear": (ex, ex + rng.normal(0, 0.4, n)),
        "lognorm_linear": (ln, 2 * ln + rng.normal(0, 0.5, n)),
        "t3_linear": (t3, t3 + rng.normal(0, 0.5, n)),
        "quadratic": (u, u**2 + rng.normal(0, 0.5, n)),
        "saturation": (
            rng.uniform(0, 10, n),
            None,  # filled below (needs its own x)
        ),
        "sigmoid": (u, 1 / (1 + np.exp(-2 * u)) + rng.normal(0, 0.05, n)),
        "log_curve": (rng.uniform(0.5, 10, n), None),
        "step_with_slope": (
            u,
            np.where(u > 0, 1.5 + 0.5 * u, 0.5 * u) + rng.normal(0, 0.2, n),
        ),
        "changept_slopes": (
            u,
            np.where(u > 0, 0.2 * u + 1.0, 0.9 * u) + rng.normal(0, 0.2, n),
        ),
        "hetero_fan": (u, 0.8 * u + rng.normal(0, 0.2 + 0.4 * np.abs(u), n)),
        "leverage_2pct": (None, None),
        "subgroup_8pct": (None, None),
    }
    xs = cases["saturation"][0]
    cases["saturation"] = (xs, 3 * (1 - np.exp(-0.8 * xs)) + rng.normal(0, 0.15, n))
    xl = cases["log_curve"][0]
    cases["log_curve"] = (xl, np.log(xl) + rng.normal(0, 0.15, n))
    xb = rng.normal(size=n)
    yb = xb + rng.normal(0, 0.5, n)
    k = max(2, n // 50)
    xb[:k] += 12
    yb[:k] += 12
    cases["leverage_2pct"] = (xb, yb)
    xg = rng.normal(size=n)
    yg = rng.normal(size=n)
    ks = max(8, int(0.08 * n))
    yg[:ks] = xg[:ks] * 0.9 + rng.normal(0, 0.1, ks)
    cases["subgroup_8pct"] = (xg, yg)
    return cases


def _flat_steps(rng: np.random.Generator, n: int) -> dict[str, tuple]:
    """Flat steps of a continuous x: the by-design overlap (reported apart)."""
    u = rng.uniform(-3, 3, size=n)
    return {
        "flat_step_noise0.10": (u, (u > 0).astype(float) + rng.normal(0, 0.10, n)),
        "flat_step_noise0.25": (u, (u > 0).astype(float) + rng.normal(0, 0.25, n)),
    }


def _positives(rng: np.random.Generator, n: int) -> dict[str, tuple]:
    cases: dict[str, tuple] = {}
    for sep in (3.0, 4.0, 5.0):
        for frac in (0.5, 0.25, 0.12):
            n1 = int(n * frac)
            gx = np.concatenate([rng.normal(0, 1, n1), rng.normal(sep, 1, n - n1)])
            gy = np.concatenate([rng.normal(0, 1, n1), rng.normal(sep, 1, n - n1)])
            cases[f"blobs_sep{sep:.0f}_frac{frac}"] = (gx, gy)
    n1 = n // 2
    wx = np.concatenate([rng.normal(0, 1, n1), rng.normal(4, 1, n - n1)])
    wy = 0.3 * wx + np.concatenate([rng.normal(0, 1, n1), rng.normal(4, 1, n - n1)])
    cases["blobs_sep4_within0.3"] = (wx, wy)
    return cases


def _run_family(title: str, build) -> None:
    print(f"\n=== {title} ===")
    print(
        f"{'case':24} {'n':>5} {'fire%':>6} {'r2(min-max)':>13} "
        f"{'valley(max)':>11} {'share(min)':>10} {'within(min)':>11}"
    )
    for n in _SIZES:
        by_case: dict[str, list] = {}
        for seed in _SEEDS:
            rng = np.random.default_rng(1000 * n + seed)
            for name, (x, y) in build(rng, n).items():
                fired, vals = _fires(np.asarray(x, float), np.asarray(y, float))
                by_case.setdefault(name, []).append((fired, vals))
        for name, rows in sorted(by_case.items()):
            fire = 100.0 * sum(f for f, _ in rows) / len(rows)
            r2s = [
                v["cluster_split_r2"]
                for _, v in rows
                if v["cluster_split_r2"] is not None
            ]
            vys = [
                v["cluster_valley_share"]
                for _, v in rows
                if v["cluster_valley_share"] is not None
            ]
            shs = [
                v["cluster_min_share"]
                for _, v in rows
                if v["cluster_min_share"] is not None
            ]
            wis = [
                v["pearson_within_cluster"]
                for _, v in rows
                if v["pearson_within_cluster"] is not None
            ]
            r2r = f"{min(r2s):.2f}-{max(r2s):.2f}" if r2s else "NA"
            vy = f"{max(vys):.3f}" if vys else "NA"
            sh = f"{min(shs):.3f}" if shs else "NA"
            wi = f"{min(wis):.3f}" if wis else "NA"
            print(
                f"{name:24} {n:>5} {fire:>5.0f}% {r2r:>13} {vy:>11} {sh:>10} {wi:>11}"
            )


def main() -> None:
    print(
        "Gate floors under test: "
        f"split_r2>{SPLIT_R2_FLOOR}, valley<{VALLEY_SHARE_CEILING}, "
        f"min_share>={MIN_SHARE_FLOOR}, |p|>={GLOBAL_PEARSON_FLOOR}, "
        f"within<={WITHIN_RATIO_CEILING}*|p|"
    )
    _run_family("NEGATIVES (target: 0% fire)", _negatives)
    _run_family(
        "FLAT STEPS (by-design overlap; same joint law as a mixture)", _flat_steps
    )
    _run_family("POSITIVES (two-blob mixtures; target: fire at sep>=4)", _positives)


if __name__ == "__main__":
    main()
