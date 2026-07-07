"""Calibration sweep for the moderate-Spearman curvature route into
``monotonic_nonlinear`` (blind-test finding: a noisy cubic, flat in the middle
and steep in the tails, depresses Spearman below Pearson, so gating the label on
a *strong* Spearman under-labels it ``mixed_or_ambiguous`` even though the
bin-LoF clearly confirms the bend).

The rule now also promotes when curvature is confirmed by BOTH the raw and the
leave-one-bin-out robust bin-LoF gain, ``max(|p|,|s|)`` is strong, and Spearman
still shows a genuine monotone trend (``>= WEAK``, no sign conflict). Because the
change only *adds* an OR branch, the sole risk is a **false positive**: promoting
a genuine straight line or a leverage artifact. This sweep quantifies that.

- NEGATIVE families (must almost never read ``monotonic_nonlinear``): bivariate
  normals across rho, linear-with-heteroscedastic-noise (Pearson high, Spearman
  can drift below it), and strong-linear-plus-outliers (leverage that inflates
  the raw bin-LoF but collapses the robust one).
- POSITIVE families (should read ``monotonic_nonlinear``): flat-middle cubics
  (the finding), exponential / logarithmic / sigmoid curves - reported split by
  whether Spearman is *moderate* (< STRONG), i.e. the regime only the new route
  reaches.

Run: ``python validation/curvature_promotion_sweep.py``  (base install only).
Excluded from the sdist via MANIFEST.in, like tests/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from corrsleuth.heuristics import classifier as clf
from corrsleuth.metrics import compute_bin_lof, compute_pearson, compute_spearman
from corrsleuth.validation.input import validate_pair

_STRONG = clf.STRONG_MAGNITUDE_THRESHOLD
_WEAK = clf.WEAK_MAGNITUDE_THRESHOLD
_BIN = clf.BIN_LOF_R2_GAIN_THRESHOLD
_RANK_GAP = clf.RANK_LINEAR_GAP_THRESHOLD
_CONFLICT = clf.CONFLICTING_SIGN_THRESHOLD


def _rule5(x: np.ndarray, y: np.ndarray) -> tuple[bool, bool, float]:
    """Evaluate rule 5 (monotonic_nonlinear) both ways on one sample, isolated
    from the rest of the cascade. Returns (old_fires, new_fires, |spearman|).

    ``new_fires and not old_fires`` is exactly what this change *adds* — so on
    linear/leverage families it is the false-positive count attributable to the
    fix (an upper bound, since earlier rules like leverage/weak may preempt)."""
    pair = validate_pair(pd.DataFrame({"x": x, "y": y}), "x", "y")
    pv = compute_pearson(pair).value
    sv = compute_spearman(pair).value
    b = compute_bin_lof(pair)
    bin_lof = b["bin_lof_r2_gain"].value
    bin_lof_robust = b["bin_lof_r2_gain_robust"].value
    p, s = abs(pv), abs(sv)
    conflict = pv * sv < 0 and p >= _CONFLICT and s >= _CONFLICT
    curved = bin_lof is not None and bin_lof > _BIN

    old = not conflict and s > _STRONG and ((s - p > _RANK_GAP) or curved)
    new = not conflict and (
        (s > _STRONG and s - p > _RANK_GAP)
        or (
            curved
            and (
                s > _STRONG
                or (
                    max(p, s) > _STRONG
                    and s >= _WEAK
                    and bin_lof_robust is not None
                    and bin_lof_robust > _BIN
                )
            )
        )
    )
    return old, new, s


# --- families --------------------------------------------------------------
def bvn(seed: int, n: int, rho: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return x, rho * x + np.sqrt(1 - rho**2) * rng.normal(size=n)


def linear_hetero(seed: int, n: int, noise: float) -> tuple[np.ndarray, np.ndarray]:
    # Linear mean, spread grows with |x| - Pearson high, no real curvature.
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return x, 0.9 * x + noise * (0.3 + np.abs(x)) * rng.normal(size=n)


def leverage(seed: int, n: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    # Strong clean linear plus k extreme outliers off the line (inflates the raw
    # bin-LoF; the robust gain should collapse).
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = x + 0.1 * rng.normal(size=n)
    idx = rng.choice(n, k, replace=False)
    y[idx] += rng.choice([-1, 1], k) * rng.uniform(6, 10, k)
    return x, y


def cubic(seed: int, n: int, noise: float) -> tuple[np.ndarray, np.ndarray]:
    # Flat in the middle, steep in the tails: monotone, but ranks compress so
    # Spearman sits below Pearson (the X9 blind-test shape).
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return x, x**3 + noise * rng.normal(size=n)


def exp_curve(seed: int, n: int, noise: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, size=n)
    return x, np.exp(x) + noise * rng.normal(size=n)


def log_curve(seed: int, n: int, noise: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.1, 6, size=n)
    return x, np.log(x) + noise * rng.normal(size=n)


def sigmoid(seed: int, n: int, noise: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-6, 6, size=n)
    return x, 1.0 / (1.0 + np.exp(-x)) + noise * rng.normal(size=n)


def _run(name, gen, params, seeds, positive):
    """Report old vs new rule-5 firing rates and the *incremental* promotions
    (new AND NOT old) this change introduces."""
    total = old_n = new_n = incr = 0
    for p in params:
        for s in seeds:
            x, y = gen(s, *p)
            old, new, _ = _rule5(x, y)
            total += 1
            old_n += old
            new_n += new
            incr += new and not old
    tag = "INCREMENTAL FP" if not positive else "detect(new)"
    print(
        f"  {name:34} old={old_n / total:.3f} new={new_n / total:.3f}  "
        f"{tag}: {incr}/{total} = {incr / total:.3f}"
    )
    return incr, total


def main() -> None:
    seeds = range(40)
    print("=" * 78)
    print("NEGATIVE families - INCREMENTAL FP (new rule fires where old did not)")
    print("must be ~0; 'old'/'new' are the total rule-5 firing rates for context.")
    neg = 0
    neg_total = 0
    for name, gen, params in (
        (
            "bivariate normal rho .5-.9",
            bvn,
            [(n, r) for n in (100, 300, 1000) for r in (0.5, 0.6, 0.7, 0.8, 0.9)],
        ),
        (
            "linear + heteroscedastic noise",
            linear_hetero,
            [(n, nz) for n in (100, 300, 1000) for nz in (0.3, 0.6, 1.0)],
        ),
        (
            "strong linear + 1-3 outliers",
            leverage,
            [(n, k) for n in (200, 500) for k in (1, 2, 3)],
        ),
    ):
        m, t = _run(name, gen, params, seeds, positive=False)
        neg += m
        neg_total += t
    print(
        f"  {'TOTAL negative':34} INCREMENTAL FP: {neg}/{neg_total} = {neg / neg_total:.4f}"
    )

    print("\n" + "=" * 78)
    print("POSITIVE families - new-route detection should be high; the incremental")
    print("column is the curves the old strong-Spearman gate under-labeled.")
    for name, gen, params in (
        (
            "cubic (flat middle, X9 shape)",
            cubic,
            [(n, nz) for n in (300, 1000) for nz in (0.5, 1.0, 2.0)],
        ),
        (
            "exponential curve",
            exp_curve,
            [(n, nz) for n in (300, 1000) for nz in (0.3, 0.8, 1.5)],
        ),
        (
            "logarithmic curve",
            log_curve,
            [(n, nz) for n in (300, 1000) for nz in (0.2, 0.5, 1.0)],
        ),
        (
            "sigmoid S-curve",
            sigmoid,
            [(n, nz) for n in (300, 1000) for nz in (0.05, 0.15, 0.3)],
        ),
    ):
        _run(name, gen, params, seeds, positive=True)

    print("\n" + "=" * 78)
    print("Rule (heuristics/classifier.py, rule 5): the moderate-Spearman route")
    print("requires bin_lof > 0.05 AND bin_lof_robust > 0.05 AND max(|p|,|s|) >")
    print("STRONG (0.50) AND |spearman| >= WEAK (0.20) AND no sign conflict. The")
    print("robust gain is what keeps leverage 'curvature' (one outlier bin) out.")


if __name__ == "__main__":
    main()
