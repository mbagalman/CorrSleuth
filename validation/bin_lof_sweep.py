"""Threshold-calibration sweep for the bin lack-of-fit diagnostic.

Rebuilds (and extends) the validation sweep that sets
``BIN_LOF_R2_GAIN_THRESHOLD`` and ``OSCILLATION_BIN_LOF_FLOOR`` in
``corrsleuth/heuristics/classifier.py``. The original run was a one-off whose
results were baked into docstrings but never committed; this script makes it
re-runnable so any future change to the statistic can be re-locked (see FU-A in
docs/development/pre-release-review/10-review-fix-sequencing.md).

It computes, for every grid cell, BOTH statistics from the same bins:

- ``raw``  — the shipped ``R²_bins - R²_linear`` (positive null bias ~(k-2)/(n-1)).
- ``adj``  — the df-adjusted (adjusted-R²) gain that replaces it.

so one run shows the before/after and drives the new threshold.

Key extension over the historical sweep: a **bivariate-normal family at moderate
ρ**. The old sweep used only ``make_relationship`` shapes, whose "linear" cases
sit at ρ≈0.87+ even at max noise, so it never exercised the ρ≈0.5-0.7 regime
where the raw statistic mislabels ordinary noisy-linear data. That blind spot is
why the bias shipped.

Run: ``python validation/bin_lof_sweep.py``  (needs the base install only).
Excluded from the sdist via MANIFEST.in, like tests/.
"""

from __future__ import annotations

import numpy as np

from corrsleuth.datasets import make_relationship

_SIZES = (100, 200, 500, 1000)
_NOISE = (0.1, 0.3, 0.5, 1.0)
_SEEDS = range(10)
_RHOS = (0.4, 0.5, 0.6, 0.7)

# Mirrors metrics/shape.py exactly.
_TARGET_POINTS_PER_BIN = 10
_MIN_BINS, _MAX_BINS = 5, 20

# Shapes whose conditional mean is a straight line or has no trend: the bin-LoF
# curvature gate must NOT fire on these (false positives here are the bug).
_CURVATURE_NEGATIVE = (
    "linear_positive",
    "linear_negative",
    "independent",
    "heteroscedastic",  # linear mean, non-constant variance
    "bowtie_variance",  # linear mean, symmetric variance
    "outlier_driven",  # leverage, not curvature
)
# Shapes with genuine smooth/step curvature the gate SHOULD catch.
_CURVATURE_POSITIVE = (
    "exponential_monotonic",
    "logarithmic_monotonic",
    "threshold_step",
    "monotonic_log",
)
# Reported separately: real curvature, but routed by sq_corr not bin-LoF.
_CURVATURE_OTHER = ("u_shape", "circular")


def _gains(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int] | None:
    """Return (raw_gain, adjusted_gain, n_bins) for one (x, y) sample, or None
    for a degenerate response. Binning identical to metrics/shape.py."""
    n = len(x)
    order = np.argsort(x, kind="mergesort")
    xs, ys = x[order], y[order]
    n_bins = int(np.clip(n // _TARGET_POINTS_PER_BIN, _MIN_BINS, _MAX_BINS))
    bins = np.array_split(np.arange(n), n_bins)
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    if ss_tot == 0.0:
        return None
    ss_res_bins = float(sum(np.sum((ys[b] - ys[b].mean()) ** 2) for b in bins))
    slope, intercept = np.polyfit(xs, ys, 1)
    ss_res_lin = float(np.sum((ys - (slope * xs + intercept)) ** 2))

    raw = (ss_res_lin - ss_res_bins) / ss_tot
    adj_bins = 1.0 - (ss_res_bins / (n - n_bins)) / (ss_tot / (n - 1))
    adj_lin = 1.0 - (ss_res_lin / (n - 2)) / (ss_tot / (n - 1))
    adj = adj_bins - adj_lin
    return raw, adj, n_bins


def _bivariate_normal(rho: float, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = rho * x + np.sqrt(1.0 - rho**2) * rng.normal(size=n)
    return x, y


# --- Heavy-tailed-Y oscillation false positive (FU-U) ------------------------
# A pathologically heavy-tailed target (exp of a wide uniform, ~150x median tail)
# against an *independent* predictor. A few extreme Y values dominate whichever
# bins they land in, occasionally pushing the raw gain over the 0.15 oscillation
# floor with >= 2 reversals -- a false "nonmonotonic/oscillating" read. The
# leave-one-bin-out robust gain (mirroring metrics/shape.py) collapses when the
# structure is carried by a single bin, which is what the classifier now gates
# the oscillation route on.
_HYSTERESIS = 0.15  # _BIN_REVERSAL_HYSTERESIS_FRACTION in metrics/shape.py
_OSC_FLOOR = 0.15  # OSCILLATION_BIN_LOF_FLOOR in heuristics/classifier.py


def _adj_gain(xs: np.ndarray, ys: np.ndarray, n_bins: int) -> float | None:
    """Df-adjusted gain for pre-sorted (xs, ys) split into n_bins equal bins."""
    n = len(xs)
    bins = np.array_split(np.arange(n), n_bins)
    if any(len(b) < 2 for b in bins):
        return None
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    if ss_tot == 0.0:
        return None
    pred = np.empty_like(ys)
    for b in bins:
        pred[b] = ys[b].mean()
    ss_bins = float(np.sum((ys - pred) ** 2))
    try:
        sl, ic = np.polyfit(xs, ys, 1)
    except np.linalg.LinAlgError:
        return None
    ss_lin = float(np.sum((ys - (sl * xs + ic)) ** 2))
    adj_bins = 1.0 - (ss_bins / (n - n_bins)) / (ss_tot / (n - 1))
    adj_lin = 1.0 - (ss_lin / (n - 2)) / (ss_tot / (n - 1))
    return adj_bins - adj_lin


def _turning_points(means: np.ndarray, hyst: float) -> int:
    r, d, anchor, extreme = 0, 0, means[0], means[0]
    for v in means[1:]:
        if d == 0:
            if abs(v - anchor) >= hyst:
                d = 1 if v > anchor else -1
                extreme = v
        elif d == 1:
            if v > extreme:
                extreme = v
            elif extreme - v >= hyst:
                r += 1
                d, extreme = -1, v
        else:
            if v < extreme:
                extreme = v
            elif v - extreme >= hyst:
                r += 1
                d, extreme = 1, v
    return r


def _osc_stats(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int] | None:
    """Return (raw_gain, robust_gain, reversals) for the oscillation gate."""
    n = len(x)
    order = np.argsort(x, kind="mergesort")
    xs, ys = x[order], y[order]
    nb = int(np.clip(n // _TARGET_POINTS_PER_BIN, _MIN_BINS, _MAX_BINS))
    bins = np.array_split(np.arange(n), nb)
    raw = _adj_gain(xs, ys, nb)
    if raw is None:
        return None
    robust = raw
    for j in range(nb):
        keep = np.concatenate([bins[i] for i in range(nb) if i != j])
        g = _adj_gain(xs[keep], ys[keep], nb - 1)
        if g is not None and g == g:
            robust = min(robust, g)
    means = np.array([ys[b].mean() for b in bins])
    rng = float(means.max() - means.min())
    rev = 0 if rng <= 0 else _turning_points(means, _HYSTERESIS * rng)
    return raw, robust, rev


def _heavy_tail(seed: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.normal(size=n), np.exp(rng.uniform(0.1, 10, size=n))


def _collect() -> dict:
    """Run the full grid, returning raw/adj gain arrays keyed by group and n."""
    out: dict = {}

    def add(group: str, n: int, raw: float, adj: float) -> None:
        out.setdefault((group, n), {"raw": [], "adj": []})
        out[(group, n)]["raw"].append(raw)
        out[(group, n)]["adj"].append(adj)

    for shape in (
        _CURVATURE_NEGATIVE + _CURVATURE_POSITIVE + _CURVATURE_OTHER + ("sinusoidal",)
    ):
        group = (
            "NEG"
            if shape in _CURVATURE_NEGATIVE
            else "POS"
            if shape in _CURVATURE_POSITIVE
            else "SIN"
            if shape == "sinusoidal"
            else shape  # u_shape / circular reported by name
        )
        for n in _SIZES:
            for noise in _NOISE:
                for seed in _SEEDS:
                    df = make_relationship(shape, n=n, noise=noise, random_state=seed)
                    g = _gains(df["x"].to_numpy(), df["y"].to_numpy())
                    if g is not None:
                        add(group, n, g[0], g[1])
                        if shape == "independent":
                            add("NOISE", n, g[0], g[1])  # pure-noise floor probe

    for rho in _RHOS:
        for n in _SIZES:
            for seed in _SEEDS:
                x, y = _bivariate_normal(rho, n, seed)
                g = _gains(x, y)
                if g is not None:
                    add(f"BVN_rho={rho}", n, g[0], g[1])
    return out


def _fp_rate(vals: list[float], thr: float) -> float:
    return float(np.mean(np.asarray(vals) > thr)) if vals else float("nan")


def main() -> None:
    data = _collect()

    def pool(group_prefix: str) -> dict[int, list[float]]:
        by_n: dict[int, dict[str, list[float]]] = {}
        for (grp, n), d in data.items():
            if grp.startswith(group_prefix):
                by_n.setdefault(n, {"raw": [], "adj": []})
                by_n[n]["raw"].extend(d["raw"])
                by_n[n]["adj"].extend(d["adj"])
        return by_n

    neg = pool("NEG")
    bvn = pool("BVN")
    pos = pool("POS")
    sin = pool("SIN")

    print("=" * 78)
    print("NEGATIVE CONTROLS (linear/no-trend shapes) — gain must stay LOW")
    print("  n     raw_mean  raw_p95   raw_max  | adj_mean  adj_p95   adj_max")
    for n in _SIZES:
        r, a = np.array(neg[n]["raw"]), np.array(neg[n]["adj"])
        print(
            f"  {n:<5} {r.mean():+.4f}  {np.percentile(r, 95):+.4f}  {r.max():+.4f} "
            f" | {a.mean():+.4f}  {np.percentile(a, 95):+.4f}  {a.max():+.4f}"
        )

    print("\nBIVARIATE NORMAL (the historical blind spot) — gain must stay LOW")
    print("  n     raw_mean  raw_p95   raw_max  | adj_mean  adj_p95   adj_max")
    for n in _SIZES:
        r, a = np.array(bvn[n]["raw"]), np.array(bvn[n]["adj"])
        print(
            f"  {n:<5} {r.mean():+.4f}  {np.percentile(r, 95):+.4f}  {r.max():+.4f} "
            f" | {a.mean():+.4f}  {np.percentile(a, 95):+.4f}  {a.max():+.4f}"
        )

    print("\nCURVATURE POSITIVES (exp/log/step/monotonic_log) — gain should be HIGH")
    print("  n     raw_min   raw_p05   raw_mean | adj_min   adj_p05   adj_mean")
    for n in _SIZES:
        r, a = np.array(pos[n]["raw"]), np.array(pos[n]["adj"])
        print(
            f"  {n:<5} {r.min():+.4f}  {np.percentile(r, 5):+.4f}  {r.mean():+.4f} "
            f" | {a.min():+.4f}  {np.percentile(a, 5):+.4f}  {a.mean():+.4f}"
        )

    print("\nSINUSOID (oscillation route) — adj gain vs the OSCILLATION floor")
    print("  n     adj_min   adj_p05   adj_mean")
    for n in _SIZES:
        a = np.array(sin[n]["adj"])
        print(f"  {n:<5} {a.min():+.4f}  {np.percentile(a, 5):+.4f}  {a.mean():+.4f}")

    noise = pool("NOISE")
    print("\nOSCILLATION FLOOR — pure-noise ceiling vs sinusoid floor (adj), by n")
    print("  n     noise_max   sin_min    sin_p05")
    for n in _SIZES:
        nz, sn = np.array(noise[n]["adj"]), np.array(sin[n]["adj"])
        print(
            f"  {n:<5} {nz.max():+.4f}    {sn.min():+.4f}   {np.percentile(sn, 5):+.4f}"
        )
    all_noise = np.concatenate([np.array(noise[n]["adj"]) for n in _SIZES])
    all_sin = np.concatenate([np.array(sin[n]["adj"]) for n in _SIZES])
    print("  floor  noise_FP  sin_detect   (chosen OSCILLATION_BIN_LOF_FLOOR = 0.15)")
    for f in (0.15, 0.20, 0.30):
        print(
            f"  {f:.2f}   {np.mean(all_noise > f):.3f}     {np.mean(all_sin > f):.3f}"
        )

    print("\n" + "=" * 78)
    print("FALSE-POSITIVE vs DETECTION at candidate curvature thresholds (adj stat)")
    all_neg = [v for n in _SIZES for v in neg[n]["adj"]] + [
        v for n in _SIZES for v in bvn[n]["adj"]
    ]
    all_pos = [v for n in _SIZES for v in pos[n]["adj"]]
    all_neg_raw = [v for n in _SIZES for v in neg[n]["raw"]] + [
        v for n in _SIZES for v in bvn[n]["raw"]
    ]
    all_pos_raw = [v for n in _SIZES for v in pos[n]["raw"]]
    print("  thr    adj_FP   adj_detect | raw_FP   raw_detect")
    for thr in (0.02, 0.03, 0.04, 0.05):
        print(
            f"  {thr:.2f}   {_fp_rate(all_neg, thr):.3f}    {_fp_rate(all_pos, thr):.3f} "
            f"     | {_fp_rate(all_neg_raw, thr):.3f}    {_fp_rate(all_pos_raw, thr):.3f}"
        )

    # Detail: FP on the moderate-rho blind spot alone, at n=100 (worst case).
    print("\nADJ false-positive on bivariate normal by (rho, n=100) at thr=0.03/0.05")
    for rho in _RHOS:
        vals = data[(f"BVN_rho={rho}", 100)]["adj"]
        raws = data[(f"BVN_rho={rho}", 100)]["raw"]
        print(
            f"  rho={rho}: adj FP@0.03={_fp_rate(vals, 0.03):.2f} FP@0.05={_fp_rate(vals, 0.05):.2f}"
            f"  | raw FP@0.05={_fp_rate(raws, 0.05):.2f}"
        )

    print("\n" + "=" * 78)
    print("HEAVY-TAILED-Y OSCILLATION FALSE POSITIVE (FU-U) — raw vs robust gain")
    print(
        "  independent predictor vs exp(uniform(0.1,10)) target; osc = rev>=2 & gain>0.15"
    )
    print(
        "  n     seeds  raw_g_p95  raw_g_max  rob_g_p95  rob_g_max  osc_raw  osc_robust"
    )
    ht_seeds = range(600)
    ht_osc_raw = ht_osc_rob = 0
    for n in _SIZES:
        raws, robs = [], []
        o_raw = o_rob = 0
        for seed in ht_seeds:
            x, y = _heavy_tail(seed, n)
            st = _osc_stats(x, y)
            if st is None:
                continue
            raw, robust, rev = st
            raws.append(raw)
            robs.append(robust)
            if rev >= 2 and raw > _OSC_FLOOR:
                o_raw += 1
            if rev >= 2 and raw > _OSC_FLOOR and robust > _OSC_FLOOR:
                o_rob += 1
        ht_osc_raw += o_raw
        ht_osc_rob += o_rob
        r, b = np.array(raws), np.array(robs)
        print(
            f"  {n:<5} {len(raws):<6} {np.percentile(r, 95):+.4f}   {r.max():+.4f}  "
            f" {np.percentile(b, 95):+.4f}   {b.max():+.4f}   {o_raw:<7}  {o_rob}"
        )
    print(
        f"  TOTAL oscillation false positives across all n: raw={ht_osc_raw}  "
        f"robust={ht_osc_rob}  (the robust gate is what the classifier applies)"
    )

    print("\n" + "=" * 78)
    print("LOCKED thresholds (heuristics/classifier.py):")
    print("  BIN_LOF_R2_GAIN_THRESHOLD = 0.05  (unchanged; ~0 FP on the moderate-rho")
    print(
        "                                     blind spot once the gain is df-adjusted)"
    )
    print("  OSCILLATION_BIN_LOF_FLOOR = 0.15  (was 0.30 on the unadjusted statistic)")
    print("  + oscillation route also requires the leave-one-bin-out ROBUST gain")
    print("    to clear the floor (FU-U), eliminating the heavy-tailed-Y artifact")
    print("    above without changing the floor or the curvature threshold.")


if __name__ == "__main__":
    main()
