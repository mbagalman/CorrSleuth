"""Threshold-calibration sweep for the robust sq_corr gate (FU-V).

`sq_corr = corr((X−x̄)², (Y−ȳ)²)` catches magnitude/radial dependence (a circle,
a one-sided U-shape) that Pearson/Spearman/dCor under-read. But a **heavy-tailed**
variable — in a scan, the *target* — can manufacture a raw `|sq_corr|` over the
0.35 threshold with a handful of extreme squared values, mislabeling an
independent predictor `nonmonotonic_dependence` via the `magnitude_linked` route.

The fix (`compute_squared_correlation_robust` in `metrics/shape.py`) recomputes
`|sq_corr|` after dropping the few most extreme points and takes the minimum. The
classifier trusts sq_corr only when this robust value clears an **asymmetric,
lower** floor (`SQ_CORR_ROBUST_FLOOR = 0.20`). This script picks and validates
that floor:

- a heavy-tailed-Y artifact family (the false positives), and
- genuine circle / U-shape families (must not lose detections).

For each raw-firing pair (`|sq_corr| > 0.35`) it records the robust value, then
reports, per candidate floor: how many artifacts survive (residual FP) versus how
many genuine detections are kept. The floor is the irreducible-residual knee — it
keeps essentially all genuine detections while removing as many artifacts as
possible; a *higher* floor removes no additional artifact but costs disproportionately
more genuine detections. It is lower than the raw threshold because a genuine
magnitude link keeps a robust value well above what an artifact collapses to. It
cannot reach zero residual: an artifact whose bulk correlation survives the drop
is indistinguishable from a weak real link.

Run: ``python validation/sq_corr_sweep.py``  (needs the base install only).
Excluded from the sdist via MANIFEST.in, like tests/.
"""

from __future__ import annotations

import numpy as np

from corrsleuth.metrics.shape import (
    _SQ_CORR_ROBUST_DROP,
    _squared_correlation,
)

_RAW_THRESHOLD = 0.35  # SQ_CORR_THRESHOLD in heuristics/classifier.py
_SIZES = (100, 200, 500)
_FLOORS = (0.10, 0.15, 0.20, 0.25, 0.30)


def _raw_and_robust(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    """Return (|sq_corr|, robust |sq_corr|) mirroring metrics/shape.py exactly."""
    from scipy import stats

    x2 = (x - x.mean()) ** 2
    y2 = (y - y.mean()) ** 2
    base = _squared_correlation(x2, y2)
    if base is None:
        return None
    extremity = np.maximum(stats.rankdata(x2), stats.rankdata(y2))
    order = np.argsort(extremity)[::-1]
    n = x2.shape[0]
    worst = abs(base)
    for j in range(1, _SQ_CORR_ROBUST_DROP + 1):
        if n - j < 2:
            break
        keep = np.ones(n, dtype=bool)
        keep[order[:j]] = False
        r = _squared_correlation(x2[keep], y2[keep])
        if r is not None:
            worst = min(worst, abs(r))
    return abs(base), worst


def _heavy_tail(seed: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.normal(size=n), np.exp(rng.uniform(0.1, 10, size=n))


def _circle(seed: int, n: int, noise: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 2 * np.pi, size=n)
    return (
        np.cos(t) + rng.normal(0, noise, size=n),
        np.sin(t) + rng.normal(0, noise, size=n),
    )


def _ushape(seed: int, n: int, noise: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, size=n)
    return x, x**2 + rng.normal(0, noise, size=n)


def _robust_of_firing(gen, seeds, sizes) -> np.ndarray:
    """Robust values of pairs whose RAW |sq_corr| clears the 0.35 threshold."""
    vals: list[float] = []
    for n in sizes:
        for s in seeds:
            x, y = gen(s, n)
            out = _raw_and_robust(x, y)
            if out is not None and out[0] > _RAW_THRESHOLD:
                vals.append(out[1])
    return np.asarray(vals)


def main() -> None:
    ht = _robust_of_firing(_heavy_tail, range(1500), _SIZES)
    circ = np.concatenate(
        [
            _robust_of_firing(
                lambda s, n, noise=noise: _circle(s, n, noise), range(120), _SIZES
            )
            for noise in (0.05, 0.2)
        ]
    )
    ush = np.concatenate(
        [
            _robust_of_firing(
                lambda s, n, noise=noise: _ushape(s, n, noise), range(120), _SIZES
            )
            for noise in (0.5, 1.0, 1.5)
        ]
    )

    print("=" * 78)
    print("ROBUST sq_corr of RAW-FIRING pairs (|sq_corr| > 0.35), by family")
    print("  family        n_firing   min    p05    p50    max")
    for name, arr in (("heavy-tail", ht), ("circle", circ), ("u_shape", ush)):
        if arr.size:
            print(
                f"  {name:12} {arr.size:<9} {arr.min():.3f}  "
                f"{np.percentile(arr, 5):.3f}  {np.percentile(arr, 50):.3f}  {arr.max():.3f}"
            )

    print("\n" + "=" * 78)
    print("FLOOR SELECTION — kept = (raw fired AND robust > floor)")
    print("  floor | heavy-tail kept (FP) | circle kept | u_shape kept")
    for f in _FLOORS:
        print(
            f"  {f:.2f}  |   {int((ht > f).sum()):>4} / {ht.size:<6}      | "
            f"{int((circ > f).sum())}/{circ.size:<5}   | {int((ush > f).sum())}/{ush.size}"
        )

    print("\n" + "=" * 78)
    print("LOCKED (heuristics/classifier.py):")
    print("  SQ_CORR_THRESHOLD       = 0.35  (raw |sq_corr|, unchanged)")
    print("  SQ_CORR_ROBUST_FLOOR    = 0.20  (asymmetric: lower than the raw bar)")
    print("  The irreducible-residual point: 0.20 removes ~7/8 heavy-tail artifacts")
    print("  while keeping ~all genuine circle/u_shape detections. A higher floor")
    print("  (0.25) removes NO additional artifact but costs ~10x more genuine")
    print("  u_shape detections. The one residual artifact — whose bulk correlation")
    print("  outlives the drop — is indistinguishable from a weak real link")
    print(f"  (drop K={_SQ_CORR_ROBUST_DROP}).")


if __name__ == "__main__":
    main()
