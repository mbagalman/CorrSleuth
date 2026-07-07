import numpy as np
import pandas as pd
import pytest

from corrsleuth.api import profile_pair
from corrsleuth.datasets import make_relationship
from corrsleuth.heuristics.classifier import (
    _dependence_type_axis,
    _functional_direction_axis,
    _mean_shape_axis,
    _outlier_sensitivity_axis,
    _variance_shape_axis,
    apply_heuristics,
    derive_diagnostic_axes,
    detect_metric_warnings,
)
from corrsleuth.result import MetricResult


def test_canonical_shapes_lite():
    # Linear Positive
    df = make_relationship("linear_positive", n=500, noise=0.1, random_state=42)
    res = profile_pair(df, "x", "y", mode="lite")
    assert res.pattern == "near_linear"


def test_canonical_shapes_standard():
    pytest.importorskip("dcor")
    pytest.importorskip("sklearn")
    # U Shape -> nonmonotonic_dependence
    df = make_relationship("u_shape", n=500, noise=0.1, random_state=42)
    res = profile_pair(df, "x", "y", mode="standard")
    assert res.pattern == "nonmonotonic_dependence"

    # Monotonic Log -> monotonic_nonlinear
    df = make_relationship("monotonic_log", n=500, noise=0.1, random_state=42)
    res = profile_pair(df, "x", "y", mode="standard")
    assert res.pattern == "monotonic_nonlinear"

    # Independent -> weak_or_no_relationship
    df = make_relationship("independent", n=500, noise=0.1, random_state=42)
    res = profile_pair(df, "x", "y", mode="standard")
    assert res.pattern == "weak_or_no_relationship"


def test_exponential_monotonic_resolves_to_monotonic_nonlinear():
    # A smooth monotonic curve over an ordinary (non-rigged) X range: Pearson
    # and Spearman stay close together (gap well under
    # RANK_LINEAR_GAP_THRESHOLD), so only bin_lof_r2_gain catches the real
    # curvature. Lite mode: no optional dependency needed.
    df = make_relationship("exponential_monotonic", n=500, noise=0.1, random_state=42)
    res = profile_pair(df, "x", "y", mode="lite")
    assert res.pattern == "monotonic_nonlinear"


def test_logarithmic_monotonic_resolves_to_monotonic_nonlinear():
    df = make_relationship("logarithmic_monotonic", n=500, noise=0.1, random_state=42)
    res = profile_pair(df, "x", "y", mode="lite")
    assert res.pattern == "monotonic_nonlinear"


def test_threshold_step_resolves_to_monotonic_nonlinear():
    # A two-level step function sits in near_linear's regime by the
    # rank-linear gap alone (Pearson and Spearman both moderately strong, gap
    # small), but bin_lof_r2_gain reveals the two flat groups a line doesn't
    # capture.
    df = make_relationship("threshold_step", n=500, noise=0.1, random_state=42)
    res = profile_pair(df, "x", "y", mode="lite")
    assert res.pattern == "monotonic_nonlinear"


def test_circular_resolves_to_nonmonotonic_dependence():
    # Points scattered around a ring: Pearson, Spearman, and distance
    # correlation on the raw values are all near zero (distance correlation is
    # structurally capped around ~0.2 for a true circular relationship), but
    # sq_corr (corr of the mean-centered squares) is strongly negative. Lite mode: no optional
    # dependency needed.
    df = make_relationship("circular", n=500, noise=0.1, random_state=42)
    res = profile_pair(df, "x", "y", mode="lite")
    assert res.pattern == "nonmonotonic_dependence"


@pytest.mark.parametrize("center", [(0.0, 0.0), (5.0, 5.0), (100.0, -40.0)])
@pytest.mark.parametrize("seed", range(5))
def test_offset_circle_still_reads_nonmonotonic_dependence(center, seed):
    """Regression lock for the sq_corr translation-invariance BLOCKER (FU-B).

    ``corr(X², Y²)`` on raw values collapses toward ``corr(X, Y)`` for data far
    from the origin, so before the fix a noisy ring centered anywhere but (0, 0)
    read ``sq_corr`` ≈ 0 and was mislabeled ``weak_or_no_relationship`` (a circle
    at (5, 5) measured −0.05 instead of −0.95). Centering X and Y before squaring
    makes the diagnostic shape-only, so the label must be
    ``nonmonotonic_dependence`` regardless of where the circle sits. The (0, 0)
    cell guards against a regression on the origin case that always worked."""
    import numpy as np
    import pandas as pd

    cx, cy = center
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, size=500)
    radius = 5.0 * (1 + rng.normal(0, 0.05, size=500))
    df = pd.DataFrame(
        {"x": cx + radius * np.cos(theta), "y": cy + radius * np.sin(theta)}
    )
    res = profile_pair(df, "x", "y", mode="lite")

    assert res.pattern == "nonmonotonic_dependence", (
        f"center={center} seed={seed} -> {res.pattern}"
    )
    assert res.diagnostics.sq_corr is not None and res.diagnostics.sq_corr < -0.5, (
        f"center={center} seed={seed} -> sq_corr={res.diagnostics.sq_corr}"
    )


def test_sinusoidal_resolves_to_nonmonotonic_dependence_in_every_mode():
    # A ~2.5-cycle sinusoid: Pearson/Spearman weak, distance correlation only
    # marginally above its floor, sq_corr blind to it. The bin-mean reversal
    # count jointly with the bin lack-of-fit gain (the lite-computable
    # oscillation route) is what labels it — previously lite and deep mode
    # read this as weak_or_no_relationship, actively underselling a strong
    # deterministic function.
    for mode in ("lite", "deep"):
        df = make_relationship("sinusoidal", n=500, noise=0.1, random_state=42)
        res = profile_pair(df, "x", "y", mode=mode)
        assert res.pattern == "nonmonotonic_dependence"
        assert res.diagnostics.dependence_type == "oscillating"
        assert res.diagnostics.bin_reversal_count >= 2
        assert res.diagnostics.bin_lof_r2_gain > 0.3
        # The label now states the dependence, so the deep-mode "xi is high
        # but the label may understate dependence" warning has nothing to
        # correct and must not fire.
        assert not any("may understate" in w for w in res.warnings)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("noise", [0.1, 0.3])
def test_sinusoidal_stays_oscillating_across_seeds_and_noise(seed, noise):
    # OSCILLATION_BIN_LOF_FLOOR / OSCILLATION_MIN_REVERSALS were locked via a
    # 2,080-run sweep; this keeps the shipped generator pinned to the gate
    # across seeds and noise levels the way the other thin-margin thresholds
    # are regression-tested.
    df = make_relationship("sinusoidal", n=500, noise=noise, random_state=seed)
    res = profile_pair(df, "x", "y", mode="lite")
    assert res.pattern == "nonmonotonic_dependence"
    assert res.diagnostics.dependence_type == "oscillating"


@pytest.mark.parametrize("shape", ["u_shape", "circular", "independent"])
@pytest.mark.parametrize("seed", range(10))
def test_non_oscillating_shapes_never_read_oscillating(shape, seed):
    # A single bend (U-shape: exactly 1 reversal), a closed loop (bin means
    # flat, gain below the floor), and pure noise (many reversals, near-zero
    # gain) must all stay out of the oscillation gate.
    df = make_relationship(shape, n=500, noise=0.1, random_state=seed)
    res = profile_pair(df, "x", "y", mode="lite")
    assert res.diagnostics.dependence_type != "oscillating"


@pytest.mark.parametrize("shape", ["linear_positive", "linear_negative"])
@pytest.mark.parametrize("seed", range(10))
def test_linear_shapes_stay_near_linear_across_seeds(shape, seed):
    # BIN_LOF_R2_GAIN_THRESHOLD (0.05) has a thinner margin than most cascade
    # thresholds (genuinely linear data measures ~-0.01 on any single seed), so
    # this regression check runs across many seeds rather than relying on one.
    df = make_relationship(shape, n=500, noise=0.1, random_state=seed)
    res = profile_pair(df, "x", "y", mode="lite")
    assert res.pattern == "near_linear"


@pytest.mark.parametrize("rho", [0.5, 0.6, 0.8])
@pytest.mark.parametrize("n", [100, 300, 500])
@pytest.mark.parametrize("seed", range(10))
def test_moderate_correlation_bivariate_normal_never_reads_curved(rho, n, seed):
    """Regression lock for the bin-lack-of-fit df-bias BLOCKER (FU-A).

    A plain bivariate normal at moderate rho is a straight line with noise. The
    old unadjusted R² gain carried a positive null bias ~(k-2)/(n-1) — above the
    0.05 curvature threshold for n < ~400 — which mislabeled ~25% of the
    rho=0.6/n=100 cells as ``monotonic_nonlinear`` (and drove ``mean_shape``
    ``curved`` on the majority of moderate-correlation pairs). Unlike the
    existing ``linear_positive`` regression, this exercises the moderate-rho
    (~0.5-0.8) regime the make_relationship shapes never reach (their "linear"
    cases sit at rho ≈ 0.87+ even at max noise), which is exactly why the bias
    shipped. Post-fix: stays ``near_linear`` (or falls through to
    ``mixed_or_ambiguous`` at the rho=0.5 boundary), never ``monotonic_nonlinear``
    or ``nonmonotonic_dependence``, and its mean is never ``curved``."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = rho * x + np.sqrt(1.0 - rho**2) * rng.normal(size=n)
    res = profile_pair(pd.DataFrame({"x": x, "y": y}), "x", "y", mode="lite")

    assert res.pattern in ("near_linear", "mixed_or_ambiguous"), (
        f"rho={rho} n={n} seed={seed} -> {res.pattern}"
    )
    assert res.diagnostics.mean_shape != "curved", (
        f"rho={rho} n={n} seed={seed} -> mean_shape={res.diagnostics.mean_shape}"
    )


@pytest.mark.parametrize("n", [100, 200])
@pytest.mark.parametrize("seed", range(10))
def test_pure_noise_mean_shape_is_not_curved(n, seed):
    """Regression lock (FU-A / Chunk 1 #3). The df-unadjusted bin gain made
    ``mean_shape="curved"`` fire on the majority of pure-noise pairs, flatly
    contradicting their ``weak_or_no_relationship`` label. The df-adjusted gain,
    plus the weak-trend structure gate in ``_mean_shape_axis``, keep ``curved``
    off noise."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    res = profile_pair(pd.DataFrame({"x": x, "y": y}), "x", "y", mode="lite")

    assert res.diagnostics.mean_shape != "curved", (
        f"n={n} seed={seed} -> mean_shape={res.diagnostics.mean_shape}"
    )


def test_conflicting_signs_warning():
    import numpy as np
    import pandas as pd

    # Create an artificial dataset where Pearson is strongly positive, but Spearman is strongly negative
    # E.g. mostly negative trend but one huge positive outlier
    x = np.arange(50, dtype=float)
    y = -np.arange(50, dtype=float)
    x[-1] = 10000
    y[-1] = 10000

    df = pd.DataFrame({"x": x, "y": y})
    res = profile_pair(df, "x", "y")

    # Check that Pearson is > 0.3 and Spearman is < -0.3
    p = float(res.metrics[res.metrics["metric"] == "pearson"]["value"].iloc[0])
    s = float(res.metrics[res.metrics["metric"] == "spearman"]["value"].iloc[0])

    assert p > 0.3
    assert s < -0.3
    assert any("conflicting directions" in w for w in res.warnings)


def test_sign_conflict_is_never_labeled_near_linear_or_monotonic():
    """Opposite-sign Pearson/Spearman must not read as agreement. With nearly
    equal magnitudes the abs gap is ~0, which previously mislabeled the pair
    near_linear; the cascade now keys off the signed conflict."""

    def metrics(p, s, k):
        return {
            "pearson": MetricResult("pearson", p, True),
            "spearman": MetricResult("spearman", s, True),
            "kendall_tau_b": MetricResult("kendall_tau_b", k, True),
        }

    conflict = metrics(0.8, -0.8, -0.6)
    # With independent leverage evidence, the conflict is a leverage signature.
    assert (
        apply_heuristics(conflict, ["pearson_trim_sensitive"], 200).label
        == "possible_outlier_or_leverage"
    )
    # Without it, it must not be near_linear/monotonic — it is ambiguous.
    assert (
        apply_heuristics(conflict, ["pearson_trim_stable"], 200).label
        == "mixed_or_ambiguous"
    )
    # Same magnitudes but agreeing signs is still near_linear (unchanged).
    assert (
        apply_heuristics(metrics(0.8, 0.8, 0.6), ["pearson_trim_stable"], 200).label
        == "near_linear"
    )


def test_disagreement_score_reflects_sign_conflict():
    """A sign conflict (e.g. +0.96 vs -0.94) must not score as agreement: the
    disagreement uses the signed Pearson-Spearman difference, so the score is
    large rather than ~0."""
    import numpy as np
    import pandas as pd

    pytest.importorskip("dcor")
    pytest.importorskip("sklearn")
    n = 200
    x = np.linspace(0, 10, n)
    y = -x.copy()
    x[-2:] = [200, 210]
    y[-2:] = [200, 210]  # leverage points flip Pearson positive
    res = profile_pair(pd.DataFrame({"x": x, "y": y}), "x", "y", mode="deep")
    metrics = {row["metric"]: row["value"] for _, row in res.metrics.iterrows()}

    assert metrics["pearson"] > 0.5 and metrics["spearman"] < -0.5
    assert res.pattern == "possible_outlier_or_leverage"
    assert res.disagreement_score > 1.0


def test_outlier_driven_uses_trim_sensitivity():
    df = make_relationship("outlier_driven", n=500, noise=0.1, random_state=42)
    res = profile_pair(df, "x", "y")

    assert res.pattern == "possible_outlier_or_leverage"
    assert res.diagnostics.pearson_trimmed is not None
    assert res.diagnostics.pearson_trim_delta is not None
    assert res.diagnostics.pearson_trim_delta > 0.20
    assert any("trimming extreme" in warning for warning in res.warnings)


def test_trim_sensitivity_detects_pearson_sign_flip():
    """A few high-leverage points can flip full-sample Pearson negative while
    the trimmed bulk stays positive. The magnitudes are nearly equal, so the
    old abs-of-abs delta would score ~0 and mislabel this 'stable'; the signed
    delta must catch it as leverage-sensitive."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 500
    x = rng.uniform(-1, 1, n)
    y = x + rng.normal(0, 0.05, n)  # tight positive bulk
    # High-leverage outliers forming a strong negative direction.
    ox = np.array([40, 42, 44, 46, 48], dtype=float)
    df = pd.DataFrame({"x": np.concatenate([x, ox]), "y": np.concatenate([y, -ox])})

    res = profile_pair(df, "x", "y")

    p = float(res.metrics[res.metrics["metric"] == "pearson"]["value"].iloc[0])
    assert p < -0.5  # full-sample Pearson is negative...
    assert res.diagnostics.pearson_trimmed > 0.5  # ...but the trimmed bulk is positive
    # Magnitudes are nearly equal, so abs(|p| - |trimmed|) would be < 0.20.
    assert abs(abs(p) - abs(res.diagnostics.pearson_trimmed)) < 0.20
    # The signed delta still flags it.
    assert res.diagnostics.pearson_trim_delta > 0.20
    assert any("trimming extreme" in warning for warning in res.warnings)


def test_detect_metric_warnings_flags_conflicting_signs():
    metrics = {
        "pearson": MetricResult("pearson", 0.6, True),
        "spearman": MetricResult("spearman", -0.5, True),
    }
    warnings = detect_metric_warnings(metrics)
    assert any("conflicting directions" in w for w in warnings)


def test_detect_metric_warnings_silent_when_below_threshold():
    metrics = {
        "pearson": MetricResult("pearson", 0.2, True),
        "spearman": MetricResult("spearman", -0.25, True),
    }
    assert detect_metric_warnings(metrics) == []


def test_detect_metric_warnings_silent_on_aligned_signs():
    metrics = {
        "pearson": MetricResult("pearson", 0.6, True),
        "spearman": MetricResult("spearman", 0.5, True),
    }
    assert detect_metric_warnings(metrics) == []


def _weak_metrics_with_xi(xi=None, xi_reverse=None):
    metrics = {
        "pearson": MetricResult("pearson", 0.05, True),
        "spearman": MetricResult("spearman", 0.04, True),
        "kendall_tau_b": MetricResult("kendall_tau_b", 0.03, True),
    }
    if xi is not None:
        metrics["chatterjee_xi"] = MetricResult("chatterjee_xi", xi, True)
    if xi_reverse is not None:
        metrics["chatterjee_xi_reverse"] = MetricResult(
            "chatterjee_xi_reverse", xi_reverse, True
        )
    return metrics


def test_detect_metric_warnings_flags_high_xi_with_weak_label():
    metrics = _weak_metrics_with_xi(xi=0.9, xi_reverse=0.3)
    warnings = detect_metric_warnings(metrics, label="weak_or_no_relationship")
    assert any("chatterjee_xi (0.900)" in w for w in warnings)
    assert any("weak_or_no_relationship" in w for w in warnings)


def test_detect_metric_warnings_xi_reports_stronger_direction():
    metrics = _weak_metrics_with_xi(xi=0.3, xi_reverse=0.8)
    warnings = detect_metric_warnings(metrics, label="mixed_or_ambiguous")
    assert any("chatterjee_xi_reverse (0.800)" in w for w in warnings)


def test_detect_metric_warnings_xi_silent_without_label():
    metrics = _weak_metrics_with_xi(xi=0.9)
    assert detect_metric_warnings(metrics) == []


def test_detect_metric_warnings_xi_silent_on_other_labels():
    metrics = _weak_metrics_with_xi(xi=0.9)
    assert detect_metric_warnings(metrics, label="near_linear") == []


def test_detect_metric_warnings_xi_silent_below_threshold():
    metrics = _weak_metrics_with_xi(xi=0.30)
    assert detect_metric_warnings(metrics, label="weak_or_no_relationship") == []


def test_detect_metric_warnings_xi_silent_when_not_computed():
    metrics = _weak_metrics_with_xi(xi=None)
    metrics["chatterjee_xi"] = MetricResult("chatterjee_xi", None, True)
    assert detect_metric_warnings(metrics, label="weak_or_no_relationship") == []


def test_classifier_treats_nan_metric_as_not_computable():
    """A NaN metric value must be treated as 'no value' (not_computable), not
    slip past the None guard and fall through to mixed_or_ambiguous."""
    metrics = {
        "pearson": MetricResult("pearson", float("nan"), True),
        "spearman": MetricResult("spearman", 0.5, True),
        "kendall_tau_b": MetricResult("kendall_tau_b", 0.4, True),
    }
    res = apply_heuristics(metrics, flags=[], n_used=100)
    assert res.label == "not_computable"


def test_detect_metric_warnings_ignores_nan_xi():
    """A NaN xi must not trigger (or crash) the dependence warning; max() over
    candidates would otherwise be unreliable with NaN present."""
    metrics = _weak_metrics_with_xi(xi=float("nan"), xi_reverse=0.30)
    warnings = detect_metric_warnings(metrics, label="weak_or_no_relationship")
    assert not any("chatterjee_xi" in w for w in warnings)


def test_deep_mode_u_shape_resolves_via_sq_corr_without_xi_warning():
    # deep mode resolves a classic U-shape to nonmonotonic_dependence (via the
    # lite-computable sq_corr route, and also distance correlation now that deep
    # is a superset of standard), so it no longer falls back to
    # weak_or_no_relationship and needs no chatterjee_xi warning as a safety net.
    pytest.importorskip("dcor")
    pytest.importorskip("sklearn")
    df = make_relationship("u_shape", n=500, noise=0.1, random_state=42)
    res = profile_pair(df, "x", "y", mode="deep")

    assert res.pattern == "nonmonotonic_dependence"
    assert not any("understate" in w for w in res.warnings)


def test_lite_mode_u_shape_resolves_via_sq_corr():
    # sq_corr needs no optional dependency, so lite mode resolves the same way.
    df = make_relationship("u_shape", n=500, noise=0.1, random_state=42)
    res = profile_pair(df, "x", "y", mode="lite")

    assert res.pattern == "nonmonotonic_dependence"
    assert not any("chatterjee_xi" in w for w in res.warnings)


def test_stable_trim_sensitivity_avoids_outlier_label():
    metrics = {
        "pearson": MetricResult("pearson", 0.80, True),
        "spearman": MetricResult("spearman", 0.52, True),
        "kendall_tau_b": MetricResult("kendall_tau_b", 0.45, True),
    }

    result = apply_heuristics(metrics, ["pearson_trim_stable"], n_used=100)

    assert result.label != "possible_outlier_or_leverage"


# --- Secondary diagnostic axes (derive_diagnostic_axes) ---


def _axis_metrics(**values):
    """Build a metrics dict of MetricResults from name=value kwargs.

    When ``bin_lof_r2_gain`` / ``sq_corr`` is given but its ``*_robust`` companion
    is not, the robust value defaults to the same value — the real-world case for
    genuine structure (a sinusoid/curve/circle), where dropping the extreme few
    bins/points barely changes the statistic. Tests exercising a robustness gate
    itself pass a lower ``*_robust`` value explicitly."""
    if "bin_lof_r2_gain" in values and "bin_lof_r2_gain_robust" not in values:
        values["bin_lof_r2_gain_robust"] = values["bin_lof_r2_gain"]
    if "sq_corr" in values and "sq_corr_robust" not in values:
        values["sq_corr_robust"] = abs(values["sq_corr"])
    return {name: MetricResult(name, value, True) for name, value in values.items()}


def test_axes_linear_pair_is_linear_monotone_low():
    metrics = _axis_metrics(
        pearson=0.95, spearman=0.95, bin_lof_r2_gain=0.0, sq_corr=0.9
    )
    axes = derive_diagnostic_axes(metrics, "near_linear", "stable")
    assert axes["mean_shape"] == "linear"
    # Strong monotone signal: not read as magnitude-linked despite a high sq_corr.
    assert axes["dependence_type"] == "monotone"
    assert axes["outlier_sensitivity"] == "low"
    assert axes["variance_shape"] is None
    assert axes["functional_direction"] is None  # no xi outside deep mode


def test_axes_curved_mean_via_bin_lof():
    metrics = _axis_metrics(
        pearson=0.90, spearman=0.95, bin_lof_r2_gain=0.12, sq_corr=0.9
    )
    axes = derive_diagnostic_axes(metrics, "monotonic_nonlinear", "stable")
    assert axes["mean_shape"] == "curved"


def test_axes_circle_is_closed_loop_and_neither_direction_in_deep_mode():
    metrics = _axis_metrics(
        pearson=0.0,
        spearman=0.0,
        bin_lof_r2_gain=0.02,
        sq_corr=-0.9,
        chatterjee_xi=0.10,
        chatterjee_xi_reverse=0.05,
    )
    axes = derive_diagnostic_axes(metrics, "nonmonotonic_dependence", "stable")
    assert axes["dependence_type"] == "closed_loop_or_multivalued"
    assert axes["functional_direction"] == "neither_direction"
    assert axes["mean_shape"] is None  # no functional mean trend for a ring


def test_axes_u_shape_is_magnitude_linked_and_y_of_x():
    # Y is a function of X (xi forward high) but X is not a function of Y
    # (xi reverse low) — so NOT a closed loop, and the direction is y_of_x.
    metrics = _axis_metrics(
        pearson=0.05,
        spearman=0.05,
        bin_lof_r2_gain=0.9,
        sq_corr=0.95,
        chatterjee_xi=0.90,
        chatterjee_xi_reverse=0.30,
    )
    axes = derive_diagnostic_axes(metrics, "nonmonotonic_dependence", "stable")
    assert axes["dependence_type"] == "magnitude_linked"
    assert axes["functional_direction"] == "y_of_x"


def test_axes_magnitude_link_without_xi_is_not_closed_loop():
    # Same magnitude signature but no xi (lite/standard mode): we cannot tell
    # closed-loop from a one-way function, so it stays magnitude_linked.
    metrics = _axis_metrics(pearson=0.0, spearman=0.0, sq_corr=-0.9)
    axes = derive_diagnostic_axes(metrics, "nonmonotonic_dependence", "stable")
    assert axes["dependence_type"] == "magnitude_linked"
    assert axes["functional_direction"] is None


def test_axes_oscillating_requires_joint_reversal_and_gain_gate():
    # Both conditions present -> oscillating.
    metrics = _axis_metrics(
        pearson=0.08, spearman=0.10, bin_lof_r2_gain=0.83, bin_reversal_count=4
    )
    axes = derive_diagnostic_axes(metrics, "nonmonotonic_dependence", "stable")
    assert axes["dependence_type"] == "oscillating"

    # One reversal (a single bend, e.g. a U-shape) is not oscillation.
    metrics = _axis_metrics(
        pearson=0.08, spearman=0.10, bin_lof_r2_gain=0.83, bin_reversal_count=1
    )
    axes = derive_diagnostic_axes(metrics, "nonmonotonic_dependence", "stable")
    assert axes["dependence_type"] != "oscillating"

    # Many reversals but negligible gain (pure noise) is not oscillation.
    metrics = _axis_metrics(
        pearson=0.05, spearman=0.05, bin_lof_r2_gain=0.06, bin_reversal_count=14
    )
    axes = derive_diagnostic_axes(metrics, "weak_or_no_relationship", "stable")
    assert axes["dependence_type"] != "oscillating"

    # Gated on weak monotone signals, like the other rule-4 routes: a strong
    # trend with wiggle is a monotone story, not an oscillation one.
    metrics = _axis_metrics(
        pearson=0.60, spearman=0.65, bin_lof_r2_gain=0.83, bin_reversal_count=4
    )
    axes = derive_diagnostic_axes(metrics, "near_linear", "stable")
    assert axes["dependence_type"] == "monotone"


def test_axes_oscillating_takes_precedence_over_nonmonotone():
    # In standard mode a sinusoid also clears the distance-correlation floor;
    # "oscillating" is the more specific description and must win over the
    # generic "nonmonotone".
    metrics = _axis_metrics(
        pearson=0.08,
        spearman=0.10,
        distance_correlation=0.43,
        bin_lof_r2_gain=0.83,
        bin_reversal_count=4,
    )
    axes = derive_diagnostic_axes(metrics, "nonmonotonic_dependence", "stable")
    assert axes["dependence_type"] == "oscillating"


def test_axes_missing_reversal_count_preserves_previous_behavior():
    # Without bin_reversal_count in the metrics (e.g. older callers), the axis
    # must fall through to the existing sq/dc logic, never oscillating.
    metrics = _axis_metrics(
        pearson=0.05, spearman=0.05, bin_lof_r2_gain=0.9, sq_corr=0.95
    )
    axes = derive_diagnostic_axes(metrics, "nonmonotonic_dependence", "stable")
    assert axes["dependence_type"] == "magnitude_linked"


def test_cascade_oscillation_route_into_nonmonotonic_dependence():
    def metrics(p, s, k, **extra):
        # Mirror _axis_metrics: robust gain defaults to the raw gain (genuine
        # multi-bin structure survives dropping any one bin).
        if "bin_lof_r2_gain" in extra and "bin_lof_r2_gain_robust" not in extra:
            extra["bin_lof_r2_gain_robust"] = extra["bin_lof_r2_gain"]
        base = {
            "pearson": MetricResult("pearson", p, True),
            "spearman": MetricResult("spearman", s, True),
            "kendall_tau_b": MetricResult("kendall_tau_b", k, True),
        }
        base.update(
            {name: MetricResult(name, value, True) for name, value in extra.items()}
        )
        return base

    # The oscillation route labels without dc or sq_corr (lite mode).
    oscillating = metrics(0.08, 0.12, 0.08, bin_lof_r2_gain=0.83, bin_reversal_count=4)
    assert (
        apply_heuristics(oscillating, ["pearson_trim_stable"], 500).label
        == "nonmonotonic_dependence"
    )

    # Reversals without the gain floor (noise) must not fire the route.
    noise = metrics(0.05, 0.06, 0.04, bin_lof_r2_gain=0.06, bin_reversal_count=14)
    assert (
        apply_heuristics(noise, ["pearson_trim_stable"], 500).label
        == "weak_or_no_relationship"
    )

    # Gain without enough reversals (a U-shape reads 1) must not fire it
    # either — the U-shape's own route is sq_corr, deliberately not this one.
    single_bend = metrics(0.05, 0.06, 0.04, bin_lof_r2_gain=0.75, bin_reversal_count=1)
    assert (
        apply_heuristics(single_bend, ["pearson_trim_stable"], 500).label
        != "nonmonotonic_dependence"
    )

    # A raw gain over the floor with enough reversals still must NOT fire the
    # route when the *robust* (leave-one-bin-out) gain collapses — the
    # heavy-tailed-Y artifact (FU-U), where one outlier bin carries the whole
    # apparent oscillation.
    artifact = metrics(
        0.08,
        0.12,
        0.08,
        bin_lof_r2_gain=0.19,
        bin_lof_r2_gain_robust=0.05,
        bin_reversal_count=6,
    )
    assert (
        apply_heuristics(artifact, ["pearson_trim_stable"], 500).label
        == "weak_or_no_relationship"
    )


def test_heavy_tailed_y_artifact_does_not_read_as_oscillation():
    """End-to-end FU-U lock: an independent predictor against a pathologically
    heavy-tailed target (whose unlucky draw inflates the raw bin-LoF gain over
    the oscillation floor) is NOT mislabeled ``nonmonotonic_dependence`` via the
    oscillation route, and its ``mean_shape``/``dependence_type`` axes do not
    read as curved/oscillating. Seeds 20 and 135 both tripped the route before
    the leave-one-bin-out robustness gate."""
    for seed in (20, 135):
        rng = np.random.default_rng(seed)
        df = pd.DataFrame(
            {"x": rng.normal(size=100), "y": np.exp(rng.uniform(0.1, 10, size=100))}
        )
        res = profile_pair(df, "x", "y", mode="lite")
        # The raw gain really is over the floor (the artifact is present)...
        assert res.diagnostics.bin_lof_r2_gain > 0.15
        # ...but the label and axes are not fooled by it.
        assert res.pattern == "weak_or_no_relationship"
        assert res.diagnostics.mean_shape is None
        assert res.diagnostics.dependence_type != "oscillating"


def test_heavy_tailed_y_artifact_does_not_read_as_magnitude_linked():
    """End-to-end FU-V lock: seed 574's heavy-tailed target inflates the raw
    sq_corr over 0.35, but the robust sq_corr collapses, so the pair reads
    ``weak_or_no_relationship`` (not ``nonmonotonic_dependence`` /
    ``magnitude_linked``) — while a genuine U-shape, whose sq_corr is robust, is
    still detected in lite mode."""
    rng = np.random.default_rng(574)
    df = pd.DataFrame(
        {"x": rng.normal(size=100), "y": np.exp(rng.uniform(0.1, 10, size=100))}
    )
    res = profile_pair(df, "x", "y", mode="lite")
    # The raw sq_corr really is over the threshold (the artifact is present)...
    assert abs(res.diagnostics.sq_corr) > 0.35
    # ...but the label and axis are not fooled by it.
    assert res.pattern == "weak_or_no_relationship"
    assert res.diagnostics.dependence_type != "magnitude_linked"

    # A genuine U-shape (sq_corr robust to dropping the extreme points) still
    # lands as magnitude-linked nonmonotonic dependence, in lite mode.
    rr = np.random.default_rng(1)
    x = rr.uniform(-3, 3, size=400)
    u = pd.DataFrame({"x": x, "y": x**2 + rr.normal(0, 0.4, size=400)})
    u_res = profile_pair(u, "x", "y", mode="lite")
    assert u_res.pattern == "nonmonotonic_dependence"
    assert u_res.diagnostics.dependence_type == "magnitude_linked"


def test_axes_outlier_sensitivity_from_trim_status():
    metrics = _axis_metrics(pearson=0.9, spearman=0.5, bin_lof_r2_gain=0.0)
    assert (
        derive_diagnostic_axes(metrics, "possible_outlier_or_leverage", "sensitive")[
            "outlier_sensitivity"
        ]
        == "high"
    )
    assert (
        derive_diagnostic_axes(metrics, "near_linear", "stable")["outlier_sensitivity"]
        == "low"
    )
    assert (
        derive_diagnostic_axes(metrics, "near_linear", "unavailable")[
            "outlier_sensitivity"
        ]
        == "unavailable"
    )


def test_axes_none_when_core_metrics_missing():
    metrics = _axis_metrics(pearson=None, spearman=None)
    axes = derive_diagnostic_axes(metrics, "not_computable", "unavailable")
    assert axes["mean_shape"] is None
    assert axes["dependence_type"] is None
    assert axes["functional_direction"] is None
    assert axes["outlier_sensitivity"] == "unavailable"


def test_axes_are_orthogonal_to_label_outlier_driven_but_linear_mean():
    # A leverage-driven pair can still have a linear conditional mean: the label
    # is possible_outlier_or_leverage, but mean_shape is linear and
    # outlier_sensitivity localizes the leverage — the axes carry what the label
    # cannot. The outlier_driven cluster reads as high_leverage_cluster.
    pytest.importorskip("dcor")
    pytest.importorskip("sklearn")
    df = make_relationship("outlier_driven", n=500, noise=0.1, random_state=42)
    res = profile_pair(df, "x", "y", mode="deep")
    assert res.pattern == "possible_outlier_or_leverage"
    assert res.diagnostics.outlier_sensitivity == "high_leverage_cluster"


# --- variance_shape axis / heteroscedasticity (ticket 1.1) ---


def test_variance_shape_axis_direction_and_effect_floor():
    linear_bin_lof = 0.0  # adequately linear mean, so variance is assessable
    # Clear funnel: BP rejects, ratio well above the floor.
    assert _variance_shape_axis(1e-10, 7.0, linear_bin_lof) == "increasing_spread"
    # Reversed funnel.
    assert _variance_shape_axis(1e-10, 0.2, linear_bin_lof) == "decreasing_spread"
    # BP rejects but the effect is negligible (ratio ~1) -> constant. This is the
    # large-n guard: BP alone would over-flag.
    assert _variance_shape_axis(1e-10, 1.1, linear_bin_lof) == "constant"
    # BP does not reject -> constant.
    assert _variance_shape_axis(0.4, 7.0, linear_bin_lof) == "constant"


def test_variance_shape_axis_gated_off_by_curved_mean():
    # A curved mean makes linear-fit residuals heteroscedastic as an artifact,
    # so variance_shape must be None regardless of the BP/GQ values.
    curved_bin_lof = 0.5
    assert _variance_shape_axis(1e-10, 7.0, curved_bin_lof) is None


def test_variance_shape_axis_none_when_not_computed():
    assert _variance_shape_axis(None, None, 0.0) is None
    assert _variance_shape_axis(1e-10, 7.0, None) is None


def test_variance_shape_axis_bowtie_direction_and_effect_floor():
    linear_bin_lof = 0.0
    # BP does not reject at all -- a real bowtie's squared-residuals-vs-x
    # relationship is not linear, so BP alone can miss it entirely. bowtie_ratio
    # still catches it independently.
    assert _variance_shape_axis(0.4, 1.1, linear_bin_lof, bowtie_ratio=11.0) == (
        "edge_high_spread"
    )
    # BP rejects but gq_ratio is inconclusive (~1, as a bowtie's low-x/high-x
    # groups have similar variance) -- bowtie_ratio still catches it.
    assert _variance_shape_axis(1e-10, 1.1, linear_bin_lof, bowtie_ratio=11.0) == (
        "edge_high_spread"
    )
    # Reversed: spread high in the middle, calm at the edges.
    assert _variance_shape_axis(0.4, 1.1, linear_bin_lof, bowtie_ratio=1.0 / 11.0) == (
        "center_high_spread"
    )
    # bowtie_ratio near 1 (no real edge-vs-middle effect) -> constant.
    assert (
        _variance_shape_axis(0.4, 1.1, linear_bin_lof, bowtie_ratio=1.1) == "constant"
    )
    # bowtie_ratio unavailable -> falls back to constant (unchanged from before
    # this axis existed).
    assert _variance_shape_axis(0.4, 1.1, linear_bin_lof) == "constant"


def test_variance_shape_axis_funnel_takes_priority_over_bowtie():
    # A genuine one-directional funnel must still report increasing_spread even
    # if bowtie_ratio happens to be non-trivial -- the existing check is not
    # replaced by the new one.
    linear_bin_lof = 0.0
    assert _variance_shape_axis(1e-10, 7.0, linear_bin_lof, bowtie_ratio=3.0) == (
        "increasing_spread"
    )


def test_heteroscedastic_shape_is_near_linear_with_increasing_spread_and_warning():
    df = make_relationship("heteroscedastic", n=500, noise=0.1, random_state=42)
    res = profile_pair(df, "x", "y", mode="lite")

    # The mean trend is linear (label unchanged) but the variance is flagged.
    assert res.pattern == "near_linear"
    assert res.diagnostics.variance_shape == "increasing_spread"
    assert res.diagnostics.mean_shape == "linear"
    assert any("residual spread" in w for w in res.warnings)


@pytest.mark.parametrize("seed", range(8))
def test_homoscedastic_linear_stays_constant_without_warning(seed):
    # Guards against false positives: a clean homoscedastic linear pair must
    # report constant variance and emit no heteroscedasticity warning, even
    # though the large-n Breusch-Pagan test occasionally rejects.
    df = make_relationship("linear_positive", n=500, noise=0.1, random_state=seed)
    res = profile_pair(df, "x", "y", mode="lite")

    assert res.diagnostics.variance_shape == "constant"
    assert not any("residual spread" in w for w in res.warnings)


def test_curved_relationship_does_not_report_variance_shape():
    # An exponential curve: the linear-fit residuals look heteroscedastic, but
    # that is a curvature artifact, so variance_shape must be None.
    df = make_relationship("exponential_monotonic", n=500, noise=0.1, random_state=42)
    res = profile_pair(df, "x", "y", mode="lite")

    assert res.pattern == "monotonic_nonlinear"
    assert res.diagnostics.variance_shape is None
    assert not any("residual spread" in w for w in res.warnings)


# --- bowtie (edge-vs-middle) variance (ticket 1.6) ---


def test_bowtie_variance_reports_edge_high_spread_with_warning():
    df = make_relationship("bowtie_variance", n=600, noise=0.1, random_state=42)
    res = profile_pair(df, "x", "y", mode="lite")

    # The mean trend is still linear (label unchanged); only the variance
    # *shape* sub-diagnosis differs from a one-directional funnel.
    assert res.pattern == "near_linear"
    assert res.diagnostics.mean_shape == "linear"
    assert res.diagnostics.variance_shape == "edge_high_spread"
    assert res.diagnostics.bowtie_ratio is not None
    assert any("extremes of x" in w for w in res.warnings)


@pytest.mark.parametrize("seed", range(6))
def test_bowtie_variance_stays_edge_high_spread_across_seeds(seed):
    df = make_relationship("bowtie_variance", n=600, noise=0.1, random_state=seed)
    res = profile_pair(df, "x", "y", mode="lite")
    assert res.diagnostics.variance_shape == "edge_high_spread"


@pytest.mark.parametrize("seed", range(6))
def test_homoscedastic_and_funnel_never_report_bowtie(seed):
    # Guards against false positives: clean homoscedastic data and a genuine
    # one-directional funnel must never be misread as a symmetric bowtie.
    for shape in ("linear_positive", "heteroscedastic"):
        df = make_relationship(shape, n=500, noise=0.1, random_state=seed)
        res = profile_pair(df, "x", "y", mode="lite")
        assert res.diagnostics.variance_shape != "edge_high_spread"
        assert res.diagnostics.variance_shape != "center_high_spread"


# --- mean_shape refinement / segmentation (ticket 1.2) ---


def test_mean_shape_axis_refines_curved_monotone_by_stepness():
    strong_s = 0.9  # monotone
    curved_bin_lof = 0.2
    # High stepness -> step/threshold.
    assert _mean_shape_axis(0.7, strong_s, curved_bin_lof, 1.0) == "step_or_threshold"
    # Low/negative stepness -> smooth curve.
    assert _mean_shape_axis(0.7, strong_s, curved_bin_lof, -0.6) == "smooth_curve"


def test_mean_shape_axis_non_monotone_curve_stays_generic_curved():
    # Weak Spearman (a U-shape): smooth-vs-step does not apply, even if the
    # stepness value happens to be high. A genuine curve's gain is robust to
    # dropping any one bin, so the leave-one-bin-out gain also clears the floor.
    assert _mean_shape_axis(0.05, 0.05, 0.9, 1.0, 0.9) == "curved"
    # But a no-trend "gain" carried by a single outlier bin (robust gain below
    # the floor) is a heavy-tailed-Y artifact, not curvature — stays unassessed.
    assert _mean_shape_axis(0.05, 0.05, 0.9, 1.0, 0.02) is None


def test_mean_shape_axis_curved_without_segmentation_is_generic_curved():
    # n below the segmentation floor: we know it is a monotone curve but cannot
    # say step vs smooth, so it stays the generic "curved".
    assert _mean_shape_axis(0.7, 0.9, 0.2, None) == "curved"


def test_threshold_step_reports_step_or_threshold_with_breakpoint():
    df = make_relationship("threshold_step", n=500, noise=0.1, random_state=42)
    res = profile_pair(df, "x", "y", mode="lite")

    # Label unchanged; the break story lives in the axis + breakpoint_x.
    assert res.pattern == "monotonic_nonlinear"
    assert res.diagnostics.mean_shape == "step_or_threshold"
    # The step is generated at x == 0; the located breakpoint is near it.
    assert res.diagnostics.breakpoint_x is not None
    assert abs(res.diagnostics.breakpoint_x) < 0.5


@pytest.mark.parametrize("shape", ["exponential_monotonic", "logarithmic_monotonic"])
def test_smooth_curves_report_smooth_curve_without_breakpoint(shape):
    df = make_relationship(shape, n=500, noise=0.1, random_state=42)
    res = profile_pair(df, "x", "y", mode="lite")

    assert res.pattern == "monotonic_nonlinear"
    assert res.diagnostics.mean_shape == "smooth_curve"
    # No spurious breakpoint reported for a smooth curve.
    assert res.diagnostics.breakpoint_x is None


def test_monotone_piecewise_linear_folds_into_smooth_curve():
    # A monotone two-slope kink is not reliably separable from a smooth bend
    # over a finite range, so it is (honestly) reported as smooth_curve, not a
    # distinct piecewise label. Its breakpoint is treated as an artifact.
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 500
    x = rng.uniform(-3, 3, n)
    y = np.where(x < 0, 0.5 * x, 3.0 * x) + rng.normal(0, 0.1, n)
    res = profile_pair(pd.DataFrame({"x": x, "y": y}), "x", "y", mode="lite")

    assert res.diagnostics.mean_shape == "smooth_curve"
    assert res.diagnostics.breakpoint_x is None


def test_u_shape_mean_stays_generic_curved_not_step():
    df = make_relationship("u_shape", n=500, noise=0.1, random_state=42)
    res = profile_pair(df, "x", "y", mode="lite")

    assert res.pattern == "nonmonotonic_dependence"
    assert res.diagnostics.mean_shape == "curved"


# --- outlier_sensitivity refinement / Cook's distance (ticket 1.3) ---


def test_outlier_sensitivity_axis_refined_by_influence_count():
    # Cook's-distance count takes precedence and localizes the influence.
    assert _outlier_sensitivity_axis("sensitive", 1) == "single_point_driven"
    assert _outlier_sensitivity_axis("sensitive", 5) == "high_leverage_cluster"
    # No influential rows -> fall back to the trim-sensitivity verdict.
    assert _outlier_sensitivity_axis("sensitive", 0) == "high"
    assert _outlier_sensitivity_axis("stable", 0) == "low"
    assert _outlier_sensitivity_axis("unavailable", None) == "unavailable"


def test_outlier_sensitivity_axis_fires_even_when_trim_says_stable():
    # Cook's distance has no 1%-trim blind spot: a leverage cluster the trim
    # check missed still surfaces on the axis.
    assert _outlier_sensitivity_axis("stable", 4) == "high_leverage_cluster"


def test_outlier_driven_reports_high_leverage_cluster():
    df = make_relationship("outlier_driven", n=500, noise=0.1, random_state=42)
    res = profile_pair(df, "x", "y", mode="lite")

    assert res.pattern == "possible_outlier_or_leverage"
    assert res.diagnostics.outlier_sensitivity == "high_leverage_cluster"
    assert res.diagnostics.n_influential_points >= 2
    assert res.diagnostics.max_cook_distance is not None


def test_single_dominant_point_reports_single_point_driven():
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 500
    x = rng.uniform(-3, 3, n)
    y = x + rng.normal(0, 0.3, n)
    x[-1], y[-1] = 20.0, -20.0  # one high-leverage, high-residual row
    res = profile_pair(pd.DataFrame({"x": x, "y": y}), "x", "y", mode="lite")

    assert res.diagnostics.outlier_sensitivity == "single_point_driven"
    assert res.diagnostics.n_influential_points == 1
    assert res.diagnostics.max_cook_distance > 1.0


def test_clean_linear_reports_low_outlier_sensitivity():
    df = make_relationship("linear_positive", n=500, noise=0.1, random_state=42)
    res = profile_pair(df, "x", "y", mode="lite")

    assert res.diagnostics.outlier_sensitivity == "low"
    assert res.diagnostics.n_influential_points == 0


# --- suppress heteroscedasticity warning when it's a leverage artifact (ticket 1.5) ---


def _het_metrics(bp, gq, bin_lof=0.0, bowtie=1.1, n_influential=None, excl=None):
    metrics = {
        "bp_pvalue": MetricResult("bp_pvalue", bp, True),
        "gq_ratio": MetricResult("gq_ratio", gq, True),
        "bin_lof_r2_gain": MetricResult("bin_lof_r2_gain", bin_lof, True),
        "bowtie_ratio": MetricResult("bowtie_ratio", bowtie, True),
    }
    if n_influential is not None:
        metrics["n_influential_points"] = MetricResult(
            "n_influential_points", n_influential, True
        )
    if excl is not None:
        bp_excl, gq_excl, bowtie_excl = excl
        metrics["bp_pvalue_excl_influential"] = MetricResult(
            "bp_pvalue_excl_influential", bp_excl, True
        )
        metrics["gq_ratio_excl_influential"] = MetricResult(
            "gq_ratio_excl_influential", gq_excl, True
        )
        metrics["bowtie_ratio_excl_influential"] = MetricResult(
            "bowtie_ratio_excl_influential", bowtie_excl, True
        )
    return metrics


def test_variance_warning_attributed_to_leverage_when_signal_vanishes_on_exclusion():
    # X13-shaped: funnel signal on the full sample, gone on the subset
    # excluding the Cook's-flagged row -- report it as the same leverage
    # artifact, not an independent-sounding heteroscedasticity warning.
    metrics = _het_metrics(
        bp=0.002,
        gq=27.5,
        n_influential=1,
        excl=(0.44, 1.2, 1.1),
    )
    warnings = detect_metric_warnings(metrics)
    assert len(warnings) == 1
    assert "same leverage issue" in warnings[0]
    assert "Pearson describes the center trend" not in warnings[0]


def test_bowtie_warning_attributed_to_leverage_when_signal_vanishes_on_exclusion():
    metrics = _het_metrics(
        bp=0.4,
        gq=1.1,
        bowtie=11.0,
        n_influential=1,
        excl=(0.5, 1.1, 1.1),
    )
    warnings = detect_metric_warnings(metrics)
    assert len(warnings) == 1
    assert "same leverage issue" in warnings[0]
    assert "invisible to a simple increasing/decreasing" not in warnings[0]


def test_bowtie_warning_not_attributed_when_bowtie_excl_missing():
    # A bowtie signal on the full sample, but the excl-influential recompute
    # produced bp/gq without a bowtie_ratio (its middle third degenerated), so the
    # bowtie check never re-ran. The signal must NOT be attributed to leverage —
    # a "constant" verdict from a check that didn't run is not evidence (Chunk 1
    # #2 / FU-G). The independent bowtie warning is kept instead.
    metrics = _het_metrics(
        bp=0.4, gq=1.1, bowtie=11.0, n_influential=1, excl=(0.5, 1.1, None)
    )
    warnings = detect_metric_warnings(metrics)
    assert len(warnings) == 1
    assert "same leverage issue" not in warnings[0]
    assert "invisible to a simple increasing/decreasing" in warnings[0]


def test_dependence_type_monotone_gates_on_spearman_not_pearson():
    # A leverage pair — strong Pearson, near-zero Spearman — has no monotone
    # trend; it must not be called "monotone" on the strength of the linear
    # artifact (Chunk 1 #9). It falls through to None.
    assert _dependence_type_axis(0.85, 0.05, None, None, None, None) is None
    # A genuine monotone pair (Spearman clears the weak floor) still reads monotone.
    assert _dependence_type_axis(0.60, 0.55, None, None, None, None) == "monotone"


def test_functional_direction_none_for_strong_linear_with_subthreshold_xi():
    # xi is only ~0.30 at rho=0.7 for a bivariate normal, below the 0.35 bar, so
    # an obviously functional noisy-linear pair must NOT read "neither_direction"
    # (Chunk 1 #6). With strong |p|/|s| and both xi below the bar the axis is
    # uninformative -> None.
    assert _functional_direction_axis(0.30, 0.29, 0.72, 0.72) is None
    # A circle (weak p/s, both xi weak) genuinely lacks a functional direction.
    assert _functional_direction_axis(0.10, 0.10, 0.05, 0.05) == "neither_direction"
    # A real functional direction still wins regardless of |p|/|s|.
    assert _functional_direction_axis(0.60, 0.10, 0.72, 0.72) == "y_of_x"


def test_variance_warning_stays_independent_when_signal_survives_exclusion():
    # The signal is still present after excluding the flagged row -- a genuine
    # leverage cluster and genuinely independent heteroscedasticity can
    # coexist, so both must be reported (not folded into one artifact claim).
    metrics = _het_metrics(
        bp=0.002,
        gq=27.5,
        n_influential=1,
        excl=(0.0009, 6.0, 1.1),
    )
    warnings = detect_metric_warnings(metrics)
    assert len(warnings) == 1
    assert "Pearson describes the center trend" in warnings[0]
    assert "same leverage issue" not in warnings[0]


def test_variance_warning_not_attributed_when_influence_unavailable():
    # No n_influential_points to reason about -- default to the ordinary
    # (independent-sounding) warning rather than guessing.
    metrics = _het_metrics(bp=0.002, gq=27.5)
    warnings = detect_metric_warnings(metrics)
    assert len(warnings) == 1
    assert "same leverage issue" not in warnings[0]


def test_variance_warning_not_attributed_when_exclusion_metrics_missing():
    # n_influential_points >= 1, but the *_excl_influential metrics were never
    # computed (e.g. the caller didn't run the recomputation) -- conservative
    # default is to keep the ordinary warning, not assume it's an artifact.
    metrics = _het_metrics(bp=0.002, gq=27.5, n_influential=1)
    warnings = detect_metric_warnings(metrics)
    assert len(warnings) == 1
    assert "same leverage issue" not in warnings[0]


def test_single_outlier_manufacturing_correlation_reports_one_attributed_warning():
    """End-to-end X13 shape: pure noise plus one extreme row. Both the funnel
    Goldfeld-Quandt signal and the leverage flag stem from the same row -- the
    warnings must say so, not read as two independent problems."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 500
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    x[-1], y[-1] = 20.0, 20.0
    df = pd.DataFrame({"x": x, "y": y})
    res = profile_pair(df, "x", "y", mode="lite")

    assert res.diagnostics.n_influential_points == 1
    assert res.diagnostics.variance_shape is not None
    variance_warnings = [
        w
        for w in res.warnings
        if "residual spread" in w.lower() or "Residual spread" in w
    ]
    assert len(variance_warnings) == 1
    assert "same leverage issue" in variance_warnings[0]


def test_outlier_cluster_with_surviving_heteroscedasticity_keeps_both_warnings():
    # The bundled outlier_driven scenario: a leverage cluster (not a single
    # row) manufactures the correlation. Excluding only the Cook's-flagged
    # rows does not fully remove the scale difference the remaining outliers
    # still carry, so the heteroscedasticity signal survives -- both warnings
    # must still be reported, not folded into one.
    df = make_relationship("outlier_driven", n=500, noise=0.1, random_state=42)
    res = profile_pair(df, "x", "y", mode="lite")

    assert res.diagnostics.outlier_sensitivity == "high_leverage_cluster"
    assert any("leverage-sensitive" in w for w in res.warnings)
    assert any("Pearson describes the center trend" in w for w in res.warnings)


def test_no_exclusion_recomputation_when_outlier_sensitivity_low():
    # Common case: clean heteroscedastic data with no elevated leverage.
    # Behavior must be unchanged from ticket 1.6 -- no attribution attempted.
    df = make_relationship("heteroscedastic", n=500, noise=0.1, random_state=42)
    res = profile_pair(df, "x", "y", mode="lite")

    assert res.diagnostics.outlier_sensitivity == "low"
    assert res.diagnostics.variance_shape == "increasing_spread"
    assert any("Pearson describes the center trend" in w for w in res.warnings)
    assert not any("same leverage issue" in w for w in res.warnings)
