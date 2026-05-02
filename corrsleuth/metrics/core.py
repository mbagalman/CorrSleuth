import scipy.stats as stats

from corrsleuth.exceptions import MetricComputationError
from corrsleuth.result import MetricResult
from corrsleuth.validation.input import CleanPair


def _safe_compute(name: str, fn, *args, **kwargs) -> float:
    """Run an underlying metric call and re-raise unexpected failures with metric context."""
    try:
        return fn(*args, **kwargs)
    except (ValueError, RuntimeError, FloatingPointError) as e:
        raise MetricComputationError(
            f"Failed to compute {name}: {type(e).__name__}: {e}"
        ) from e


def compute_pearson(pair: CleanPair) -> MetricResult:
    """Pearson product-moment correlation. Returns ``value=None`` for constant inputs."""
    if pair.x_is_constant or pair.y_is_constant:
        return MetricResult(name="pearson", value=None, available=True)
    r, _ = _safe_compute("pearson", stats.pearsonr, pair.x, pair.y)
    return MetricResult(name="pearson", value=float(r), available=True)


def compute_spearman(pair: CleanPair) -> MetricResult:
    """Spearman rank correlation. Returns ``value=None`` for constant inputs."""
    if pair.x_is_constant or pair.y_is_constant:
        return MetricResult(name="spearman", value=None, available=True)
    rho, _ = _safe_compute("spearman", stats.spearmanr, pair.x, pair.y)
    return MetricResult(name="spearman", value=float(rho), available=True)


def compute_kendall(pair: CleanPair) -> MetricResult:
    """Kendall tau-b rank correlation. Returns ``value=None`` for constant inputs."""
    if pair.x_is_constant or pair.y_is_constant:
        return MetricResult(name="kendall_tau_b", value=None, available=True)
    tau, _ = _safe_compute("kendall_tau_b", stats.kendalltau, pair.x, pair.y)
    return MetricResult(name="kendall_tau_b", value=float(tau), available=True)
