import scipy.stats as stats
from corrsleuth.result import MetricResult
from corrsleuth.validation.input import CleanPair

def compute_pearson(pair: CleanPair) -> MetricResult:
    if pair.x_is_constant or pair.y_is_constant:
        return MetricResult(name="pearson", value=None, available=True)
    r, _ = stats.pearsonr(pair.x, pair.y)
    return MetricResult(name="pearson", value=float(r), available=True)

def compute_spearman(pair: CleanPair) -> MetricResult:
    if pair.x_is_constant or pair.y_is_constant:
        return MetricResult(name="spearman", value=None, available=True)
    rho, _ = stats.spearmanr(pair.x, pair.y)
    return MetricResult(name="spearman", value=float(rho), available=True)

def compute_kendall(pair: CleanPair) -> MetricResult:
    if pair.x_is_constant or pair.y_is_constant:
        return MetricResult(name="kendall_tau_b", value=None, available=True)
    tau, _ = stats.kendalltau(pair.x, pair.y)
    return MetricResult(name="kendall_tau_b", value=float(tau), available=True)
