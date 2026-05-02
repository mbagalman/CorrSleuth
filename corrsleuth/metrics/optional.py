import warnings
from typing import Optional
from corrsleuth.result import MetricResult
from corrsleuth.validation.input import CleanPair
from corrsleuth.exceptions import OptionalDependencyError

def compute_distance_correlation(pair: CleanPair, mode: str = "lite", max_n_for_dcor: Optional[int] = 20000) -> MetricResult:
    try:
        import dcor
    except ImportError:
        if mode == "standard":
            raise OptionalDependencyError("dcor is required for standard mode. Run `pip install corrsleuth[standard]`")
        return MetricResult(name="distance_correlation", value=None, available=False)
        
    if pair.x_is_constant or pair.y_is_constant:
        return MetricResult(name="distance_correlation", value=None, available=True)
        
    x = pair.x.values
    y = pair.y.values
    
    # Downsampling guardrail
    if max_n_for_dcor is not None and pair.n_used > max_n_for_dcor:
        pair.warnings.append(f"n_used > {max_n_for_dcor}. Automatically downsampling to {max_n_for_dcor} for distance correlation.")
        import numpy as np
        rng = np.random.default_rng(42) # Deterministic downsampling for tests
        idx = rng.choice(pair.n_used, max_n_for_dcor, replace=False)
        x = x[idx]
        y = y[idx]
        
    dc = dcor.distance_correlation(x, y)
    return MetricResult(name="distance_correlation", value=float(dc), available=True)

def compute_mutual_information(pair: CleanPair, mode: str = "lite") -> MetricResult:
    try:
        from sklearn.feature_selection import mutual_info_regression
    except ImportError:
        if mode == "standard":
            raise OptionalDependencyError("scikit-learn is required for standard mode. Run `pip install corrsleuth[standard]`")
        return MetricResult(name="mutual_information", value=None, available=False)
        
    if pair.x_is_constant or pair.y_is_constant:
        return MetricResult(name="mutual_information", value=None, available=True)
        
    x = pair.x.values.reshape(-1, 1)
    y = pair.y.values
    
    mi = mutual_info_regression(x, y, random_state=42)[0]
    return MetricResult(name="mutual_information", value=float(mi), available=True)
