import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List
from corrsleuth.exceptions import InputError

@dataclass
class CleanPair:
    """
    Internal contract representing a validated, paired, and cleaned numerical dataset.
    """
    x: pd.Series
    y: pd.Series
    x_name: str
    y_name: str
    n_original: int
    n_used: int
    missing_count: int
    missing_ratio: float
    x_unique_ratio: float
    y_unique_ratio: float
    x_is_constant: bool
    y_is_constant: bool
    flags: List[str]          # machine-readable
    warnings: List[str]       # raw validation warnings only

def validate_pair(data: pd.DataFrame, x: str, y: str, missing: str = "pairwise") -> CleanPair:
    if x not in data.columns:
        raise InputError(f"Column '{x}' not found in data.")
    if y not in data.columns:
        raise InputError(f"Column '{y}' not found in data.")
        
    s_x = data[x]
    s_y = data[y]
    
    if not pd.api.types.is_numeric_dtype(s_x):
        raise InputError(f"Column '{x}' is not numeric.")
    if not pd.api.types.is_numeric_dtype(s_y):
        raise InputError(f"Column '{y}' is not numeric.")
        
    if np.isinf(s_x).any() or np.isinf(s_y).any():
        raise InputError("Input data contains infinite values.")
        
    n_original = len(data)
    
    # Missing value handling
    if missing not in ["pairwise", "listwise", "raise"]:
        raise InputError(f"Unsupported missing mode: '{missing}'. Supported modes are 'pairwise', 'listwise', and 'raise'.")
        
    df_pair = data[[x, y]].copy()
    if missing in ["pairwise", "listwise"]:
        df_pair = df_pair.dropna()
    elif missing == "raise":
        if df_pair.isna().any().any():
            raise InputError("Missing values found and missing='raise'.")
            
    n_used = len(df_pair)
    missing_count = n_original - n_used
    missing_ratio = missing_count / n_original if n_original > 0 else 1.0
    
    flags = []
    warnings = []
    
    if n_used < 2:
        raise InputError(
            f"At least 2 valid observations are required to profile a relationship; "
            f"got n_used={n_used} after handling missing values."
        )
        
    x_clean = df_pair[x].astype(float)
    y_clean = df_pair[y].astype(float)
    
    x_unique_ratio = x_clean.nunique() / n_used
    y_unique_ratio = y_clean.nunique() / n_used
    
    x_is_constant = x_clean.std() == 0 or x_clean.nunique() <= 1
    y_is_constant = y_clean.std() == 0 or y_clean.nunique() <= 1
    
    if missing_ratio > 0.5:
        flags.append("high_missingness")
        warnings.append(f">50% missing data ({missing_ratio:.1%} missing).")
        
    if x_unique_ratio < 0.05 or y_unique_ratio < 0.05:
        flags.append("low_unique_ratio")
        warnings.append("Low unique value ratio (< 0.05). Rank-based metrics may be unstable due to ties.")
        
    if n_used < 30:
        flags.append("low_n")
        warnings.append(f"Small sample size (n={n_used}). Interpret metrics with caution.")
        
    if x_is_constant or y_is_constant:
        flags.append("constant_input")
        warnings.append("One or both variables are constant. Metrics may not be computable.")
        
    return CleanPair(
        x=x_clean,
        y=y_clean,
        x_name=x,
        y_name=y,
        n_original=n_original,
        n_used=n_used,
        missing_count=missing_count,
        missing_ratio=missing_ratio,
        x_unique_ratio=x_unique_ratio,
        y_unique_ratio=y_unique_ratio,
        x_is_constant=x_is_constant,
        y_is_constant=y_is_constant,
        flags=flags,
        warnings=warnings
    )
