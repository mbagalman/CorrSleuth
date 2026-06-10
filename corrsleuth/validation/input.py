import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List
from corrsleuth.exceptions import InputError

_TIE_RATE_WARN_THRESHOLD = 0.30


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
    x_tie_rate: float
    y_tie_rate: float
    flags: List[str]          # machine-readable
    warnings: List[str]       # raw validation warnings only


def is_constant_series(series: pd.Series) -> bool:
    return series.nunique() <= 1 or series.std() == 0


def compute_tie_rate(series: pd.Series) -> float:
    """Return the fraction of observations whose value is shared with another row.

    A series with all unique values has tie rate 0; a series of all-equal values
    has tie rate 1. Useful as a rank-metric reliability signal — Spearman and
    Kendall handle ties via tie-correction, but high tie rates compress the rank
    space and reduce their effective resolution.
    """
    n = len(series)
    if n == 0:
        return 0.0
    counts = series.value_counts()
    n_singletons = int((counts == 1).sum())
    return float((n - n_singletons) / n)


def compute_heuristic_flags(pair: "CleanPair") -> List[str]:
    """Return the subset of CleanPair flags that the heuristic cascade reads.

    Validation also emits warning-only flags (``high_missingness``,
    ``low_unique_ratio``) that ``apply_heuristics`` ignores; this helper exposes
    the heuristic-relevant flags so bootstrap replicates can synthesize the same
    decision context as the original validated pair.
    """
    flags: List[str] = []
    if pair.x_is_constant or pair.y_is_constant:
        flags.append("constant_input")
    if pair.n_used < 30:
        flags.append("low_n")
    return flags


def validate_pair(data: pd.DataFrame, x: str, y: str, missing: str = "pairwise") -> CleanPair:
    if x == y:
        raise InputError(
            f"x and y must be different columns; got '{x}' for both."
        )
    if x not in data.columns:
        raise InputError(f"Column '{x}' not found in data.")
    if y not in data.columns:
        raise InputError(f"Column '{y}' not found in data.")

    s_x = data[x]
    s_y = data[y]

    for name, selected in ((x, s_x), (y, s_y)):
        if isinstance(selected, pd.DataFrame):
            raise InputError(
                f"Column '{name}' matches multiple columns in data; "
                f"column names must be unique."
            )

    if not pd.api.types.is_numeric_dtype(s_x):
        raise InputError(f"Column '{x}' is not numeric.")
    if not pd.api.types.is_numeric_dtype(s_y):
        raise InputError(f"Column '{y}' is not numeric.")

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

    # Checked after missing-value handling so an inf in a row that the missing
    # policy drops anyway does not abort the profile.
    if np.isinf(df_pair[x]).any() or np.isinf(df_pair[y]).any():
        raise InputError(
            "Input data contains infinite values in the rows used for profiling."
        )

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

    x_is_constant = is_constant_series(x_clean)
    y_is_constant = is_constant_series(y_clean)

    x_tie_rate = compute_tie_rate(x_clean)
    y_tie_rate = compute_tie_rate(y_clean)

    if missing_ratio > 0.5:
        flags.append("high_missingness")
        warnings.append(f">50% missing data ({missing_ratio:.1%} missing).")

    if x_unique_ratio < 0.05 or y_unique_ratio < 0.05:
        flags.append("low_unique_ratio")
        warnings.append("Low unique value ratio (< 0.05). Rank-based metrics may be unstable due to ties.")

    for name, tie_rate in ((x, x_tie_rate), (y, y_tie_rate)):
        if tie_rate > _TIE_RATE_WARN_THRESHOLD:
            warnings.append(
                f"'{name}' has a high tie rate ({tie_rate:.1%}); rank metrics "
                f"like Spearman and Kendall may be less informative due to "
                f"repeated values."
            )
    if x_tie_rate > _TIE_RATE_WARN_THRESHOLD or y_tie_rate > _TIE_RATE_WARN_THRESHOLD:
        flags.append("high_tie_rate")

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
        x_tie_rate=x_tie_rate,
        y_tie_rate=y_tie_rate,
        flags=flags,
        warnings=warnings
    )
