from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd

from corrsleuth.exceptions import InputError

#: Minimum number of paired observations before metrics are considered to have
#: adequate power. Below this the ``low_n`` flag is set, which the heuristic
#: cascade maps to ``low_power_or_uncertain``. n = 30 is the conventional
#: "small sample" rule of thumb (the point near which the t-distribution
#: approaches normal); it is intentionally a floor, not a guarantee — see the
#: caveat in docs/interpretation-guide.md and docs/thresholds-and-rationale.md.
LOW_N_THRESHOLD = 30

#: Fraction of an input column that may be tied (non-unique) before a rank
#: metric reliability warning is emitted. Above ~30% ties, Spearman/Kendall
#: tie-correction is working hard enough that their effective resolution drops.
_TIE_RATE_WARN_THRESHOLD = 0.30

#: Missing-data fraction above which a high-missingness warning is emitted. At
#: more than half missing, listwise/pairwise deletion has removed most of the
#: data and any coefficient is computed on an unrepresentative remainder.
_HIGH_MISSINGNESS_THRESHOLD = 0.5

#: Unique-value ratio (distinct values / n) below which a column is treated as
#: near-discrete and rank metrics are warned as tie-unstable. 5% distinct
#: values means each value is shared by ~20 rows on average.
_LOW_UNIQUE_RATIO_THRESHOLD = 0.05


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
    flags: list[str]  # machine-readable
    warnings: list[str]  # raw validation warnings only


class DtypeProblem(NamedTuple):
    """Why a column fails the real-numeric gate."""

    error_type: str  # "NonNumeric" or "ComplexDtype"
    message: str


def real_numeric_problem(
    series: pd.Series, label: str, context: str = "profiling"
) -> DtypeProblem | None:
    """Return why ``series`` fails the real-numeric gate, or None if it passes.

    Every gate that decides whether a column is profilable routes through this
    classifier so the acceptance policy, the ``error_type`` code, and the
    user-facing wording live in one place.

    ``pd.api.types.is_numeric_dtype`` treats complex dtypes as numeric, but every
    metric CorrSleuth computes is defined for real-valued data. Casting a complex
    column to ``float`` silently discards the imaginary part (pandas emits a
    ``ComplexWarning``), so complex columns are rejected up front rather than
    projected onto the real axis without the caller's knowledge.

    ``label`` is the display name used in the message (e.g. ``"Column 'x'"`` or
    ``"Target column 'y'"``); ``context`` is the operation the caller was asked
    to perform (``"profiling"`` or ``"scanning"``).
    """
    if not pd.api.types.is_numeric_dtype(series):
        return DtypeProblem("NonNumeric", f"{label} is not numeric.")
    if pd.api.types.is_complex_dtype(series):
        return DtypeProblem(
            "ComplexDtype",
            f"{label} has a complex dtype; CorrSleuth only supports real-valued "
            f"numeric data. Cast to a real dtype explicitly (e.g. take the real "
            f"part or magnitude) before {context}.",
        )
    return None


def is_real_numeric_dtype(series: pd.Series) -> bool:
    """Return True for real-valued numeric columns (see ``real_numeric_problem``)."""
    return real_numeric_problem(series, "") is None


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


def compute_heuristic_flags(pair: "CleanPair") -> list[str]:
    """Return the subset of CleanPair flags that the heuristic cascade reads.

    Validation also emits warning-only flags (``high_missingness``,
    ``low_unique_ratio``) that ``apply_heuristics`` ignores; this helper exposes
    the heuristic-relevant flags so bootstrap replicates can synthesize the same
    decision context as the original validated pair.
    """
    flags: list[str] = []
    if pair.x_is_constant or pair.y_is_constant:
        flags.append("constant_input")
    if pair.n_used < LOW_N_THRESHOLD:
        flags.append("low_n")
    return flags


def validate_pair(
    data: pd.DataFrame, x: str, y: str, missing: str = "pairwise"
) -> CleanPair:
    """Validate and clean a numeric ``x``/``y`` pair into a :class:`CleanPair`.

    Parameters
    ----------
    data : pd.DataFrame
        Source data containing both columns.
    x, y : str
        Names of the two real-valued numeric columns to validate.
    missing : {"pairwise", "listwise", "raise"}, default "pairwise"
        Missing-value policy. ``"pairwise"`` drops rows missing in ``x`` or
        ``y`` only. ``"listwise"`` drops rows missing in *any* column of
        ``data`` (complete-case deletion) before selecting the pair, so the two
        coincide only when ``data`` contains just ``x`` and ``y``. ``"raise"``
        errors if the pair contains any missing values.

    Returns
    -------
    CleanPair
        The cleaned pair with derived statistics (counts, unique/tie rates,
        constant-column flags) and validation warnings.

    Raises
    ------
    InputError
        If ``x == y``, a column is missing, non-numeric, or complex-valued
        (only real-valued numeric data is supported), a name matches
        multiple columns, ``missing`` is not a supported mode, missing values
        are present when ``missing="raise"``, the rows used contain infinite
        values, or fewer than two valid observations remain.
    """
    if x == y:
        raise InputError(f"x and y must be different columns; got '{x}' for both.")
    if x not in data.columns:
        raise InputError(f"Column '{x}' not found in data.")
    if y not in data.columns:
        raise InputError(f"Column '{y}' not found in data.")

    s_x = data[x]
    s_y = data[y]

    for name, selected in ((x, s_x), (y, s_y)):
        # The DataFrame check must precede the dtype check: data[name] returns
        # a DataFrame for a duplicated name, which the dtype predicates cannot
        # classify.
        if isinstance(selected, pd.DataFrame):
            raise InputError(
                f"Column '{name}' matches multiple columns in data; "
                f"column names must be unique."
            )
        problem = real_numeric_problem(selected, f"Column '{name}'")
        if problem is not None:
            raise InputError(problem.message)

    n_original = len(data)

    # Missing value handling
    if missing not in ["pairwise", "listwise", "raise"]:
        raise InputError(
            f"Unsupported missing mode: '{missing}'. Supported modes are 'pairwise', 'listwise', and 'raise'."
        )

    if missing == "listwise":
        # Complete-case deletion: drop any row that is missing a value in ANY
        # column of `data`, then restrict to the pair. This differs from
        # "pairwise" whenever `data` carries columns beyond x/y (e.g. a
        # multi-column scan), where it yields a common complete-case sample
        # across pairs rather than dropping only on x/y.
        df_pair = data.dropna()[[x, y]].copy()
    elif missing == "pairwise":
        # Drop rows missing in x or y only; other columns are ignored.
        df_pair = data[[x, y]].dropna().copy()
    else:  # missing == "raise"
        df_pair = data[[x, y]].copy()
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

    if missing_ratio > _HIGH_MISSINGNESS_THRESHOLD:
        flags.append("high_missingness")
        warnings.append(f">50% missing data ({missing_ratio:.1%} missing).")

    if (
        x_unique_ratio < _LOW_UNIQUE_RATIO_THRESHOLD
        or y_unique_ratio < _LOW_UNIQUE_RATIO_THRESHOLD
    ):
        flags.append("low_unique_ratio")
        warnings.append(
            "Low unique value ratio (< 0.05). Rank-based metrics may be unstable due to ties."
        )

    for name, tie_rate in ((x, x_tie_rate), (y, y_tie_rate)):
        if tie_rate > _TIE_RATE_WARN_THRESHOLD:
            warnings.append(
                f"'{name}' has a high tie rate ({tie_rate:.1%}); rank metrics "
                f"like Spearman and Kendall may be less informative due to "
                f"repeated values."
            )
    if x_tie_rate > _TIE_RATE_WARN_THRESHOLD or y_tie_rate > _TIE_RATE_WARN_THRESHOLD:
        flags.append("high_tie_rate")

    if n_used < LOW_N_THRESHOLD:
        flags.append("low_n")
        warnings.append(
            f"Small sample size (n={n_used}). Interpret metrics with caution."
        )

    if x_is_constant or y_is_constant:
        flags.append("constant_input")
        warnings.append(
            "One or both variables are constant. Metrics may not be computable."
        )

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
        warnings=warnings,
    )
