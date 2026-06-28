"""
Custom exceptions for CorrSleuth.
"""


class CorrSleuthError(Exception):
    """Base exception for all CorrSleuth errors."""

    pass


class InputError(CorrSleuthError):
    """Raised when the input data is invalid (e.g., non-numeric, infinite, completely missing)."""

    pass


class OptionalDependencyError(CorrSleuthError):
    """Raised when an optional dependency is required but not installed."""

    pass


class MetricComputationError(CorrSleuthError):
    """Raised when a specific metric fails to compute unexpectedly."""

    pass
