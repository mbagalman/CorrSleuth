"""
CorrSleuth: A relationship diagnosis engine for pandas users.
"""

__version__ = "0.2.0"

from corrsleuth.api import profile_pair
from corrsleuth.datasets import make_relationship
from corrsleuth.exceptions import (
    CorrSleuthError,
    InputError,
    MetricComputationError,
    OptionalDependencyError,
)
from corrsleuth.result import CorrSleuthResult, MetricDiagnostics
from corrsleuth.scan import (
    CorrSleuthTargetReport,
    TargetScanEntry,
    scan_target,
)

__all__ = [
    "profile_pair",
    "make_relationship",
    "scan_target",
    "CorrSleuthResult",
    "MetricDiagnostics",
    "CorrSleuthTargetReport",
    "TargetScanEntry",
    "CorrSleuthError",
    "InputError",
    "MetricComputationError",
    "OptionalDependencyError",
]
