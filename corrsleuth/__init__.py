"""
CorrSleuth: A relationship diagnosis engine for pandas users.
"""

__version__ = "0.1.0"

from corrsleuth.api import profile_pair
from corrsleuth.datasets import make_relationship

__all__ = ["profile_pair", "make_relationship"]
