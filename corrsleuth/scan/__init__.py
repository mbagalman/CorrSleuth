"""Target-oriented scanning workflow.

Profiles every eligible numeric predictor in a DataFrame against a single
numeric target column, returning a report that can be inspected as a tidy
DataFrame, a text summary, Markdown, or a grid of scatter panels. Per-column
failures are captured rather than aborting the scan unless the caller asks for
``errors="raise"``.

The implementation is split across:

- :mod:`corrsleuth.scan.core` — orchestration (:func:`scan_target`) and the
  :class:`TargetScanEntry` record.
- :mod:`corrsleuth.scan.report` — :class:`CorrSleuthTargetReport` and its
  ``to_frame`` / ``summary`` / ``to_markdown`` / ``pearson_underrated``
  rendering.
- :mod:`corrsleuth.scan.plot` — the ``plot_top`` figure builder.
"""

from corrsleuth.scan.core import TargetScanEntry, scan_target
from corrsleuth.scan.report import CorrSleuthTargetReport

__all__ = [
    "scan_target",
    "CorrSleuthTargetReport",
    "TargetScanEntry",
]
