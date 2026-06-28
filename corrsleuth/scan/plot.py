"""Scatter-panel plotting for a target scan.

:func:`build_scan_figure` backs :meth:`CorrSleuthTargetReport.plot_top`. It is a
free function taking the report so the (expensive) matplotlib import and all
figure-layout logic stay out of :mod:`corrsleuth.scan.report`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import pandas as pd

from corrsleuth.exceptions import InputError
from corrsleuth.scan.core import TargetScanEntry, metrics_map

if TYPE_CHECKING:
    from corrsleuth.scan.report import CorrSleuthTargetReport

#: Valid ``sort_by`` keys for :meth:`CorrSleuthTargetReport.plot_top`. Metric
#: names are sorted by absolute value descending; ``disagreement_score`` is
#: sorted by raw value descending.
_VALID_SORT_KEYS: tuple[str, ...] = (
    "disagreement_score",
    "pearson",
    "spearman",
    "kendall_tau_b",
    "distance_correlation",
    "mutual_information",
    "pearson_trimmed_1pct",
    "pearson_winsorized_1pct",
    "biweight_midcorrelation",
    "pearson_median_clipped_20pct",
    "chatterjee_xi",
    "chatterjee_xi_reverse",
)


def build_scan_figure(
    report: CorrSleuthTargetReport,
    *,
    n: int = 12,
    sort_by: str = "disagreement_score",
    patterns: Sequence[str] | None = None,
    ncols: int = 3,
    figsize: tuple[float, float] | None = None,
    show: bool = False,
) -> Any:
    """Build the top-``n`` scatter-panel figure for ``report``.

    See :meth:`CorrSleuthTargetReport.plot_top` for the parameter and return
    semantics; this function holds the implementation.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise InputError("n must be a positive integer.")
    if isinstance(ncols, bool) or not isinstance(ncols, int) or ncols < 1:
        raise InputError("ncols must be a positive integer.")
    if sort_by not in _VALID_SORT_KEYS:
        raise InputError(
            f"Unknown sort_by: {sort_by!r}. Supported values are {_VALID_SORT_KEYS}."
        )

    if isinstance(patterns, str):
        patterns = [patterns]

    candidates = [
        e
        for e in report.successes
        if e.result_data._clean_x is not None and e.result_data._clean_y is not None
    ]
    if patterns is not None:
        pattern_set = set(patterns)
        candidates = [e for e in candidates if e.result_data.pattern in pattern_set]

    candidates.sort(key=lambda e: (-_sort_value(e, sort_by), e.column))
    candidates = candidates[:n]

    # matplotlib is a hard dependency, but importing pyplot is expensive
    # (hundreds of ms), so it stays deferred here to keep `import corrsleuth`
    # and non-plotting workflows fast.
    import matplotlib.pyplot as plt

    if not candidates:
        fig, ax = plt.subplots(figsize=figsize or (6, 4))
        ax.text(
            0.5,
            0.5,
            "No variables to plot.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
            color="dimgray",
        )
        ax.set_axis_off()
        fig.suptitle(
            f"Target scan: {report.target}",
            fontsize=12,
            fontweight="bold",
        )
        if show:
            plt.show()
        return fig

    nrows = math.ceil(len(candidates) / ncols)
    if figsize is None:
        figsize = (4 * ncols, 3 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    flat_axes = axes.flatten()

    # flat_axes is padded to a full grid, so it is intentionally longer than
    # candidates; strict=False lets zip stop at the shorter candidates list.
    for ax, entry in zip(flat_axes, candidates, strict=False):
        result = entry.result_data
        # candidates were filtered on both clean series being present.
        assert result._clean_x is not None and result._clean_y is not None
        # scan_target() calls profile_pair(data, target, col), so
        # _clean_x holds the target's data and _clean_y holds the
        # candidate's. EDA convention puts the candidate (predictor) on x
        # and the target on y, so swap here.
        target_data = result._clean_x.to_numpy()
        candidate_data = result._clean_y.to_numpy()
        n_pts = len(candidate_data)

        if n_pts > 5000:
            ax.hexbin(
                candidate_data,
                target_data,
                gridsize=30,
                cmap="Blues",
                mincnt=1,
            )
        else:
            alpha = min(1.0, 100 / n_pts) if n_pts > 0 else 1.0
            ax.scatter(
                candidate_data,
                target_data,
                alpha=alpha,
                edgecolor="none",
                color="steelblue",
                s=8,
            )

        ax.set_title(_panel_title(entry), fontsize=9)
        ax.set_xlabel(entry.column, fontsize=8)
        ax.set_ylabel(report.target, fontsize=8)
        ax.tick_params(labelsize=7)

    for ax in flat_axes[len(candidates) :]:
        ax.set_axis_off()

    fig.suptitle(
        f"Target scan: {report.target} (top by {sort_by})",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if show:
        plt.show()

    return fig


def _sort_value(entry: TargetScanEntry, sort_by: str) -> float:
    if sort_by == "disagreement_score":
        return float(entry.result_data.disagreement_score)
    metrics = metrics_map(entry)
    value = metrics.get(sort_by)
    if value is None or pd.isna(value):
        return 0.0
    return abs(float(value))


def _panel_title(entry: TargetScanEntry) -> str:
    metrics = metrics_map(entry)
    pearson = metrics.get("pearson")
    spearman = metrics.get("spearman")

    def _fmt(value: Any) -> str:
        if value is None or pd.isna(value):
            return "NA"
        return f"{value:.2f}"

    return (
        f"{entry.column}\n{entry.result_data.pattern} | "
        f"p={_fmt(pearson)} s={_fmt(spearman)}"
    )
