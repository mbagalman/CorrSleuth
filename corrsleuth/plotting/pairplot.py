import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

_LOWESS_SUBSAMPLE_SEED = 42


def _format_value(value) -> str:
    return f"{value:.3f}" if value is not None and pd.notna(value) else "NA"


def _add_text(
    ax,
    y_pos: float,
    text: str,
    *,
    x_pos: float = 0.0,
    line_height: float = 0.06,
    **kwargs,
) -> float:
    ax.text(x_pos, y_pos, text, **kwargs)
    return y_pos - line_height


def plot_pair(result, show: bool = False):
    """Create a compact 1x3 diagnostic plot for a profiled pair.

    Parameters
    ----------
    result : CorrSleuthResult
        A result from :func:`corrsleuth.profile_pair`. Must retain the cleaned
        data (``_clean_x``/``_clean_y``) used to draw the panels.
    show : bool, default False
        If ``True``, display the figure via ``matplotlib.pyplot.show()``.

    Returns
    -------
    matplotlib.figure.Figure
        The figure with three panels: a scatter of ``x`` vs ``y``, a
        rank-transformed scatter, and a text summary of the metrics and label.
    """
    x_name = result.x_name
    y_name = result.y_name

    # 1. Setup Data
    x = result._clean_x.values
    y = result._clean_y.values
    n = len(x)

    # 2. Setup Figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Relationship Profile: {x_name} vs {y_name}", fontsize=14, fontweight="bold"
    )

    ax_scatter = axes[0]
    ax_rank = axes[1]
    ax_text = axes[2]

    # 3. Scatter Plot
    if n > 5000:
        ax_scatter.hexbin(x, y, gridsize=30, cmap="Blues", mincnt=1)
    else:
        alpha = min(1.0, 100 / n) if n > 0 else 1.0
        ax_scatter.scatter(x, y, alpha=alpha, edgecolor="none", color="steelblue")

    ax_scatter.set_title("Raw Data Scatter")
    ax_scatter.set_xlabel(x_name)
    ax_scatter.set_ylabel(y_name)

    try:
        import statsmodels.api as sm

        lowess = sm.nonparametric.lowess
        # Subsample for LOWESS to keep large-n plots responsive. Seeded so
        # repeated plot() calls produce the same smoother.
        n_lowess = min(n, 1000)
        if n > n_lowess:
            rng = np.random.default_rng(_LOWESS_SUBSAMPLE_SEED)
            idx = rng.choice(n, n_lowess, replace=False)
        else:
            idx = np.arange(n)
        try:
            # Degenerate inputs (e.g. a constant variable) make statsmodels'
            # lowess divide by zero internally; suppress the numpy warning and
            # let the except handle anything unrecoverable.
            with np.errstate(invalid="ignore", divide="ignore"):
                z = lowess(y[idx], x[idx], frac=0.3)
            order = np.argsort(z[:, 0])
            ax_scatter.plot(z[order, 0], z[order, 1], color="darkorange", linewidth=2)
        except Exception:
            pass
    except ImportError:
        pass

    # 4. Rank Plot
    if n > 0:
        rx = stats.rankdata(x)
        ry = stats.rankdata(y)
        if n > 5000:
            ax_rank.hexbin(rx, ry, gridsize=30, cmap="Purples", mincnt=1)
        else:
            alpha = min(1.0, 100 / n) if n > 0 else 1.0
            ax_rank.scatter(
                rx, ry, alpha=alpha, edgecolor="none", color="rebeccapurple"
            )

    ax_rank.set_title("Ranked Data Scatter")
    ax_rank.set_xlabel(f"Rank({x_name})")
    ax_rank.set_ylabel(f"Rank({y_name})")

    # 5. Text Summary Panel
    ax_text.axis("off")

    y_pos = 0.97
    line_height = 0.055

    y_pos = _add_text(
        ax_text,
        y_pos,
        "Primary Pattern",
        fontweight="bold",
        fontsize=11,
        line_height=line_height,
    )
    y_pos = _add_text(
        ax_text,
        y_pos,
        result.pattern,
        x_pos=0.05,
        fontsize=10,
        color="firebrick",
        line_height=line_height,
    )
    y_pos = _add_text(
        ax_text,
        y_pos,
        f"n_used: {n}",
        x_pos=0.05,
        fontsize=9,
        color="dimgray",
        line_height=line_height,
    )
    y_pos -= line_height * 0.4

    y_pos = _add_text(
        ax_text,
        y_pos,
        "Metrics",
        fontweight="bold",
        fontsize=11,
        line_height=line_height,
    )

    for _, row in result.metrics.iterrows():
        m_name = row["metric"].replace("_", " ").title()
        m_val = _format_value(row["value"])
        ax_text.text(0.05, y_pos, f"{m_name}:", fontsize=10)
        ax_text.text(0.5, y_pos, m_val, fontsize=10, fontweight="bold")
        y_pos -= line_height

    y_pos -= line_height * 0.4
    y_pos = _add_text(
        ax_text,
        y_pos,
        "Diagnostics",
        fontweight="bold",
        fontsize=11,
        line_height=line_height,
    )
    diagnostic_rows = [
        ("Disagreement", result.diagnostics.disagreement_score),
        ("Rank-linear gap", result.diagnostics.rank_linear_gap),
        ("Nonmonotonic gap", result.diagnostics.nonmonotonic_gap),
        ("Trim delta", result.diagnostics.pearson_trim_delta),
    ]
    for label, value in diagnostic_rows:
        ax_text.text(0.05, y_pos, f"{label}:", fontsize=9)
        ax_text.text(0.55, y_pos, _format_value(value), fontsize=9, fontweight="bold")
        y_pos -= line_height

    y_pos -= line_height * 0.4

    if result.warnings:
        ax_text.text(
            0.0, y_pos, "Warnings", fontweight="bold", fontsize=11, color="darkorange"
        )
        y_pos -= line_height
        for w in result.warnings[:3]:  # Show max 3 warnings to avoid overflow
            ax_text.text(0.05, y_pos, f"- {w}", fontsize=9, wrap=True)
            y_pos -= line_height
    else:
        y_pos = _add_text(
            ax_text,
            y_pos,
            "Warnings",
            fontweight="bold",
            fontsize=11,
            color="darkorange",
            line_height=line_height,
        )
        _add_text(
            ax_text,
            y_pos,
            "None",
            x_pos=0.05,
            fontsize=9,
            color="dimgray",
            line_height=line_height,
        )

    plt.tight_layout(rect=(0, 0, 1, 0.95))

    if show:
        plt.show()

    return fig
