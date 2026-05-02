import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

def plot_pair(result, show: bool = False):
    """
    Creates a compact 1x3 diagnostic plot.
    """
    x_name = result.x_name
    y_name = result.y_name
    
    # 1. Setup Data
    x = result._clean_x.values
    y = result._clean_y.values
    n = len(x)
    
    # 2. Setup Figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Relationship Profile: {x_name} vs {y_name}", fontsize=14, fontweight="bold")
    
    ax_scatter = axes[0]
    ax_rank = axes[1]
    ax_text = axes[2]
    
    # 3. Scatter Plot
    if n > 5000:
        ax_scatter.hexbin(x, y, gridsize=30, cmap='Blues', mincnt=1)
    else:
        alpha = min(1.0, 100 / n) if n > 0 else 1.0
        ax_scatter.scatter(x, y, alpha=alpha, edgecolor="none", color="steelblue")
        
    ax_scatter.set_title("Raw Data Scatter")
    ax_scatter.set_xlabel(x_name)
    ax_scatter.set_ylabel(y_name)
    
    try:
        import statsmodels.api as sm
        lowess = sm.nonparametric.lowess
        # Subsample for LOWESS if too large to avoid hanging
        n_lowess = min(n, 1000)
        import numpy as np
        idx = np.random.choice(n, n_lowess, replace=False) if n > n_lowess else np.arange(n)
        try:
            z = lowess(y[idx], x[idx], frac=0.3)
            # sort for plotting line
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
            ax_rank.hexbin(rx, ry, gridsize=30, cmap='Purples', mincnt=1)
        else:
            alpha = min(1.0, 100 / n) if n > 0 else 1.0
            ax_rank.scatter(rx, ry, alpha=alpha, edgecolor="none", color="rebeccapurple")
            
    ax_rank.set_title("Ranked Data Scatter")
    ax_rank.set_xlabel(f"Rank({x_name})")
    ax_rank.set_ylabel(f"Rank({y_name})")
    
    # 5. Text Summary Panel
    ax_text.axis('off')
    
    y_pos = 0.95
    line_height = 0.07
    
    ax_text.text(0.0, y_pos, "Primary Pattern:", fontweight='bold', fontsize=11)
    y_pos -= line_height
    ax_text.text(0.05, y_pos, result.pattern, fontsize=10, color="firebrick")
    y_pos -= line_height * 1.5
    
    ax_text.text(0.0, y_pos, "Metrics:", fontweight='bold', fontsize=11)
    y_pos -= line_height
    
    for _, row in result.metrics.iterrows():
        m_name = row['metric'].replace('_', ' ').title()
        m_val = f"{row['value']:.3f}" if pd.notna(row['value']) else "NA"
        ax_text.text(0.05, y_pos, f"{m_name}:", fontsize=10)
        ax_text.text(0.5, y_pos, m_val, fontsize=10, fontweight='bold')
        y_pos -= line_height
        
    y_pos -= line_height * 0.5
    
    if result.warnings:
        ax_text.text(0.0, y_pos, "Warnings:", fontweight='bold', fontsize=11, color="darkorange")
        y_pos -= line_height
        for w in result.warnings[:3]: # Show max 3 warnings to avoid overflow
            ax_text.text(0.05, y_pos, f"- {w}", fontsize=9, wrap=True)
            y_pos -= line_height
            
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if show:
        plt.show()
        
    return fig
