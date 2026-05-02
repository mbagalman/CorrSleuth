import pytest
import numpy as np
from corrsleuth.datasets import make_relationship
from corrsleuth.api import profile_pair
import matplotlib.pyplot as plt
import pandas as pd

def test_explain_caveat():
    df = make_relationship("linear_positive", n=100)
    res = profile_pair(df, "x", "y")
    
    exp_with = res.explain(include_caveat=True)
    assert "causally without proper design" in exp_with
    
    exp_without = res.explain(include_caveat=False)
    assert "causally without proper design" not in exp_without

def test_summary():
    df = make_relationship("linear_positive", n=100)
    res = profile_pair(df, "x", "y")
    
    summary_text = res.summary()
    assert "Relationship Profile: x vs y" in summary_text
    assert "Metrics:" in summary_text
    assert "Caveat:" in summary_text
    
def test_plot_returns_figure():
    df = make_relationship("linear_positive", n=100)
    res = profile_pair(df, "x", "y")
    
    fig = res.plot(show=False)
    assert isinstance(fig, plt.Figure)

def test_serialization():
    df = make_relationship("linear_positive", n=100)
    res = profile_pair(df, "x", "y")
    
    d = res.to_dict()
    assert d["x"] == "x"
    assert d["y"] == "y"
    assert "pattern" in d
    assert isinstance(d["metrics"], list)
    
    frame = res.to_frame()
    assert "pattern" in frame.columns
    assert "value" in frame.columns
    assert len(frame) >= 3 # at least core metrics

def test_constant_input_safe_rendering():
    df = pd.DataFrame({"x": [1, 1, 1, 1], "y": [1, 2, 3, 4]})
    res = profile_pair(df, "x", "y")
    assert res.pattern == "not_computable"
    summary_text = res.summary()
    assert "NA" in summary_text
    fig = res.plot()
    assert isinstance(fig, plt.Figure)

def test_plotting_uses_clean_data():
    df = make_relationship("linear_positive", n=100)
    df.loc[0, "x"] = np.nan
    res = profile_pair(df, "x", "y")
    
    # Mutate df after profiling
    df["x"] = 0
    
    fig = res.plot()
    # It shouldn't crash or plot zeroes if it stored clean data properly
    assert isinstance(fig, plt.Figure)
