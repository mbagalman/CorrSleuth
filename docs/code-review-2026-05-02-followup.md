# CorrSleuth Code Review Follow-Up - 2026-05-02

## Summary

The previous review findings have mostly been addressed:

- `missing="listwise"` and invalid missing modes are now handled.
- `not_computable` metric values now render as `NA` in `summary()` and the plot text panel.
- `max_n_for_dcor` is now exposed by `profile_pair()` and passed into distance correlation.
- Conflicting Pearson/Spearman direction warnings have been added.
- Plotting now uses preserved cleaned data instead of re-cleaning the original DataFrame.
- `include_caveat` is now stored as a result-level default.
- `make_relationship` is exported from the package root.

Current verification:

```text
C:\Users\micha\AppData\Local\Python\pythoncore-3.14-64\python.exe -m pytest -q
21 passed in 11.74s
```

## Finding

### P2 - Optional LOWESS plotting path now crashes when `statsmodels` is installed

File: `corrsleuth/plotting/pairplot.py`

`plot_pair()` now converts the cleaned pandas Series to NumPy arrays:

```python
x = result._clean_x.values
y = result._clean_y.values
```

That is fine for scatter and rank plotting, but the optional LOWESS block still treats `x` and `y` like pandas Series:

```python
z = lowess(y.values[idx], x.values[idx], frac=0.3)
```

Because `x` and `y` are already NumPy arrays, they do not have `.values`. The current test environment does not appear to have `statsmodels` installed, so this path is skipped and the test suite passes. In an environment with `statsmodels`, `result.plot()` will raise `AttributeError` instead of returning a figure.

Suggested fix:

- Change the LOWESS call to use the arrays directly:

```python
z = lowess(y[idx], x[idx], frac=0.3)
```

- Consider catching a narrow plotting exception around optional LOWESS so an optional smoother cannot break the required base plot contract.
- Add a regression test that monkeypatches a fake `statsmodels.api` module, or add a conditional test when `statsmodels` is installed.

## Remaining Test Gaps

- `max_n_for_dcor=None` disables downsampling.
- `max_n_for_dcor=...` downsampling appends a warning.
- Conflicting Pearson/Spearman signs append the intended warning.
- Plotting with the optional LOWESS path enabled.
