# Pre-PyPI Release Code Review — 2026-06-10

Scope: full review of package source, tests, packaging metadata, CI, and
public documentation ahead of the 0.1.0 PyPI release.

## Verification performed

- Full test suite: **140 passed** (local, Python 3.14, numpy 2.4 / pandas 2.3 /
  scipy 1.17 / dcor 0.7 / scikit-learn 1.8).
- `python -m build`: sdist + wheel build cleanly; `twine check`: **PASSED** for both.
- Wheel contents verified: only `corrsleuth/` sources, `py.typed`, and LICENSE —
  no stray files.
- Clean-venv install of the wheel with **base dependencies only** (Python 3.14):
  `lite` and `deep` modes work, `mode="standard"` raises `OptionalDependencyError`
  with install instructions, `scan_target`, `plot()` (without statsmodels),
  `summary()`, `explain()`, and `to_markdown()` all work.
- README quickstart and canonical examples reproduce the documented patterns.
- README API documentation checked against actual signatures of `profile_pair`,
  `scan_target`, `CorrSleuthResult`, and `CorrSleuthTargetReport` — accurate.
- Metadata: PEP 639 license expression + `license-files`, absolute GitHub URLs
  (render correctly on PyPI), version single-sourced from `corrsleuth.__version__`,
  CHANGELOG consistent at 0.1.0.

## Issues

### P1 — fix before release

#### 1. Deep mode silently mislabels strong nonmonotonic dependence — **FIXED 2026-06-10**

> Resolution: `detect_metric_warnings()` now accepts the assigned label and
> emits a warning when Chatterjee's ξ (either direction) exceeds 0.35 while the
> label is `weak_or_no_relationship` or `mixed_or_ambiguous`. Covered by new
> tests in `tests/test_heuristics.py` (unit + deep-mode U-shape regression).
> README, interpretation guide, and CHANGELOG updated. The fuller cascade
> change (ξ triggering `nonmonotonic_dependence` in deep mode) remains deferred
> to 0.2.0.

```python
df = make_relationship("u_shape", n=500, noise=0.1, random_state=42)
r = profile_pair(df, "x", "y", mode="deep")
# r.pattern        -> "weak_or_no_relationship"
# chatterjee_xi    -> 0.936
# r.warnings       -> []   (nothing flags the contradiction)
```

The heuristic cascade (`corrsleuth/heuristics/classifier.py`) only consults
Pearson/Spearman/Kendall/dCor, so in deep mode a textbook U-shape gets the label
`weak_or_no_relationship` while the same output table shows `chatterjee_xi = 0.94`.
No warning is emitted. Given the package's core promise — "tells you where the
correlation matrix may be misleading" — deep mode producing a misleading label
with no caveat undercuts the headline use case.

Lowest-risk pre-release fix: extend `detect_metric_warnings()` to emit a warning
when `chatterjee_xi` (either direction) exceeds a threshold (e.g. 0.35) but the
assigned label is `weak_or_no_relationship` or `mixed_or_ambiguous`. A cascade
change (letting ξ trigger `nonmonotonic_dependence` in deep mode) is the fuller
fix but changes documented labeling behavior — defensible to defer to 0.2.0.

### P2 — should fix before release

#### 2. `profile_pair(df, "x", "x")` crashes with a raw pandas error — **FIXED 2026-06-10**

> Resolution: `validate_pair` now raises `InputError("x and y must be
> different columns...")` for `x == y` and a clear "matches multiple columns"
> `InputError` when duplicate column names make `data[x]` resolve to a
> DataFrame. `scan_target` got the same duplicate-name guard for the target
> column. Covered by new tests in `tests/test_validation.py` and
> `tests/test_scan.py`.

`profile_pair(df, "x", "x")` raises
`ValueError: The truth value of a Series is ambiguous...` because
`data[[x, y]]` with `x == y` produces duplicate columns and `df_pair[x]`
returns a DataFrame. Same root cause gives a *misleading* `InputError`
("Column 'x' is not numeric.") when the input DataFrame itself has duplicate
column names. Fix in `validate_pair`: raise a clear `InputError` when
`x == y`, and when `data[x]` / `data[y]` resolves to more than one column.

#### 3. `scan_target(max_pairs=-1)` silently drops the last candidate — **FIXED 2026-06-10**

> Resolution: `scan_target` now validates `max_pairs` and `sample_size` as
> positive int or `None`, raising `InputError` otherwise (same pattern as the
> existing `top_n` / `n` / `ncols` / `bootstrap` validation). Docstrings
> updated; covered by new tests in `tests/test_scan.py`.

`candidates[:max_pairs]` with a negative value slices from the end, so
`max_pairs=-1` silently profiles all-but-one column instead of raising.
Likewise `sample_size=-5` escapes as a raw pandas `ValueError` rather than
`InputError`. Validate both as positive int or `None` (the codebase already
does exactly this for `top_n`, `n`, `ncols`, and `bootstrap` — these two are
the only gaps).

#### 4. The sdist ships a broken test suite — **FIXED 2026-06-10**

> Resolution: added `MANIFEST.in` with `prune tests` (the partial inclusion
> came from distutils' legacy default of auto-including `tests/test*.py`) and
> `include CHANGELOG.md`. Verified: rebuilt sdist contains no test files,
> includes the changelog, and both artifacts still pass `twine check`.

The sdist includes `tests/test_*.py` but **not** `tests/statsmodels/` (the
LOWESS mock). `test_plot_lowess_subsample_is_deterministic` asserts that a
LOWESS line exists, which requires that mock — so anyone running the test
suite from the published sdist gets a failure. Either exclude tests from the
sdist entirely (common, recommended: `tool.setuptools` exclude or MANIFEST.in
`prune tests`) or include the whole `tests/` tree.

### P3 — nice to have, fine post-release

#### 5. Mock statsmodels shadows the real one; real-statsmodels path never tested — **FIXED 2026-06-10**

> Resolution: mock moved to `tests/_mocks/statsmodels/` and scoped to the two
> LOWESS tests via a `fake_statsmodels` fixture that also evicts statsmodels
> from `sys.modules` on teardown. Added `test_plot_lowess_real_statsmodels`
> (skipped unless statsmodels is installed) which asserts the smoother is
> actually drawn — necessary because `pairplot` deliberately swallows LOWESS
> failures. CI gained a real-statsmodels cell. Verified in a venv with real
> statsmodels: full suite passes and the real LOWESS path draws the line. Also
> silenced a harmless numpy divide-by-zero RuntimeWarning that real
> statsmodels emits for constant inputs.

Because `tests/` has no `__init__.py`, pytest prepends `tests/` to `sys.path`
during collection, so the fake `statsmodels` package shadows a real install for
the *entire* test session (the `monkeypatch.syspath_prepend` calls in
`test_output.py` are effectively redundant). Consequence: the LOWESS code path
in `pairplot.py` has never been exercised against real statsmodels in any
environment (it is not installed locally and not in the CI matrix) — the exact
path that shipped a crash once before (see archived 2026-05-02 review). Suggest:
move the mock to `tests/_mocks/statsmodels/` and prepend `tests/_mocks` only in
the two tests that need it, plus add one CI cell that installs real statsmodels.

#### 6. Useful names missing from the top-level namespace — **FIXED 2026-06-10**

> Resolution: `corrsleuth` now exports `CorrSleuthResult`, `MetricDiagnostics`,
> `CorrSleuthError`, `InputError`, `MetricComputationError`, and
> `OptionalDependencyError`. README notes the import path.

`corrsleuth.__init__` exports `profile_pair`, `make_relationship`, `scan_target`,
`CorrSleuthTargetReport`, `TargetScanEntry` — but not `CorrSleuthResult`,
`MetricDiagnostics`, or the exceptions. The README tells users CorrSleuth
"raises `OptionalDependencyError`", but catching it requires knowing to import
from `corrsleuth.exceptions`. Consider exporting the exceptions (and
`CorrSleuthResult` for type hints) at top level.

#### 7. Python 3.14 works but is unadvertised — **FIXED 2026-06-10**

> Resolution: added the `3.14` classifier (verified present in the rebuilt
> wheel metadata) and `3.14` to the CI test matrix. dcor's numba dependency
> supports 3.14 (verified by the passing local 3.14 standard-mode run).

The clean-venv smoke test above ran on Python 3.14 successfully. Consider adding
the `3.14` classifier and a CI matrix entry.

#### 8. Infinity check runs before missing-value handling — **FIXED 2026-06-10**

> Resolution: the check now runs after the missing policy is applied, so an
> `inf` in a row that pairwise/listwise handling drops no longer aborts the
> profile; `inf` in rows actually used still raises `InputError`. Both
> behaviors covered by new tests in `tests/test_validation.py`.

`validate_pair` raises `InputError` for `inf` anywhere in either column, even in
rows that `missing="pairwise"` would drop. Verified: a row with `inf` in `x` and
`NaN` in `y` aborts the profile. Either run the check after the dropna or
document the behavior.

#### 9. Release mechanics — **MOSTLY FIXED 2026-06-10**

- ~~CHANGELOG date is 2026-05-31; update to the actual release date.~~
  Updated to 2026-06-10; bump again if release day slips.
- No `v0.1.0` git tag exists yet; the CHANGELOG link targets
  `releases/tag/v0.1.0` — tag after (or as part of) the release. **Still open
  by design**: publishing the GitHub release (tag `v0.1.0`) now also triggers
  the publish workflow below.
- ~~No publish workflow~~ Added `.github/workflows/publish.yml` using PyPI
  Trusted Publishing, triggered by publishing a GitHub release. One-time
  manual setup required: register the trusted publisher on PyPI (owner
  `mbagalman`, repo `CorrSleuth`, workflow `publish.yml`, environment `pypi`;
  use "pending publishers" since the project doesn't exist yet) and create
  the `pypi` environment in the GitHub repo settings.

#### 10. CI breadth — **FIXED 2026-06-10**

CI is ubuntu-only. The package is pure Python so risk is low, but a single
Windows or macOS cell would catch platform issues (this review was conducted on
Windows without problems). Adding a `python -m build && twine check dist/*` step
would have caught issue 4.

> Resolution: CI matrix now covers Python 3.10–3.14 (lite + standard on
> ubuntu), one Windows cell, one macOS cell, and one real-statsmodels cell.
> A new `build` job runs `python -m build`, `twine check`, and fails if the
> sdist contains test files (regression guard for issue 4).

#### 11. Minor documentation drift — **FIXED 2026-06-10**

README's description of `report.summary()` omits the "Other or inconclusive"
section that the code emits for `low_power_or_uncertain` / `mixed_or_ambiguous` /
`not_computable` patterns.

> Resolution: README now documents the `Other or inconclusive` section.

## Edge cases tested and found healthy

- Nullable dtypes (`Int64`/`Float64` with `pd.NA`) profile correctly.
- Bool columns are treated as numeric and work.
- Datetime columns are correctly excluded from default scans.
- Constant target → all entries `not_computable`, no crash.
- `bootstrap=True` (bool) correctly rejected; bootstrap validation is thorough.
- Post-profile mutation of the source DataFrame does not affect `plot()`.
- `plot_top()` with empty filter returns a placeholder figure as documented.
- Markdown escaping handles pipes/underscores/backticks.
