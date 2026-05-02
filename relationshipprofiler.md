# Correlation Profiler: updated product description

## Working package name

For now, I’ll use:

```text
CorrSleuth
```

The best positioning is not “correlation package.” Python has plenty of those.

The positioning should be:

> **A relationship diagnosis engine for pandas users.**

Or:

> **Correlation is not one number. It is a diagnostic panel.**

## Core idea

Most Python tools answer:

> “What is the correlation between these variables?”

`relationprofiler` would answer:

> “What kind of relationship appears to exist between these variables, and which statistical measures are agreeing, disagreeing, or warning us that the usual correlation matrix is hiding something?”

The package would compute several association/dependence measures, compare their pattern of agreement and disagreement, and translate that pattern into:

* diagnostic labels,
* plain-English explanations,
* confidence/stability notes,
* warnings,
* recommended plots,
* next analytical steps.

The point is not to replace `pandas`, `scipy`, `dcor`, `hyppo`, `phik`, `pingouin`, or EDA tools. The point is to sit **above** them.

Those packages mostly provide numbers.

`relationprofiler` would provide interpretation.

## The gap in the Python ecosystem

Existing tools do important pieces of this, but none seem to own the full interpretive workflow.

### Existing tools mostly stop here

```text
Pearson = 0.12
Spearman = 0.71
Kendall = 0.52
Distance correlation = 0.69
```

That is useful, but it still leaves the analyst asking:

> “Fine. What does that mean?”

The missing layer is:

```text
Pattern: monotonic nonlinear

Evidence:
  Pearson is weak, but rank-based and nonlinear dependence measures are strong.

Interpretation:
  X and Y appear to move together directionally, but not in a straight-line way.
  Pearson likely understates the relationship.

Recommended next steps:
  Inspect smoother plot.
  Consider log/spline/piecewise transformation.
  Check whether the pattern holds within important segments.
```

That is the open space.

The package should explicitly position itself as **not another coefficient calculator** and **not another EDA report generator**.

It is a tool for identifying where the correlation matrix is misleading.

## Target users

### Primary users

1. **Data analysts and data scientists doing exploratory data analysis**
   They want to know which relationships deserve further investigation before modeling.

2. **SAS/SPSS/R users moving into Python**
   They are used to high-level procedures that produce analytical summaries. Python gives them powerful ingredients, but not always a satisfying “procedure-like” workflow.

3. **Feature-engineering teams**
   They need to know which variables are related to a target, which relationships are nonlinear, and which variables are redundant.

4. **Teachers, writers, and explainers**
   This package could be excellent for teaching why Pearson correlation is not “correlation,” full stop.

### Secondary users

* Marketing analytics teams
* Survey researchers
* Product analysts
* Risk analysts
* Bioinformatics/social science analysts
* Business analysts who know enough statistics to be dangerous, which is the most common kind of dangerous

## Product promise

A good README statement:

```markdown
# relationprofiler

Correlation is not one number.

`relationprofiler` is a Python toolkit for profiling relationships among variables using multiple association and dependence measures. It compares Pearson, Spearman, Kendall tau-b, distance correlation, mutual information, and related statistics to help identify linear, monotonic, nonlinear, nonmonotonic, outlier-sensitive, and weak relationships.

Most tools give you a correlation matrix.

`relationprofiler` tells you where the correlation matrix may be misleading.
```

A more memorable hook:

```text
Pearson says nothing is happening.
Spearman says something is happening.
Distance correlation is screaming from the basement.

relationprofiler helps you figure out which one to believe.
```

## MVP principle

The biggest risk is scope creep.

The full vision is compelling:

* target scans,
* HTML reports,
* bootstrap stability,
* feature screening,
* variable clustering,
* sklearn selectors,
* mixed-type support,
* gallery plots,
* simulation-based classifiers.

But the first version should be brutally focused.

## Version 0.1 should do one thing extremely well

### v0.1 core feature

```python
from relationprofiler import profile_pair

result = profile_pair(df, x="discount_depth", y="sales")

result.summary()
result.explain()
result.plot()
```

The first release should focus on **numeric-vs-numeric pairwise profiling** only.

That is enough.

Do not start with categorical variables. Do not start with full automated reports. Do not start with an sklearn transformer. Do not build the Death Star when what users need first is a flashlight.

## v0.1 metrics

The v0.1 metric set should be small and defensible.

### Required metrics

```text
Pearson
Spearman
Kendall tau-b
Distance correlation
Mutual information
```

### Optional / later metrics

```text
Hoeffding’s D
HSIC
MIC / MINE
Robust correlations
Biweight midcorrelation
Partial correlations
Cramér’s V
Phik
Correlation ratio / eta-squared
```

The MVP should not depend on every advanced package. Keep the base install lightweight.

## Dependency strategy

Base install:

```text
numpy
pandas
scipy
matplotlib
scikit-learn
```

Optional extras:

```bash
pip install relationprofiler[dcor]
pip install relationprofiler[plots]
pip install relationprofiler[advanced]
pip install relationprofiler[reports]
pip install relationprofiler[all]
```

Possible optional dependencies:

```text
dcor          # distance correlation
statsmodels   # LOWESS smoother
hyppo         # independence tests
phik          # mixed-type association later
dython        # categorical/mixed-type association later
jinja2        # HTML reports later
joblib        # parallelization
```

This matters because install friction kills adoption. A user should be able to run:

```bash
pip install relationprofiler
```

and get the core experience without dragging in half of PyPI and a small weather system.

## v0.1 API

### Pairwise profiling

```python
from relationprofiler import profile_pair

result = profile_pair(
    data=df,
    x="marketing_spend",
    y="sales",
    methods="default",
    missing="pairwise",
    plots=True
)
```

### Example output

```text
Relationship profile: marketing_spend vs sales

Primary pattern:
  Evidence consistent with a monotonic nonlinear relationship

Metrics:
  Pearson:              0.42
  Spearman:             0.78
  Kendall tau-b:         0.59
  Distance correlation:  0.74
  Mutual information:    0.31

Interpretation:
  marketing_spend and sales appear to have a strong directional relationship,
  but Pearson is substantially weaker than the rank-based and nonlinear
  dependence measures. This suggests the relationship may be monotonic but
  not well summarized by a straight line.

Warnings:
  - Pearson may understate this relationship.
  - Do not interpret this association causally without design or controls.
  - Check whether this pattern holds within important segments.

Recommended next steps:
  - Inspect the smoother plot.
  - Consider log, spline, or piecewise transformations.
  - Check outliers and leverage points.
```

Notice the phrase:

```text
Evidence consistent with...
```

That should be everywhere.

Not:

```text
This is definitely nonlinear.
```

Not:

```text
This variable causes sales.
```

Not:

```text
Congratulations, you have discovered truth.
```

The package should be useful, not drunk.

## Core diagnostic labels

The v0.1 classifier can be rule-based. That is fine. It is interpretable, testable, and easy to document.

Suggested labels:

```text
near_linear
monotonic_nonlinear
possible_outlier_or_leverage
nonmonotonic_dependence
weak_or_no_pairwise_relationship
rank_instability_or_ties
low_power_or_uncertain
```

### Label: `near_linear`

Typical pattern:

```text
Pearson high
Spearman high
Kendall moderate/high
Distance correlation high
Pearson and Spearman close
```

Interpretation:

```text
Evidence consistent with an approximately linear or near-linear relationship.
```

### Label: `monotonic_nonlinear`

Typical pattern:

```text
Spearman high
Kendall moderate/high
Distance correlation high
Pearson meaningfully lower
```

Interpretation:

```text
Evidence consistent with a directional relationship that is not well summarized by a straight line.
```

Common examples:

```text
log relationship
exponential relationship
saturation effect
diminishing returns
S-curve
```

### Label: `possible_outlier_or_leverage`

Typical pattern:

```text
Pearson high
Spearman/Kendall much lower
Pearson changes materially after robust/winsorized sensitivity check
```

Interpretation:

```text
The apparent linear association may be driven by outliers or high-leverage observations.
```

### Label: `nonmonotonic_dependence`

Typical pattern:

```text
Pearson low
Spearman low
Kendall low
Distance correlation moderate/high
Mutual information moderate/high
```

Interpretation:

```text
Evidence consistent with a relationship that is not simply increasing or decreasing.
```

Common examples:

```text
U-shaped
inverted U-shaped
circular
periodic
clustered
threshold-based
```

### Label: `rank_instability_or_ties`

Typical pattern:

```text
Spearman and Kendall differ unusually
High tie rate
Low unique-value ratio
```

Interpretation:

```text
Rank-based measures may be affected by ties, discreteness, or ordinal compression.
```

### Label: `weak_or_no_pairwise_relationship`

Typical pattern:

```text
All measures low
Adequate sample size
No obvious instability
```

Interpretation:

```text
Little evidence of pairwise association in the observed data.
```

Important caveat:

```text
This does not rule out interactions, conditional effects, segment-specific effects,
lagged effects, causal effects, or relationships hidden by measurement problems.
```

### Label: `low_power_or_uncertain`

Typical pattern:

```text
Small n
Wide bootstrap intervals
Unstable metric estimates
Conflicting diagnostics
```

Interpretation:

```text
The evidence is too unstable to confidently describe the relationship shape.
```

## Pattern classification rules

Internally, v0.1 might use simple thresholds.

For example:

```python
r_p = abs(pearson)
r_s = abs(spearman)
r_k = abs(kendall)
r_d = distance_corr

diff_rank_linear = r_s - r_p
diff_dcor_rank = r_d - r_s
```

Possible starting rules:

```text
near_linear:
  r_p >= 0.5
  r_s >= 0.5
  abs(r_p - r_s) <= 0.15

monotonic_nonlinear:
  r_s >= 0.5
  r_d >= 0.5
  r_s - r_p >= 0.20

possible_outlier_or_leverage:
  r_p >= 0.5
  r_p - r_s >= 0.20

nonmonotonic_dependence:
  r_p < 0.25
  r_s < 0.25
  r_d >= 0.35

weak_or_no_pairwise_relationship:
  r_p < 0.20
  r_s < 0.20
  r_d < 0.20
```

These are not sacred. They are a starting point. The docs should call them heuristics, not commandments delivered on stone tablets.

## Simulated datasets should be in v0.1

The feedback is right: the simulation generator should not be postponed. It should be part of the first release because it supports:

* documentation,
* examples,
* testing,
* validation of rules,
* teaching,
* blog posts,
* user trust.

API:

```python
from relationprofiler.datasets import make_relationship

df = make_relationship("u_shape", n=500, noise=0.2, random_state=42)

result = profile_pair(df, x="x", y="y")
result.summary()
result.plot()
```

Initial relationship generators:

```text
linear_positive
linear_negative
monotonic_log
monotonic_exp
s_curve
u_shape
inverted_u
sinusoidal
circle
threshold
clustered
heteroscedastic
outlier_driven
independent
```

This is also how you can test whether your diagnostic rules behave sensibly.

Example test:

```python
def test_u_shape_flagged_as_nonmonotonic():
    df = make_relationship("u_shape", n=1000, noise=0.1, random_state=1)
    result = profile_pair(df, "x", "y")
    assert result.pattern == "nonmonotonic_dependence"
```

## v0.1 visual outputs

The first plot should be excellent.

### `result.plot()`

For v0.1, create a compact diagnostic figure with:

1. scatterplot,
2. smoother line if optional smoother dependency is installed,
3. rank-rank plot,
4. metric summary,
5. diagnostic label and warnings.

Possible title:

```text
discount_depth vs sales
Evidence consistent with: monotonic nonlinear
Pearson = 0.42 | Spearman = 0.78 | dCor = 0.74
```

For large n, the package should use alpha blending or hexbin-style plotting.

```python
result.plot(kind="scatter")
result.plot(kind="rank")
result.plot(kind="diagnostic")
```

Do not overcomplicate v0.1 with a 12-panel dashboard. The plot should answer:

> “Why did the package label the relationship this way?”

## v0.1 object design

The result object should be friendly and inspectable.

```python
result.metrics
```

Returns a DataFrame:

| metric        | value | p_value | n_used | notes                              |
| ------------- | ----: | ------: | -----: | ---------------------------------- |
| pearson       |  0.42 |   0.003 |    500 | linear association                 |
| spearman      |  0.78 |  <0.001 |    500 | rank monotonic association         |
| kendall_tau_b |  0.59 |  <0.001 |    500 | concordance-based rank association |
| distance_corr |  0.74 |      NA |    500 | nonlinear dependence               |
| mutual_info   |  0.31 |      NA |    500 | general dependence estimate        |

```python
result.pattern
```

```text
monotonic_nonlinear
```

```python
result.warnings
```

```python
[
    "Pearson is substantially weaker than rank-based measures.",
    "Relationship may be monotonic but nonlinear.",
    "Association is not causation."
]
```

```python
result.recommendations
```

```python
[
    "Inspect smoother plot.",
    "Consider nonlinear transformations.",
    "Check whether the relationship holds within important segments."
]
```

```python
result.explain()
```

Returns a readable paragraph.

```python
result.to_dict()
result.to_frame()
```

For integration.

## v0.2: target-oriented scan

This should probably be the next release, because it will be the killer practical workflow.

```python
from relationprofiler import scan_target

report = scan_target(
    df,
    target="sales",
    methods="default",
    mode="lite"
)

report.summary()
report.to_frame()
report.plot_top(n=12)
```

Output sections:

```text
Strongest near-linear relationships
Strongest monotonic nonlinear relationships
Potential nonmonotonic relationships
Variables Pearson may underrate
Possible outlier-driven relationships
Variables with tie/discreteness warnings
Variables with insufficient data
```

This is what users will want in real EDA:

> “Here are the variables related to my target, and here’s how they seem to be related.”

## v0.2 performance modes

Performance needs to be designed early.

For many columns, full metric calculation can become expensive. Distance correlation, mutual information, bootstrap, and permutation tests can hurt.

Modes:

```python
scan_target(df, target="sales", mode="lite")
scan_target(df, target="sales", mode="standard")
scan_target(df, target="sales", mode="deep")
```

Possible behavior:

### Lite mode

```text
Pearson
Spearman
Kendall
basic warnings
no bootstrap
no permutation
optional sample cap
```

### Standard mode

```text
Pearson
Spearman
Kendall
distance correlation
mutual information
basic plot recommendations
```

### Deep mode

```text
All standard metrics
bootstrap intervals
permutation tests where applicable
outlier sensitivity
stability score
```

Performance options:

```python
scan_target(
    df,
    target="sales",
    n_jobs=-1,
    sample_size=10000,
    cache=True
)
```

The documentation should be blunt:

```text
Use profile_pair for careful inspection.
Use scan_target for practical feature screening.
Use scan_pairs carefully on wide data.
```

## v0.3: bootstrap stability

Bootstrap stability is a major differentiator, but it can wait until after v0.1 if needed.

API:

```python
result = profile_pair(
    df,
    x="discount_depth",
    y="sales",
    bootstrap=500
)
```

Output:

```text
Pearson:              0.42 [0.31, 0.53]
Spearman:             0.78 [0.71, 0.83]
Distance correlation: 0.74 [0.68, 0.80]

Pattern stability:
  monotonic_nonlinear in 91% of bootstrap samples
```

That last line is powerful. It turns classification from a brittle label into a stability-aware diagnostic.

## v0.3/v0.4: HTML reports

HTML reports are not v0.1. But they are a strong adoption feature later.

```python
report.to_html("relationship_profile.html")
```

Report sections:

```text
Dataset overview
Missingness warnings
Strongest relationships
Relationships Pearson may miss
Likely monotonic nonlinear relationships
Likely nonmonotonic relationships
Possible outlier-driven relationships
Recommended plots
Method appendix
```

This should not try to compete with ydata-profiling or Sweetviz. It should be narrower and sharper:

> “Here are the relationship patterns you should inspect.”

## Missing data policy

The package should be explicit and annoying in the right way.

```python
profile_pair(
    df,
    x="income",
    y="spend",
    missing="pairwise"  # "pairwise", "listwise", "raise"
)
```

Warnings:

```text
23% of rows were dropped because of missing values.
Results may be biased if missingness is related to x or y.
```

Do not silently make missingness disappear like a consulting deck with bad news.

## Outlier sensitivity

Even in v0.1, basic outlier sensitivity should be considered, or at least planned.

At minimum:

```text
Pearson much stronger than Spearman/Kendall → warn about possible leverage/outliers.
```

Later:

```python
profile_pair(..., outlier_check=True)
```

Could compute:

* winsorized Pearson,
* Pearson after trimming extreme x/y quantiles,
* robust correlation,
* influence diagnostics.

Output:

```text
Pearson drops from 0.81 to 0.29 after trimming extreme 1% values.
The apparent linear relationship may be leverage-sensitive.
```

## Ties and discreteness

Rank-based methods can behave differently with many ties.

The package should compute:

```text
unique_x_ratio
unique_y_ratio
tie_rate_x
tie_rate_y
```

Warning:

```text
Variable x has only 5 unique values across 10,000 rows.
Rank-based measures may be affected by ties or ordinal compression.
Consider categorical/ordinal association methods.
```

## Mixed-type variables

Do **not** include mixed-type support in v0.1.

Later, support:

```text
numeric-numeric
numeric-categorical
categorical-categorical
ordinal-numeric
ordinal-ordinal
```

But for mixed types, avoid reinventing everything. Consider optional integrations with:

```text
phik
dython
scipy.stats
sklearn.feature_selection
```

Possible later API:

```python
profile_pair(df, "region", "sales")
```

Output:

```text
Relationship type: categorical-numeric
Suggested measures: eta-squared, mutual information, Kruskal-Wallis
```

But later. Numeric-only first.

## Full pairwise scan

This is useful but potentially expensive.

```python
from relationprofiler import scan_pairs

pairs = scan_pairs(
    df,
    columns=None,
    mode="lite",
    max_pairs=5000,
    n_jobs=-1
)
```

Outputs:

```python
pairs.to_frame()
pairs.plot_disagreement_heatmap()
pairs.plot_metric_heatmap(metric="spearman")
pairs.plot_metric_heatmap(metric="distance_corr")
```

The best plot here is the **disagreement heatmap**.

Example disagreement score:

```python
disagreement_score = (
    abs(abs(spearman) - abs(pearson))
    + max(0, distance_corr - abs(spearman))
)
```

Interpretation:

```text
High disagreement means the relationship may not be well summarized by ordinary Pearson correlation.
```

This score should be presented as a heuristic.

Again: useful, not divinely revealed.

## Feature-screening mode

This is probably the package’s long-term practical sweet spot.

```python
from relationprofiler import screen_features

screen = screen_features(
    df,
    target="churn",
    mode="standard"
)
```

Output:

```text
Recommended candidates for linear models:
  tenure, income, monthly_spend

Recommended candidates for nonlinear modeling:
  age, visit_frequency, discount_depth

Variables Pearson may underrate:
  loyalty_score, usage_frequency

Potentially redundant variables:
  impressions, reach, ad_views

Variables requiring caution:
  zip_income_proxy: high missingness
  campaign_count: outlier-sensitive
  satisfaction_score: many ties
```

This starts moving beyond diagnostics into modeling workflow support.

## Variable clustering mode

This could eventually connect back to your `PROC VARCLUS` question.

Instead of clustering variables only by Pearson correlation, cluster them by richer dependence profiles.

```python
from relationprofiler import cluster_variables

clusters = cluster_variables(
    df,
    similarity="profile",
    methods=["pearson", "spearman", "distance_corr"]
)
```

Output:

```text
Cluster 1:
  variables: impressions, reach, ad_views
  pattern: near-linear redundancy
  representative: reach

Cluster 2:
  variables: tenure, loyalty_score, repeat_purchase_rate
  pattern: monotonic nonlinear redundancy
  representative: loyalty_score
```

This should not be in v0.1, but it is a natural expansion.

## sklearn-style selector

Later:

```python
from relationprofiler import RelationshipSelector

selector = RelationshipSelector(
    target="sales",
    include_patterns=[
        "near_linear",
        "monotonic_nonlinear",
        "nonmonotonic_dependence"
    ],
    remove_redundant=True
)

X_selected = selector.fit_transform(X, y)
```

This could be useful, but it should come after the diagnostic layer is solid.

## Proposed package architecture

```text
relationprofiler/
  __init__.py

  pair.py
  scan.py

  metrics/
    __init__.py
    pearson.py
    spearman.py
    kendall.py
    distance.py
    mutual_info.py

  diagnostics/
    __init__.py
    classify.py
    disagreement.py
    warnings.py
    missingness.py
    ties.py
    outliers.py
    bootstrap.py

  plotting/
    __init__.py
    pairplot.py
    rankplot.py
    heatmap.py
    gallery.py

  datasets/
    __init__.py
    simulations.py

  reports/
    __init__.py
    html.py
    markdown.py

  sklearn/
    __init__.py
    selector.py

  utils/
    validation.py
    types.py
```

For v0.1, only some of that needs to exist:

```text
pair.py
metrics/
diagnostics/classify.py
diagnostics/warnings.py
plotting/pairplot.py
datasets/simulations.py
utils/validation.py
```

## Testing strategy

The simulation datasets make testing straightforward.

Examples:

```python
def test_linear_positive_near_linear():
    df = make_relationship("linear_positive", n=1000, noise=0.1, random_state=42)
    result = profile_pair(df, "x", "y")
    assert result.pattern == "near_linear"


def test_u_shape_nonmonotonic():
    df = make_relationship("u_shape", n=1000, noise=0.1, random_state=42)
    result = profile_pair(df, "x", "y")
    assert result.pattern == "nonmonotonic_dependence"


def test_outlier_driven_warning():
    df = make_relationship("outlier_driven", n=500, random_state=42)
    result = profile_pair(df, "x", "y")
    assert "outlier" in " ".join(result.warnings).lower()
```

Also test:

```text
missing values
constant variables
small sample size
many ties
all-null columns
infinite values
non-numeric columns in v0.1
```

The package should fail gracefully.

Bad:

```text
ValueError: array must not contain infs or NaNs
```

Better:

```text
Variable 'income' contains infinite values. Replace or remove them before profiling.
```

## Documentation plan

The docs should be example-driven.

Suggested pages:

```text
Quickstart
What relationprofiler does
What relationprofiler does not do
Understanding the diagnostic labels
When Pearson lies
Monotonic vs nonmonotonic relationships
Outlier-sensitive correlations
Using simulated datasets
Performance modes
Missing data and ties
API reference
```

The examples should use the simulation generator heavily.

Example docs section:

```python
from relationprofiler.datasets import make_relationship
from relationprofiler import profile_pair

df = make_relationship("u_shape", n=500, noise=0.15, random_state=42)
result = profile_pair(df, "x", "y")

result.summary()
result.plot()
```

Then show:

```text
Pearson near zero.
Spearman near zero.
Distance correlation high.
Pattern: evidence consistent with nonmonotonic dependence.
```

That is instantly understandable.

## Adoption strategy

This package has unusually good content potential.

Possible launch articles:

```text
Correlation Is Not One Number
When Pearson Lies
Five Ways Your Correlation Matrix Is Gaslighting You
How to Find U-Shaped Relationships Before Your Model Embarrasses You
The Analyst’s Guide to Pearson, Spearman, Kendall, and Distance Correlation
```

GitHub README should show one visual example immediately.

The first screen should not be a wall of math. It should be:

1. one sentence,
2. one code snippet,
3. one diagnostic output,
4. one plot.

## Suggested README opening

```markdown
# relationprofiler

Correlation is not one number.

`relationprofiler` profiles pairwise relationships in pandas DataFrames by comparing multiple association and dependence measures — Pearson, Spearman, Kendall tau-b, distance correlation, and mutual information — and translating their agreement or disagreement into practical diagnostics.

Most tools give you this:

| x | y | correlation |
|---|---:|---:|
| discount_depth | sales | 0.42 |

`relationprofiler` gives you this:

**Pattern:** evidence consistent with monotonic nonlinear association

Pearson is moderate, but Spearman, Kendall, and distance correlation are strong. This suggests the relationship is directional but not well summarized by a straight line.

**Recommended next steps:**
- Inspect the smoother plot
- Consider nonlinear transformations
- Check whether the pattern holds within segments
```

## One-page product definition

```text
relationprofiler is a Python package for diagnosing pairwise relationships among numeric variables. It computes multiple association and dependence measures, compares their agreement and disagreement, assigns cautious diagnostic labels, generates explanatory narratives, and recommends plots or next analytical checks.

It is designed for pandas users who want more than a correlation matrix but less than a full custom statistical investigation for every pair of variables.
```

## Revised roadmap

### v0.1 — Pairwise numeric profiler

Must have:

```text
profile_pair()
Pearson
Spearman
Kendall tau-b
Distance correlation if available
Mutual information
Rule-based diagnostic labels
Warnings for missingness/ties/small n
Scatter/rank diagnostic plot
explain()
summary()
make_relationship()
```

Nice to have:

```text
LOWESS smoother
basic outlier warning
clean result DataFrame
```

Do not include:

```text
HTML reports
full pairwise scan
feature selection
categorical support
sklearn selector
variable clustering
```

### v0.2 — Target scan

```text
scan_target()
lite/standard modes
top relationships by pattern
variables Pearson may underrate
basic gallery plots
disagreement score
```

### v0.3 — Stability and performance

```text
bootstrap confidence intervals
pattern stability
n_jobs
subsampling
caching
deep mode
```

### v0.4 — Reporting

```text
HTML reports
Markdown reports
plot galleries
method appendix
```

### v0.5 — Feature screening / redundancy

```text
screen_features()
redundancy detection
representative variable suggestions
possible VARCLUS-style expansion
```

### v1.0 — Broader relationship intelligence

```text
mixed-type support
optional phik/dython integrations
sklearn transformer
simulation-calibrated classifier
variable clustering
```

## The key design philosophy

The package should always be:

```text
cautious
interpretable
pandas-native
visual
educational
practical
```

It should never be:

```text
overconfident
causal-sounding
black-boxy
dependency-bloated
trying to replace full EDA tools
```

## Final updated verdict

This is a strong GitHub/PyPI idea.

The review strengthens the case, but it also sharpens the execution strategy: **ship the pairwise profiler first**. The winning wedge is not a giant automated EDA system. It is one excellent, trustworthy function:

```python
profile_pair(df, "x", "y")
```

That function should give the analyst what they actually need:

```text
Here are the measures.
Here is how they agree or disagree.
Here is what that pattern may suggest.
Here are the warnings.
Here is the plot you should inspect.
Here is what not to overclaim.
```

That alone would be useful.

Everything else can grow from there.
