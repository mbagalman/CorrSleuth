# Idea Summary: CorrSleuth

## 1. Core Concept
[cite_start]**CorrSleuth** is a relationship diagnosis engine designed for pandas users to bridge the gap between raw statistical coefficients and actionable analytical insights. [cite: 6] [cite_start]Instead of providing a single, potentially misleading correlation number, it functions as a "diagnostic panel" that compares multiple association measures—such as Pearson, Spearman, and Distance Correlation—to identify hidden non-linear patterns and "hidden gems" in data. [cite: 6] [cite_start]The product's value lies in its interpretive layer, translating complex statistical disagreements into plain-English warnings and recommended next steps for data scientists. [cite: 6]

## 2. Problem Space
* [cite_start]**Pain Point:** The "Pearson Trap" [cite: 5][cite_start]: Analysts often rely on a single correlation matrix that can report near-zero values for high-value non-linear relationships (e.g., U-shapes or saturation curves), leading them to prematurely discard predictive features. [cite: 5, 6]
* [cite_start]**Opportunity:** As machine learning workflows shift toward high-capacity non-linear models like XGBoost and Random Forests, the value of identifying non-linear signals—which standard Pearson tests "gaslight" the analyst into ignoring—has become a critical competitive advantage in feature engineering. [cite: 6]

## 3. Target Audience
* [cite_start]**Data Scientists & Feature Engineering Teams:** Professionals who need to mitigate the risk of missing non-linear signals during the exploratory data analysis (EDA) phase. [cite: 6]
* [cite_start]**Analysts Transitioning to Python:** Former users of SAS, SPSS, or R who expect high-level "procedural" summaries and interpretative guidance rather than fragmented, DIY metric calculation. [cite: 6]

## 4. Primary Capabilities
* [cite_start]**`profile_pair` Diagnosis:** A core function that computes a suite of dependence measures (Pearson, Spearman, Kendall, Distance Correlation, Mutual Information) and outputs a structured diagnostic label (e.g., `monotonic_nonlinear`). [cite: 6]
* [cite_start]**Automated Interpretation (`explain`):** Generates natural-language summaries detailing why specific metrics disagree and what that implies about the underlying data structure. [cite: 6]
* [cite_start]**Relationship Simulator (`make_relationship`):** A utility for generating 14+ distinct relationship types (e.g., sinusoidal, heteroscedastic) to help users validate their diagnostic intuition and test model sensitivity. [cite: 6]

## 5. Differentiation
* [cite_start]**Interpretation over Calculation:** Unlike existing Python libraries (SciPy, dCor) that focus on the "what" (the number), CorrSleuth focuses on the "why" (the pattern). [cite: 6]
* [cite_start]**Metric Disagreement Engine:** It explicitly highlights where the correlation matrix is misleading, using the delta between linear and non-linear measures as a heuristic for discovery. [cite: 6]
* [cite_start]**Safety-First Bias:** The tool is designed to be "cautiously helpful," using language like "evidence consistent with" to prevent analysts from over-claiming causality or truth. [cite: 6]

## 6. MVP Scope (Feasibility)
* [cite_start]**Core:** Numeric-vs-numeric pairwise profiling, five core metrics (Pearson, Spearman, Kendall, Distance Correlation, Mutual Information), rule-based diagnostic labels, and basic scatter/rank visual diagnostics. [cite: 6, 7]
* [cite_start]**Omitted:** Categorical/mixed-type support, automated HTML reports, multi-core target scanning, and Scikit-Learn transformers (reserved for v0.2+). [cite: 6, 7]
* [cite_start]**Complexity Drivers:** High-level performance challenges related to the computational cost of Distance Correlation and Mutual Information on large datasets, necessitating future subsampling or "Lite" modes. [cite: 6]
