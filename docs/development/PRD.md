## **PRD: CorrSleuth**

### **1\. Overview**

* **Purpose:** To serve as a relationship diagnosis engine for pandas users that interprets statistical associations using a "diagnostic panel" approach.  
* **Scope:** v0.1 covers numeric-vs-numeric pairwise profiling, heuristic pattern labeling, a relationship simulator, and basic visual diagnostics.  
* **Problem:** Standard correlation matrices can obscure nonlinear or nonmonotonic relationships, potentially misleading analysts into overlooking high-value features.  
* **Target Users:** Data Scientists, Feature Engineering teams, and Analysts transitioning from SAS/R/SPSS to Python.

### **2\. Goals**

#### **2.1 Business goals**

* Reach 100+ GitHub stars within 3 months of launch.  
* Achieve ≥ 500 monthly PyPI downloads within 3 months.

#### **2.2 User goals**

* Quickly identify where a standard correlation matrix is incomplete or potentially misleading.  
* Standardize the interpretation of complex dependence measures into actionable insights.

#### **2.3 Product principles**

1. **Diagnostic, not causal**: Identify patterns, never claim causation.  
2. **Cautious language**: Use "evidence consistent with" instead of "this is."  
3. **Explain the disagreement**: The value is in *why* metrics differ, not the raw numbers.  
4. **Useful warnings over silent failure**: Flag missingness, ties, and small samples explicitly.  
5. **Lightweight first**: Minimize core dependencies; use optional extras for heavy computations.

### **3\. User personas**

#### **3.1 Key user types**

* The Exploratory Analyst (EDA)  
* The SAS/R Migrator  
* The Feature Engineering Specialist

#### **3.2 Persona descriptions**

* **The SAS/R Migrator**: Accustomed to high-level "procedures" that provide summary interpretations and diagnostic checks in a single call.

#### **3.3 Role-based access**

* **Standard User**: Local execution via Python environment.

### **4\. Functional requirements**

#### **Diagnostic labeling & heuristics (Priority: High)**

* **FR-001**: CorrSleuth must assign one primary diagnostic label using absolute metric strengths: ![][image1], ![][image2], ![][image3], ![][image4]. Rules are applied in the following priority order:

| Priority | Label | Trigger Condition (Provisional) |
| :---- | :---- | :---- |
| 1 | not\_computable | All-null, nonnumeric, infinite, or constant (std=0) inputs. |
| 2 | low\_power\_or\_uncertain | ![][image5] or required metrics unavailable. |
| 3 | possible\_outlier\_or\_leverage | ![][image6] AND (![][image7] OR ![][image8]). |
| 4 | nonmonotonic\_dependence | ![][image9] AND ![][image10] AND ![][image11] (if available). |
| 5 | monotonic\_nonlinear | ![][image12] AND (![][image13]) AND ![][image11] (if available). |
| 6 | near\_linear | ![][image6] AND ![][image12] AND $ |
| 7 | weak\_or\_no\_relationship | ![][image9] AND ![][image10] AND (![][image14] unavailable OR ![][image15]). |
| 8 | mixed\_or\_ambiguous | Fallback for cases not meeting above thresholds. |

* **FR-001.1**: Calculate a disagreement\_score attribute: abs(p \- s) \+ max(0, dc \- s).

#### **Out-of-scope for v0.1**

* Target-oriented scans (scan\_target).  
* Full pairwise matrix / disagreement heatmaps.  
* Bootstrap stability and permutation tests.  
* Categorical or mixed-type support.  
* HTML reports and Scikit-Learn transformers.

#### **Result object structure (Priority: High)**

* **FR-002**: profile\_pair() returns a CorrSleuthResult object containing:  
  * **Attributes**: x\_name, y\_name, metrics (DataFrame), pattern, warnings, recommendations, disagreement\_score.  
  * **Methods**:  
    * .summary(): Tabular view of metrics and primary label.  
    * .explain(): 2-3 sentence narrative explaining metric disagreement and pattern evidence.  
    * .plot(): Multi-panel scatter \+ rank plot with pattern annotation.

#### **Performance & dependencies (Priority: Medium)**

* **FR-003**: **Core Dependencies**: pandas, numpy, scipy, matplotlib.  
* **FR-004**: **Optional Extras**: dcor (Distance Correlation), scikit-learn (Mutual Info), statsmodels (LOWESS).  
* **FR-005**: If mode="standard" is requested without dcor or scikit-learn, raise OptionalDependencyError.

### **5\. User stories**

#### **Story: Detecting U-Shapes**

* **ID**: US-001 | **Acceptance Criteria**: For a simulated U-shape, label is nonmonotonic\_dependence. .explain() highlights high ![][image14] vs low ![][image16].

#### **Story: Conflicting Directionality**

* **ID**: US-002 | **Acceptance Criteria**: If sign(P) \!= sign(S) AND both ![][image17], append a "Conflicting Directional Evidence" warning.

#### **Story: Simulation Consistency**

* **ID**: US-003 | **Acceptance Criteria**: make\_relationship(..., random\_state=42) produces identical DataFrames. Every simulated type must produce its expected label with random\_state=42.

### **6\. User experience**

#### **6.1 Entry points**

* pip install corrsleuth\[standard\]

#### **6.2 Core flow (API Signatures)**

\# API Signature  
result \= cs.profile\_pair(  
    data=df,   
    x="column\_a",   
    y="column\_b",   
    mode="standard", \# "lite", "standard"  
    missing="pairwise", \# "pairwise", "drop"  
    random\_state=42  
)

\# Output Signatures  
result.explain() \# Returns str  
result.plot()    \# Returns (fig, ax), calls plt.show()

#### **6.3 Edge cases**

* **Ties**: Unique ratio \< 0.05 triggers a warning about rank-metric instability.  
* **NaN-heavy**: Columns with \> 50% missing data after pairwise dropping trigger a low\_power\_or\_uncertain warning.  
* **Infinities**: Raise InputError specifically naming the offending variable.

#### **6.4 Visual requirements**

* Plots must function in both CLI (standard window) and Jupyter (inline) environments. Use plt.show() pattern.

### **7\. Success metrics**

#### **7.1 User/Technical metrics**

* **Performance**: mode="lite" \< 2 seconds for 100k rows on consumer hardware.  
* **Survey**: ≥ 80% of early beta users report the tool saved time in identifying non-linear patterns.  
* **Trust**: 100% of simulator relationship types must map to the correct heuristic label in automated tests.

### **8\. Implementation requirements**

#### **8.1 Packaging**

* pyproject.toml; MIT License; Python 3.10+ support; GitHub Actions CI.

#### **8.2 Documentation**

* README quickstart; API reference; Worked examples for 5 key patterns (Linear, Log, U-Shape, S-Curve, Outlier).  
* **Standard Footer**: All .explain() and .summary() outputs must contain a non-causal disclaimer.

### **9\. Risks and assumptions**

* **Risk**: Users treat diagnostics as proof of causality.  
* **Mitigation**: Standardized footer disclaimer and cautious narrative generation.

### **10\. Milestones**

* **Phase 1A**: Core metrics, deterministic make\_relationship, and baseline test suite.  
* **Phase 1B**: Priority-based heuristic engine & narrative generation logic.  
* **Phase 1C**: Visual diagnostic panels and PyPI/GitHub release readiness.

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAHgAAAAYCAYAAAAxkDmIAAAEFUlEQVR4Xu2YWahOURTHFzJnyJDMUYSQmWR4IDwgImTqCi8iypQXN5kzPHgwPQjxhMxDSV3KUFLkxVSSEJklMq//3Xvdu876zjnf+Urfvd+951f/vrP/e53znbPX2Wevc4hSUlJSUlJyprM1UiKpYY1C4JQ1FK1ZHVmdvLDdltVSB1Uj+lABTojT1lD8jdEDVuPy0GpBXyrABJ+xhuE4uYRahpPz59mOKkw/KsAEn7WGQWZsGNLXzXZUUfpTASb4nDUMSOAha3rQ94lV33ZUUQZQggRPZT1kdfDt+azPrDVlEfnlvDUU7cglEQWWpZhcXxvbwYxlbSV3XRtMn7CD9YzcoIEuqm8w6zZrj2+vZD1n9S6LKGcuuWUE/UtNnzCO9YT1lLXL9ME/zGru29tY78idfzMJ8gykBAn+xTrJuk5uAD+QW8cwWE1VXL6IS/Bsin48f/OyoNLEPhi4maw3rDqBCJewt6xZrK+sTeTGRXjP2kfuOENZL1hFrPsqBjQhF3OHtZj1M9hdSg9yMWtZG8mdM84LTGat8P13WXNYN8idnxxXkzXBWKQHkZvFclABj8qwAdNgUKKEgXzNesV6SW5QknDBGgokCee5l7WbtZ912XtbVJwwnTJvCFwvBkyYQsEYKdYkeajMF/pt+B/VNhKoZxW8uqbdSLUneU9TS3n6Fxrj29rT4FpiEywVJ+6M76x6qq+EMg+YDy5aQ4Hz+WLNGBC/TrUnUHBmAsRg4AWZQdN82w5yL7+tEyegH49mzMwGFBzPq76/SHkCfMz+0aqNSaIJSzCWjtgEC9ixOMTLZTD/F5esocA5rbZmBMPIxa8idyNjpurZBTZT5qDhcWw9gBolzNfgiSKJgE6ovrAEgfEU9PG4RnuZ8lp574rywBDKIcEjQzysy/kmKsEorHBOuGuTEPZ4tpRQZkxUIrAehvkafDo8yHpEmcexbWEnBX25SfARQ1jiPVkqhEQJlsJAg0oUni1GLCgUclESohJ8hDLPM44RFB0v33BvUWaMToR+nGP9R10RBfbBfwo9vSeEJRhFFbxjykPbfs2DJ59w8ZbT0G8nSvB2cgeorTy0dSGST8ISjNeRsAHKBuLtO/FjcoMPJlLwmOhDG5U89lvg/RneX+Tblvbk+msqD0UqKmkBNY7+L6zPaKMA1cCTdV57Lfz2b+UnSjB2uEZuvf3B+kPl74IVgU4wBgGvLfh4gdc3nCNmVXcVEwcqXAwO9sXvzWB3KQfIjQGqYyToHrlYfNsWjnovDhwHMRg//MrNoZG3FVwTqnRU0Jaw/8GbBfbRyQWJEowDjrJmBRI2g1PCSZxglPSVhTTBycma4OUU/kioSNIEJydrgteT+zaL77CVhTTBycma4MpIV2ukRKKr9pTqxD/bSiDcmgu0mgAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAIcAAAAYCAYAAADQ1+6cAAAE7ElEQVR4Xu2Zach1UxTHl3komcocIoQMIcrwhTKkJEISIT4gQnyQobzGTJGMn95kLMrMByQyZc5YpiTz9Jrnl/W7ay9n3fWcc+59PrzXfT37V//u2f+z73nWPXudvdc+j0ilUqlUKv97tlctk83KImGjbEw7C1RHZzOxpuruonnFW1ossSrjw/1brPhO+pNjSdX3qtdUD6g+Ul2vukm1TuhXGc092Zh2GPi+5PhbtVzyLi9+ZXbcm41p5wfpTo43VXdms1CTY/bcl41p50fpTg4SgOWjjRuzURnJ/dnoYz3V1aovVVelc5PiJ+lPDnRCPhE4T/Vx+YRzxJaqZ/7t0bCEai/Vs6qnVesOnx6A94TYjLWDaoPh07KW6lPV/qX9uOoL1b7eQblC9ZI0Me2uell1s+qI4vF3iOMtsXHogiX0Q2li2SSc21H1uTSF+Upiddk7qs28U4BzY/ONWEF4uNgNXmP49EToS45fpUmQt1UHyMz6g99wjFifnVTvqk4Xu6HXhn5AQtDvGtXFqueGTw8gnrNUd5XjP6UpfJcSS7rjVQtVN4g9VIep/ih9gHtJHPwttulc41Cx6/lyyDHXYemkyF69+BF+x1di1/dYkPOt6sryuanY3zxV7Hcxrhs3XQeMnRwUJ+eGNkGfH9oZbsyLYsG2idmHLP5M9YnYDdp58M1+fpbu5ICvpUkQ1+blHDdi2XKMP78cA/H6QACDyVMYieeBPgyEs5sM93lSdUs5xj9QbPAZHO93gdgMdXvxmGWc04oXr7laac8PHhBL7OexsGsDZokjxRIgXxNoP5a8B1O7k9/EAmDvyx/IT+Sk+EX6k8PZUCxmfjSfcFH5XKX4EQYNb0XVdeXYWVVs2udpdp6SmdfwwXRYgldQHZV83rk4PMnQNmCPFI+EcPjteMcFz2PxpQs8loNLe8/ySe2Fv01pwx7F8/vjPJTanWwpzQ9Avw+fnhgsHTk5qBsOSp7D+p5v+mUt3onB45Np9iSxgd3WOwXo80po836FJeuN4Dnvic2SfbQlR5vHzMj1IjkW6IqFvvnl1utihT6JHHk4tXuZp3pU2oOeFG3JQaKenDyHqfKv5D0vNgNFWDZjcrQVqBH6+FMPJBAe9UkG/45sJujDgGYv32faF7Z4MRb3umI5pcXz5S8yVnKQrXk9ykG3cazYUz2umL5H0ZYcxHJG8mArsXNxumWXgBfrJ6ZePNZ94DhPsUBt4NAnFoW8TcRbW2wKZ4138LcI7Qx1Sx40aiM8rxlg+eI5fpxjcY9YIBal+MxyETxf6mLfsZKDL8f/S+yq+iC0J0lODmoE4os3DRhIvFjgwQvFvzV4tL0uAXYVeWbZWqyQdPjOPuV4/dL2GOIMwFY2x5ZhZ0WBHpOPHRDf2y947Eb8WiQ+tQLEWICtqfdjqWB3BocE3/EHA9h9sow6YyUHuwguwE6BT/bf/xU5OVjqtlOtLLZ8MMjvl+P8NIEP4tli2z1+U9v/XHhvwTVICPrsPXx68GTzlHGNM8WKxlfF4ou7LmZPZt4+iIf3KRF2OrclD+i7UHVp8DwW4l0gNjN4LCSewwPB9jzDd7lmXlrGSo5pIifHbOHmxlmi0s2cSg5/R8DWrzKaOZUcvMUkOVh7K6OZM8nBGk19whvJS2S40Ku0s9glxy4y82VNZdHA/18qldnxD8Cacc5oImNGAAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAHUAAAAYCAYAAADEbrI4AAAEYklEQVR4Xu2YV6gdVRSGlyZKghgswRgLPlhATEFJsD3ZEgvYUYwkgcRK0PhgQSxoBBEsBPJgxQdF0BBFRFREkxCMUayQB7GAYC8xFrChMVmfe6971qy755wByRm9zgc/Z/a/9ty796y9Z9aMSEdHR0dHxyiOikZHhSnRKDFdda7qMtWkEGuDrdHIjFfto9pDtbtqTxk9wZ1zbKKkuexVDW83pqqOVC1QnRZigzhIdYZqseoI5x+b/WudB4+EdhEuomm3EGuDuqReJNWxmjxvhtjv1fB2Y7P0/ufdITYIP94Ls8dGq5tjo6RC6eS2GDQOG+u4GJC0Q/9SHRADQ4JxTY5mA2xReHbI/qfBH3NJ3VHqxzpH9Vw0h8jRUh5XE0pzujJ7Fwe/UVKPl3TyAzHQEnFynqWS4t85j2cnK/pU57XBi9J/7HWwIznv8+Bvyn6kUVLXSTp539ym+PhIddZIj+FSmohhE70it89TPd4L92U/1ROSLt5jIcZ8P5ZUfMGdkhbORkmFWQn6El+jOkbSuO6v9Eicr3pfdbOkQm5CNfz3XEo7srR7oVFSKSY+dG0uHJUYq78NShMxbKKHSaqGOV5W6VGGi/mL6gXVJarnpbdo+b1a9ZLqXdV81auqa1SfqN7K/SI/q25UrVe9ImksF1R6pMoWf6HqK9Va1b2+g/KkpD709fyjpHIiEwEmYR4XoR/zJC2Akr5VfS1pIl+qvpA0+CaUJmIQY7exM1bndr/+sIukPrwOeUgK2PksZI55Nhuzsue5J3tWqUJpHBuyt5PzaJ/j2uY9GjyrfqMPA5PKrYCTuYW953ze9doiXhxjpqTYT6qzs8dOwWPn1vGH9BLI7r5V9aek3Qsn5t8PJC1Iz20yejylBEbvpNxm5xtW5EXwDgzeqhofBiaVCXLyO/nXJ7YtShOHp6Ucs0TXfTgh/pTqUkkXuwR3Hfpd5TxqC5L/svNukNHJ4tUKj+eywXl4ZzqPvx3HP63gAV5d8ur8EdZK74/yQOf40JFoO5QmCd9LOYaHrHiKEJsbzQCVP/0Od16pgFmZveXOm529h5xHm/dlX2SVFuWSggd4i6KZGZhUTr4vtO/Kx9y2+nGIpIquqfyq7UdpkoD/WTQlXXRLbAn8WHGC/3hROt97PG95Nr+WPauS4dns7S/pebxrbq9wfQDPqmMKMfhGenPiVm/4sbzhjqFRUvlu6dunZO9k5w+TeHGBL0X4l8dAxhIwIwaUX1W3uDbJvEP1oPMGJXVL/j09e1wjYGHTfia37Y3hbdXr+Riul9SPXX2dVAvSm6T64eJgd4xPsebpm1RbUR52E15dGT8M/JjYGRQ5P6p+kFSR87yyQokXd3yrsDmmb4RqHJ+/zQ6KhSD+7cHj6xS+JdR4WNIYfpP0jOUWSz/aVOUGO9AWBo80XqU45s7IuOEESbdpFsHe2QPGS52AH+mb1H8rcaF1VOmSOgbpkjoG6ZI6BvlPJvW4aHRU4NWp4//ANq8QViPIDOVQAAAAAElFTkSuQmCC>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGMAAAAYCAYAAADu3kOXAAADaUlEQVR4Xu2YWchNURiGP0NkTC5EKVEuyTwmF+LChWQIKbPMF0SJ4r8SN4hyQ4aQIVeiSCKZbtwgSfJfGDLP8/y91lr963/32vvsU2f/Zx/tp97O+d619zlr7W+db611RAoKCgoKMqIZGzXGQDZK0VE1VrWcG3JAP1UvNgN0U41XreQGordqi+q2aie1TaW4ErxloxRnVH+s8kZ/SZeM11J6DGdVv63Oqz6pdqvaqrarRjdcWjHesZGG9ZI8kGoxQNIlA5wT86CZCWLGtkPVntpm2basxv6ejTR8VN1hMweg5qZNBh7oLvJQ5uAPJt9nnWSXjA9spAGdmc5mDhgk5SWjqxd3sN4BzwsxUnWJzQqBSV6SuaorqhuqVhI/M6aoHoipyVuprSnAjI5LxgbVS9UF1QiJjuF5wAsxTkxCQtSp3qiOkj9UVS9mTWutWqN6qOrrXyRmXUoEu46fYmroMdVFCXcasxJ+nWq26qmYxa4piUsGSgsSMU/MpLos0TEgZi8tnVQnVPdV01SPVNtUzW07JifiV6prqseqOapbtt2RmIyJEu3g54A3kzx0DvEYz2PwcOKEWfpMTEKfiOl8GoZINBn4hXJ/Ed/z4i7Ww2Qrl55i7j1NPjz0HUeBhWJ+BfBc//D+h33vwLONBTdgG8cetnjszSAvaRHMCpQDTgb6hknFHiaQY5H1kiZPHLhvP5tifAhlDWAt8icF1ijmCxsO1EXc3Jl8eN29eJn18sAwaZyM0Ba8RcBbZT3M4iRwKFvsxcPF3OfKkQOf45LhwPvjXhziKxuOpRLtdGgg+wJeteBkYPDcN/xi2ZtkvT7kM0hGOy/eKNHPAu6sUu95iJd4cYjYZGyW6Bet9jy32BzyvHLA7qYcpYGTcV2ifTvlediYOODt8WLmiEQ3JJsk+vkA5wXsKv1SFLqOiU2GW9QcOI0idqdEJAH0sD6DLV6oLmYJJ8PNUAe2k4hPqtqoFnht820bbzfBYdUoNqWhUvh/UK5QffdigDNZ6BkxsckAbj3Aqn9QzJcifuFfpExW/RKz+0EbOl8NOBlgr5i+ocSgtmMNxBju+hdZcD/aIPzD8E11U9XSv4jA2cE9I7yubdz8DzyPq2wGSExGrRFKRi1RJCNHFMnIEUUycsR/lQw+fNUacX9AFuSBv8Gs53QvX0YhAAAAAElFTkSuQmCC>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADIAAAAVCAYAAAAElr0/AAABF0lEQVR4Xu2Uu64BURSGV1xyEo1EoZGj8BIqtUdQKBQeQFROp6OnEgnzEhIh4QEUCgUPoDiHQqKgOcG/smeSsRBjYibB/pIvM/tfu9iXmUWk0Wg0mg8iKYNXIgincCALzyQB/2HJHAdgD+5gwZrkkh/4Cw2Re8IM5uGR1KbGsAobZuaWGqnDaMqCV3TMJy96BEO2Gmcx29gJBvyDUZF7Cv94XzBHl6fPG5LZLTKk5s7p/CB8hxewFVmdnG8kDQ9wAcOi5iu84MqVrGW+R+yFO7Thinz+tCx40dwiZcYdLEvqtB+FGwb/7Nw0fCFOl58Q34CVDUm1YzeU4RJ2ZcELinAjQ7CGe5iSBRfwzU5gXxZemW8ZaN6NE8opMdb99OXFAAAAAElFTkSuQmCC>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAD0AAAAVCAYAAAD1neayAAABK0lEQVR4Xu2Wr0tDURiGj4JBlDEZBhfNltWVVV0Rs8G/wGR3YtLFwRAMJotZo0mLwbBVmyKDscFgQRwq8/m4Rzh+iByHcI/e88DDdt73cDkfdz+uMZFIJBKJZIY8TukwCzSwgzldhEwR33DPrks4wtbHBk92sY9NXYTINa7jGLfxGVfxAQ+cfT7M4SOe6iI0LuyrDL3l5Es2m4RpbOMVrqkudQom+TGSg+kBV77IJqFikuts6iJtbvBFZcfmd4aumuQ6G7pIGzmU/v5KNlCZLzN4h+dYVl0wyIDLznrRZrNO5sMC9vBIF6Exb5IBd5zsErvO2oc6DvFQFyEih5Shz/DJvq992vE9J3hvfv6pSJVXvNWhJ/LX9CcfQ+XOysNJZpCBRbnb+6r7l7wDS283DF6WH5oAAAAASUVORK5CYII=>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFoAAAAVCAYAAADGpvm7AAABlklEQVR4Xu2WzysFURTHj5QSESUpKTZEWdooFhaysrCiKFvZWdm9kkiKnaWNHTv5AyRsLWUjZWEjkt+/v6cz5c55L+++l5m56nzq03v3e+a9mbkz9weRYRiGYRiGYQTHiA6M5NiHB7BCF0KiDX7BtajdB9/hOWVz4d3wEt7BCVUrxjb8gDO6EALHcIiks+fgMxyEp3DTOS4NpuETXCbprNd42YtOkt/N60LW7Eaf3NGTTl4XZWnijqzeqF0uTfAWbsB2VUudBlgPhyn/pjoKZC49sL8EfeDzsTlYFS+VTTWcJZlS+OFlyhHJvOyyTr93dDPJw/DVhwv66Wx2PF4uC36zF+EjbFW11OGbWi2Q3assSaZU+wa+qawU3LmaR20QcKd2Oe3GKKt1siTJUf7o2YErKvNhgOS/3PUmCGpILsxdpffgtdNOGl6wrlT2AFtUVoxDknsZ1YUQWCK5uC2SeYy/L8SOSIczknPz9vKFZD/tQyXJVpRHQNDwIniiw3+E7wPJHH6LxnRo/C2fJB3NQ5W3QEYCfAPkMFZcrzPliQAAAABJRU5ErkJggg==>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFsAAAAVCAYAAAApZJKFAAAB4ElEQVR4Xu2YSyhFQRjHP3mXSLEgG4+tFAtFKXktbFC2srBgw0KSshMpG7GwUMpKHlE2HguxICQ7W2WHhSQlb/+vmaM5c+8993Sce+5R86tf98z3zWmaufPqEBkMBoPB4I00OAKn4ZaWM/hMHfxWNCSYdPgMb/SEwX/GSczqPj3hgnI9YHDmhbxvIc0k3m2HKVouNGTAT7gpy8XwCT7CPKtSQOj79QT8gAVKLB7VcB1ewywtl3RmYBOJTrbAe9gN9+GlUi8I1MGugCdwCA781nDPEryl4CeMIw/ylzu5SPYlyLFUpZxIGki0twJrYJeMc6xXPnthksT2tKAnkkEVzKHoeyXHuOPRKIH1Li2V7zhxQKK9bdioxHOV57+QTeKWs6YngmaKIge7I0pMhZdnmUvz5TtOcFvncFU+z9rTf6aQxDk0pyeC5g1eaDE+YJwG22+4Lf5jmExZtuCB8grv3XdwTE8kC+5Yv1LmfZtjlUoskdSSfXCt9plWWKTk3LILr0h8BggV3LE9pTxK4joYFMMUuYqs8oYtGh++Zx/DUwrhfbuNRMcGSZzaX3DZViPxHMIdLcbb2ivs1OKx6CExQWId6KHgjMQA/3f4dhR6eFbP60GD/7yTGGxefkdazuAjP8eWY2P0TsHmAAAAAElFTkSuQmCC>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAD0AAAAVCAYAAAD1neayAAABLElEQVR4XmNgGAWjYBSMglEwCkYBFuCMLjBcgSAQVwPxDyBWQpMbUCAFxH+BuAHKNwTin0B8AaaADKDKADGjjgHi8UEHDgOxPxD/B+JcBkiseADxIyDuQFJHLDBggARiIbrEYAJboDTI0/FI4pJQMWIBKPBA6sPRJQYbEAZiASD2ZMD0oA4WMXTAAsQ3gHg7usRQACeA+Dea2CwGwp5mBOLTQHwIXWIoAJDn0PMvSOw9mhg+AMomID2J6BKDFYAci1yliELFOJHEiAXKDJDCsApdYjABHgaIB4uRxPYA8UskPqkAVAh+AuI2IBZDkxsUoJMB4umVQPwNyq5HUUE+4ALiHCD+BcQaaHIDCv4A8Rl0QRoAc3SBgQSgmAU1TkYMAHkYhEGx3YQmNywBALm3N3DMH6BMAAAAAElFTkSuQmCC>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAD0AAAAVCAYAAAD1neayAAAA+0lEQVR4Xu2VrYqCQRSGj4ogKqjBvQ+z2LWZzCbTNqN34BWsbBAxeQMmLUaTUZPFoggii8Hgz3uYXRhP8BtZhPn0PPCEOe8MnMMwDJGiKIqiKIIIzMriq9KCa/glA1/g25jAHZzC5G38EN9wC3My8I0xHMEG7MH+bezMAK5gSga+UYBnaz2EJ2sdRAVe4AxGReYtTTJNd2BNZPeokzlXEvXQwM3b8hsPokpmb1kGYcBuOk9mEL59V4pkzsxhXGReciDTsA2vE6LmCn9PG5iRgU/wgPzF/BH7rf2HNtzDDxn4Ag/5A49khl2Q23t24RMuYVcG70JaFhTleVwBdT8r0USjiRkAAAAASUVORK5CYII=>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEUAAAAVCAYAAAAQAyPeAAABiElEQVR4Xu2VPSiFURjHH98ySAxIGQzsDAaSRdkMirIwsZmUgXJtFmURZoOBwSKTspFsBkWUxSAGUcjn/99zrs59ruh+5L63e371632fj+U87znvEQkEAoH/YxIuwyVbKGQ+nVu2UOhwKO02Wci0iA4lU7psIp9Zl+wMpQ2+wQtYaWqRpxU+iA5i3j13EzqUJngsWr+GtYnlX1mE97DBFqJIEfyAY7Bb9Mty0dNeDykRHdwq7IVTojsgFWLwSfR4RhoOYMSLuWuYK/VyHAhzfS7udDGHmQ7jogPdsYUosC/J/45Rk6uGN5I8gDoTpwN36QE8tIVcwsWfmtwlfPTiTdG+OS+XLSpEd8y2LeQSLnb4h9yMey92MW387sicGLyDKyYfCbjYei/mwpkrh7Oi2zt+PfMYWXgDpcKa6FGssYUowcUOufdmeO5y5MQ9++E7HHBxnAlJ7Ue5Aa9glS1EDd4wr/AZLsAyuAdf4KDXR87grejQ+MV5S/1Fj2j/keiuC4AOmwgE8p8v7XpSgoGAPq0AAAAASUVORK5CYII=>

[image12]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAD0AAAAVCAYAAAD1neayAAAA/ElEQVR4Xu2UPQuBURTHjyxiMiiDhawGq0VM+AB8AGUzSJl8EZnkOyjFIpOsBt9BKZK3vPxP13CfM3i8DRf3V7+65/zr6Tn36TxEFovFYrH8DVnZ+AcicAsbMvgWPHAEl3AC/c74LkG4IHUBUZEZzRAOYAV2YNcZuxKANXiECZEZSRKetboHT1r9LCV4gXkZmESd1Eu2YFFk75CGfTiDXpEZAQ+tyzv+LgU4hlP6zPM+Sk47h0gNzV//Vcqk1iUlA1PYkBpSh2uf6LnBf/Am3MGYyIyDB2xrNe+evAQ3wnB1k8/Gw0Ou4Z7UsHN6fP/i8ACrMvhlMrJhsZjBFZAVLAgeLHkCAAAAAElFTkSuQmCC>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFoAAAAVCAYAAADGpvm7AAABnElEQVR4Xu2XPShFYRjHH1+DlK+yWUgpg5RR2RRFJiOZJVkZbGxSDBQTI1nETGIgMzGZbFJEkfg/PQfvec65594j5yM9v/p17/v8z73vR/e8571EhmEYhmEYhpE7BnTBSI4jeAzLdJBHKuAFfIFXsNEfJ8omvIdNXnsNfsDl7ytKYxu+wwkd5An+RezBMbgC9/1xotzBQ5L+++AZ7CdZ7Iafy0qiHb7CGR3kgS6SSX2xq9pJMgur4CJJn+tOdg6fSO62uPDd8QBXYYvKMmOKZJJbcNwfFaQnhlHb0LT3+kaydbnckoyrW9XjUA0nSbaUTpVlAk/ItdIfB2iNIU+2GNznYEjtWtXiwr/sefgMm1WWOkPO+3qSCc45taSpofCtimu/Pb65e3WdyjKB90A9SW7XqlqSLFBwDAckY4tLL8l3jeoga3hQG0673KulySkF++T2iKoV44Tkc8M6yAO8sI8ktxkP8sarpQn3u0TywOJzPI+lzXdFYfhEwuf+HR0YQXiho04mUXToghEOP4j1tmH8MZcki8zyn4t/ySd49lfBuTHasAAAAABJRU5ErkJggg==>

[image14]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABIAAAAZCAYAAAA8CX6UAAABIklEQVR4XmNgGAWUAg4g1gdiZ3QJUoAkEG8G4v9QTBHgZ4AYshtdglTgwwAxqBZdglRwFoi/AzE7ugQhAArcdCA+A8TqDBDXNCArQAIJQHwZiPvQxBnqgfg5A8QAEHjJADEIZDgyOAgVN4LyTwHxKoQ0RDIGiV8MFUMGnFCx90hiIP46GGcTVAAZdGIR+wvEV9HEQpA5IA0nkQWA4DcQH0ITA6kzRhNDASAFIBegizUj8eWhYkxIYhgApIAXixgXECcD8VMg1oSKYQOMMAZIgT2SBCzGQOAPAyKvgcRYoGwY2AHEHjAOKIpBAfkTiN9Axe4xQBKjGkwREPAB8TsGiLrPQHwYiFmR5MEApMidAZGKQWGBLWBB3tAFYk90iVEw1AAADEM9YlRotXYAAAAASUVORK5CYII=>

[image15]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEUAAAAVCAYAAAAQAyPeAAABkklEQVR4Xu2WPShFYRjHH5+lZCBJymBQVovFYFEWGRTFwCKbUmSgZFAyKSk2ZfSxyaTuRInJoIjBYJAsJN/8/z3v0Xufe4dzU7f3ds+vft3zfHQ779M573tEEhISEvLHBFyDK7ZQzPw4d2yh2OFQ2m2ymGkRHUq+qIaVNhkaW5KfobTBNzhtCyHQCp9EB7Hgfg/SOpQmeCpav4O16eXYpET/Y9jkg6EEfsNR2Ak/RW94xushZaKDW4ddcApe+w0xKIVn8Bj2mlpQcABDXsynhrlyL8eBMNft4g4Xc5hxqIK3cNcWQiQlmXvHiMnVwHvJHECdibOxCF/gqi2EDBd/YXI38NmLt0X75r1cXMZFh1kQT0gEFzuYJTfrrrkHMKaNfx25w33rBB7ZQohwsQ1ezIUzx2+GOdHFRMczXyMLT6Bc2YNXcMwWQoGLHXDXzaI3G+0n5+63B37BPhdH8NXYN7m4bMJ3OGnyQcAT5gO+wiVYAQ9FP6j6vT5yCR9Eh7Yhekr9l3r4CJdtIUGPfJqQUAD8Au1BUwsDXDE5AAAAAElFTkSuQmCC>

[image16]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAB0AAAAYCAYAAAAGXva8AAABbElEQVR4Xu2UvStFYRzHf8RKWZQyK8rgGozeNpkNDNwB/wGRRUxSslmsFi9ZpCSDlOGalMFkEosiUfL2/fU8R7/zPc/p6uqcuuVTn+55vr/nrfs854j8UwX0cZAHmxxkTScc4DBrbjnIg0sOsmYctnL4W5rgPZzw7TZ4B3d/eoT55IDYhtdwB9ZQTQ7hFPyCRfgCR+ETHDP9GO2Xxh68gLPwGM7HyyIH/jdaNKLbZyEKcIhDTxd8M+0z+Gra0izu7x2W5AK9gUypE3ccaSyIG7cM+6kWoyS0G3FnEVp0C15xSOg4a5B3eELZg4QH6FnOcUiMiLtE0aI98bJDC/wN1WyaMoU3Z1mX5Ea13U6ZNPrCpMnO4Y1pR8zABg4NOs+Gadf7LMGquMI+fPbPOnmI4AQGvWQ6h94P7Zv6xfqApxwGqIVrHFaK7kivdzkGxb2Df6ZF3KL6vpbjkYNKWYKLcAV2UI054iAP9CZWJ98J40+7bfCtPgAAAABJRU5ErkJggg==>

[image17]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAG0AAAAYCAYAAADwF3MkAAAD+0lEQVR4Xu2ZSagVRxSGjwNRI06IaBxQAxIUnHARNRgeaFwqEgRxERwWggMILly4EBHUhSvjBJIEBRGiS5fBPFw4YoKi4EpFBBEUjXFCjXr+V93v1v3vOdX97m3lRe4HB17/p/vvuqequqvribRp06ZNmxKMZqFFmvGbw0KTVOXT6znGQsYAjQkaEzUmZ4HjMRqjovMYzy/FExYMpml0ahzXmJ1pX2l8kZ8g5Xw+C7wiH9R4n4ibGkO7z67h+aX4hwUC93mncU7jT43XGms1rsUnSbHPp2Ccxg4Jg6wnzNXYojGMExZFRT4loZOYBRL01aQX+Vk8ZSHiljTev4/GZUNP+XxsvpYwsDZnxzs1HtTSSf7TmJ79/beE37W3lm4kVWQ8EvOZZWHlUn4e/7KQMVaC/0ZOKFM0bpPm+XwK0E7Ui7UlpDEdEs7DJAAYkM8yzSVV5F8lXHyUExnI8SMp5eeBRlqclHCPNZyQsOBZSZrnw+zTeCnhcVQFaItV5Dsa91kkBkq4dnx2PCk7tvy6SRU5v5hHENguIYfZEJPy83jOQgYGS96GHyln4fl4jNB4JKETW2Gr2EU+L7ae4oCEay5yIiZV5FSPY6QimJSfh1fsZVJrAwJF+L7ujHo8nxSDNe6J/zQpw89i1+ms2HqKNxKuwYLGJVXkvFiHJawmf9N4lWl7ovNiUn4eL1iIOCJhtRh33l8Sis2kfIrop3FVgv9iyhVxQuzOOSO2bnFD46GEVTk/9hvwivyThBv29OXu+aWwZqzFLKl1XGd9qouyPkV0SLjHCtI98Hi1OqdTbD3FNxqPpWAAekW+LeGGeF73BM8vBWYvg1ltge8ftAvvIsbyaYaZGm81NnDCYb3YnYP3kqUXsU3CddaquQuvyPmI/pYTBXh+Kaxiez8WS2LkLnFCbJ+eMl+C/yrSUywXu73XxdZj8DG9kLT8XY7Vu4lXZFx0msUSWH7YEkthFdv7sZskPDqwhcVYPmXoq3FFwjvoB8qVxWovZuvvpMXbbiCfHDHrMm036d1YRZ4hzc0ywH4oMrx45ySGi71IwjW8ATwk0/uTnsM+RQzSuCuNbW4GPK4PkYa2fhkdY88U2v5Iw+IHC5kczDysZrkj64gbjA89LJvxwYzNV7zYMVqmRucUwQVAYbAqwkj24GKf1Riu8YeExmMxhC0hbFN5HQbYxwN7g3jZ7+JEi6B22MpC56Hd8+rTXdyRxv+E4DGKOl+QcB02CfAacOEit4rn9wsLEWWLXURZHwzO/zVekZvF8sM30FIWI8oWu4iqfHo9VpFbwfIr+n6qqthV+fR6rCK3guU3kgWiqmJX5dPrwX+jq6QZv+9YaJKqfNq0aZ0Pvhb+wC2+fpUAAAAASUVORK5CYII=>