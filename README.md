# 42578 Advanced Business Analytics — Project

Analysis of NHS England A&E (Accident & Emergency) trust performance 2017–2019, focused on how hospital trusts respond to demand shocks and how resilience varies across the system.

## Status

Early-stage exploration. Data pipelines and metrics are built; a confirmed research question and modelling strategy are not yet settled.

## Data

Monthly A&E statistics published by NHS England, covering January 2017 – December 2019 (pre-COVID). Each month is one `.xls` file, organised under `data/YY/YY-mon.xls`. Raw source files are version-controlled; generated intermediate CSVs are gitignored.

Key variables extracted per trust per month:

- Total A&E attendances
- 4-hour wait performance (proportion seen within 4 hours)
- Emergency admissions
- Patients waiting >4h and >12h from decision to admit

## Pipeline

```
build_dataset.py              → ae_2017_2019_full_panel.csv
prepare_analysis_dataset.py   → ae_2017_2019_analysis_ready.csv
shock_analysis.py             → ae_2017_2019_with_shocks.csv
                                trust_shock_summary.csv
shock_resilience.py           → trust_resilience_scores.csv
trust_clustering.py           → trust_resilience_clusters.csv
```

Run scripts in the order above. `audit_full.py` is a standalone data quality checker.

## Core Concepts

**Demand shock** — a month where a trust's attendances are significantly above its own historical baseline, defined by a z-score threshold (≥ 1.5 SD above the trust's rolling 12-month mean). A ratio-based definition is also computed for comparison.

**Resilience score** — how much a trust's performance deteriorates above its own expected baseline during shock months. Measured on both 4-hour breach rate and 12-hour wait rate; combined into a composite score weighted 70/30. Estimates are regularised with shrinkage to handle trusts with few shock observations.

**Clustering** — K-means grouping of trusts by resilience features to identify structural archetypes (e.g. high-resilience vs. fragile-under-shock).

## Dependencies

Standard scientific Python stack:

```
pandas
numpy
matplotlib
scikit-learn
openpyxl / xlrd   # for .xls reading
```

## Extending This Project

Future work may include:

- Regression or panel models explaining resilience (staffing, capacity, geography, etc.)
- Incorporating pre/post shock recovery dynamics
- Integrating additional NHS datasets (workforce, finance, CQC ratings)
- Extending the time window (post-2019 data requires format checks)

Add new analysis scripts following the same pattern: read from an existing pipeline CSV, write a new output CSV to `data/`, use `Path(__file__).resolve().parent` for all paths.
