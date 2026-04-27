# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

This is a university data analytics project (DTU 42578 - Advanced Business Analytics) analysing NHS England Accident & Emergency (A&E) monthly performance data from 2017–2019. The goal is to study how hospital trusts respond to demand shocks — periods of unusually high A&E attendance — and to rank and cluster trusts by resilience.

**The research direction is not yet fixed.** The current scripts are exploratory pipelines that build data infrastructure (shock detection, resilience scoring, clustering) without a confirmed hypothesis or modelling strategy. Expect significant iteration.

## Running the Pipeline

Scripts must be run in order — each step's output is the next step's input:

```bash
python scripts/build_dataset.py            # raw .xls → ../ae_2017_2019_full_panel.csv
python scripts/prepare_analysis_dataset.py # full panel → ../ae_2017_2019_analysis_ready.csv
python scripts/shock_analysis.py           # analysis-ready → ../ae_2017_2019_with_shocks.csv + ../trust_shock_summary.csv
python scripts/shock_resilience.py         # with_shocks → ../trust_resilience_scores.csv
python scripts/trust_clustering.py         # resilience scores → ../trust_resilience_clusters.csv
```

`audit_full.py` is a standalone data quality checker — run it ad hoc against any intermediate CSV.

Generated CSVs are gitignored (`data/*.csv`). Raw `.xls` source files are tracked.

## Data

- Source: NHS England A&E monthly statistics, one `.xls` file per month
- Location: `data/17/`, `data/18/`, `data/19/` (years 2017–2019)
- File naming: `YY-mon.xls` (e.g. `18-mar.xls`). `build_dataset.py` parses year and month from the filename.
- Raw files have 15 header rows before data starts — all reads use `skiprows=15`
- Column names are matched by exact string after lowercasing/stripping; the `find_exact_col` helper in `build_dataset.py` handles whitespace/newline variants

## Architecture

### Key Columns (canonical names after `prepare_analysis_dataset.py`)

| Column | Meaning |
|---|---|
| `provider_code` | NHS trust identifier |
| `attendances_total` | Total A&E attendances |
| `four_hour_performance_all` | Proportion seen within 4 hours (0–1) |
| `emergency_admissions_total` | Emergency admissions |
| `wait_over_4h_decision_to_admit` | Patients waiting >4h from decision to admit |
| `wait_over_12h_decision_to_admit` | Patients waiting >12h from decision to admit |

Note: `shock_analysis.py` reads from `ae_2017_2019_analysis_ready.csv` but still uses the old pre-rename column names (`att_total`, `pct_4hr_all`, etc.). Both naming schemes coexist across scripts.

### Shock Detection (`shock_analysis.py`)

A "shock" month is defined by **z-score** of attendances relative to the trust's own rolling 12-month baseline (lagged by 1 month to avoid look-ahead):

- `demand_zscore >= 1.5` → `shock = True`
- A ratio-based definition (`demand_ratio >= 1.15`) also exists as `shock_ratio` for comparison
- Z-scores > 10 are flagged as `structural_break_flag` (data anomalies / outliers)
- Z-scores are capped at ±5 (`demand_zscore_capped`) for downstream use

### Resilience Scoring (`shock_resilience.py`)

For each trust, resilience is measured as performance deterioration **above the trust's own expected baseline** during shock months:

- `breach_impact` = actual 4-hr breach rate − expected (rolling 12m mean, lagged)
- `wait_12hr_impact` = actual 12-hr wait rate − expected
- Estimates are shrunk toward the cross-trust mean using a James-Stein-style shrinkage (`SHRINKAGE_K = 5`) to stabilise trusts with few shock observations
- Composite score = 70% breach resilience + 30% 12-hr wait resilience (percentile-ranked, higher = more resilient)
- Trusts are tagged `low/medium/high` evidence based on shock count

### Clustering (`trust_clustering.py`)

- K-means (k=4) on 6 features: breach score, wait-12h score, capped demand z-score, structural break share, average shock attendance, number of shocks
- By default only "medium" and "high" evidence trusts are clustered (`USE_RELIABLE_ONLY = True`)
- Silhouette scores for k=2–6 are printed to help choose k

## Column Naming Convention

All scripts use **short column names** throughout the pipeline (`att_total`, `pct_4hr_all`, `emerg_adm_total`, `wait_4hr_dta`, `wait_12hr_dta`). Do not introduce long-name aliases — that broke the pipeline previously.