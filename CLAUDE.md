# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

This is a university data analytics project (DTU 42578 - Advanced Business Analytics) analysing NHS England Accident & Emergency (A&E) performance data from **January 2014 – December 2019** (72 months). The goal is to study how hospital trusts respond to demand shocks — periods of unusually high A&E attendance — and to rank and cluster trusts by resilience.

The primary deliverable is **`Gustav-main-book-v3.ipynb`** — a self-contained end-to-end notebook covering data build, shock detection, resilience scoring, clustering, and validation. The standalone scripts in `scripts/` are superseded by the notebook but kept for reference.

## Primary Notebook

**`Gustav-main-book-v3.ipynb`** — 34 cells, 10 sections:

| Section | Cells | Content |
|---|---|---|
| 1. Build Dataset | 3–5 | Weekly + monthly ingestion, dual-format reader, merger splitting |
| 2. Prepare | 7–9 | Provider filters (Y0-prefix, always-perfect), overview chart |
| 3. Data Audit | 11 | Missing values, duplicate checks |
| 4. Shock Detection | 13–16 | Seasonal z-score, structural break flagging, shock trend chart |
| 5. Resilience Scoring | 18–20 | James-Stein shrinkage, breach/12hr impact, visualisation |
| 6. Trust Clustering | 22–23 | K-means k=3, PCA scatter, boxplot |
| 7. Trust Heterogeneity (ICC) | 25 | Mixed-effects null model, ICC ≈ 0.59 |
| 8. Cluster Stability | 28 | ARI bootstrap 30 seeds, mean ARI ≈ 0.957 |
| 9. Threshold Robustness | 30 | Spearman ρ = 0.70–0.82 across z≥1.0/1.5/2.0 |
| 10. Geographic Map | 33 | Folium interactive map → `trust_resilience_map.html` |

Run cells top-to-bottom. Cells 28 (ARI) depends on `X_scaled` from cell 22.

## Running the Legacy Scripts

Scripts must be run in order — each step's output is the next step's input:

```bash
python scripts/build_dataset.py            # raw .xls → ../ae_panel_full.csv
python scripts/prepare_analysis_dataset.py # full panel → ../ae_panel_analysis_ready.csv
python scripts/shock_analysis.py           # analysis-ready → ../ae_panel_with_shocks.csv
python scripts/shock_resilience.py         # with_shocks → ../trust_resilience_scores.csv
python scripts/trust_clustering.py         # resilience scores → ../trust_resilience_clusters.csv
```

`audit_full.py` is a standalone data quality checker — run it ad hoc against any intermediate CSV.

Generated CSVs are gitignored. Raw `.xls` source files are tracked.

## Data Sources

- **Monthly files:** `data/15/`, `data/16/`, `data/17/`, `data/18/`, `data/19/` — one `.xls` per month, `skiprows=15`
- **Weekly files:** `data/weekly/` — 78 files covering Jan 2014 – Jun 2015. Two formats:
  - **2014 format:** "Area Team" layout, split sub-columns for `>4hr DTA wait`; no `Total Emergency Admissions` column
  - **2015 format:** "Region" layout, combined columns
  - The `read_weekly_file()` function in the notebook handles both formats automatically
- **Succession file:** `data/succ.csv` — NHS predecessor/successor mapping (11536 rows) used for merger splitting

## Architecture

### Key Columns (short names used throughout)

| Column | Meaning |
|---|---|
| `provider_code` | NHS ODS trust identifier |
| `att_total` | Total A&E attendances |
| `pct_4hr_all` | Proportion seen within 4 hours (0–1) |
| `four_hour_breach_rate` | 1 − pct_4hr_all |
| `emerg_adm_total` | Emergency admissions |
| `wait_4hr_dta` | Patients waiting >4h from decision to admit |
| `wait_12hr_dta` | Patients waiting >12h from decision to admit |
| `series_id` | Trust series identifier (e.g. `RM3_s1`) — trusts split at merger/rename events |

**Do not introduce long-name aliases** — that broke the pipeline previously.

### Shock Detection

Shocks use a **same-calendar-month (seasonal) z-score** — July is compared only to prior Julys, not to the rolling 12-month average:

- `expanding(min_periods=3)` per (series_id, month_num) group
- `demand_zscore >= 2.0` → `shock = True` (raised from 1.5; z≥1.5 showed no significant breach impact p=0.61)
- `demand_zscore > 10` → `structural_break_flag` (data anomaly)
- Z-scores capped at ±5 → `demand_zscore_capped`
- `MIN_PERIODS = 12` in params cell — first 12 months have no z-score (ensures full seasonal cycle before classification)

### Resilience Scoring

- `breach_impact` = shock-month breach rate − non-shock baseline breach rate
- James-Stein shrinkage: `shrunk = (n × impact + k × global_mean) / (n + k)`, `k=5`
- Evidence tiers: low (<3 shocks), medium (3–5), high (≥6)
- Composite resilience score: 70% breach + 30% 12hr wait (percentile-ranked, higher = more resilient)

### Clustering

- K-means **k=3** on 2 features: `shrunk_breach_impact` + `mean_demand_zscore_capped`
- Only medium/high evidence trusts with `structural_break_share == 0` are clustered (103 trusts)
- Silhouette: k=2 = 0.447 (statistical optimum), k=3 = 0.402 (chosen for interpretability)
- Labels assigned by `shrunk_breach_impact` rank: resilient / moderate / fragile

### Validation Results (Sections 7–9)

- **ICC = 0.59** — 59% of breach rate variance is between-trust (justifies trust-level analysis)
- **Mean ARI = 0.957** — k=3 clustering is highly stable across random seeds
- **Threshold ρ = 0.70–0.82** — trust rankings are consistent across z≥1.0/1.5/2.0

### Geographic Map

`trust_resilience_map.html` is generated by Section 10. Coordinates are approximate ODS-code centroids hardcoded in the notebook (from training data knowledge). Covers all 103 clustered trusts.
