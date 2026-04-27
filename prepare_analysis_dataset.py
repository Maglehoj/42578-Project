import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

INPUT_PATH = DATA_DIR / "ae_2017_2019_full_panel.csv"
OUTPUT_PATH = DATA_DIR / "ae_2017_2019_analysis_ready.csv"

df = pd.read_csv(INPUT_PATH)

# dates
df["month"] = pd.to_datetime(df["month"])

# sort as provider-month panel
df = df.sort_values(["provider_code", "month"])

# remove obvious total / aggregate rows
df = df[
    ~df["provider_name"].str.contains("total", case=False, na=False)
].copy()

# core variables for your project
core_cols = [
    "provider_code",
    "region",
    "provider_name",
    "month",
    "att_total",
    "pct_4hr_all",
    "emerg_adm_total",
    "wait_4hr_dta",
    "wait_12hr_dta",
]

df_core = df[core_cols].copy()

# rename to clearer analysis names
df_core = df_core.rename(columns={
    "att_total": "attendances_total",
    "pct_4hr_all": "four_hour_performance_all",
    "emerg_adm_total": "emergency_admissions_total",
    "wait_4hr_dta": "wait_over_4h_decision_to_admit",
    "wait_12hr_dta": "wait_over_12h_decision_to_admit",
})

# convert numeric columns safely
numeric_cols = [
    "attendances_total",
    "four_hour_performance_all",
    "emergency_admissions_total",
    "wait_over_4h_decision_to_admit",
    "wait_over_12h_decision_to_admit",
]

for col in numeric_cols:
    df_core[col] = pd.to_numeric(df_core[col], errors="coerce")

# remove rows without core demand measure
df_core = df_core.dropna(subset=["attendances_total"])

# keep only rows with valid 4-hour performance for main analysis
df_analysis = df_core.dropna(subset=["four_hour_performance_all"]).copy()

# standardise 4-hour performance if stored as 0-100 instead of 0-1
if df_analysis["four_hour_performance_all"].max() > 1:
    df_analysis["four_hour_performance_all"] = (
        df_analysis["four_hour_performance_all"] / 100
    )

# derived metrics
df_analysis["four_hour_breach_rate"] = 1 - df_analysis["four_hour_performance_all"]

df_analysis["admission_rate"] = (
    df_analysis["emergency_admissions_total"] /
    df_analysis["attendances_total"]
)

df_analysis["wait_over_12h_rate"] = (
    df_analysis["wait_over_12h_decision_to_admit"] /
    df_analysis["attendances_total"]
)

# basic panel diagnostics
print("\nFinal analysis shape:")
print(df_analysis.shape)

print("\nDate range:")
print(df_analysis["month"].min(), "to", df_analysis["month"].max())

print("\nNumber of providers:")
print(df_analysis["provider_code"].nunique())

print("\nMissing values:")
print(df_analysis.isna().sum())

print("\n4-hour performance summary:")
print(df_analysis["four_hour_performance_all"].describe())

print("\nAttendances summary:")
print(df_analysis["attendances_total"].describe())

# missingness by month before dropping 4hr rows
missing_by_month = (
    df_core.groupby("month")[[
        "four_hour_performance_all",
        "emergency_admissions_total",
        "wait_over_12h_decision_to_admit"
    ]]
    .apply(lambda x: x.isna().mean())
)

print("\nMissingness by month:")
print(missing_by_month)

# monthly overview
monthly_avg = (
    df_analysis.groupby("month")[[
        "attendances_total",
        "four_hour_performance_all",
        "four_hour_breach_rate",
        "wait_over_12h_rate"
    ]]
    .mean()
    .reset_index()
)

# save cleaned analysis data
df_analysis.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved analysis-ready file to: {OUTPUT_PATH}")

# plots
plt.figure(figsize=(10, 5))
plt.plot(monthly_avg["month"], monthly_avg["attendances_total"])
plt.title("Average Monthly Attendances per Provider")
plt.xlabel("Month")
plt.ylabel("Attendances")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(monthly_avg["month"], monthly_avg["four_hour_performance_all"])
plt.title("Average 4-Hour Performance Over Time")
plt.xlabel("Month")
plt.ylabel("4-Hour Performance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(monthly_avg["month"], monthly_avg["four_hour_breach_rate"])
plt.title("Average 4-Hour Breach Rate Over Time")
plt.xlabel("Month")
plt.ylabel("Breach Rate")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()