import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

OUTPUT_DIR  = BASE_DIR.parent
INPUT_PATH  = OUTPUT_DIR / "ae_2017_2019_full_panel.csv"
OUTPUT_PATH = OUTPUT_DIR / "ae_2017_2019_analysis_ready.csv"

df = pd.read_csv(INPUT_PATH)

df["month"] = pd.to_datetime(df["month"])
df = df.sort_values(["provider_code", "month"])

df = df[~df["provider_name"].str.contains("total", case=False, na=False)].copy()

core_cols = [
    "provider_code", "region", "provider_name", "month",
    "att_total", "pct_4hr_all", "emerg_adm_total",
    "wait_4hr_dta", "wait_12hr_dta",
]

df_core = df[core_cols].copy()

numeric_cols = [
    "att_total", "pct_4hr_all", "emerg_adm_total",
    "wait_4hr_dta", "wait_12hr_dta",
]

for col in numeric_cols:
    df_core[col] = pd.to_numeric(df_core[col], errors="coerce")

df_core = df_core.dropna(subset=["att_total"])
df_analysis = df_core.dropna(subset=["pct_4hr_all"]).copy()

if df_analysis["pct_4hr_all"].max() > 1:
    df_analysis["pct_4hr_all"] = df_analysis["pct_4hr_all"] / 100

df_analysis["four_hour_breach_rate"] = 1 - df_analysis["pct_4hr_all"]
df_analysis["admission_rate"]        = df_analysis["emerg_adm_total"] / df_analysis["att_total"]
df_analysis["wait_12hr_rate"]        = df_analysis["wait_12hr_dta"]   / df_analysis["att_total"]

print("\nFinal analysis shape:")
print(df_analysis.shape)

print("\nDate range:")
print(df_analysis["month"].min(), "to", df_analysis["month"].max())

print("\nNumber of providers:")
print(df_analysis["provider_code"].nunique())

print("\nMissing values:")
print(df_analysis.isna().sum())

print("\n4-hour performance summary:")
print(df_analysis["pct_4hr_all"].describe())

print("\nAttendances summary:")
print(df_analysis["att_total"].describe())

missing_by_month = (
    df_core.groupby("month")[["pct_4hr_all", "emerg_adm_total", "wait_12hr_dta"]]
    .apply(lambda x: x.isna().mean())
)

print("\nMissingness by month:")
print(missing_by_month)

monthly_avg = (
    df_analysis.groupby("month")[[
        "att_total", "pct_4hr_all",
        "four_hour_breach_rate", "wait_12hr_rate",
    ]]
    .mean()
    .reset_index()
)

df_analysis.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved analysis-ready file to: {OUTPUT_PATH}")

plt.figure(figsize=(10, 5))
plt.plot(monthly_avg["month"], monthly_avg["att_total"])
plt.title("Average Monthly Attendances per Provider")
plt.xlabel("Month")
plt.ylabel("Attendances")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(monthly_avg["month"], monthly_avg["pct_4hr_all"])
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
