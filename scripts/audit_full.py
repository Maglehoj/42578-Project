import pandas as pd

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

OUTPUT_DIR = BASE_DIR.parent
PATH = OUTPUT_DIR / "ae_2017_2019_full_panel.csv"
OUTPUT_PATH = OUTPUT_DIR / "ae_2017_2019_analysis_ready.csv"

df = pd.read_csv(PATH)

df = df[
    (df["provider_code"] != "-") &
    (df["provider_name"].str.lower() != "england")
].copy()

df["month"] = pd.to_datetime(df["month"])

df["year"] = df["month"].dt.year
df["month_num"] = df["month"].dt.month

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissingness:")
print(df.isna().mean().sort_values(ascending=False))

print("\nNumeric summaries:")
print(df.describe().T)

print("\nRows with impossible 4-hour performance:")
print(df[
    (df["pct_4hr_all"] < 0) | (df["pct_4hr_all"] > 1)
][["provider_code", "provider_name", "month", "pct_4hr_all"]])

print("\nAttendance consistency check:")
if {"att_total", "under4_total", "over4_total"}.issubset(df.columns):
    df["attendance_check_diff"] = (
        df["att_total"] - df["under4_total"] - df["over4_total"]
    )
    print(df["attendance_check_diff"].describe())
    print(df[df["attendance_check_diff"].abs() > 1][[
        "provider_code", "provider_name", "month",
        "att_total", "under4_total", "over4_total",
        "attendance_check_diff"
    ]].head(30))

print("\n4-hour percentage consistency check:")
if {"under4_total", "att_total", "pct_4hr_all"}.issubset(df.columns):
    df["pct_check"] = df["under4_total"] / df["att_total"]
    df["pct_diff"] = df["pct_4hr_all"] - df["pct_check"]

    print(df["pct_diff"].describe())
    print(df[df["pct_diff"].abs() > 0.01][[
        "provider_code", "provider_name", "month",
        "att_total", "under4_total", "pct_4hr_all", "pct_check", "pct_diff"
    ]].head(30))

print(
    df.sort_values("att_total", ascending=False)[[
        "provider_code",
        "region",
        "provider_name",
        "month",
        "att_total",
        "emerg_adm_total"
    ]].head(30)
)

print(df.groupby("provider_code")["month"].nunique().describe())


df.to_csv(OUTPUT_PATH, index=False)

print(f"\nSaved clean analysis dataset to: {OUTPUT_PATH}")
print(f"Final shape: {df.shape}")