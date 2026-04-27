import pandas as pd
import glob
import os

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

DATA_PATH = str(DATA_DIR / "**/*.xls")
OUTPUT_PATH = DATA_DIR / "ae_2017_2019_full_panel.csv"

month_map = {
    "jan": "01", "feb": "02", "mar": "03",
    "apr": "04", "may": "05", "jun": "06",
    "jul": "07", "aug": "08", "sep": "09",
    "oct": "10", "nov": "11", "dec": "12"
}

files = sorted(glob.glob(DATA_PATH, recursive=True))
all_dfs = []

def clean_col(c):
    return str(c).lower().strip().replace("\n", " ")

def find_exact_col(raw_df, target):
    target_clean = clean_col(target)
    for col in raw_df.columns:
        if clean_col(col) == target_clean:
            return col
    return None

for file in files:
    filename = os.path.basename(file)

    try:
        year = "20" + filename[:2]
        month_str = filename[3:6].lower()
        month = month_map[month_str]
        file_month = pd.to_datetime(f"{year}-{month}-01")

        raw = pd.read_excel(file, skiprows=15)
        raw = raw.dropna(axis=0, how="all")

        # exact header-based extraction for key columns
        col_code = find_exact_col(raw, "Code")
        col_region = find_exact_col(raw, "Region")
        col_name = find_exact_col(raw, "Name")
        col_att_total = find_exact_col(raw, "Total attendances")
        col_pct_all = find_exact_col(raw, "Percentage in 4 hours or less (all)")
        col_adm_total = find_exact_col(raw, "Total Emergency Admissions")
        col_wait4 = find_exact_col(raw, "Number of patients spending >4 hours from decision to admit to admission")
        col_wait12 = find_exact_col(raw, "Number of patients spending >12 hours from decision to admit to admission")

        required = {
            "provider_code": col_code,
            "region": col_region,
            "provider_name": col_name,
            "att_total": col_att_total,
            "pct_4hr_all": col_pct_all,
            "emerg_adm_total": col_adm_total,
            "wait_4hr_dta": col_wait4,
            "wait_12hr_dta": col_wait12,
        }

        missing = [k for k, v in required.items() if v is None]
        if missing:
            print(f"SKIPPED {filename}: missing columns {missing}")
            print(list(raw.columns))
            continue

        df = pd.DataFrame({
            "provider_code": raw[col_code],
            "region": raw[col_region],
            "provider_name": raw[col_name],
            "att_total": raw[col_att_total],
            "pct_4hr_all": raw[col_pct_all],
            "emerg_adm_total": raw[col_adm_total],
            "wait_4hr_dta": raw[col_wait4],
            "wait_12hr_dta": raw[col_wait12],
        })

        df["month"] = file_month
        df["source_file"] = filename

        all_dfs.append(df)

        print(f"Loaded {filename}: {df.shape[0]} rows")

    except Exception as e:
        print(f"ERROR with {file}: {e}")

if not all_dfs:
    raise ValueError("No files loaded.")

combined = pd.concat(all_dfs, ignore_index=True)

combined = combined[
    combined["provider_code"].notna()
    & combined["provider_name"].notna()
].copy()

combined["provider_code"] = combined["provider_code"].astype(str).str.strip()
combined["region"] = combined["region"].astype(str).str.strip()
combined["provider_name"] = combined["provider_name"].astype(str).str.strip()

numeric_cols = [
    "att_total",
    "pct_4hr_all",
    "emerg_adm_total",
    "wait_4hr_dta",
    "wait_12hr_dta"
]

for col in numeric_cols:
    combined[col] = pd.to_numeric(combined[col], errors="coerce")

combined.to_csv(OUTPUT_PATH, index=False)

print("\nDone.")
print(f"Saved to: {OUTPUT_PATH}")
print(f"Final shape: {combined.shape}")

print("\n4-hour performance check:")
print(combined["pct_4hr_all"].describe())

print("\nMissing values:")
print(combined[[
    "provider_code",
    "provider_name",
    "month",
    "att_total",
    "pct_4hr_all",
    "emerg_adm_total",
    "wait_12hr_dta"
]].isna().sum())