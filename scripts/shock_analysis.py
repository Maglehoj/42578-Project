import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

OUTPUT_DIR = BASE_DIR.parent
INPUT_PATH = OUTPUT_DIR / "ae_2017_2019_analysis_ready.csv"
OUTPUT_DATA_PATH = OUTPUT_DIR / "ae_2017_2019_with_shocks.csv"
OUTPUT_SUMMARY_PATH = OUTPUT_DIR / "trust_shock_summary.csv"

SHOCK_THRESHOLD = 1.15
ROLLING_WINDOW = 12
MIN_PERIODS = 6


def main():
    df = pd.read_csv(INPUT_PATH)

    df["month"] = pd.to_datetime(df["month"])

    numeric_cols = [
        "att_total",
        "pct_4hr_all",
        "emerg_adm_total",
        "wait_4hr_dta",
        "wait_12hr_dta",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # keep rows usable for main shock/performance analysis
    df = df.dropna(subset=["att_total", "pct_4hr_all"]).copy()

    # defensive aggregate removal
    df = df[
        (df["provider_code"] != "-") &
        (~df["provider_name"].str.contains("england", case=False, na=False))
    ].copy()

    df = df.sort_values(["provider_code", "month"]).copy()

    # core derived metrics
    df["four_hour_breach_rate"] = 1 - df["pct_4hr_all"]
    df["admission_rate"] = df["emerg_adm_total"] / df["att_total"]
    df["wait_4hr_rate"] = df["wait_4hr_dta"] / df["att_total"]
    df["wait_12hr_rate"] = df["wait_12hr_dta"] / df["att_total"]

    # demand baseline: rolling mean and rolling standard deviation
    df["att_rolling_mean_12m"] = (
        df.groupby("provider_code")["att_total"]
        .transform(lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=MIN_PERIODS).mean())
    )

    df["att_rolling_std_12m"] = (
        df.groupby("provider_code")["att_total"]
        .transform(lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=MIN_PERIODS).std())
    )

    df["demand_ratio"] = df["att_total"] / df["att_rolling_mean_12m"]

    df["demand_zscore"] = (
        (df["att_total"] - df["att_rolling_mean_12m"]) /
        df["att_rolling_std_12m"]
    )

    #capped z-scores
    df["demand_zscore_capped"] = df["demand_zscore"].clip(-5, 5)

    df["structural_break_flag"] = df["demand_zscore"] > 10

    # old definition: high demand relative to rolling average
    df["shock_ratio"] = df["demand_ratio"] >= SHOCK_THRESHOLD

    # new definition: unusually high demand relative to trust's own volatility
    ZSCORE_THRESHOLD = 1.5
    df["shock_zscore"] = df["demand_zscore"] >= ZSCORE_THRESHOLD

    # main shock definition for downstream scripts
    df["shock"] = df["shock_zscore"]

    # response metrics
    df["breach_change_vs_prev_month"] = (
        df.groupby("provider_code")["four_hour_breach_rate"].diff()
    )

    df["performance_change_vs_prev_month"] = (
        df.groupby("provider_code")["pct_4hr_all"].diff()
    )

    df["wait_12hr_rate_change_vs_prev_month"] = (
        df.groupby("provider_code")["wait_12hr_rate"].diff()
    )

    # trust-level shock summary
    shock_summary = (
        df[df["shock"]]
        .groupby(["provider_code", "provider_name"])
        .agg(
            shock_months=("shock", "sum"),
            avg_demand_ratio=("demand_ratio", "mean"),
            avg_breach_change=("breach_change_vs_prev_month", "mean"),
            avg_performance_change=("performance_change_vs_prev_month", "mean"),
            avg_wait_12hr_rate=("wait_12hr_rate", "mean"),
            avg_wait_12hr_rate_change=("wait_12hr_rate_change_vs_prev_month", "mean"),
            avg_attendances=("att_total", "mean"),
        )
        .reset_index()
    )

    # require at least 2 shocks for ranking stability
    shock_summary_ranked = (
        shock_summary[shock_summary["shock_months"] >= 2]
        .sort_values("avg_breach_change", ascending=False)
    )

    # diagnostics
    print("\nDataset shape:")
    print(df.shape)

    print("\nDate range:")
    print(df["month"].min(), "to", df["month"].max())

    print("\nProviders:")
    print(df["provider_code"].nunique())

    print("\nShock threshold:")
    print(SHOCK_THRESHOLD)

    print("\nRows with usable rolling baseline:")
    print(df["att_rolling_mean_12m"].notna().sum())

    print("\nShock share:")
    print(df["shock"].mean())

    print("\nNumber of shock months:")
    print(df["shock"].sum())

    print("\nProviders with at least one shock:")
    print((df.groupby("provider_code")["shock"].sum() > 0).sum())

    print("\nTop fragile trusts during shocks:")
    print(shock_summary_ranked.head(15))

    print("\nTop resilient trusts during shocks:")
    print(shock_summary_ranked.tail(15))

    print("\nRatio shock share:")
    print(df["shock_ratio"].mean())

    print("\nZ-score shock share:")
    print(df["shock_zscore"].mean())

    print("\nRatio shock months:")
    print(df["shock_ratio"].sum())

    print("\nZ-score shock months:")
    print(df["shock_zscore"].sum())

    print("\nOverlap between shock definitions:")
    print(pd.crosstab(df["shock_ratio"], df["shock_zscore"]))

    # save outputs
    df.to_csv(OUTPUT_DATA_PATH, index=False)
    shock_summary.to_csv(OUTPUT_SUMMARY_PATH, index=False)

    print(f"\nSaved shock dataset to: {OUTPUT_DATA_PATH}")
    print(f"Saved trust shock summary to: {OUTPUT_SUMMARY_PATH}")

    # monthly diagnostics
    monthly = (
    df.groupby("month")
        .agg(
            avg_attendances=("att_total", "mean"),
            avg_4hr_performance=("pct_4hr_all", "mean"),
            avg_breach_rate=("four_hour_breach_rate", "mean"),
            shock_ratio_share=("shock_ratio", "mean"),
            shock_zscore_share=("shock_zscore", "mean"),
            avg_wait_12hr_rate=("wait_12hr_rate", "mean"),
            avg_demand_zscore_capped=("demand_zscore_capped", "mean"),
            structural_break_share=("structural_break_flag", "mean"),
        )
        .reset_index()
    )

    plot_monthly(monthly)

    # plot provider with most shocks
    print("\nExtreme z-score months > 10:")
    print(df["structural_break_flag"].sum())

    print("\nProviders with structural break flags:")
    print((df.groupby("provider_code")["structural_break_flag"].sum() > 0).sum())


def plot_monthly(monthly):
    plt.figure(figsize=(10, 5))
    plt.plot(monthly["month"], monthly["avg_attendances"])
    plt.title("Average Attendances per Provider")
    plt.xlabel("Month")
    plt.ylabel("Attendances")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(monthly["month"], monthly["avg_4hr_performance"])
    plt.title("Average 4-Hour Performance")
    plt.xlabel("Month")
    plt.ylabel("4-Hour Performance")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(monthly["month"], monthly["shock_ratio_share"], label="Ratio shock")
    plt.plot(monthly["month"], monthly["shock_zscore_share"], label="Z-score shock")
    plt.title("Share of Providers Experiencing Demand Shock")
    plt.legend()
    plt.xlabel("Month")
    plt.ylabel("Shock Share")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_provider_with_most_shocks(df):
    shock_counts = df.groupby("provider_code")["shock"].sum()

    if shock_counts.max() == 0:
        print("\nNo provider has shock months, skipping provider shock plot.")
        return

    sample_provider = shock_counts.sort_values(ascending=False).index[0]
    sample = df[df["provider_code"] == sample_provider].copy()

    print("\nProvider plotted for shock sanity check:")
    print(sample["provider_code"].iloc[0], "-", sample["provider_name"].iloc[0])
    print("Shock months:", sample["shock"].sum())

    plt.figure(figsize=(10, 5))
    plt.plot(sample["month"], sample["att_total"], label="Actual attendances")
    plt.plot(sample["month"], sample["att_rolling_mean_12m"], label="12m rolling baseline")

    shock_rows = sample[sample["shock"]]

    plt.scatter(
        shock_rows["month"],
        shock_rows["att_total"],
        color="red",
        label="Shock month",
        zorder=5,
    )

    plt.title(f"Demand Shock Check: {sample['provider_name'].iloc[0]}")
    plt.xlabel("Month")
    plt.ylabel("Attendances")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()