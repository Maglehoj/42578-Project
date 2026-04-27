import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

INPUT_PATH = DATA_DIR / "ae_2017_2019_with_shocks.csv"
OUTPUT_PATH = DATA_DIR / "trust_resilience_scores.csv"

ROLLING_WINDOW = 12
MIN_PERIODS = 6
SHRINKAGE_K = 5

BREACH_WEIGHT = 0.7
WAIT12_WEIGHT = 0.3


def shrink_metric(df, metric_col, count_col="n_shocks"):
    overall = df[metric_col].mean()

    return (
        df[count_col] * df[metric_col]
        + SHRINKAGE_K * overall
    ) / (
        df[count_col] + SHRINKAGE_K
    )


def percentile_score(series):
    # lower impact = better resilience
    return 100 * series.rank(pct=True, ascending=False)


def main():
    df = pd.read_csv(INPUT_PATH)
    df["month"] = pd.to_datetime(df["month"])

    numeric_cols = [
        "att_total",
        "pct_4hr_all",
        "emerg_adm_total",
        "wait_4hr_dta",
        "wait_12hr_dta",
        "four_hour_breach_rate",
        "wait_12hr_rate",
        "demand_ratio",
        "demand_zscore",
        "demand_zscore_capped",
        "structural_break_flag",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["provider_code", "month"]).copy()

    # Defensive metric creation
    if "four_hour_breach_rate" not in df.columns:
        df["four_hour_breach_rate"] = 1 - df["pct_4hr_all"]

    if "wait_12hr_rate" not in df.columns:
        df["wait_12hr_rate"] = df["wait_12hr_dta"] / df["att_total"]

    # Expected baseline: previous 12 months only
    df["expected_breach_rate"] = (
        df.groupby("provider_code")["four_hour_breach_rate"]
        .transform(lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=MIN_PERIODS).mean())
    )

    df["expected_wait_12hr_rate"] = (
        df.groupby("provider_code")["wait_12hr_rate"]
        .transform(lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=MIN_PERIODS).mean())
    )

    # Shock-period deterioration
    df["breach_impact"] = (
        df["four_hour_breach_rate"] - df["expected_breach_rate"]
    )

    df["wait_12hr_impact"] = (
        df["wait_12hr_rate"] - df["expected_wait_12hr_rate"]
    )

    shock_df = df[
        (df["shock"] == True)
        & df["breach_impact"].notna()
        & df["wait_12hr_impact"].notna()
    ].copy()

    if shock_df.empty:
        raise ValueError("No valid shock months found for resilience scoring.")

    trust_resilience = (
        shock_df
        .groupby(["provider_code", "provider_name"])
        .agg(
            n_shocks=("shock", "sum"),

            mean_breach_impact=("breach_impact", "mean"),
            median_breach_impact=("breach_impact", "median"),
            mean_wait_12hr_impact=("wait_12hr_impact", "mean"),
            median_wait_12hr_impact=("wait_12hr_impact", "median"),

            mean_breach_during_shock=("four_hour_breach_rate", "mean"),
            mean_expected_breach=("expected_breach_rate", "mean"),

            mean_wait_12hr_rate_during_shock=("wait_12hr_rate", "mean"),
            mean_expected_wait_12hr_rate=("expected_wait_12hr_rate", "mean"),

            mean_demand_ratio=("demand_ratio", "mean"),
            mean_demand_zscore=("demand_zscore", "mean"),
            avg_attendances_during_shock=("att_total", "mean"),

            mean_demand_zscore_capped=("demand_zscore_capped", "mean"),
            structural_break_months=("structural_break_flag", "sum"),
            structural_break_share=("structural_break_flag", "mean"),
        )
        .reset_index()
    )

    # Shrink noisy trust-level estimates toward overall mean
    trust_resilience["shrunk_breach_impact"] = shrink_metric(
        trust_resilience,
        "mean_breach_impact"
    )

    trust_resilience["shrunk_wait_12hr_impact"] = shrink_metric(
        trust_resilience,
        "mean_wait_12hr_impact"
    )

    # Component scores
    trust_resilience["breach_resilience_score"] = percentile_score(
        trust_resilience["shrunk_breach_impact"]
    )

    trust_resilience["wait_12hr_resilience_score"] = percentile_score(
        trust_resilience["shrunk_wait_12hr_impact"]
    )

    # Composite resilience score
    trust_resilience["resilience_score"] = (
        BREACH_WEIGHT * trust_resilience["breach_resilience_score"]
        + WAIT12_WEIGHT * trust_resilience["wait_12hr_resilience_score"]
    )

    trust_resilience["evidence_strength"] = pd.cut(
        trust_resilience["n_shocks"],
        bins=[0, 2, 5, 100],
        labels=["low", "medium", "high"]
    )

    trust_resilience = trust_resilience.sort_values(
        "resilience_score",
        ascending=False
    )

    trust_resilience.to_csv(OUTPUT_PATH, index=False)

    print("\nSaved resilience scores to:")
    print(OUTPUT_PATH)

    print("\nNumber of trusts scored:")
    print(len(trust_resilience))

    print("\nShock count distribution:")
    print(trust_resilience["n_shocks"].describe())

    print("\nTop resilient trusts:")
    print(trust_resilience.head(15))

    print("\nMost fragile trusts:")
    print(trust_resilience.tail(15))

    reliable = trust_resilience[
        trust_resilience["evidence_strength"].isin(["medium", "high"])
    ].copy()

    print("\nTop resilient trusts, medium/high evidence only:")
    print(reliable.head(15))

    print("\nMost fragile trusts, medium/high evidence only:")
    print(reliable.tail(15))

    plot_outputs(trust_resilience)


def plot_outputs(trust_resilience):
    plt.figure(figsize=(8, 5))
    trust_resilience["resilience_score"].hist(bins=25)
    plt.title("Distribution of Composite Resilience Scores")
    plt.xlabel("Composite Resilience Score")
    plt.ylabel("Number of Trusts")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.scatter(
        trust_resilience["breach_resilience_score"],
        trust_resilience["wait_12hr_resilience_score"],
        alpha=0.7
    )
    plt.title("4-Hour vs 12-Hour Resilience")
    plt.xlabel("4-Hour Breach Resilience Score")
    plt.ylabel("12-Hour Wait Resilience Score")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.scatter(
        trust_resilience["n_shocks"],
        trust_resilience["resilience_score"],
        alpha=0.7
    )
    plt.title("Composite Resilience Score by Number of Shock Months")
    plt.xlabel("Number of Shock Months")
    plt.ylabel("Composite Resilience Score")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()