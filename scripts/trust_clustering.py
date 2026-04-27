import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


OUTPUT_DIR = BASE_DIR.parent
INPUT_PATH = OUTPUT_DIR / "trust_resilience_scores.csv"
OUTPUT_PATH = OUTPUT_DIR / "trust_resilience_clusters.csv"

N_CLUSTERS = 4
RANDOM_STATE = 42

USE_RELIABLE_ONLY = True


def main():
    df = pd.read_csv(INPUT_PATH)

    print("\nLoaded resilience data:")
    print(df.shape)

    if USE_RELIABLE_ONLY:
        df_cluster = df[
            df["evidence_strength"].isin(["medium", "high"])
        ].copy()
    else:
        df_cluster = df.copy()

    print("\nTrusts used for clustering:")
    print(df_cluster.shape[0])

    features = [
        "breach_resilience_score",
        "wait_12hr_resilience_score",
        "mean_demand_zscore_capped",
        "structural_break_share",
        "avg_attendances_during_shock",
        "n_shocks",
    ]

    df_cluster = df_cluster.dropna(subset=features).copy()

    X = df_cluster[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Optional check: compare silhouette scores for 2–6 clusters
    print("\nSilhouette scores:")
    for k in range(2, 7):
        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=20
        )
        labels = model.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        print(f"k={k}: {score:.3f}")

    # Main clustering
    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=20
    )

    df_cluster["cluster"] = kmeans.fit_predict(X_scaled)

    # Cluster profiles
    profile_cols = [
        "resilience_score",
        "breach_resilience_score",
        "wait_12hr_resilience_score",
        "shrunk_breach_impact",
        "shrunk_wait_12hr_impact",
        "mean_demand_zscore",
        "mean_demand_ratio",
        "avg_attendances_during_shock",
        "n_shocks",
        "mean_demand_zscore_capped",
        "structural_break_share",
        "structural_break_months",
    ]

    cluster_profile = (
        df_cluster
        .groupby("cluster")[profile_cols]
        .mean()
        .round(3)
    )

    cluster_sizes = df_cluster["cluster"].value_counts().sort_index()
    cluster_profile.insert(0, "n_trusts", cluster_sizes)

    print("\nCluster profiles:")
    print(cluster_profile)

    # Add simple descriptive labels
    df_cluster = add_cluster_labels(df_cluster, cluster_profile)

    # Save clustered data
    df_cluster.to_csv(OUTPUT_PATH, index=False)

    print("\nSaved clustered trust file to:")
    print(OUTPUT_PATH)

    print("\nCluster labels:")
    print(
        df_cluster[["cluster", "cluster_label"]]
        .drop_duplicates()
        .sort_values("cluster")
    )

    print("\nExample trusts by cluster:")
    for c in sorted(df_cluster["cluster"].unique()):
        print(f"\nCluster {c}:")
        print(
            df_cluster[df_cluster["cluster"] == c][
                [
                    "provider_code",
                    "provider_name",
                    "resilience_score",
                    "breach_resilience_score",
                    "wait_12hr_resilience_score",
                    "mean_demand_zscore",
                    "n_shocks",
                    "cluster_label",
                ]
            ]
            .sort_values("resilience_score", ascending=False)
            .head(10),
            df_cluster[df_cluster["cluster"] == 5][[
                "provider_name",
                "avg_attendances_during_shock",
                "mean_demand_zscore",
                "n_shocks"
            ]]
        )
    df_cluster.groupby("cluster")[
        ["resilience_score", "structural_break_share"]
    ].mean()        
    plot_clusters(df_cluster, X_scaled)


def add_cluster_labels(df_cluster, cluster_profile):
    labels = {}

    for cluster_id, row in cluster_profile.iterrows():
        resilience = row["resilience_score"]
        demand = row["mean_demand_zscore"]
        waits = row["wait_12hr_resilience_score"]

        if resilience >= cluster_profile["resilience_score"].quantile(0.75):
            label = "high resilience"
        elif resilience <= cluster_profile["resilience_score"].quantile(0.25):
            label = "fragile under shock"
        else:
            label = "moderate resilience"

        if demand >= cluster_profile["mean_demand_zscore"].quantile(0.75):
            label += " / high shock intensity"

        if waits <= cluster_profile["wait_12hr_resilience_score"].quantile(0.25):
            label += " / 12h wait pressure"

        labels[cluster_id] = label

    df_cluster["cluster_label"] = df_cluster["cluster"].map(labels)

    return df_cluster


def plot_clusters(df_cluster, X_scaled):
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)

    df_cluster["pca1"] = coords[:, 0]
    df_cluster["pca2"] = coords[:, 1]

    plt.figure(figsize=(8, 6))

    for cluster in sorted(df_cluster["cluster"].unique()):
        subset = df_cluster[df_cluster["cluster"] == cluster]
        plt.scatter(
            subset["pca1"],
            subset["pca2"],
            label=f"Cluster {cluster}",
            alpha=0.75
        )

    plt.title("Trust Resilience Clusters")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    df_cluster.boxplot(
        column="resilience_score",
        by="cluster"
    )
    plt.title("Resilience Score by Cluster")
    plt.suptitle("")
    plt.xlabel("Cluster")
    plt.ylabel("Composite Resilience Score")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

