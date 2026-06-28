import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


INPUT_PATH = "data/forbes_global_2000_2026_scored.csv"
OUTPUT_PATH = "data/forbes_global_2000_2026_clustered.csv"
CHART_DIR = "outputs/charts"


def prepare_clustering_data(df: pd.DataFrame):
    features = [
        "sales_b",
        "profit_b",
        "assets_b",
        "market_value_b",
        "profit_margin",
        "return_on_assets",
        "asset_turnover",
        "market_to_sales",
        "ai_company_strength_score"
    ]

    X = df[features].copy()
    X = X.replace([float("inf"), float("-inf")], pd.NA)
    X = X.fillna(X.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, features


def run_kmeans_clustering(df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
    df = df.copy()

    X_scaled, features = prepare_clustering_data(df)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    df["company_cluster"] = kmeans.fit_predict(X_scaled)
    cluster_labels = {
    0: "Global Scale Giants & Financial Leaders",
    1: "Mega-Cap AI / Platform Leaders",
    2: "Operating Scale / Consumer & Industrial Firms",
    3: "Asset-Heavy Financial & Infrastructure Base",
    4: "High-Margin Innovation Companies"
}

    df["cluster_label"] = df["company_cluster"].map(cluster_labels)


    return df, X_scaled


def create_pca_chart(df: pd.DataFrame, X_scaled):
    os.makedirs(CHART_DIR, exist_ok=True)

    pca = PCA(n_components=2, random_state=42)
    pca_values = pca.fit_transform(X_scaled)

    df["pca_1"] = pca_values[:, 0]
    df["pca_2"] = pca_values[:, 1]

    plt.figure(figsize=(10, 7))

    for cluster in sorted(df["company_cluster"].unique()):
        cluster_df = df[df["company_cluster"] == cluster]
        plt.scatter(
            cluster_df["pca_1"],
            cluster_df["pca_2"],
            label=f"Cluster {cluster}",
            alpha=0.7
        )

    plt.title("Company Clusters Using PCA")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.tight_layout()

    output_path = f"{CHART_DIR}/company_clusters_pca.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    return df


def summarize_clusters(df: pd.DataFrame):
    summary = (
        df.groupby("company_cluster")
        .agg(
            company_count=("company", "count"),
            avg_sales_b=("sales_b", "mean"),
            avg_profit_b=("profit_b", "mean"),
            avg_assets_b=("assets_b", "mean"),
            avg_market_value_b=("market_value_b", "mean"),
            avg_profit_margin=("profit_margin", "mean"),
            avg_ai_score=("ai_company_strength_score", "mean")
        )
        .sort_values("avg_ai_score", ascending=False)
    )

    print("\nCluster Summary:")
    print(summary.to_string())

    print("\nTop industries by cluster:")
    for cluster in sorted(df["company_cluster"].unique()):
        print(f"\nCluster {cluster}:")
        print(df[df["company_cluster"] == cluster]["industry"].value_counts().head(5).to_string())

    print("\nTop companies by cluster:")
    for cluster in sorted(df["company_cluster"].unique()):
        print(f"\nCluster {cluster}:")
        print(
            df[df["company_cluster"] == cluster]
            .sort_values("ai_company_strength_score", ascending=False)
            [["company", "country", "industry", "ai_company_strength_score"]]
            .head(5)
            .to_string(index=False)
        )


def main():
    df = pd.read_csv(INPUT_PATH)

    clustered_df, X_scaled = run_kmeans_clustering(df, n_clusters=5)
    clustered_df = create_pca_chart(clustered_df, X_scaled)

    clustered_df.to_csv(OUTPUT_PATH, index=False)

    print("\nClustering completed successfully!")
    print(f"Saved clustered data to: {OUTPUT_PATH}")
    print(f"Saved PCA chart to: {CHART_DIR}/company_clusters_pca.png")

    summarize_clusters(clustered_df)


if __name__ == "__main__":
    main()