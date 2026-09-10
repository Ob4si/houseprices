"""
Part 3: Customer Segmentation
--------------------------------------------------------------------
Model: K-Means Clustering (scikit-learn)

DATA SOURCE NOTE:
The dataset below is SYNTHETICALLY GENERATED (n=200) using numpy, built
around three plausible underlying customer archetypes (budget-conscious,
average, and high-spending/frequent buyers) with random noise layered on
top -- this is a common approach for demonstrating segmentation when a
specific real dataset (e.g., the "Mall Customer Segmentation" Kaggle
dataset) isn't provided. If your instructor's announcement points to a
specific real dataset, swap `generate_customer_data()` for:
    df = pd.read_csv("your_downloaded_file.csv")

Improvements over generic starter code:
  - 200 records instead of a handful, built from 3 latent archetypes + noise
    so the elbow plot and clusters show a genuine "elbow" instead of an
    assumed K=3
  - Reports silhouette score in addition to inertia to justify chosen K
  - Cluster analysis includes per-cluster size and rounded, human-readable stats
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless rendering for saving to file
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42


def generate_customer_data(n_records: int = 200) -> pd.DataFrame:
    """Generate a realistic synthetic customer dataset with 3 latent segments."""
    rng = np.random.default_rng(RANDOM_SEED)
    regions = ["North", "South", "East", "West"]

    # Archetype centers: (annual_spending, purchase_frequency, age)
    archetypes = {
        "budget": {"spending": 1200, "frequency": 4, "age": 35, "weight": 0.40},
        "average": {"spending": 4500, "frequency": 12, "age": 42, "weight": 0.35},
        "high_value": {"spending": 12000, "frequency": 28, "age": 38, "weight": 0.25},
    }

    rows = []
    archetype_names = list(archetypes.keys())
    weights = [archetypes[a]["weight"] for a in archetype_names]
    assignments = rng.choice(archetype_names, size=n_records, p=weights)

    for archetype in assignments:
        a = archetypes[archetype]
        spending = max(100, rng.normal(a["spending"], a["spending"] * 0.25))
        frequency = max(1, rng.normal(a["frequency"], a["frequency"] * 0.3))
        age = int(np.clip(rng.normal(a["age"], 10), 18, 75))
        region = rng.choice(regions)
        rows.append(
            {
                "annual_spending": round(spending, 2),
                "purchase_frequency": round(frequency, 1),
                "age": age,
                "region": region,
            }
        )

    return pd.DataFrame(rows)


def main():
    df = generate_customer_data(200)
    print(f"Dataset shape: {df.shape}")
    print(df.head(), "\n")

    features = ["annual_spending", "purchase_frequency", "age"]
    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow method: inertia for K=1..8, plus silhouette score for K=2..8
    inertias = []
    silhouette_scores = {}
    k_range = range(1, 9)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        if k >= 2:
            silhouette_scores[k] = silhouette_score(X_scaled, labels)

    print("Silhouette scores by K:")
    for k, score in silhouette_scores.items():
        print(f"  K={k}: {score:.3f}")

    best_k = max(silhouette_scores, key=silhouette_scores.get)
    print(f"\nBest K by silhouette score: {best_k}\n")

    plt.figure(figsize=(7, 5))
    plt.plot(list(k_range), inertias, marker="o")
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal K")
    plt.axvline(x=best_k, color="red", linestyle="--", label=f"Chosen K={best_k}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("elbow_plot.png", dpi=120)
    print("Saved elbow_plot.png")

    # Final clustering with chosen K
    final_km = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=10)
    df["cluster"] = final_km.fit_predict(X_scaled)

    print(f"\nCluster analysis (K={best_k}):")
    cluster_summary = df.groupby("cluster")[features].mean().round(1)
    cluster_summary["count"] = df.groupby("cluster").size()
    print(cluster_summary, "\n")

    for cluster_id, row in cluster_summary.iterrows():
        if row["annual_spending"] > 8000:
            strategy = "High-value segment -> exclusive promotions, loyalty perks, early access to new products"
        elif row["annual_spending"] > 3000:
            strategy = "Mid-value segment -> upsell bundles, moderate discounts to increase frequency"
        else:
            strategy = "Budget-conscious segment -> value deals, free-shipping thresholds, entry-level offers"
        print(f"Cluster {cluster_id} (n={int(row['count'])}): {strategy}")

    df.to_csv("customer_segments.csv", index=False)
    print("\nSaved customer_segments.csv")


if __name__ == "__main__":
    main()
