"""
segmentation.py — RFM-based K-Means customer segmentation.

Segment labels are assigned dynamically based on cluster centroid ranking
across Monetary, Frequency, and Recency — NOT randomly.

Label map (5 clusters):
  Rank 5 → VIP Customers          (highest spending, most frequent, most recent)
  Rank 4 → Loyal Customers
  Rank 3 → Regular Customers
  Rank 2 → Occasional Customers
  Rank 1 → Low Engagement Customers

With 3 clusters (default):
  Rank 3 → High-Value Customers
  Rank 2 → Regular Customers
  Rank 1 → Low-Value Customers
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from src.utils import safe_empty, build_rfm


_LABEL_MAP_3 = {2: 'High-Value Customers', 1: 'Regular Customers', 0: 'Low-Value Customers'}
_LABEL_MAP_4 = {3: 'VIP Customers', 2: 'Loyal Customers', 1: 'Regular Customers', 0: 'Occasional Customers'}
_LABEL_MAP_5 = {4: 'VIP Customers', 3: 'Loyal Customers', 2: 'Regular Customers',
                1: 'Occasional Customers', 0: 'Low Engagement Customers'}


def perform_clustering(customer_summary: pd.DataFrame, n_clusters: int = 3):
    """
    Segment customers via K-Means on Purchase_Frequency, Total_Spending,
    and Average_Purchase_Value.

    Preserves API compatibility with the existing dashboard.

    Returns (customer_summary_with_segment, kmeans_model)
    """
    if safe_empty(customer_summary):
        return None, None

    features = ['Purchase_Frequency', 'Total_Spending', 'Average_Purchase_Value']
    avail = [f for f in features if f in customer_summary.columns]
    if not avail:
        return customer_summary, None

    X = customer_summary[avail].copy().fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    customer_summary = customer_summary.copy()
    customer_summary['Cluster'] = kmeans.fit_predict(X_scaled)

    # Rank clusters by average Total_Spending → assign meaningful labels
    cluster_rank = (customer_summary.groupby('Cluster')['Total_Spending']
                                    .mean()
                                    .rank(method='first', ascending=True)
                                    .astype(int) - 1)   # 0 = lowest

    label_maps = {3: _LABEL_MAP_3, 4: _LABEL_MAP_4, 5: _LABEL_MAP_5}
    lmap = label_maps.get(n_clusters, None)

    customer_summary['Customer Segment'] = customer_summary['Cluster'].map(
        lambda c: lmap[cluster_rank[c]] if lmap else f"Segment {cluster_rank[c]+1}"
    )

    return customer_summary, kmeans


def perform_rfm_clustering(df: pd.DataFrame, n_clusters: int = 4):
    """
    Full RFM-based segmentation directly from the transaction DataFrame.

    Returns rfm_df with columns: Customer ID, Recency, Frequency, Monetary,
                                  Cluster, Customer Segment, …
    """
    if safe_empty(df):
        return pd.DataFrame(), None

    rfm = build_rfm(df)
    if rfm.empty:
        return rfm, None

    features = ['Recency', 'Frequency', 'Monetary']
    X = rfm[features].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(X_scaled)

    # Rank by Monetary desc (highest spend → highest rank)
    cluster_rank = (rfm.groupby('Cluster')['Monetary']
                       .mean()
                       .rank(method='first', ascending=True)
                       .astype(int) - 1)

    label_maps = {3: _LABEL_MAP_3, 4: _LABEL_MAP_4, 5: _LABEL_MAP_5}
    lmap = label_maps.get(n_clusters, None)

    rfm['Customer Segment'] = rfm['Cluster'].map(
        lambda c: lmap[cluster_rank[c]] if lmap else f"Segment {cluster_rank[c]+1}"
    )

    return rfm, kmeans


def segment_statistics(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate segment-level statistics for display."""
    if safe_empty(rfm_df) or 'Customer Segment' not in rfm_df.columns:
        return pd.DataFrame()

    return (rfm_df.groupby('Customer Segment').agg(
        Customer_Count=('Customer ID', 'count'),
        Avg_Revenue=('Monetary', 'mean'),
        Total_Revenue=('Monetary', 'sum'),
        Avg_Frequency=('Frequency', 'mean'),
        Avg_Recency=('Recency', 'mean'),
    ).round(2).reset_index()
      .sort_values('Total_Revenue', ascending=False))


def find_optimal_clusters(customer_summary: pd.DataFrame, max_clusters: int = 10, save_path=None):
    """Elbow method — kept for notebook/analysis use."""
    if safe_empty(customer_summary):
        return

    features = ['Purchase_Frequency', 'Total_Spending', 'Average_Purchase_Value']
    X = customer_summary[features].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    wcss = []
    for i in range(1, max_clusters + 1):
        km = KMeans(n_clusters=i, random_state=42, n_init=10)
        km.fit(X_scaled)
        wcss.append(km.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_clusters + 1), wcss, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
