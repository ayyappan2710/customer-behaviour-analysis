import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def perform_clustering(customer_summary, n_clusters=3):
    """Segment customers using K-Means."""
    if customer_summary is None or customer_summary.empty:
        return None, None
        
    # Select features for clustering
    features = ['Purchase_Frequency', 'Total_Spending', 'Average_Purchase_Value']
    X = customer_summary[features].copy()
    
    # Handle missing values if any
    X = X.fillna(X.median())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    customer_summary['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Map cluster to segment name based on Total Spending
    cluster_stats = customer_summary.groupby('Cluster')[features].mean()
    
    # Simple dynamic logic: highest spending is High-Value, lowest is Low-Value
    sorted_clusters = cluster_stats['Total_Spending'].sort_values().index.tolist()
    
    segment_mapping = {}
    if n_clusters == 3:
        segment_mapping = {
            sorted_clusters[0]: 'Low-Value Customers',
            sorted_clusters[1]: 'Regular Customers',
            sorted_clusters[2]: 'High-Value Customers'
        }
    else:
        for i, cluster_id in enumerate(sorted_clusters):
            segment_mapping[cluster_id] = f"Segment Level {i+1}"
            
    customer_summary['Customer Segment'] = customer_summary['Cluster'].map(segment_mapping)
    
    return customer_summary, kmeans

def find_optimal_clusters(customer_summary, max_clusters=10, save_path=None):
    """Use elbow method to find optimal clusters."""
    if customer_summary is None or customer_summary.empty:
        return
        
    features = ['Purchase_Frequency', 'Total_Spending', 'Average_Purchase_Value']
    X = customer_summary[features].fillna(customer_summary[features].median())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    wcss = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
        
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
