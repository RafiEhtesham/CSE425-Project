"""
Evaluation metrics for clustering
"""

import numpy as np
from sklearn.metrics import (
    silhouette_score, 
    calinski_harabasz_score, 
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)


def compute_silhouette_score(features, labels):
    """
    Compute Silhouette Score
    
    Measures how similar an object is to its own cluster compared to other clusters.
    Range: [-1, 1], higher is better
    """
    if len(np.unique(labels)) < 2:
        return 0.0
    return silhouette_score(features, labels)


def compute_calinski_harabasz_score(features, labels):
    """
    Compute Calinski-Harabasz Index
    
    Ratio of between-cluster variance to within-cluster variance.
    Higher is better
    """
    if len(np.unique(labels)) < 2:
        return 0.0
    return calinski_harabasz_score(features, labels)


def compute_davies_bouldin_score(features, labels):
    """
    Compute Davies-Bouldin Index
    
    Average similarity between each cluster and its most similar cluster.
    Lower is better
    """
    if len(np.unique(labels)) < 2:
        return float('inf')
    return davies_bouldin_score(features, labels)


def compute_adjusted_rand_score(true_labels, pred_labels):
    """
    Compute Adjusted Rand Index (ARI)
    
    Measures similarity between predicted clusters and ground truth labels.
    Range: [-1, 1], higher is better (1 = perfect match, 0 = random)
    """
    return adjusted_rand_score(true_labels, pred_labels)


def compute_nmi(true_labels, pred_labels):
    """
    Compute Normalized Mutual Information (NMI)
    
    Measures mutual information between predicted clusters and true labels.
    Range: [0, 1], higher is better
    """
    return normalized_mutual_info_score(true_labels, pred_labels)


def compute_cluster_purity(true_labels, pred_labels):
    """
    Compute Cluster Purity
    
    Fraction of the dominant class in each cluster.
    Range: [0, 1], higher is better
    """
    # Create contingency matrix
    n = len(true_labels)
    clusters = np.unique(pred_labels)
    classes = np.unique(true_labels)
    
    purity_sum = 0
    for cluster in clusters:
        cluster_mask = pred_labels == cluster
        cluster_true_labels = true_labels[cluster_mask]
        
        if len(cluster_true_labels) == 0:
            continue
        
        # Find dominant class in this cluster
        max_count = 0
        for class_label in classes:
            count = np.sum(cluster_true_labels == class_label)
            max_count = max(max_count, count)
        
        purity_sum += max_count
    
    return purity_sum / n


def evaluate_clustering(features, pred_labels, true_labels=None):
    """
    Compute all clustering evaluation metrics
    
    Args:
        features: Feature matrix
        pred_labels: Predicted cluster labels
        true_labels: Ground truth labels (optional)
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    metrics = {}
    
    # Unsupervised metrics (don't require true labels)
    try:
        metrics['silhouette_score'] = compute_silhouette_score(features, pred_labels)
    except:
        metrics['silhouette_score'] = 0.0
    
    try:
        metrics['calinski_harabasz_score'] = compute_calinski_harabasz_score(features, pred_labels)
    except:
        metrics['calinski_harabasz_score'] = 0.0
    
    try:
        metrics['davies_bouldin_score'] = compute_davies_bouldin_score(features, pred_labels)
    except:
        metrics['davies_bouldin_score'] = float('inf')
    
    # Supervised metrics (require true labels)
    if true_labels is not None:
        try:
            metrics['adjusted_rand_score'] = compute_adjusted_rand_score(true_labels, pred_labels)
        except:
            metrics['adjusted_rand_score'] = 0.0
        
        try:
            metrics['nmi'] = compute_nmi(true_labels, pred_labels)
        except:
            metrics['nmi'] = 0.0
        
        try:
            metrics['cluster_purity'] = compute_cluster_purity(true_labels, pred_labels)
        except:
            metrics['cluster_purity'] = 0.0
    
    return metrics


def print_metrics(metrics, title="Clustering Metrics"):
    """
    Print clustering metrics in a formatted way
    """
    print(f"\n{title}")
    print("=" * 60)
    
    if 'silhouette_score' in metrics:
        print(f"Silhouette Score:          {metrics['silhouette_score']:.4f} (higher is better)")
    
    if 'calinski_harabasz_score' in metrics:
        print(f"Calinski-Harabasz Index:   {metrics['calinski_harabasz_score']:.4f} (higher is better)")
    
    if 'davies_bouldin_score' in metrics:
        print(f"Davies-Bouldin Index:      {metrics['davies_bouldin_score']:.4f} (lower is better)")
    
    if 'adjusted_rand_score' in metrics:
        print(f"Adjusted Rand Index (ARI): {metrics['adjusted_rand_score']:.4f} (higher is better)")
    
    if 'nmi' in metrics:
        print(f"Normalized Mutual Info:    {metrics['nmi']:.4f} (higher is better)")
    
    if 'cluster_purity' in metrics:
        print(f"Cluster Purity:            {metrics['cluster_purity']:.4f} (higher is better)")
    
    print("=" * 60)


def compare_methods(results_dict):
    """
    Compare clustering results from different methods
    
    Args:
        results_dict: Dictionary of {method_name: metrics}
    
    Returns:
        comparison_df: DataFrame with comparison
    """
    import pandas as pd
    
    df = pd.DataFrame(results_dict).T
    
    # Sort by silhouette score (if available)
    if 'silhouette_score' in df.columns:
        df = df.sort_values('silhouette_score', ascending=False)
    
    return df
