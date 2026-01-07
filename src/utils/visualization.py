"""
Visualization utilities
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Try to import umap, set flag if not available
UMAP_AVAILABLE = False
try:
    import umap #type: ignore
    UMAP_AVAILABLE = True
except Exception as e:
    UMAP_AVAILABLE = False
    print("Warning: umap-learn not installed. UMAP visualizations will be disabled.")


def plot_latent_space(latent_features, labels, method='tsne', title='Latent Space Visualization', save_path=None, genre_names=None):
    """
    Visualize latent space using t-SNE or UMAP
    
    Args:
        latent_features: Latent representations
        labels: Cluster or true labels
        method: 'tsne' or 'umap'
        title: Plot title
        save_path: Path to save figure
        genre_names: List of genre names for legend
    """
    plt.figure(figsize=(12, 8))
    
    # Dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        embedded = reducer.fit_transform(latent_features)
    elif method == 'umap':
        if not UMAP_AVAILABLE:
            print(f"Warning: UMAP not available, falling back to t-SNE")
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            embedded = reducer.fit_transform(latent_features)
        else:
            reducer = umap.UMAP(n_components=2, random_state=42)
            embedded = reducer.fit_transform(latent_features)
    elif method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        embedded = reducer.fit_transform(latent_features)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create scatter plot
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels))) #type: ignore
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        label_name = genre_names[label] if genre_names is not None else f'Cluster {label}'
        plt.scatter(embedded[mask, 0], embedded[mask, 1], 
                    c=[colors[i]], label=label_name, alpha=0.6, s=50)
    
    plt.xlabel(f'{method.upper()} Component 1', fontsize=12)
    plt.ylabel(f'{method.upper()} Component 2', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_cluster_distribution(labels, genre_names=None, title='Cluster Distribution', save_path=None):
    """
    Plot cluster distribution as bar chart
    """
    plt.figure(figsize=(10, 6))
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    if genre_names is not None:
        label_names = [genre_names[i] for i in unique_labels]
    else:
        label_names = [f'Cluster {i}' for i in unique_labels]
    
    plt.bar(label_names, counts, color='steelblue', alpha=0.7)
    plt.xlabel('Cluster', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(true_labels, pred_labels, genre_names=None, 
                            title='Confusion Matrix', save_path=None):
    """
    Plot confusion matrix between true and predicted labels
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(true_labels, pred_labels)
    
    plt.figure(figsize=(10, 8))
    
    if genre_names is not None:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=[f'C{i}' for i in range(cm.shape[1])],
                   yticklabels=genre_names)
    else:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    
    plt.xlabel('Predicted Cluster', fontsize=12)
    plt.ylabel('True Genre', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_training_history(history, save_path=None):
    """
    Plot training history (loss curves)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Total loss
    axes[0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Total Loss', fontsize=12)
    axes[0].set_title('Total Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Reconstruction loss
    axes[1].plot(history['train_recon_loss'], label='Train', linewidth=2)
    axes[1].plot(history['val_recon_loss'], label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Reconstruction Loss', fontsize=12)
    axes[1].set_title('Reconstruction Loss', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # KL divergence
    axes[2].plot(history['train_kld'], label='Train', linewidth=2)
    axes[2].plot(history['val_kld'], label='Validation', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('KL Divergence', fontsize=12)
    axes[2].set_title('KL Divergence', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_reconstruction_comparison(original, reconstructed, n_samples=5, save_path=None):
    """
    Plot original vs reconstructed features
    """
    fig, axes = plt.subplots(n_samples, 2, figsize=(12, n_samples * 2))
    
    for i in range(n_samples):
        # Original
        axes[i, 0].plot(original[i], linewidth=1.5, color='steelblue')
        axes[i, 0].set_title(f'Original Sample {i+1}', fontsize=10)
        axes[i, 0].set_ylabel('Feature Value', fontsize=9)
        axes[i, 0].grid(alpha=0.3)
        
        # Reconstructed
        axes[i, 1].plot(reconstructed[i], linewidth=1.5, color='coral')
        axes[i, 1].set_title(f'Reconstructed Sample {i+1}', fontsize=10)
        axes[i, 1].set_ylabel('Feature Value', fontsize=9)
        axes[i, 1].grid(alpha=0.3)
    
    axes[-1, 0].set_xlabel('Feature Index', fontsize=10)
    axes[-1, 1].set_xlabel('Feature Index', fontsize=10)
    
    plt.suptitle('Original vs Reconstructed Features', fontsize=14, fontweight='bold', y=1.001)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_latent_traversal(model, latent_dim, device='cpu', save_path=None):
    """
    Plot latent space traversal for disentanglement analysis
    """
    import torch
    
    n_steps = 10
    traversal_range = (-3, 3)
    
    fig, axes = plt.subplots(latent_dim, n_steps, figsize=(n_steps * 2, latent_dim * 2))
    
    for dim in range(latent_dim):
        z = torch.zeros(n_steps, latent_dim).to(device)
        z[:, dim] = torch.linspace(traversal_range[0], traversal_range[1], n_steps)
        
        with torch.no_grad():
            if hasattr(model, 'decode'):
                recon = model.decode(z).cpu().numpy()
            else:
                recon = model(z).cpu().numpy()
        
        for step in range(n_steps):
            ax = axes[dim, step] if latent_dim > 1 else axes[step]
            ax.plot(recon[step], linewidth=1)
            ax.set_xticks([])
            ax.set_yticks([])
            
            if step == 0:
                ax.set_ylabel(f'Dim {dim}', fontsize=10)
            if dim == 0:
                ax.set_title(f'Step {step+1}', fontsize=9)
    
    plt.suptitle('Latent Space Traversal', fontsize=14, fontweight='bold', y=1.001)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_metrics_comparison(metrics_dict, save_path=None):
    """
    Compare metrics across different methods
    """
    import pandas as pd
    
    df = pd.DataFrame(metrics_dict).T
    
    # Select key metrics
    key_metrics = ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score']
    available_metrics = [m for m in key_metrics if m in df.columns]
    
    if not available_metrics:
        print("No metrics available for comparison")
        return
    
    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(available_metrics):
        df[metric].plot(kind='bar', ax=axes[i], color='steelblue', alpha=0.7)
        axes[i].set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Method', fontsize=10)
        axes[i].set_ylabel('Score', fontsize=10)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_genre_cluster_heatmap(true_labels, pred_labels, genre_names, save_path=None):
    """
    Plot heatmap showing how genres are distributed across clusters
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(true_labels, pred_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=[f'Cluster {i}' for i in range(cm.shape[1])],
                yticklabels=genre_names,
                cbar_kws={'label': 'Proportion'})
    
    plt.xlabel('Predicted Cluster', fontsize=12)
    plt.ylabel('True Genre', fontsize=12)
    plt.title('Genre Distribution Across Clusters', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
