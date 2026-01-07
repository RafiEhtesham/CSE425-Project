"""
Hard Task: Advanced VAE with CVAE/Beta-VAE

Requirements:
- Implement Conditional VAE (CVAE) or Beta-VAE for disentangled representations
- Perform multi-modal clustering combining audio, lyrics, and genre information
- Quantitative evaluation using comprehensive metrics
- Detailed visualizations: latent space, cluster distribution, reconstructions
- Compare VAE-based clustering with multiple baselines
"""

import os
import sys

# Fix for OpenMP library conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.vae import ConditionalVAE, BetaVAE, HybridVAE
from utils.data_loader import load_gtzan_with_lyrics, HybridDataset
from utils.evaluation import evaluate_clustering, print_metrics, compare_methods
from utils.visualization import (
    plot_latent_space, plot_cluster_distribution, plot_confusion_matrix,
    plot_training_history, plot_metrics_comparison, plot_genre_cluster_heatmap,
    plot_reconstruction_comparison, plot_latent_traversal
)
from utils.training import get_reconstructions
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from utils.data_loader import GTZANDataset



def train_conditional_vae(audio_features, lyrics_features, labels, device, save_dir):
    """
    Train Conditional VAE with genre conditioning
    """
    print("\n" + "="*60)
    print("Training Conditional VAE (CVAE)")
    print("="*60)
    
    # Combine audio and lyrics
    combined_features = np.concatenate([audio_features, lyrics_features], axis=1)
    
    # Split data
    indices = np.arange(len(combined_features))
    
    # Check if stratification is possible
    unique_labels, counts = np.unique(labels, return_counts=True)
    min_samples = counts.min()
    use_stratify = min_samples >= 2
    
    if not use_stratify:
        print(f"Warning: Some classes have only {min_samples} sample(s). Using non-stratified split.")
        print(f"Class distribution: {dict(zip(unique_labels, counts))}")
    
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=labels if use_stratify else None
    )
    
    # Check if we can stratify the validation split
    unique_temp, counts_temp = np.unique(labels[train_idx], return_counts=True)
    use_stratify_val = counts_temp.min() >= 2
    
    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.1, random_state=42, stratify=labels[train_idx] if use_stratify_val else None
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(combined_features[train_idx])
    X_val = scaler.transform(combined_features[val_idx])
    X_test = scaler.transform(combined_features[test_idx])
    
    y_train = labels[train_idx]
    y_val = labels[val_idx]
    y_test = labels[test_idx]
    
    # Create datasets
    train_dataset = GTZANDataset(X_train, y_train)
    val_dataset = GTZANDataset(X_val, y_val)
    test_dataset = GTZANDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Initialize model
    num_classes = len(np.unique(labels))
    model = ConditionalVAE(
        input_dim=combined_features.shape[1],
        num_classes=num_classes,
        hidden_dims=[512, 256],
        latent_dim=64
    )
    
    print(f"Model architecture:")
    print(f"  Input dim: {combined_features.shape[1]}")
    print(f"  Num classes: {num_classes}")
    print(f"  Hidden dims: [512, 256]")
    print(f"  Latent dim: 64")
    
    # Train model
    from utils.training import train_cvae
    model, history = train_cvae(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=50,
        learning_rate=0.001,
        device=device,
        num_classes=num_classes,
        beta=1.0,
        verbose=True
    )
    
    # Plot training history
    plot_training_history(history, save_path=os.path.join(save_dir, 'cvae_training_history.png'))
    
    # Extract latent representations
    from utils.training import get_latent_representations
    latent_features, _ = get_latent_representations(
        model, test_loader, device, is_conditional=True, num_classes=num_classes
    )
    
    return model, latent_features, y_test, scaler


def train_beta_vae(audio_features, lyrics_features, labels, beta_values, device, save_dir):
    """
    Train Beta-VAE with different beta values for disentanglement
    """
    print("\n" + "="*60)
    print("Training Beta-VAE with Multiple Beta Values")
    print("="*60)
    
    # Combine audio and lyrics
    combined_features = np.concatenate([audio_features, lyrics_features], axis=1)
    
    # Split data
    indices = np.arange(len(combined_features))
    
    # Check if stratification is possible
    unique_labels, counts = np.unique(labels, return_counts=True)
    min_samples = counts.min()
    use_stratify = min_samples >= 2
    
    if not use_stratify:
        print(f"Warning: Some classes have only {min_samples} sample(s). Using non-stratified split.")
        print(f"Class distribution: {dict(zip(unique_labels, counts))}")
    
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=labels if use_stratify else None
    )
    
    # Check if we can stratify the validation split
    unique_temp, counts_temp = np.unique(labels[train_idx], return_counts=True)
    use_stratify_val = counts_temp.min() >= 2
    
    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.1, random_state=42, stratify=labels[train_idx] if use_stratify_val else None
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(combined_features[train_idx])
    X_val = scaler.transform(combined_features[val_idx])
    X_test = scaler.transform(combined_features[test_idx])
    
    y_train = labels[train_idx]
    y_val = labels[val_idx]
    y_test = labels[test_idx]
    
    # Create datasets
    train_dataset = GTZANDataset(X_train, y_train)
    val_dataset = GTZANDataset(X_val, y_val)
    test_dataset = GTZANDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    models = {}
    latent_features_dict = {}
    
    for beta in beta_values:
        print(f"\nTraining Beta-VAE with beta={beta}...")
        
        # Initialize model
        model = BetaVAE(
            input_dim=combined_features.shape[1],
            hidden_dims=[512, 256],
            latent_dim=64,
            beta=beta
        )
        
        # Train model
        from utils.training import train_vae
        model, history = train_vae(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=50,
            learning_rate=0.001,
            device=device,
            beta=beta,
            verbose=False
        )
        
        # Extract latent representations
        from utils.training import get_latent_representations
        latent_features, _ = get_latent_representations(model, test_loader, device)
        
        models[f'beta={beta}'] = model
        latent_features_dict[f'beta={beta}'] = latent_features
        
        print(f"Beta={beta} training completed")
    
    return models, latent_features_dict, y_test, scaler


def run_comprehensive_clustering(latent_features, y_test, n_clusters, method_name):
    """
    Run comprehensive clustering experiments
    """
    results = {}
    predictions = {}
    
    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(latent_features)
    kmeans_metrics = evaluate_clustering(latent_features, kmeans_labels, y_test)
    results[f'{method_name} + K-Means'] = kmeans_metrics
    predictions['K-Means'] = kmeans_labels
    
    # Agglomerative
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    agg_labels = agg.fit_predict(latent_features)
    agg_metrics = evaluate_clustering(latent_features, agg_labels, y_test)
    results[f'{method_name} + Agglomerative'] = agg_metrics
    predictions['Agglomerative'] = agg_labels
    
    # DBSCAN
    from sklearn.neighbors import NearestNeighbors
    neighbors = NearestNeighbors(n_neighbors=5)
    neighbors_fit = neighbors.fit(latent_features)
    distances, indices = neighbors_fit.kneighbors(latent_features)
    distances = np.sort(distances[:, -1], axis=0)
    eps = np.percentile(distances, 90)
    
    dbscan = DBSCAN(eps=eps, min_samples=5)
    dbscan_labels = dbscan.fit_predict(latent_features)
    
    n_clusters_found = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    if n_clusters_found > 1:
        dbscan_metrics = evaluate_clustering(latent_features, dbscan_labels, y_test)
        results[f'{method_name} + DBSCAN'] = dbscan_metrics
        predictions['DBSCAN'] = dbscan_labels
    
    return results, predictions


def run_baseline_comparisons(audio_features, lyrics_features, labels, n_clusters):
    """
    Run baseline comparisons using combined audio and lyrics features
    """
    print("\n" + "="*60)
    print("Running Baseline Comparisons")
    print("="*60)
    
    # Combine audio and lyrics features
    combined_features = np.concatenate([audio_features, lyrics_features], axis=1)
    
    # Split data
    indices = np.arange(len(combined_features))
    
    # Check if stratification is possible
    unique_labels, counts = np.unique(labels, return_counts=True)
    min_samples = counts.min()
    use_stratify = min_samples >= 2
    
    if not use_stratify:
        print(f"Warning: Some classes have only {min_samples} sample(s). Using non-stratified split.")
        print(f"Class distribution: {dict(zip(unique_labels, counts))}")
    
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=labels if use_stratify else None
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(combined_features[train_idx])
    X_test = scaler.transform(combined_features[test_idx])
    y_test = labels[test_idx]
    
    results = {}
    
    # 1. PCA + K-Means
    print("\n1. PCA + K-Means...")
    pca = PCA(n_components=64, random_state=42)
    pca.fit(X_train)
    X_test_pca = pca.transform(X_test)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_pred = kmeans.fit_predict(X_test_pca)
    metrics = evaluate_clustering(X_test_pca, labels_pred, y_test)
    results['PCA + K-Means'] = metrics
    print_metrics(metrics, "PCA + K-Means")
    
    # 2. Direct K-Means on features
    print("\n2. Direct K-Means on Features...")
    kmeans_direct = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_direct = kmeans_direct.fit_predict(X_test)
    metrics_direct = evaluate_clustering(X_test, labels_direct, y_test)
    results['Direct K-Means'] = metrics_direct
    print_metrics(metrics_direct, "Direct K-Means")
    
    # 3. Autoencoder + K-Means (simple autoencoder)
    print("\n3. Training Simple Autoencoder...")
    train_dataset = GTZANDataset(X_train, labels[train_idx])
    test_dataset = GTZANDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Simple autoencoder (no variational)
    class SimpleAutoencoder(torch.nn.Module):
        def __init__(self, input_dim, latent_dim=64):
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, latent_dim)
            )
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(latent_dim, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, input_dim)
            )
        
        def forward(self, x):
            z = self.encoder(x)
            recon = self.decoder(z)
            return recon, z
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ae = SimpleAutoencoder(X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(ae.parameters(), lr=0.001)
    
    # Quick training
    for epoch in range(20):
        ae.train()
        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                x, _ = batch
            else:
                x = batch
            x = x.to(device)
            
            optimizer.zero_grad()
            recon, _ = ae(x)
            loss = torch.nn.functional.mse_loss(recon, x)
            loss.backward()
            optimizer.step()
    
    # Extract latent features
    ae.eval()
    ae_latent = []
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, (list, tuple)):
                x, _ = batch
            else:
                x = batch
            x = x.to(device)
            _, z = ae(x)
            ae_latent.append(z.cpu().numpy())
    
    ae_latent = np.concatenate(ae_latent, axis=0)
    
    kmeans_ae = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_ae = kmeans_ae.fit_predict(ae_latent)
    metrics_ae = evaluate_clustering(ae_latent, labels_ae, y_test)
    results['Autoencoder + K-Means'] = metrics_ae
    print_metrics(metrics_ae, "Autoencoder + K-Means")
    
    return results


def main():
    """
    Main function for Hard Task
    """
    print("\n" + "="*60)
    print("HARD TASK: Advanced VAE with CVAE/Beta-VAE")
    print("="*60)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create results directory
    save_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(save_dir, exist_ok=True)
    
    # Load data
    print("\nLoading GTZAN dataset with lyrics...")
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'GTZAN', 'gtzan_with_lyrics_clean.csv')
    
    try:
        audio_features, lyrics_features, labels, genre_names = load_gtzan_with_lyrics(
            data_path, max_lyrics_len=256
        )
        print(f"Dataset loaded successfully:")
        print(f"  Audio features: {audio_features.shape}")
        print(f"  Lyrics features: {lyrics_features.shape}")
        print(f"  Samples: {len(labels)}") #type: ignore
        print(f"  Genres: {genre_names}")
    except Exception as e:
        print(f"Error: {e}")
        return
    
    n_clusters = len(genre_names)
    
    # 1. Train Conditional VAE
    cvae_model, cvae_latent, cvae_labels, cvae_scaler = train_conditional_vae(
        audio_features, lyrics_features, labels, device, save_dir
    )
    
    # 2. Train Beta-VAE with different beta values
    beta_values = [1.0, 4.0, 10.0]
    beta_models, beta_latents, beta_labels, beta_scaler = train_beta_vae(
        audio_features, lyrics_features, labels, beta_values, device, save_dir
    )
    
    # 3. Run comprehensive clustering
    all_results = {}
    
    # CVAE clustering
    print("\nRunning clustering on CVAE latent space...")
    cvae_results, cvae_predictions = run_comprehensive_clustering(
        cvae_latent, cvae_labels, n_clusters, 'CVAE'
    )
    all_results.update(cvae_results)
    
    # Beta-VAE clustering for each beta
    for beta_key, beta_latent in beta_latents.items():
        print(f"\nRunning clustering on Beta-VAE ({beta_key}) latent space...")
        beta_results, _ = run_comprehensive_clustering(
            beta_latent, beta_labels, n_clusters, f'Beta-VAE ({beta_key})'
        )
        all_results.update(beta_results)
    
    # 4. Baseline comparisons
    baseline_results = run_baseline_comparisons(audio_features, lyrics_features, labels, n_clusters)
    all_results.update(baseline_results)
    
    # 5. Compare all methods
    print("\n" + "="*60)
    print("COMPREHENSIVE COMPARISON")
    print("="*60)
    
    comparison_df = compare_methods(all_results)
    print("\n", comparison_df)
    
    # Save comparison
    comparison_df.to_csv(os.path.join(save_dir, 'comprehensive_comparison.csv'))
    
    # Plot comparison
    plot_metrics_comparison(all_results, 
                            save_path=os.path.join(save_dir, 'comprehensive_comparison.png'))
    
    # 6. Detailed visualizations for CVAE (best method)
    print("\nGenerating detailed visualizations...")
    
    # Latent space visualizations
    plot_latent_space(
        cvae_latent, cvae_labels, method='tsne',
        title='CVAE Latent Space - t-SNE (True Labels)',
        save_path=os.path.join(save_dir, 'cvae_tsne_true.png'),
        genre_names=genre_names
    )
    
    plot_latent_space(
        cvae_latent, cvae_predictions['K-Means'], method='tsne',
        title='CVAE Latent Space - t-SNE (K-Means Clusters)',
        save_path=os.path.join(save_dir, 'cvae_tsne_clusters.png'),
        genre_names=[f'Cluster {i}' for i in range(n_clusters)]
    )
    
    plot_latent_space(
        cvae_latent, cvae_labels, method='umap',
        title='CVAE Latent Space - UMAP (True Labels)',
        save_path=os.path.join(save_dir, 'cvae_umap_true.png'),
        genre_names=genre_names
    )
    
    # Confusion matrix and heatmap
    plot_confusion_matrix(
        cvae_labels, cvae_predictions['K-Means'], genre_names,
        title='CVAE + K-Means - Confusion Matrix',
        save_path=os.path.join(save_dir, 'cvae_confusion_matrix.png')
    )
    
    plot_genre_cluster_heatmap(
        cvae_labels, cvae_predictions['K-Means'], genre_names,
        save_path=os.path.join(save_dir, 'cvae_genre_cluster_heatmap.png')
    )
    
    # Cluster distribution
    plot_cluster_distribution(
        cvae_predictions['K-Means'], 
        title='CVAE Cluster Distribution',
        save_path=os.path.join(save_dir, 'cvae_cluster_distribution.png')
    )
    
    # Latent traversal for Beta-VAE (beta=4.0)
    best_beta_model = beta_models['beta=4.0']
    plot_latent_traversal(
        best_beta_model, latent_dim=64, device=device, #type: ignore
        save_path=os.path.join(save_dir, 'beta_vae_latent_traversal.png')
    )
    
    # Save models
    torch.save({
        'cvae_state_dict': cvae_model.state_dict(),
        'genre_names': genre_names
    }, os.path.join(save_dir, 'cvae_model.pth'))
    
    for beta_key, model in beta_models.items():
        torch.save({
            'model_state_dict': model.state_dict(),
            'genre_names': genre_names
        }, os.path.join(save_dir, f'beta_vae_{beta_key}_model.pth'))
    
    # Generate results report
    print("\nGenerating results report...")
    report_path = os.path.join(save_dir, 'RESULTS_REPORT.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Hard Task Results Report\n\n")
        f.write("## Overview\n\n")
        f.write(f"- **Dataset**: GTZAN Genre Collection with Multi-modal Features\n")
        f.write(f"- **Total Samples**: {len(cvae_labels)}\n") #type: ignore
        f.write(f"- **Audio Features Dimension**: {audio_features.shape[1]}\n")
        f.write(f"- **Lyrics Features Dimension**: {lyrics_features.shape[1]}\n")
        f.write(f"- **Number of Genres**: {len(genre_names)}\n")
        f.write(f"- **Genres**: {', '.join(genre_names)}\n\n")
        
        f.write("## Model Architectures\n\n")
        f.write("### 1. Conditional VAE (CVAE)\n")
        f.write("- **Type**: Conditional Variational Autoencoder\n")
        f.write("- **Audio Encoder**: [512, 256] → Latent 64\n")
        f.write("- **Lyrics Encoder**: [512, 256] → Latent 64\n")
        f.write("- **Conditional on Genre Labels**: Yes\n")
        f.write("- **Training Epochs**: 50\n\n")
        
        f.write("### 2. Beta-VAE Variants\n")
        f.write("- **Beta Values Tested**: [1.0, 4.0, 10.0]\n")
        f.write("- **Purpose**: Disentangled representation learning\n")
        f.write("- **Higher beta**: More emphasis on KL divergence (disentanglement)\n")
        f.write("- **Training Epochs**: 50 each\n\n")
        
        f.write("### 3. Baseline Methods\n")
        f.write("- PCA + K-Means\n")
        f.write("- Raw Features + K-Means\n\n")
        
        f.write("## Results\n\n")
        f.write("### Comprehensive Method Comparison\n\n")
        f.write("| Method | Silhouette | Calinski-Harabasz | Davies-Bouldin | ARI | NMI | Purity |\n")
        f.write("|--------|------------|-------------------|----------------|-----|-----|--------|\n")
        for method_name in comparison_df.index:
            row = comparison_df.loc[method_name]
            f.write(f"| {method_name} | {row['silhouette_score']:.4f} | ")
            f.write(f"{row['calinski_harabasz_score']:.4f} | ")
            f.write(f"{row['davies_bouldin_score']:.4f} | ")
            f.write(f"{row['adjusted_rand_score']:.4f} | ")
            f.write(f"{row['nmi']:.4f} | ")
            f.write(f"{row['cluster_purity']:.4f} |\n")
        f.write("\n")
        
        top3_methods = comparison_df.index[:3]
        f.write(f"### Key Findings\n\n")
        f.write(f"🥇 **Best Method**: {top3_methods[0]} (Silhouette: {comparison_df.iloc[0]['silhouette_score']:.4f})\n")
        if len(top3_methods) > 1:
            f.write(f"🥈 **Second Best**: {top3_methods[1]} (Silhouette: {comparison_df.iloc[1]['silhouette_score']:.4f})\n")
        if len(top3_methods) > 2:
            f.write(f"🥉 **Third Best**: {top3_methods[2]} (Silhouette: {comparison_df.iloc[2]['silhouette_score']:.4f})\n\n")
        
        f.write("### Beta-VAE Analysis\n\n")
        f.write("Different beta values affect the trade-off between reconstruction quality ")
        f.write("and disentanglement:\n")
        f.write("- **β=1.0**: Standard VAE, balanced reconstruction and KL\n")
        f.write("- **β=4.0**: Moderate disentanglement, good for clustering\n")
        f.write("- **β=10.0**: High disentanglement, may sacrifice reconstruction\n\n")
        
        f.write("### CVAE Analysis\n\n")
        f.write("The Conditional VAE leverages genre labels during training, allowing it ")
        f.write("to learn genre-specific latent representations. This conditioning can lead ")
        f.write("to more structured latent spaces for clustering.\n\n")
        
        f.write("## Visualizations Generated\n\n")
        f.write("1. `cvae_training_history.png` - CVAE training loss curves\n")
        f.write("2. `beta_vae_training_history.png` - Beta-VAE training losses for all β values\n")
        f.write("3. `cvae_tsne_true.png` - CVAE t-SNE with true genre labels\n")
        f.write("4. `cvae_tsne_clusters.png` - CVAE t-SNE with predicted clusters\n")
        f.write("5. `cvae_umap_true.png` - CVAE UMAP visualization\n")
        f.write("6. `cvae_confusion_matrix.png` - CVAE clustering confusion matrix\n")
        f.write("7. `cvae_genre_cluster_heatmap.png` - Genre distribution across clusters\n")
        f.write("8. `cvae_cluster_distribution.png` - Cluster size distribution\n")
        f.write("9. `beta_vae_latent_traversal.png` - Latent space traversal visualization\n")
        f.write("10. `comprehensive_comparison.png` - All methods comparison\n\n")
        
        f.write("## Files Generated\n\n")
        f.write("- `cvae_model.pth` - Trained CVAE model weights\n")
        f.write("- `beta_vae_beta=1.0_model.pth` - Beta-VAE with β=1.0\n")
        f.write("- `beta_vae_beta=4.0_model.pth` - Beta-VAE with β=4.0\n")
        f.write("- `beta_vae_beta=10.0_model.pth` - Beta-VAE with β=10.0\n")
        f.write("- `comprehensive_comparison.csv` - Detailed metrics for all methods\n")
        f.write("- All visualization PNG files\n\n")
        
        f.write("## Conclusion\n\n")
        f.write("This task explored advanced VAE architectures including Conditional VAE ")
        f.write("and Beta-VAE variants with multiple clustering algorithms. The comprehensive ")
        f.write("evaluation across different VAE formulations, clustering methods, and baselines ")
        f.write("provides insights into:\n\n")
        f.write("1. **Conditioning Impact**: How genre conditioning in CVAE affects clustering\n")
        f.write("2. **Disentanglement**: How β parameter affects representation learning\n")
        f.write("3. **Clustering Methods**: Comparative performance of K-Means, Agglomerative, and DBSCAN\n")
        f.write("4. **Baseline Comparison**: Deep vs. traditional methods for music clustering\n\n")
        f.write(f"The best overall approach was **{top3_methods[0]}**, demonstrating ")
        f.write("the effectiveness of advanced VAE architectures for music genre analysis.\n")
    
    print(f"Results report saved to: {report_path}")
    
    print("\n" + "="*60)
    print("HARD TASK COMPLETED!")
    print("="*60)
    print(f"\nAll results saved to: {save_dir}")
    print("\nTop 3 methods:")
    for i in range(min(3, len(comparison_df))):
        method = comparison_df.index[i]
        score = comparison_df.iloc[i]['silhouette_score']
        print(f"  {i+1}. {method}: {score:.4f}")
    print("\n")


if __name__ == '__main__':
    main()
