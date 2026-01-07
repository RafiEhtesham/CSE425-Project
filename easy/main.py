"""
Easy Task: Basic VAE for Music Clustering

Requirements:
- Implement a basic VAE for feature extraction from music data
- Use GTZAN hybrid language music dataset
- Perform clustering using K-Means on latent features
- Visualize clusters using t-SNE or UMAP
- Compare with baseline (PCA + K-Means) using metrics
"""

import os
import sys

# Fix for OpenMP library conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.vae import BasicVAE
from utils.data_loader import load_gtzan_csv, prepare_data, create_data_loaders
from utils.training import train_vae, get_latent_representations
from utils.evaluation import evaluate_clustering, print_metrics, compare_methods
from utils.visualization import (
    plot_latent_space, plot_cluster_distribution, plot_confusion_matrix,
    plot_training_history, plot_metrics_comparison, plot_genre_cluster_heatmap
)


def run_baseline_pca_kmeans(X_train, X_test, y_test, n_clusters, genre_names):
    """
    Baseline: PCA + K-Means clustering
    """
    print("\n" + "="*60)
    print("Running Baseline: PCA + K-Means")
    print("="*60)
    
    # Apply PCA
    pca = PCA(n_components=64, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    
    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred_labels = kmeans.fit_predict(X_test_pca)
    
    # Evaluate
    metrics = evaluate_clustering(X_test_pca, pred_labels, y_test)
    print_metrics(metrics, "PCA + K-Means Results")
    
    return pred_labels, X_test_pca, metrics


def run_vae_clustering(X_train, X_val, X_test, y_train, y_val, y_test, 
                       n_clusters, genre_names, device, save_dir):
    """
    VAE-based clustering
    """
    print("\n" + "="*60)
    print("Running VAE-based Clustering")
    print("="*60)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, X_val, X_test, y_train, y_val, y_test, batch_size=16
    )
    
    # Initialize model
    input_dim = X_train.shape[1]
    model = BasicVAE(
        input_dim=input_dim,
        hidden_dims=[512, 256, 128],
        latent_dim=128,
        dropout=0.3
    )
    
    print(f"Model architecture:")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dims: [512, 256, 128]")
    print(f"  Latent dim: 128")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\nTraining VAE...")
    model, history = train_vae(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=150,
        learning_rate=0.0005,
        device=device,
        beta=1.0,
        verbose=True
    )
    
    # Plot training history
    plot_training_history(history, save_path=os.path.join(save_dir, 'training_history.png'))
    
    # Extract latent representations
    print("\nExtracting latent representations...")
    latent_features, _ = get_latent_representations(model, test_loader, device)
    
    # K-Means clustering on latent features
    print("\nPerforming K-Means clustering on latent features...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred_labels = kmeans.fit_predict(latent_features)
    
    # Evaluate
    metrics = evaluate_clustering(latent_features, pred_labels, y_test)
    print_metrics(metrics, "VAE + K-Means Results")
    
    return pred_labels, latent_features, metrics, model


def main():
    """
    Main function for Easy Task
    """
    print("\n" + "="*60)
    print("EASY TASK: Basic VAE for Music Clustering")
    print("="*60)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create results directory
    save_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(save_dir, exist_ok=True)
    
    # Load data
    print("\nLoading GTZAN dataset...")
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'GTZAN', 'gtzan_with_lyrics_clean.csv')
    audio_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'GTZAN', 'genres_original')
    
    # Try to load from CSV and extract features from audio files
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded dataset with {len(df)} samples")
        
        # Check if we need to extract features from audio files
        print("Extracting audio features from WAV files...")
        from utils.data_loader import extract_audio_features
        
        features_list = []
        labels_list = []
        valid_samples = 0
        
        for idx, row in df.iterrows():
            audio_file = row['ref']
            genre = row['genre']
            audio_path = os.path.join(audio_dir, genre, audio_file)
            
            if os.path.exists(audio_path):
                try:
                    # Extract features
                    audio_features = extract_audio_features(audio_path)
                    features_list.append(audio_features)
                    labels_list.append(genre)
                    valid_samples += 1
                    
                    if valid_samples % 100 == 0:
                        print(f"Processed {valid_samples} files...")
                        
                except Exception as e:
                    # Skip files with errors
                    continue
            
            # Limit to first 1000 samples for faster processing
            if valid_samples >= 1000:
                break
        
        if len(features_list) == 0:
            print("Error: No audio features extracted. Check if audio files exist.")
            print(f"Looking for audio files in: {audio_dir}")
            return
        
        features = np.array(features_list)
        
        # Extract labels
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels_list)
        genre_names = label_encoder.classes_
        
        print(f"Successfully extracted features from {len(features)} audio files")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please ensure the GTZAN dataset is properly prepared.")
        import traceback
        traceback.print_exc()
        return
    
    print(f"Features shape: {features.shape}")
    print(f"Number of genres: {len(genre_names)}")
    print(f"Genres: {genre_names}")
    
    # Prepare data
    print("\nPreparing data...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(
        features, labels, test_size=0.2, val_size=0.1, random_state=42
    )
    
    print(f"Train set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    n_clusters = len(genre_names)
    
    # 1. Baseline: PCA + K-Means
    baseline_labels, baseline_features, baseline_metrics = run_baseline_pca_kmeans(
        X_train, X_test, y_test, n_clusters, genre_names
    )
    
    # 2. VAE + K-Means
    vae_labels, vae_features, vae_metrics, model = run_vae_clustering(
        X_train, X_val, X_test, y_train, y_val, y_test,
        n_clusters, genre_names, device, save_dir
    )
    
    # Compare methods
    print("\n" + "="*60)
    print("COMPARISON OF METHODS")
    print("="*60)
    
    results = {
        'PCA + K-Means': baseline_metrics,
        'VAE + K-Means': vae_metrics
    }
    
    comparison_df = compare_methods(results)
    print("\n", comparison_df)
    
    # Save comparison
    comparison_df.to_csv(os.path.join(save_dir, 'metrics_comparison.csv'))
    
    # Plot metrics comparison
    plot_metrics_comparison(results, save_path=os.path.join(save_dir, 'metrics_comparison.png'))
    
    # Visualizations
    print("\nGenerating visualizations...")
    
    # t-SNE visualization for baseline
    plot_latent_space(
        baseline_features, baseline_labels, method='tsne',
        title='Baseline (PCA + K-Means) - t-SNE Visualization',
        save_path=os.path.join(save_dir, 'baseline_tsne_clusters.png'),
        genre_names=[f'Cluster {i}' for i in range(n_clusters)]
    )
    
    # t-SNE visualization for VAE with true labels
    plot_latent_space(
        vae_features, y_test, method='tsne',
        title='VAE Latent Space - t-SNE (True Labels)',
        save_path=os.path.join(save_dir, 'vae_tsne_true.png'),
        genre_names=genre_names
    )
    
    # t-SNE visualization for VAE with predicted clusters
    plot_latent_space(
        vae_features, vae_labels, method='tsne',
        title='VAE Latent Space - t-SNE (Predicted Clusters)',
        save_path=os.path.join(save_dir, 'vae_tsne_clusters.png'),
        genre_names=[f'Cluster {i}' for i in range(n_clusters)]
    )
    
    # UMAP visualization for VAE
    plot_latent_space(
        vae_features, y_test, method='umap',
        title='VAE Latent Space - UMAP (True Labels)',
        save_path=os.path.join(save_dir, 'vae_umap_true.png'),
        genre_names=genre_names
    )
    
    # Cluster distribution
    plot_cluster_distribution(
        vae_labels, title='VAE Cluster Distribution',
        save_path=os.path.join(save_dir, 'cluster_distribution.png')
    )
    
    # Confusion matrices
    plot_confusion_matrix(
        y_test, baseline_labels, genre_names,
        title='Baseline (PCA + K-Means) - Confusion Matrix',
        save_path=os.path.join(save_dir, 'baseline_confusion_matrix.png')
    )
    
    plot_confusion_matrix(
        y_test, vae_labels, genre_names,
        title='VAE + K-Means - Confusion Matrix',
        save_path=os.path.join(save_dir, 'vae_confusion_matrix.png')
    )
    
    # Genre-cluster heatmap
    plot_genre_cluster_heatmap(
        y_test, vae_labels, genre_names,
        save_path=os.path.join(save_dir, 'genre_cluster_heatmap.png')
    )
    
    # Save model
    model_path = os.path.join(save_dir, 'vae_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'genre_names': genre_names
    }, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Generate results report
    print("\nGenerating results report...")
    report_path = os.path.join(save_dir, 'RESULTS_REPORT.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Easy Task Results Report\n\n")
        f.write("## Overview\n\n")
        f.write(f"- **Dataset**: GTZAN Genre Collection\n")
        f.write(f"- **Total Samples**: {len(features)}\n")
        f.write(f"- **Features Dimension**: {features.shape[1]}\n")
        f.write(f"- **Number of Genres**: {len(genre_names)}\n")
        f.write(f"- **Genres**: {', '.join(genre_names)}\n\n")
        
        f.write("## Model Architecture\n\n")
        f.write("- **Type**: BasicVAE (Variational Autoencoder)\n")
        f.write("- **Input Dimension**: 71 (audio features)\n")
        f.write("- **Hidden Layers**: [512, 256]\n")
        f.write("- **Latent Dimension**: 64\n")
        f.write("- **Training Epochs**: 50\n")
        f.write("- **Optimizer**: Adam (lr=0.001)\n")
        f.write("- **Loss**: Reconstruction Loss + KL Divergence (β=1.0)\n\n")
        
        f.write("## Results\n\n")
        f.write("### Clustering Performance\n\n")
        f.write("| Method | Silhouette | Calinski-Harabasz | Davies-Bouldin | ARI | NMI | Purity |\n")
        f.write("|--------|------------|-------------------|----------------|-----|-----|--------|\n")
        f.write(f"| PCA + K-Means | {baseline_metrics.get('silhouette_score', 0):.4f} | ")
        f.write(f"{baseline_metrics.get('calinski_harabasz_score', 0):.4f} | ")
        f.write(f"{baseline_metrics.get('davies_bouldin_score', 0):.4f} | ")
        f.write(f"{baseline_metrics.get('adjusted_rand_score', 0):.4f} | ")
        f.write(f"{baseline_metrics.get('nmi', 0):.4f} | ")
        f.write(f"{baseline_metrics.get('cluster_purity', 0):.4f} |\n")
        f.write(f"| VAE + K-Means | {vae_metrics.get('silhouette_score', 0):.4f} | ")
        f.write(f"{vae_metrics.get('calinski_harabasz_score', 0):.4f} | ")
        f.write(f"{vae_metrics.get('davies_bouldin_score', 0):.4f} | ")
        f.write(f"{vae_metrics.get('adjusted_rand_score', 0):.4f} | ")
        f.write(f"{vae_metrics.get('nmi', 0):.4f} | ")
        f.write(f"{vae_metrics.get('cluster_purity', 0):.4f} |\n\n")
        
        if vae_metrics['silhouette_score'] > baseline_metrics['silhouette_score']:
            improvement = ((vae_metrics['silhouette_score'] - baseline_metrics['silhouette_score']) / 
                          baseline_metrics['silhouette_score'] * 100)
            f.write(f"### Key Findings\n\n")
            f.write(f"✅ **VAE shows {improvement:.2f}% improvement over PCA baseline!**\n\n")
        else:
            f.write(f"### Key Findings\n\n")
            f.write(f"⚠️ PCA baseline performs better than VAE on this dataset.\n\n")
        
        f.write("## Visualizations Generated\n\n")
        f.write("1. `training_history.png` - VAE training loss curves\n")
        f.write("2. `vae_tsne_true.png` - t-SNE visualization with true genre labels\n")
        f.write("3. `vae_tsne_clusters.png` - t-SNE visualization with predicted clusters\n")
        f.write("4. `vae_umap_true.png` - UMAP visualization with true labels\n")
        f.write("5. `baseline_tsne_clusters.png` - Baseline PCA clustering visualization\n")
        f.write("6. `vae_confusion_matrix.png` - Confusion matrix for VAE clustering\n")
        f.write("7. `baseline_confusion_matrix.png` - Confusion matrix for baseline\n")
        f.write("8. `genre_cluster_heatmap.png` - Genre distribution across clusters\n")
        f.write("9. `cluster_distribution.png` - Cluster size distribution\n")
        f.write("10. `metrics_comparison.png` - Visual comparison of methods\n\n")
        
        f.write("## Files Generated\n\n")
        f.write("- `vae_model.pth` - Trained VAE model weights\n")
        f.write("- `metrics_comparison.csv` - Detailed metrics table\n")
        f.write("- All visualization PNG files\n\n")
        
        f.write("## Conclusion\n\n")
        f.write("The Basic VAE successfully learned latent representations of music features. ")
        f.write("The model was trained for 50 epochs and used for unsupervised clustering. ")
        if vae_metrics['silhouette_score'] > baseline_metrics['silhouette_score']:
            f.write("The VAE-based approach outperformed the PCA baseline, demonstrating ")
            f.write("the effectiveness of deep generative models for music feature learning.\n")
        else:
            f.write("While the VAE learned meaningful representations, the simple PCA baseline ")
            f.write("achieved better clustering performance on this particular dataset.\n")
    
    print(f"Results report saved to: {report_path}")
    
    print("\n" + "="*60)
    print("EASY TASK COMPLETED!")
    print("="*60)
    print(f"\nAll results saved to: {save_dir}")
    print("\nKey findings:")
    print(f"  - PCA + K-Means Silhouette Score: {baseline_metrics['silhouette_score']:.4f}")
    print(f"  - VAE + K-Means Silhouette Score: {vae_metrics['silhouette_score']:.4f}")
    
    if vae_metrics['silhouette_score'] > baseline_metrics['silhouette_score']:
        improvement = ((vae_metrics['silhouette_score'] - baseline_metrics['silhouette_score']) / 
                      baseline_metrics['silhouette_score'] * 100)
        print(f"  - VAE shows {improvement:.2f}% improvement over baseline!")
    
    print("\n")


if __name__ == '__main__':
    main()
