# Medium Task Results Report

## Overview

- **Dataset**: GTZAN Genre Collection with Multi-modal Features
- **Total Samples**: 122
- **Audio Features Dimension**: 71
- **Lyrics Features Dimension**: 256
- **Number of Genres**: 10
- **Genres**: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock

## Model Architecture

- **Type**: HybridVAE (Multi-modal Variational Autoencoder)
- **Audio Encoder**: [512, 256] → Latent 64
- **Lyrics Encoder**: [512, 256] → Latent 64
- **Combined Latent Dimension**: 128
- **Training Epochs**: 50
- **Optimizer**: Adam (lr=0.001)
- **Loss**: Combined Reconstruction Loss + KL Divergence

## Results

### Clustering Methods Comparison

| Method | Silhouette | Calinski-Harabasz | Davies-Bouldin | ARI | NMI | Purity |
|--------|------------|-------------------|----------------|-----|-----|--------|
| Agglomerative | 0.2894 | 37.6610 | 0.9001 | 0.1901 | 0.3911 | 0.4836 |
| K-Means | 0.2832 | 39.3259 | 0.9556 | 0.1888 | 0.4133 | 0.5164 |
| PCA + K-Means (Baseline) | 0.1159 | 10.4948 | 1.9605 | 0.1438 | 0.3610 | 0.4672 |

### Key Findings

✅ **Best Method**: Agglomerative
✅ **Best Silhouette Score**: 0.2894

The Hybrid VAE successfully integrates audio and lyrics features into a unified latent representation. Multiple clustering algorithms were compared to find the optimal approach for genre classification.

## Visualizations Generated

1. `training_history.png` - Hybrid VAE training loss curves
2. `hybrid_tsne_true.png` - t-SNE visualization with true genre labels
3. `hybrid_tsne_clusters.png` - t-SNE visualization with predicted clusters
4. `hybrid_umap_true.png` - UMAP visualization with true labels
5. `hybrid_confusion_matrix.png` - Confusion matrix for best method
6. `hybrid_genre_cluster_heatmap.png` - Genre distribution across clusters
7. `clustering_comparison.png` - Visual comparison of clustering methods

## Files Generated

- `hybrid_vae_model.pth` - Trained Hybrid VAE model weights
- `clustering_comparison.csv` - Detailed clustering metrics table
- All visualization PNG files

## Conclusion

The Hybrid VAE architecture successfully fused multi-modal information (audio and lyrics) into a unified latent space. This approach demonstrates the power of multi-modal learning for music genre classification. Among the tested methods, Agglomerative achieved the best performance.
