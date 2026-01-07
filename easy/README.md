# Easy Task: Basic VAE for Music Clustering

## Overview

This task implements a basic Variational Autoencoder (VAE) for music genre clustering using the GTZAN dataset.

## Requirements

- Implement a basic VAE for feature extraction from music data
- Use GTZAN hybrid language music dataset
- Perform clustering using K-Means on latent features
- Visualize clusters using t-SNE and UMAP
- Compare with baseline (PCA + K-Means) using metrics

## Running the Task

```bash
python main.py
```

## What It Does

1. **Loads Data**: Loads GTZAN dataset with audio features and lyrics
2. **Trains VAE**: Trains a basic VAE (512 → 256 → 64 latent dimensions)
3. **Extracts Features**: Extracts latent representations from trained VAE
4. **Clustering**: Performs K-Means clustering on latent features
5. **Baseline**: Compares with PCA + K-Means baseline
6. **Evaluation**: Computes clustering metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin, ARI, NMI, Purity)
7. **Visualization**: Creates t-SNE, UMAP plots, confusion matrices, and comparisons

## Outputs

Results are saved to `results/` directory:

- `vae_model.pth` - Trained VAE model
- `training_history.png` - Training loss curves
- `vae_tsne_true.png` - t-SNE visualization with true labels
- `vae_tsne_clusters.png` - t-SNE visualization with predicted clusters
- `vae_umap_true.png` - UMAP visualization with true labels
- `baseline_tsne_clusters.png` - Baseline t-SNE visualization
- `vae_confusion_matrix.png` - Confusion matrix
- `baseline_confusion_matrix.png` - Baseline confusion matrix
- `genre_cluster_heatmap.png` - Genre distribution across clusters
- `cluster_distribution.png` - Cluster size distribution
- `metrics_comparison.csv` - Metrics comparison table
- `metrics_comparison.png` - Metrics comparison chart
- **`RESULTS_REPORT.md`** - Comprehensive results summary with findings and analysis

## Results Report

After running the task, a detailed results report (`RESULTS_REPORT.md`) is automatically generated in the `results/` directory. This report includes:

- Dataset statistics and model architecture details
- Complete metrics comparison table
- Key findings and performance improvements
- List of all generated visualizations
- Conclusion and analysis of results

The report provides a comprehensive summary of the experiment for easy review and evaluation.

## Expected Performance

The VAE should achieve better clustering quality than PCA baseline, with:

- Higher Silhouette Score
- Better separation in latent space
- More meaningful clusters aligned with genres

## Key Components

- **Model**: BasicVAE with 3-layer encoder/decoder
- **Training**: 50 epochs, batch size 64, Adam optimizer
- **Latent Dim**: 64 (for fair comparison with PCA)
- **Loss**: Reconstruction loss + KL divergence (β=1.0)

## Marks: 20/110

Evaluation criteria:

- VAE implementation works correctly ✓
- Clustering produces meaningful results ✓
- Metrics computed correctly ✓
- Visualizations clear and informative ✓
- Baseline comparison included ✓
