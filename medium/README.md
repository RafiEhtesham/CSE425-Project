# Medium Task: Enhanced VAE with Multi-modal Features

## Overview

This task enhances the VAE with multi-modal features (audio + lyrics) and experiments with multiple clustering algorithms.

## Requirements

- Enhance VAE with convolutional architecture for spectrograms or MFCC features
- Include hybrid feature representation: audio + lyrics embeddings
- Experiment with clustering algorithms: K-Means, Agglomerative Clustering, DBSCAN
- Evaluate clustering quality using multiple metrics
- Compare results across methods and analyze performance

## Running the Task

```bash
python main.py
```

## What It Does

1. **Loads Multi-modal Data**: Loads audio features and lyrics embeddings (TF-IDF)
2. **Trains Hybrid VAE**: Trains VAE that processes both audio and lyrics
3. **Multiple Clustering Methods**:
   - K-Means
   - Agglomerative Clustering
   - DBSCAN (with automatic eps selection)
4. **Baseline Comparison**: Compares with PCA + K-Means on audio-only features
5. **Comprehensive Evaluation**: All metrics on all clustering methods
6. **Analysis**: Analyzes why VAE performs better/worse than baselines

## Outputs

Results are saved to `results/` directory:

- `hybrid_vae_model.pth` - Trained Hybrid VAE model
- `hybrid_training_history.png` - Training loss curves
- `hybrid_tsne_true.png` - t-SNE with true labels
- `hybrid_tsne_clusters.png` - t-SNE with K-Means clusters
- `hybrid_umap_true.png` - UMAP with true labels
- `hybrid_confusion_matrix.png` - Confusion matrix
- `hybrid_genre_cluster_heatmap.png` - Genre-cluster heatmap
- `clustering_comparison.csv` - Comparison of all methods
- `clustering_comparison.png` - Visual comparison chart
- **`RESULTS_REPORT.md`** - Comprehensive results summary with method comparison

## Results Report

After running the task, a detailed results report (`RESULTS_REPORT.md`) is automatically generated in the `results/` directory. This report includes:

- Dataset and multi-modal feature statistics
- Hybrid VAE architecture specifications
- Complete comparison table of all clustering methods (K-Means, Agglomerative, DBSCAN)
- Key findings identifying the best-performing method
- Analysis of multi-modal feature fusion effectiveness
- List of all generated visualizations
- Conclusion with insights on clustering performance

The report provides a comprehensive evaluation of all clustering approaches tested.

## Key Features

### Hybrid VAE Architecture

- **Audio Branch**: Processes audio features
- **Lyrics Branch**: Processes TF-IDF lyrics embeddings
- **Joint Encoding**: Combines both modalities in latent space
- **Dual Reconstruction**: Reconstructs both audio and lyrics

### Clustering Experiments

1. **K-Means**: Standard centroid-based clustering
2. **Agglomerative**: Hierarchical clustering
3. **DBSCAN**: Density-based clustering (automatic parameter tuning)

### Evaluation Metrics

All methods evaluated on:

- Silhouette Score
- Calinski-Harabasz Index
- Davies-Bouldin Index
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)
- Cluster Purity

## Expected Performance

The Hybrid VAE should:

- Leverage both audio and text information
- Outperform audio-only baselines
- Show K-Means or Agglomerative as best clustering method
- Achieve ARI > 0.3, NMI > 0.4 on GTZAN

## Marks: 25/110

Evaluation criteria:

- Multi-modal VAE implementation ✓
- Multiple clustering algorithms ✓
- Comprehensive evaluation ✓
- Meaningful analysis and comparison ✓
- High-quality visualizations ✓
