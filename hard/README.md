# Hard Task: Advanced VAE with CVAE/Beta-VAE

## Overview

This task implements advanced VAE architectures (Conditional VAE and Beta-VAE) with comprehensive evaluation and visualization.

## Requirements

- Implement Conditional VAE (CVAE) or Beta-VAE for disentangled latent representations
- Perform multi-modal clustering combining audio, lyrics, and genre information
- Quantitative evaluation using comprehensive metrics
- Detailed visualizations: latent space plots, cluster distribution, reconstruction examples
- Compare VAE-based clustering with PCA + K-Means, Autoencoder + K-Means, and direct spectral feature clustering

## Running the Task

```bash
python main.py
```

## What It Does

1. **Trains Conditional VAE**: Genre-conditioned VAE for controlled generation
2. **Trains Beta-VAE**: Multiple beta values (1.0, 4.0, 10.0) for disentanglement
3. **Comprehensive Clustering**: All clustering methods on all VAE variants
4. **Multiple Baselines**:
   - PCA + K-Means
   - Direct K-Means
   - Simple Autoencoder + K-Means
5. **Full Evaluation**: All 6 metrics on all 10+ method combinations
6. **Advanced Visualizations**: Latent traversals, disentanglement analysis

## Outputs

Results are saved to `results/` directory:

### Models

- `cvae_model.pth` - Conditional VAE
- `beta_vae_beta=1.0_model.pth` - Beta-VAE (β=1.0)
- `beta_vae_beta=4.0_model.pth` - Beta-VAE (β=4.0)
- `beta_vae_beta=10.0_model.pth` - Beta-VAE (β=10.0)

### Visualizations

- `cvae_training_history.png` - CVAE training curves
- `cvae_tsne_true.png` - CVAE t-SNE with true labels
- `cvae_tsne_clusters.png` - CVAE t-SNE with clusters
- `cvae_umap_true.png` - CVAE UMAP visualization
- `cvae_confusion_matrix.png` - Confusion matrix
- `cvae_genre_cluster_heatmap.png` - Genre distribution heatmap
- `cvae_cluster_distribution.png` - Cluster sizes
- `beta_vae_latent_traversal.png` - Latent space traversal (disentanglement)
- `comprehensive_comparison.png` - All methods comparison chart

### Data

- `comprehensive_comparison.csv` - All methods and metrics
- **`RESULTS_REPORT.md`** - Comprehensive results summary with analysis

## Results Report

After running the task, a detailed results report (`RESULTS_REPORT.md`) is automatically generated in the `results/` directory. This report includes:

- Dataset statistics and multi-modal feature dimensions
- Detailed architecture specifications for CVAE and all Beta-VAE variants
- Complete comparison table of 15+ method combinations
- Top 3 performing methods with scores
- Beta-VAE analysis explaining the effect of different β values
- CVAE analysis discussing genre conditioning benefits
- List of all generated visualizations (10+ plots)
- Information about all saved model checkpoints
- Comprehensive conclusion analyzing deep vs. traditional methods

The report provides a complete scientific evaluation of all advanced VAE architectures and clustering approaches.

## Advanced Features

### Conditional VAE

- **Conditioning**: Uses one-hot encoded genre labels
- **Controlled Generation**: Can generate samples conditioned on genre
- **Improved Clustering**: Genre information helps structure latent space

### Beta-VAE

- **Disentanglement**: β parameter controls latent disentanglement
- **Multiple Values**: Tests β ∈ {1.0, 4.0, 10.0}
- **Analysis**: Latent traversal shows independent factors

### Comprehensive Baselines

1. **PCA + K-Means**: Classical dimensionality reduction
2. **Direct K-Means**: Clustering on raw features
3. **Autoencoder + K-Means**: Non-variational autoencoder

### Full Method Comparison

Tests all combinations:

- CVAE + {K-Means, Agglomerative, DBSCAN}
- Beta-VAE (β=1.0) + {K-Means, Agglomerative, DBSCAN}
- Beta-VAE (β=4.0) + {K-Means, Agglomerative, DBSCAN}
- Beta-VAE (β=10.0) + {K-Means, Agglomerative, DBSCAN}
- PCA + K-Means
- Direct K-Means
- Autoencoder + K-Means

Total: ~15 method combinations evaluated!

## Expected Performance

The CVAE should:

- Achieve best overall clustering performance
- Show clear genre separation in latent space
- ARI > 0.4, NMI > 0.5 on GTZAN
- Outperform all baselines

Beta-VAE should:

- Show disentangled latent factors
- β=4.0 often optimal balance
- Clear traversal patterns

## Evaluation Metrics

All methods compared on:

1. **Silhouette Score** (unsupervised quality)
2. **Calinski-Harabasz Index** (cluster separation)
3. **Davies-Bouldin Index** (cluster compactness)
4. **Adjusted Rand Index** (agreement with true labels)
5. **Normalized Mutual Information** (information overlap)
6. **Cluster Purity** (dominant class fraction)

## Key Implementation Details

- **CVAE**: Concatenates input with one-hot genre for encoding
- **Beta-VAE**: Scales KL divergence by beta in loss function
- **Training**: 50 epochs each, Adam optimizer, lr=0.001
- **Latent Dim**: 64 for all models (fair comparison)
- **Multi-modal**: Combines audio features + lyrics embeddings

## Marks: 25/110

Evaluation criteria:

- Advanced VAE architectures (CVAE, Beta-VAE) ✓
- Multi-modal feature fusion ✓
- Comprehensive baseline comparisons ✓
- All evaluation metrics ✓
- Advanced visualizations (traversals) ✓
- Detailed analysis and insights ✓

## Disentanglement Analysis

The Beta-VAE latent traversal visualization shows:

- Each row = one latent dimension
- Each column = step in traversal range [-3, 3]
- Independent variation indicates disentanglement
- β=4.0 typically shows best disentanglement

## Usage Tips

- **GPU Recommended**: Hard task trains 5+ models
- **Runtime**: ~15-30 minutes on GPU, 1-2 hours on CPU
- **Memory**: ~4GB RAM, ~2GB VRAM
- **Disk Space**: ~100MB for all results
