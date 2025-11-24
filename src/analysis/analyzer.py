"""
Model Analyzer

Tools for analyzing and visualizing hierarchical model behavior.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import pearsonr

from ..models.hierarchical_model import HierarchicalModel


class HierarchicalModelAnalyzer:
    """
    Analyzer for hierarchical model

    Features:
    1. Representation extraction (h_channel, h_LA)
    2. Clustering analysis
    3. Correlation with throughput
    4. Auxiliary task performance
    5. Prediction quality metrics
    6. Layer contribution analysis
    """

    def __init__(
        self,
        model: HierarchicalModel,
        device: str = 'cuda',
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def extract_representations(
        self,
        dataloader,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract representations from all samples

        Args:
            dataloader: DataLoader with samples

        Returns:
            h_channel: [N, channel_dim]
            h_LA: [N, channel_dim]
            throughput: [N]
        """
        h_channels = []
        h_LAs = []
        throughputs = []

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                h_channel, h_LA = self.model.get_representations(batch)

                h_channels.append(h_channel.cpu().numpy())
                h_LAs.append(h_LA.cpu().numpy())
                throughputs.append(batch['throughput'].cpu().numpy())

        h_channel = np.concatenate(h_channels, axis=0)
        h_LA = np.concatenate(h_LAs, axis=0)
        throughput = np.concatenate(throughputs, axis=0)

        return h_channel, h_LA, throughput

    def plot_representation_pca(
        self,
        h_channel: np.ndarray,
        h_LA: np.ndarray,
        throughput: np.ndarray,
        output_dir: str,
    ):
        """Plot PCA of representations colored by throughput"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # PCA for h_channel
        pca = PCA(n_components=2)
        h_channel_pca = pca.fit_transform(h_channel)

        scatter = axes[0].scatter(
            h_channel_pca[:, 0],
            h_channel_pca[:, 1],
            c=throughput,
            cmap='viridis',
            alpha=0.5,
        )
        axes[0].set_title('h_channel PCA (colored by throughput)')
        axes[0].set_xlabel('PC1')
        axes[0].set_ylabel('PC2')
        plt.colorbar(scatter, ax=axes[0])

        # PCA for h_LA
        pca = PCA(n_components=2)
        h_LA_pca = pca.fit_transform(h_LA)

        scatter = axes[1].scatter(
            h_LA_pca[:, 0],
            h_LA_pca[:, 1],
            c=throughput,
            cmap='viridis',
            alpha=0.5,
        )
        axes[1].set_title('h_LA PCA (colored by throughput)')
        axes[1].set_xlabel('PC1')
        axes[1].set_ylabel('PC2')
        plt.colorbar(scatter, ax=axes[1])

        plt.tight_layout()
        plt.savefig(output_dir / 'representation_pca.png', dpi=300)
        plt.close()

    def plot_clustering_analysis(
        self,
        h_channel: np.ndarray,
        h_LA: np.ndarray,
        throughput: np.ndarray,
        output_dir: str,
        n_clusters: int = 5,
    ):
        """Cluster representations and analyze cluster properties"""
        output_dir = Path(output_dir)

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        for idx, (h, name) in enumerate([(h_channel, 'h_channel'), (h_LA, 'h_LA')]):
            # Cluster
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(h)

            # Plot cluster distributions
            cluster_throughputs = [throughput[labels == i] for i in range(n_clusters)]

            axes[idx].boxplot(cluster_throughputs)
            axes[idx].set_title(f'{name} Cluster Analysis')
            axes[idx].set_xlabel('Cluster')
            axes[idx].set_ylabel('Throughput (log)')
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'clustering_analysis.png', dpi=300)
        plt.close()

    def plot_correlation_analysis(
        self,
        h_channel: np.ndarray,
        h_LA: np.ndarray,
        throughput: np.ndarray,
        output_dir: str,
    ):
        """Analyze correlation between representations and throughput"""
        output_dir = Path(output_dir)

        # Compute correlations for each dimension
        h_channel_corrs = [
            pearsonr(h_channel[:, i], throughput)[0]
            for i in range(h_channel.shape[1])
        ]
        h_LA_corrs = [
            pearsonr(h_LA[:, i], throughput)[0]
            for i in range(h_LA.shape[1])
        ]

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # h_channel correlations
        axes[0].bar(range(len(h_channel_corrs)), h_channel_corrs)
        axes[0].set_title('h_channel Dimension Correlations with Throughput')
        axes[0].set_xlabel('Dimension')
        axes[0].set_ylabel('Pearson Correlation')
        axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.3)
        axes[0].grid(True, alpha=0.3)

        # h_LA correlations
        axes[1].bar(range(len(h_LA_corrs)), h_LA_corrs)
        axes[1].set_title('h_LA Dimension Correlations with Throughput')
        axes[1].set_xlabel('Dimension')
        axes[1].set_ylabel('Pearson Correlation')
        axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.3)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_analysis.png', dpi=300)
        plt.close()

        # Print summary statistics
        print("\nCorrelation Summary:")
        print(f"h_channel: mean={np.mean(np.abs(h_channel_corrs)):.4f}, "
              f"max={np.max(np.abs(h_channel_corrs)):.4f}")
        print(f"h_LA: mean={np.mean(np.abs(h_LA_corrs)):.4f}, "
              f"max={np.max(np.abs(h_LA_corrs)):.4f}")

    def analyze(
        self,
        dataloader,
        output_dir: str,
    ):
        """
        Run full analysis pipeline

        Args:
            dataloader: DataLoader with validation data
            output_dir: Directory to save plots
        """
        print("Extracting representations...")
        h_channel, h_LA, throughput = self.extract_representations(dataloader)

        print(f"Extracted {len(h_channel)} samples")
        print(f"h_channel shape: {h_channel.shape}")
        print(f"h_LA shape: {h_LA.shape}")

        print("\nGenerating PCA plots...")
        self.plot_representation_pca(h_channel, h_LA, throughput, output_dir)

        print("Running clustering analysis...")
        self.plot_clustering_analysis(h_channel, h_LA, throughput, output_dir)

        print("Computing correlation analysis...")
        self.plot_correlation_analysis(h_channel, h_LA, throughput, output_dir)

        print(f"\nAnalysis complete! Results saved to {output_dir}")
