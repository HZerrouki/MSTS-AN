"""
EEG Graph Construction for MSTS-AN.

Implements graph construction based on the 10-20 electrode placement system.
Nodes represent EEG channels and edges represent spatial relationships.
"""

import numpy as np
import torch
from scipy.spatial.distance import cdist
from typing import List, Tuple, Optional, Dict
import networkx as nx


class EEGGraphBuilder:
    """
    Build graph structure for EEG electrodes based on 10-20 system.

    The graph represents spatial relationships between EEG channels
    for use in Graph Convolutional Networks.

    Args:
        channel_names: List of EEG channel names (default: standard 19-channel 10-20)
        graph_type: Type of graph construction ('distance', 'knn', 'fully_connected')
        k: Number of nearest neighbors for KNN graph (default: 3)
        sigma: Distance scaling parameter (default: 1.0)
        self_loops: Whether to include self-loops (default: True)
    """

    # Standard 10-20 electrode coordinates (approximate, in 2D projection)
    # Coordinates are relative positions on the scalp
    _DEFAULT_COORDS_2D = {
        'Fp1': (-0.5, 0.9), 'Fpz': (0, 1.0), 'Fp2': (0.5, 0.9),
        'AF7': (-0.7, 0.7), 'AF3': (-0.35, 0.75), 'AFz': (0, 0.8), 'AF4': (0.35, 0.75), 'AF8': (0.7, 0.7),
        'F7': (-0.8, 0.5), 'F5': (-0.6, 0.55), 'F3': (-0.4, 0.6), 'F1': (-0.2, 0.65),
        'Fz': (0, 0.7), 'F2': (0.2, 0.65), 'F4': (0.4, 0.6), 'F6': (0.6, 0.55), 'F8': (0.8, 0.5),
        'FT7': (-0.9, 0.2), 'FC5': (-0.7, 0.25), 'FC3': (-0.45, 0.3), 'FC1': (-0.22, 0.35),
        'FCz': (0, 0.4), 'FC2': (0.22, 0.35), 'FC4': (0.45, 0.3), 'FC6': (0.7, 0.25), 'FT8': (0.9, 0.2),
        'T7': (-1.0, 0.0), 'C5': (-0.75, 0.0), 'C3': (-0.5, 0.0), 'C1': (-0.25, 0.0),
        'Cz': (0, 0.0), 'C2': (0.25, 0.0), 'C4': (0.5, 0.0), 'C6': (0.75, 0.0), 'T8': (1.0, 0.0),
        'TP7': (-0.9, -0.2), 'CP5': (-0.7, -0.25), 'CP3': (-0.45, -0.3), 'CP1': (-0.22, -0.35),
        'CPz': (0, -0.4), 'CP2': (0.22, -0.35), 'CP4': (0.45, -0.3), 'CP6': (0.7, -0.25), 'TP8': (0.9, -0.2),
        'P7': (-0.8, -0.5), 'P5': (-0.6, -0.55), 'P3': (-0.4, -0.6), 'P1': (-0.2, -0.65),
        'Pz': (0, -0.7), 'P2': (0.2, -0.65), 'P4': (0.4, -0.6), 'P6': (0.6, -0.55), 'P8': (0.8, -0.5),
        'PO7': (-0.7, -0.7), 'PO3': (-0.35, -0.75), 'POz': (0, -0.8), 'PO4': (0.35, -0.75), 'PO8': (0.7, -0.7),
        'O1': (-0.5, -0.9), 'Oz': (0, -1.0), 'O2': (0.5, -0.9)
    }

    # Standard 19-channel subset
    _STANDARD_19 = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
        'T7', 'C3', 'Cz', 'C4', 'T8',
        'P7', 'P3', 'Pz', 'P4', 'P8',
        'O1', 'O2'
    ]

    # Standard 32-channel subset
    _STANDARD_32 = [
        'Fp1', 'AF3', 'F3', 'F7', 'FT7', 'FC3', 'C3', 'T7', 'TP7', 'CP3',
        'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz', 'CPz', 'Fpz', 'AFz', 'Fz',
        'FCz', 'Cz', 'O2', 'PO4', 'P4', 'P8', 'CP4', 'TP8', 'C4', 'T8',
        'FT8', 'FC4', 'F4', 'F8', 'AF4', 'Fp2'
    ]

    def __init__(
        self,
        channel_names: Optional[List[str]] = None,
        graph_type: str = 'distance',
        k: int = 3,
        sigma: float = 1.0,
        self_loops: bool = True,
        normalize: bool = True
    ):
        self.channel_names = channel_names or self._STANDARD_19
        self.graph_type = graph_type
        self.k = k
        self.sigma = sigma
        self.self_loops = self_loops
        self.normalize = normalize

        # Get coordinates for specified channels
        self.coordinates = self._get_coordinates()

    def _get_coordinates(self) -> np.ndarray:
        """Get 2D coordinates for specified channels."""
        coords = []
        for ch in self.channel_names:
            if ch in self._DEFAULT_COORDS_2D:
                coords.append(self._DEFAULT_COORDS_2D[ch])
            else:
                # Assign random coordinates for unknown channels
                coords.append((np.random.randn() * 0.5, np.random.randn() * 0.5))
        return np.array(coords)

    def build_adjacency_matrix(self) -> np.ndarray:
        """
        Build adjacency matrix for the EEG graph.

        Returns:
            Adjacency matrix of shape (n_channels, n_channels)
        """
        n_channels = len(self.channel_names)
        adj = np.zeros((n_channels, n_channels))

        if self.graph_type == 'distance':
            adj = self._build_distance_graph()
        elif self.graph_type == 'knn':
            adj = self._build_knn_graph()
        elif self.graph_type == 'fully_connected':
            adj = self._build_fully_connected_graph()
        elif self.graph_type == 'functional':
            # Placeholder for functional connectivity graph
            # Would need actual EEG data to compute correlations
            adj = self._build_distance_graph()
        else:
            raise ValueError(f"Unknown graph type: {self.graph_type}")

        # Add self-loops
        if self.self_loops:
            np.fill_diagonal(adj, 1.0)

        # Normalize adjacency matrix
        if self.normalize:
            adj = self._normalize_adjacency(adj)

        return adj

    def _build_distance_graph(self) -> np.ndarray:
        """
        Build graph based on spatial distance between electrodes.

        Edge weights are Gaussian functions of Euclidean distance.
        """
        n_channels = len(self.channel_names)
        adj = np.zeros((n_channels, n_channels))

        # Compute pairwise distances
        distances = cdist(self.coordinates, self.coordinates, metric='euclidean')

        # Convert distances to edge weights using Gaussian kernel
        adj = np.exp(-distances**2 / (2 * self.sigma**2))

        # Threshold to create sparse graph (keep only local connections)
        threshold = np.exp(-0.5**2 / (2 * self.sigma**2))  # Distance of 0.5
        adj[adj < threshold] = 0

        return adj

    def _build_knn_graph(self) -> np.ndarray:
        """Build K-nearest neighbors graph."""
        n_channels = len(self.channel_names)
        adj = np.zeros((n_channels, n_channels))

        # Compute pairwise distances
        distances = cdist(self.coordinates, self.coordinates, metric='euclidean')

        # Find k nearest neighbors for each node
        for i in range(n_channels):
            # Get indices of k+1 nearest (including self)
            knn_indices = np.argsort(distances[i])[:self.k+1]
            # Exclude self
            knn_indices = knn_indices[knn_indices != i][:self.k]

            # Add edges with inverse distance weights
            for j in knn_indices:
                adj[i, j] = 1.0 / (distances[i, j] + 1e-6)

        # Make symmetric
        adj = np.maximum(adj, adj.T)

        return adj

    def _build_fully_connected_graph(self) -> np.ndarray:
        """Build fully connected graph with distance-based weights."""
        n_channels = len(self.channel_names)

        # Compute pairwise distances
        distances = cdist(self.coordinates, self.coordinates, metric='euclidean')

        # Convert to edge weights
        adj = np.exp(-distances**2 / (2 * self.sigma**2))

        return adj

    def _normalize_adjacency(self, adj: np.ndarray) -> np.ndarray:
        """
        Normalize adjacency matrix using symmetric normalization.

        A_norm = D^(-1/2) * A * D^(-1/2)

        where D is the degree matrix.
        """
        # Compute degree matrix
        degree = np.sum(adj, axis=1)
        degree_inv_sqrt = np.power(degree, -0.5)
        degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0

        # Symmetric normalization
        D_inv_sqrt = np.diag(degree_inv_sqrt)
        adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt

        return adj_norm

    def build_edge_index(self) -> torch.Tensor:
        """
        Build edge index for PyTorch Geometric.

        Returns:
            Edge index tensor of shape (2, n_edges)
        """
        adj = self.build_adjacency_matrix()

        # Find non-zero entries
        row, col = np.nonzero(adj)

        # Create edge index
        edge_index = torch.LongTensor([row, col])

        return edge_index

    def get_graph_info(self) -> Dict:
        """Get information about the graph structure."""
        adj = self.build_adjacency_matrix()
        edge_index = self.build_edge_index()

        return {
            'n_nodes': len(self.channel_names),
            'n_edges': edge_index.shape[1],
            'channel_names': self.channel_names,
            'coordinates': self.coordinates.tolist(),
            'adjacency_sparsity': 1.0 - (np.count_nonzero(adj) / adj.size)
        }

    def visualize_graph(self, save_path: Optional[str] = None):
        """
        Visualize the EEG electrode graph.

        Args:
            save_path: Path to save the figure (optional)
        """
        import matplotlib.pyplot as plt

        adj = self.build_adjacency_matrix()
        coords = self.coordinates

        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot nodes
        ax.scatter(coords[:, 0], coords[:, 1], s=200, c='lightblue',
                   edgecolors='black', linewidths=2, zorder=3)

        # Plot edges
        for i in range(len(self.channel_names)):
            for j in range(i+1, len(self.channel_names)):
                if adj[i, j] > 0:
                    ax.plot([coords[i, 0], coords[j, 0]],
                           [coords[i, 1], coords[j, 1]],
                           'gray', alpha=adj[i, j] * 0.5, linewidth=1, zorder=1)

        # Add labels
        for i, name in enumerate(self.channel_names):
            ax.annotate(name, (coords[i, 0], coords[i, 1]),
                       textcoords="offset points", xytext=(0, 10),
                       ha='center', fontsize=8, fontweight='bold')

        # Draw head outline (approximate circle)
        theta = np.linspace(0, 2*np.pi, 100)
        radius = 1.1
        ax.plot(radius * np.cos(theta), radius * np.sin(theta),
               'k-', linewidth=2, zorder=0)

        # Draw nose
        ax.plot([0, -0.1, 0, 0.1, 0], [radius, radius + 0.15, radius, radius + 0.15, radius],
               'k-', linewidth=2, zorder=0)

        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'EEG Electrode Graph ({len(self.channel_names)} channels)',
                    fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()


def get_standard_10_20_graph(n_channels: int = 19) -> EEGGraphBuilder:
    """
    Get a standard 10-20 graph builder.

    Args:
        n_channels: Number of channels (19 or 32)

    Returns:
        EEGGraphBuilder instance
    """
    if n_channels == 19:
        channel_names = EEGGraphBuilder._STANDARD_19
    elif n_channels == 32:
        channel_names = EEGGraphBuilder._STANDARD_32
    else:
        raise ValueError(f"Unsupported number of channels: {n_channels}")

    return EEGGraphBuilder(channel_names=channel_names, graph_type='distance')


if __name__ == "__main__":
    # Test graph builder
    builder = get_standard_10_20_graph(n_channels=19)

    # Build adjacency matrix
    adj = builder.build_adjacency_matrix()
    print(f"Adjacency matrix shape: {adj.shape}")
    print(f"Adjacency matrix sparsity: {1 - np.count_nonzero(adj) / adj.size:.2%}")

    # Get graph info
    info = builder.get_graph_info()
    print(f"\nGraph info: {info}")

    # Build edge index
    edge_index = builder.build_edge_index()
    print(f"\nEdge index shape: {edge_index.shape}")

    # Visualize
    builder.visualize_graph("/tmp/eeg_graph.png")
    print("\nGraph visualization saved to /tmp/eeg_graph.png")
