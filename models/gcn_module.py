"""
Graph Convolutional Network (GCN) Module for MSTS-AN.

Implements spatial feature extraction using graph convolutions on
EEG electrode topology. Based on the normalized graph Laplacian
approach described in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm
from typing import List, Optional, Tuple


class GCNModule(nn.Module):
    """
    Graph Convolutional Network for EEG spatial feature extraction.

    Processes EEG channel data on a graph structure where nodes are
    electrodes and edges represent spatial relationships.

    Args:
        in_channels: Number of input features per node
        hidden_dims: List of hidden layer dimensions
        out_channels: Number of output features per node
        dropout: Dropout probability (default: 0.3)
        use_batch_norm: Whether to use batch normalization (default: True)
        activation: Activation function (default: 'relu')
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dims: List[int] = [64, 128, 256],
        out_channels: int = 256,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
        activation: str = 'relu'
    ):
        super(GCNModule, self).__init__()

        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.out_channels = out_channels
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm

        # Build GCN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None

        dims = [in_channels] + hidden_dims + [out_channels]

        for i in range(len(dims) - 1):
            self.convs.append(GCNConv(dims[i], dims[i + 1]))
            if use_batch_norm:
                self.batch_norms.append(BatchNorm(dims[i + 1]))

        # Activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        elif activation == 'elu':
            self.activation = F.elu
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for conv in self.convs:
            nn.init.xavier_uniform_(conv.lin.weight)
            if conv.lin.bias is not None:
                nn.init.zeros_(conv.lin.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the GCN.

        Args:
            x: Node features of shape (batch_size * n_nodes, in_channels) or
               (n_nodes, in_channels) for single graph
            edge_index: Graph connectivity of shape (2, n_edges)
            edge_weight: Optional edge weights of shape (n_edges,)

        Returns:
            Output node features of shape (batch_size * n_nodes, out_channels)
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight)

            if self.use_batch_norm:
                x = self.batch_norms[i](x)

            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Final layer without activation
        x = self.convs[-1](x, edge_index, edge_weight)

        return x

    def get_embeddings(self, x, edge_index, edge_weight=None):
        """Get intermediate embeddings (before final layer)."""
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight)

            if self.use_batch_norm:
                x = self.batch_norms[i](x)

            x = self.activation(x)

        return x


class MultiScaleGCN(nn.Module):
    """
    Multi-scale GCN that processes each frequency band separately.

    Args:
        n_bands: Number of frequency bands
        in_channels: Number of input features per node
        hidden_dims: List of hidden layer dimensions
        out_channels: Number of output features per node
        dropout: Dropout probability
    """

    def __init__(
        self,
        n_bands: int = 4,
        in_channels: int = 1024,
        hidden_dims: List[int] = [64, 128],
        out_channels: int = 256,
        dropout: float = 0.3,
        use_batch_norm: bool = True
    ):
        super(MultiScaleGCN, self).__init__()

        self.n_bands = n_bands

        # Create a GCN for each frequency band
        self.gcns = nn.ModuleList([
            GCNModule(
                in_channels=in_channels,
                hidden_dims=hidden_dims,
                out_channels=out_channels,
                dropout=dropout,
                use_batch_norm=use_batch_norm
            )
            for _ in range(n_bands)
        ])

    def forward(
        self,
        band_data: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for multi-scale GCN.

        Args:
            band_data: Input tensor of shape (batch_size, n_bands, n_channels, n_samples)
            edge_index: Graph connectivity
            edge_weight: Optional edge weights

        Returns:
            Output tensor of shape (batch_size, n_bands, n_channels, out_channels)
        """
        batch_size, n_bands, n_channels, n_samples = band_data.shape

        outputs = []
        for i in range(n_bands):
            # Get data for this band: (batch_size, n_channels, n_samples)
            x = band_data[:, i, :, :]

            # Reshape for GCN: (batch_size * n_channels, n_samples)
            x = x.reshape(batch_size * n_channels, n_samples)

            # Apply GCN
            out = self.gcns[i](x, edge_index, edge_weight)

            # Reshape back: (batch_size, n_channels, out_channels)
            out = out.reshape(batch_size, n_channels, -1)
            outputs.append(out)

        # Stack outputs: (batch_size, n_bands, n_channels, out_channels)
        return torch.stack(outputs, dim=1)


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer for learning adaptive edge weights.

    Implements attention mechanism similar to GAT but simplified
    for EEG electrode graphs.
    """

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.3):
        super(GraphAttentionLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout

        self.W = nn.Linear(in_features, out_features)
        self.a = nn.Linear(2 * out_features, 1)

        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with attention.

        Returns:
            Tuple of (output_features, attention_weights)
        """
        h = self.W(x)  # (N, out_features)
        N = h.size(0)

        # Compute attention coefficients
        src, dst = edge_index

        # Get source and destination node features
        h_src = h[src]  # (E, out_features)
        h_dst = h[dst]  # (E, out_features)

        # Concatenate and compute attention scores
        edge_h = torch.cat([h_src, h_dst], dim=1)  # (E, 2*out_features)
        edge_e = self.leaky_relu(self.a(edge_h))  # (E, 1)

        # Softmax normalization per destination node
        attention = F.softmax(edge_e, dim=0)
        attention = F.dropout(attention, p=self.dropout, training=self.training)

        # Aggregate messages
        out = torch.zeros_like(h)
        for i in range(edge_index.shape[1]):
            out[dst[i]] += attention[i] * h[src[i]]

        return out, attention


if __name__ == "__main__":
    # Test GCN module
    batch_size = 8
    n_nodes = 19
    in_channels = 1024
    out_channels = 256

    # Create sample data
    x = torch.randn(batch_size * n_nodes, in_channels)

    # Create simple chain graph
    edge_index = torch.LongTensor([
        [i for i in range(n_nodes - 1)] + [i for i in range(1, n_nodes)],
        [i for i in range(1, n_nodes)] + [i for i in range(n_nodes - 1)]
    ])

    # Repeat for batch
    edge_index_batch = []
    for b in range(batch_size):
        offset = b * n_nodes
        edge_index_batch.append(edge_index + offset)
    edge_index_batch = torch.cat(edge_index_batch, dim=1)

    # Create GCN
    gcn = GCNModule(
        in_channels=in_channels,
        hidden_dims=[64, 128],
        out_channels=out_channels,
        dropout=0.3
    )

    # Forward pass
    output = gcn(x, edge_index_batch)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: ({batch_size * n_nodes}, {out_channels})")
