"""
MSTS-AN: Multi-Scale Temporal-Spatial Attention Network

Main model architecture that combines:
1. Multi-scale feature extraction (wavelet decomposition)
2. GCN spatial encoding
3. ViT temporal encoding
4. Hybrid attention fusion
5. Classification head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .gcn_module import GCNModule
from .vit_module import ViTModule, BandSpecificViT
from .attention_fusion import AttentionFusion


class MSTSAN(nn.Module):
    """
    Multi-Scale Temporal-Spatial Attention Network for EEG-based Alzheimer's detection.

    Architecture:
        Input (C, T)
            |
        Multi-scale wavelet decomposition (4 bands: delta, theta, alpha, beta)
            |
        GCN Spatial Encoding (per band)
            |
        ViT Temporal Encoding (per band)
            |
        Hybrid Attention Fusion
            |
        Classification Head
            |
        Output (3 classes: HC, MCI, AD)

    Args:
        n_channels: Number of EEG channels (default: 19)
        seq_length: Length of temporal sequence (default: 1024)
        n_bands: Number of frequency bands (default: 4)
        n_classes: Number of output classes (default: 3)
        gcn_hidden_dims: Hidden dimensions for GCN
        gcn_out_dim: Output dimension of GCN
        vit_embed_dim: Embedding dimension for ViT
        vit_num_heads: Number of attention heads in ViT
        vit_num_layers: Number of transformer layers
        fusion_hidden_dim: Hidden dimension for fusion
        dropout: Dropout probability
    """

    def __init__(
        self,
        n_channels: int = 19,
        seq_length: int = 1024,
        n_bands: int = 4,
        n_classes: int = 3,
        gcn_hidden_dims: List[int] = [64, 128],
        gcn_out_dim: int = 256,
        vit_embed_dim: int = 256,
        vit_num_heads: int = 8,
        vit_num_layers: int = 4,
        fusion_hidden_dim: int = 128,
        classifier_hidden_dims: List[int] = [256, 128],
        dropout: float = 0.3,
        use_batch_norm: bool = True
    ):
        super(MSTSAN, self).__init__()

        self.n_channels = n_channels
        self.seq_length = seq_length
        self.n_bands = n_bands
        self.n_classes = n_classes

        # Multi-scale GCN (one per band)
        self.gcns = nn.ModuleList([
            GCNModule(
                in_channels=seq_length,
                hidden_dims=gcn_hidden_dims,
                out_channels=gcn_out_dim,
                dropout=dropout,
                use_batch_norm=use_batch_norm
            )
            for _ in range(n_bands)
        ])

        # Band-specific ViT
        self.vits = BandSpecificViT(
            n_bands=n_bands,
            seq_length=n_channels,
            in_channels=gcn_out_dim,
            embed_dim=vit_embed_dim,
            num_heads=vit_num_heads,
            num_layers=vit_num_layers,
            dropout=dropout
        )

        # Hybrid attention fusion
        self.attention_fusion = AttentionFusion(
            n_bands=n_bands,
            n_channels=n_channels,
            feature_dim=vit_embed_dim,
            hidden_dim=fusion_hidden_dim,
            dropout=dropout
        )

        # Classification head
        classifier_dims = [fusion_hidden_dim] + classifier_hidden_dims + [n_classes]
        self.classifier = self._build_classifier(classifier_dims, dropout)

        self._init_weights()

    def _build_classifier(
        self,
        dims: List[int],
        dropout: float
    ) -> nn.Module:
        """Build classification head MLP."""
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # No activation/dropout after last layer
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def _init_weights(self):
        """Initialize weights."""
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        band_data: Dict[str, torch.Tensor],
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        return_features: bool = False,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through MSTS-AN.

        Args:
            band_data: Dictionary with keys 'delta', 'theta', 'alpha', 'beta'
                      Each value is tensor of shape (batch_size, n_channels, seq_length)
            edge_index: Graph connectivity of shape (2, n_edges)
            edge_weight: Optional edge weights
            return_features: Whether to return intermediate features
            return_attention: Whether to return attention weights

        Returns:
            logits: Classification logits of shape (batch_size, n_classes)
            features: (optional) Intermediate features
            attention_weights: (optional) Band attention weights
        """
        batch_size = list(band_data.values())[0].shape[0]
        band_order = ['delta', 'theta', 'alpha', 'beta']

        # Process each band through GCN
        gcn_outputs = []
        for i, band in enumerate(band_order):
            x = band_data[band]  # (batch, n_channels, seq_length)

            # Reshape for GCN: (batch * n_channels, seq_length)
            x = x.reshape(batch_size * self.n_channels, self.seq_length)

            # Apply GCN
            x = self.gcns[i](x, edge_index, edge_weight)

            # Reshape back: (batch, n_channels, gcn_out_dim)
            x = x.reshape(batch_size, self.n_channels, -1)
            gcn_outputs.append(x)

        # Stack GCN outputs: (batch, n_bands, n_channels, gcn_out_dim)
        gcn_features = torch.stack(gcn_outputs, dim=1)

        # Process through ViT (per band)
        vit_features = self.vits(gcn_features)
        # Output: (batch, n_bands, n_channels, vit_embed_dim)

        # Apply hybrid attention fusion
        fused_features, band_weights = self.attention_fusion(vit_features)
        # fused_features: (batch, fusion_hidden_dim)

        # Classification
        logits = self.classifier(fused_features)

        outputs = [logits]

        if return_features:
            outputs.append(fused_features)

        if return_attention:
            outputs.append(band_weights)

        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    def get_embeddings(
        self,
        band_data: Dict[str, torch.Tensor],
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get feature embeddings before classification."""
        _, features = self.forward(
            band_data, edge_index, edge_weight,
            return_features=True
        )
        return features

    def get_band_attention(
        self,
        band_data: Dict[str, torch.Tensor],
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get band attention weights."""
        _, band_weights = self.forward(
            band_data, edge_index, edge_weight,
            return_attention=True
        )
        return band_weights

    def predict(
        self,
        band_data: Dict[str, torch.Tensor],
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get class predictions."""
        logits = self.forward(band_data, edge_index, edge_weight)
        probs = F.softmax(logits, dim=-1)
        predictions = torch.argmax(probs, dim=-1)
        return predictions

    def predict_proba(
        self,
        band_data: Dict[str, torch.Tensor],
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get class probabilities."""
        logits = self.forward(band_data, edge_index, edge_weight)
        probs = F.softmax(logits, dim=-1)
        return probs

    def get_attention_maps(
        self,
        band_data: Dict[str, torch.Tensor],
        edge_index: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Get attention maps for explainability.

        Returns:
            Dictionary containing:
                - band_weights: Band-specific attention weights
                - spatial_attention: Placeholder for spatial attention
        """
        band_weights = self.get_band_attention(band_data, edge_index)

        return {
            'band_weights': band_weights,
            'spatial_attention': None  # Could extract from GCN/ViT if needed
        }


def create_msts_an(config: Optional[Dict] = None) -> MSTSAN:
    """
    Factory function to create MSTS-AN model from configuration.

    Args:
        config: Configuration dictionary (optional)

    Returns:
        MSTSAN model instance
    """
    default_config = {
        'n_channels': 19,
        'seq_length': 1024,
        'n_bands': 4,
        'n_classes': 3,
        'gcn_hidden_dims': [64, 128],
        'gcn_out_dim': 256,
        'vit_embed_dim': 256,
        'vit_num_heads': 8,
        'vit_num_layers': 4,
        'fusion_hidden_dim': 128,
        'classifier_hidden_dims': [256, 128],
        'dropout': 0.3
    }

    if config is not None:
        default_config.update(config)

    return MSTSAN(**default_config)


if __name__ == "__main__":
    # Test MSTS-AN model
    batch_size = 4
    n_channels = 19
    seq_length = 1024

    # Create sample band data
    band_data = {
        'delta': torch.randn(batch_size, n_channels, seq_length),
        'theta': torch.randn(batch_size, n_channels, seq_length),
        'alpha': torch.randn(batch_size, n_channels, seq_length),
        'beta': torch.randn(batch_size, n_channels, seq_length)
    }

    # Create simple graph (chain)
    edge_index = torch.LongTensor([
        [i for i in range(n_channels - 1)] + [i for i in range(1, n_channels)],
        [i for i in range(1, n_channels)] + [i for i in range(n_channels - 1)]
    ])

    # Repeat for batch
    edge_index_batch = []
    for b in range(batch_size):
        offset = b * n_channels
        edge_index_batch.append(edge_index + offset)
    edge_index_batch = torch.cat(edge_index_batch, dim=1)

    # Create model
    model = MSTSAN(
        n_channels=n_channels,
        seq_length=seq_length,
        n_bands=4,
        n_classes=3
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Forward pass
    logits, features, band_weights = model(
        band_data, edge_index_batch,
        return_features=True, return_attention=True
    )

    print(f"\nLogits shape: {logits.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Band weights shape: {band_weights.shape}")
    print(f"Band weights (sample 0): {band_weights[0]}")

    # Test prediction
    predictions = model.predict(band_data, edge_index_batch)
    print(f"\nPredictions: {predictions}")
