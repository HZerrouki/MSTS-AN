"""
Hybrid Attention Fusion Module for MSTS-AN.

Implements band-specific attention mechanism that adaptively integrates
multi-scale EEG features across frequency bands.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BandAttention(nn.Module):
    """
    Band-specific attention mechanism.

    Learns adaptive weights for each frequency band based on their
    importance for classification.

    Args:
        n_bands: Number of frequency bands (default: 4)
        hidden_dim: Hidden dimension for attention computation
        feature_dim: Dimension of input features per band
        dropout: Dropout probability
    """

    def __init__(
        self,
        n_bands: int = 4,
        hidden_dim: int = 128,
        feature_dim: int = 256,
        dropout: float = 0.2
    ):
        super(BandAttention, self).__init__()

        self.n_bands = n_bands
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim

        # Attention projection layers
        self.query = nn.Linear(feature_dim, hidden_dim)
        self.key = nn.Linear(feature_dim, hidden_dim)
        self.value = nn.Linear(feature_dim, hidden_dim)

        # Band importance scoring
        self.band_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.dropout = nn.Dropout(dropout)
        self.scale = hidden_dim ** -0.5

    def forward(self, band_features: torch.Tensor) -> torch.Tensor:
        """
        Compute band-specific attention weights.

        Args:
            band_features: Input of shape (batch_size, n_bands, feature_dim)

        Returns:
            Attention weights of shape (batch_size, n_bands)
        """
        batch_size = band_features.shape[0]

        # Project to query, key, value
        Q = self.query(band_features)  # (batch, n_bands, hidden_dim)
        K = self.key(band_features)
        V = self.value(band_features)

        # Compute attention scores
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attended = torch.bmm(attn_weights, V)  # (batch, n_bands, hidden_dim)

        # Compute band importance scores
        importance = self.band_scorer(attended).squeeze(-1)  # (batch, n_bands)
        band_weights = F.softmax(importance, dim=-1)

        return band_weights, attended


class ChannelAttention(nn.Module):
    """
    Channel-wise attention for spatial feature weighting.

    Learns importance weights for each EEG channel based on
    their contribution to classification.

    Args:
        n_channels: Number of EEG channels
        feature_dim: Feature dimension per channel
        reduction: Channel reduction ratio for bottleneck
    """

    def __init__(
        self,
        n_channels: int = 19,
        feature_dim: int = 256,
        reduction: int = 4
    ):
        super(ChannelAttention, self).__init__()

        self.n_channels = n_channels
        self.feature_dim = feature_dim

        # Global average pooling + MLP
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // reduction),
            nn.ReLU(),
            nn.Linear(feature_dim // reduction, feature_dim),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel attention.

        Args:
            x: Input of shape (batch_size, n_channels, feature_dim)

        Returns:
            Attended features of same shape
        """
        # Compute channel weights
        weights = self.attention(x)  # (batch, n_channels, feature_dim)

        # Apply attention
        return x * weights


class AttentionFusion(nn.Module):
    """
    Hybrid Attention Fusion module for multi-band EEG features.

    Combines band-specific attention and channel attention to produce
    a unified representation for classification.

    Args:
        n_bands: Number of frequency bands
        n_channels: Number of EEG channels
        feature_dim: Feature dimension per channel per band
        hidden_dim: Hidden dimension for attention computation
        dropout: Dropout probability
    """

    def __init__(
        self,
        n_bands: int = 4,
        n_channels: int = 19,
        feature_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.2
    ):
        super(AttentionFusion, self).__init__()

        self.n_bands = n_bands
        self.n_channels = n_channels
        self.feature_dim = feature_dim

        # Band attention
        self.band_attention = BandAttention(
            n_bands=n_bands,
            hidden_dim=hidden_dim,
            feature_dim=feature_dim * n_channels,  # Flattened channel features
            dropout=dropout
        )

        # Channel attention per band
        self.channel_attention = nn.ModuleList([
            ChannelAttention(n_channels, feature_dim)
            for _ in range(n_bands)
        ])

        # Feature fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(n_bands * n_channels * feature_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, band_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse multi-band features using attention mechanisms.

        Args:
            band_features: Input of shape (batch_size, n_bands, n_channels, feature_dim)

        Returns:
            Fused feature representation of shape (batch_size, hidden_dim)
        """
        batch_size = band_features.shape[0]

        # Apply channel attention to each band
        channel_attended = []
        for i in range(self.n_bands):
            band_feat = band_features[:, i, :, :]  # (batch, n_channels, feature_dim)
            attended = self.channel_attention[i](band_feat)
            channel_attended.append(attended)

        channel_attended = torch.stack(channel_attended, dim=1)

        # Flatten for band attention: (batch, n_bands, n_channels * feature_dim)
        flattened = channel_attended.reshape(batch_size, self.n_bands, -1)

        # Compute band attention weights
        band_weights, band_attended = self.band_attention(flattened)

        # Apply band weights to features
        weighted_features = channel_attended * band_weights.view(
            batch_size, self.n_bands, 1, 1
        )

        # Flatten and fuse
        fused_input = weighted_features.reshape(batch_size, -1)
        fused_output = self.fusion(fused_input)
        fused_output = self.output_norm(fused_output)

        return fused_output, band_weights


class AdaptiveFusion(nn.Module):
    """
    Adaptive fusion that learns to combine band features dynamically.

    Similar to squeeze-and-excitation networks but adapted for multi-band EEG.

    Args:
        n_bands: Number of frequency bands
        feature_dim: Feature dimension per band
        reduction: Reduction ratio for bottleneck
    """

    def __init__(
        self,
        n_bands: int = 4,
        feature_dim: int = 256,
        reduction: int = 4
    ):
        super(AdaptiveFusion, self).__init__()

        self.n_bands = n_bands
        self.feature_dim = feature_dim

        # Global pooling + attention
        self.global_attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // reduction),
            nn.ReLU(),
            nn.Linear(feature_dim // reduction, n_bands),
            nn.Softmax(dim=-1)
        )

    def forward(self, band_features: torch.Tensor) -> torch.Tensor:
        """
        Adaptively fuse band features.

        Args:
            band_features: Input of shape (batch_size, n_bands, feature_dim)

        Returns:
            Fused features of shape (batch_size, feature_dim)
        """
        # Global average pooling across bands
        global_feat = band_features.mean(dim=1)  # (batch, feature_dim)

        # Compute attention weights
        band_weights = self.global_attention(global_feat)  # (batch, n_bands)

        # Apply weights and sum
        fused = (band_features * band_weights.unsqueeze(-1)).sum(dim=1)

        return fused, band_weights


if __name__ == "__main__":
    # Test attention fusion
    batch_size = 8
    n_bands = 4
    n_channels = 19
    feature_dim = 256

    # Create sample input
    x = torch.randn(batch_size, n_bands, n_channels, feature_dim)

    # Create fusion module
    fusion = AttentionFusion(
        n_bands=n_bands,
        n_channels=n_channels,
        feature_dim=feature_dim,
        hidden_dim=128
    )

    # Forward pass
    output, band_weights = fusion(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Band weights shape: {band_weights.shape}")
    print(f"Band weights (first sample): {band_weights[0]}")
