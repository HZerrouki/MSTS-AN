"""
Vision Transformer (ViT) Module for MSTS-AN.

Implements temporal feature extraction using Vision Transformer architecture.
Processes spatially-encoded EEG features as sequences of temporal patches.

Based on "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2021).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class PatchEmbedding(nn.Module):
    """
    Convert temporal sequence into patches and project to embedding space.

    Args:
        seq_length: Length of input sequence
        patch_size: Size of each patch
        in_channels: Number of input channels
        embed_dim: Embedding dimension
    """

    def __init__(
        self,
        seq_length: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int
    ):
        super(PatchEmbedding, self).__init__()

        self.seq_length = seq_length
        self.patch_size = patch_size
        self.n_patches = seq_length // patch_size

        # Linear projection of flattened patches
        self.projection = nn.Linear(patch_size * in_channels, embed_dim)

        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Positional embedding
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.n_patches + 1, embed_dim)
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input of shape (batch_size, n_channels, seq_length)

        Returns:
            Patch embeddings of shape (batch_size, n_patches + 1, embed_dim)
        """
        batch_size, n_channels, seq_length = x.shape

        # Reshape to patches: (batch_size, n_patches, patch_size, n_channels)
        x = x.reshape(batch_size, n_channels, self.n_patches, self.patch_size)

        # Transpose: (batch_size, n_patches, n_channels, patch_size)
        x = x.permute(0, 2, 1, 3)

        # Flatten: (batch_size, n_patches, n_channels * patch_size)
        x = x.reshape(batch_size, self.n_patches, -1)

        # Project: (batch_size, n_patches, embed_dim)
        x = self.projection(x)

        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embedding
        x = x + self.pos_embedding

        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super(MultiHeadAttention, self).__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Args:
            x: Input of shape (batch_size, seq_len, embed_dim)
            return_attention: Whether to return attention weights

        Returns:
            Output tensor and optionally attention weights
        """
        batch_size, seq_len, embed_dim = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = attn @ v  # (batch, heads, seq_len, head_dim)

        # Concatenate heads
        out = out.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)

        # Final projection
        out = self.proj(out)

        if return_attention:
            return out, attn
        return out


class FeedForward(nn.Module):
    """
    Feed-Forward Network (FFN) in Transformer.

    Args:
        embed_dim: Embedding dimension
        hidden_dim: Hidden layer dimension (typically 4 * embed_dim)
        dropout: Dropout probability
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        dropout: float = 0.1
    ):
        super(FeedForward, self).__init__()

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single Transformer Encoder Block.

    Consists of Multi-Head Self-Attention followed by Feed-Forward Network,
    with Layer Normalization and residual connections.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        mlp_ratio: Ratio of FFN hidden dim to embed dim
        dropout: Dropout probability
        attention_dropout: Attention dropout probability
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1
    ):
        super(TransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, attention_dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, int(embed_dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Args:
            x: Input of shape (batch_size, seq_len, embed_dim)
            return_attention: Whether to return attention weights

        Returns:
            Output tensor and optionally attention weights
        """
        # Multi-Head Attention with residual
        if return_attention:
            attn_out, attn_weights = self.attn(self.norm1(x), return_attention=True)
        else:
            attn_out = self.attn(self.norm1(x))

        x = x + attn_out

        # Feed-Forward with residual
        x = x + self.ffn(self.norm2(x))

        if return_attention:
            return x, attn_weights
        return x


class ViTModule(nn.Module):
    """
    Vision Transformer Module for temporal feature extraction.

    Processes spatially-encoded EEG features as sequences of patches
    and captures long-range temporal dependencies through self-attention.

    Args:
        seq_length: Length of input sequence (after spatial pooling)
        patch_size: Size of temporal patches
        in_channels: Number of input channels (from GCN output)
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        mlp_ratio: FFN hidden dim ratio
        dropout: Dropout probability
        attention_dropout: Attention dropout probability
    """

    def __init__(
        self,
        seq_length: int = 19,  # Number of EEG channels after spatial processing
        patch_size: int = 1,   # Each channel is a patch
        in_channels: int = 256,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1
    ):
        super(ViTModule, self).__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = nn.Linear(in_channels, embed_dim)

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, seq_length, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.pos_embed, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x: Input of shape (batch_size, seq_length, in_channels)
               or (batch_size, n_channels, embed_dim)
            return_attention: Whether to return attention weights from last layer

        Returns:
            Output of shape (batch_size, seq_length, embed_dim)
        """
        batch_size = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_dropout(x)

        # Apply transformer blocks
        attn_weights = None
        for block in self.blocks:
            if return_attention and block == self.blocks[-1]:
                x, attn_weights = block(x, return_attention=True)
            else:
                x = block(x)

        x = self.norm(x)

        if return_attention:
            return x, attn_weights
        return x


class BandSpecificViT(nn.Module):
    """
    Band-specific Vision Transformer for multi-scale temporal processing.

    Processes each frequency band with its own ViT, allowing
    band-specific temporal feature learning.

    Args:
        n_bands: Number of frequency bands
        seq_length: Length of input sequence per band
        in_channels: Number of input channels per band
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        dropout: Dropout probability
    """

    def __init__(
        self,
        n_bands: int = 4,
        seq_length: int = 19,
        in_channels: int = 256,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super(BandSpecificViT, self).__init__()

        self.n_bands = n_bands

        # Create ViT for each band
        self.vits = nn.ModuleList([
            ViTModule(
                seq_length=seq_length,
                patch_size=1,
                in_channels=in_channels,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout
            )
            for _ in range(n_bands)
        ])

    def forward(self, band_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            band_features: Input of shape (batch_size, n_bands, n_channels, in_channels)

        Returns:
            Output of shape (batch_size, n_bands, n_channels, embed_dim)
        """
        batch_size, n_bands, n_channels, in_channels = band_features.shape

        outputs = []
        for i in range(n_bands):
            x = band_features[:, i, :, :]  # (batch_size, n_channels, in_channels)
            out = self.vits[i](x)  # (batch_size, n_channels, embed_dim)
            outputs.append(out)

        # Stack: (batch_size, n_bands, n_channels, embed_dim)
        return torch.stack(outputs, dim=1)


if __name__ == "__main__":
    # Test ViT module
    batch_size = 8
    n_channels = 19
    in_channels = 256
    embed_dim = 256

    # Create sample data (after GCN processing)
    x = torch.randn(batch_size, n_channels, in_channels)

    # Create ViT
    vit = ViTModule(
        seq_length=n_channels,
        in_channels=in_channels,
        embed_dim=embed_dim,
        num_heads=8,
        num_layers=4
    )

    # Forward pass
    output = vit(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Test with attention
    output, attn = vit(x, return_attention=True)
    print(f"Attention shape: {attn.shape}")
