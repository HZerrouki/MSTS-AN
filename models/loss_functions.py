"""
Hybrid Loss Functions for MSTS-AN.

Implements the combined loss function described in the paper:
L_total = L_CE + λ_center * L_center + λ_triplet * L_triplet

Components:
- CrossEntropyLoss: Standard classification loss
- CenterLoss: Encourages compact feature clusters
- TripletLoss: Enforces inter-class separability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CenterLoss(nn.Module):
    """
    Center Loss for feature learning.

    Encourages features of the same class to be close to their class center,
    promoting compact and discriminative feature representations.

    Reference:
        Wen et al., "A Discriminative Feature Learning Approach for Deep Face Recognition"

    Args:
        num_classes: Number of classes
        feat_dim: Feature dimension
        alpha: Center update rate (0 < alpha < 1)
    """

    def __init__(
        self,
        num_classes: int = 3,
        feat_dim: int = 128,
        alpha: float = 0.5
    ):
        super(CenterLoss, self).__init__()

        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.alpha = alpha

        # Initialize class centers
        self.centers = nn.Parameter(
            torch.randn(num_classes, feat_dim)
        )

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute center loss.

        Args:
            features: Feature embeddings of shape (batch_size, feat_dim)
            labels: Ground truth labels of shape (batch_size,)

        Returns:
            Center loss value
        """
        batch_size = features.size(0)

        # Compute distances to centers
        # centers[i] is the center for class i
        centers_batch = self.centers[labels]  # (batch_size, feat_dim)

        # Squared Euclidean distance
        distances = torch.sum((features - centers_batch) ** 2, dim=1)

        # Average over batch
        loss = torch.mean(distances)

        return loss

    def update_centers(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ):
        """
        Update class centers (alternative implementation with explicit updates).

        Note: In practice, centers are typically updated via gradient descent
        along with other parameters. This method is kept for reference.
        """
        with torch.no_grad():
            for c in range(self.num_classes):
                mask = (labels == c)
                if mask.sum() > 0:
                    class_features = features[mask]
                    center_update = class_features.mean(dim=0)
                    self.centers[c] = (1 - self.alpha) * self.centers[c] + \
                                      self.alpha * center_update


class TripletLoss(nn.Module):
    """
    Triplet Loss for metric learning.

    Enforces that the distance between anchor and positive samples
    is smaller than distance between anchor and negative samples by a margin.

    Reference:
        Schroff et al., "FaceNet: A Unified Embedding for Face Recognition and Clustering"

    Args:
        margin: Margin for triplet loss (default: 0.2)
        distance_metric: Distance metric ('euclidean' or 'cosine')
        mining_strategy: Mining strategy ('batch_all', 'batch_hard')
    """

    def __init__(
        self,
        margin: float = 0.2,
        distance_metric: str = 'euclidean',
        mining_strategy: str = 'batch_all'
    ):
        super(TripletLoss, self).__init__()

        self.margin = margin
        self.distance_metric = distance_metric
        self.mining_strategy = mining_strategy

    def _compute_distances(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        """Compute pairwise distances."""
        if self.distance_metric == 'euclidean':
            # Efficient computation: ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x.y
            squared_norm = torch.sum(features ** 2, dim=1, keepdim=True)
            distances = squared_norm + squared_norm.T - 2 * torch.mm(features, features.T)
            distances = torch.sqrt(torch.clamp(distances, min=1e-8))
        elif self.distance_metric == 'cosine':
            # Cosine distance = 1 - cosine similarity
            normalized = F.normalize(features, p=2, dim=1)
            similarity = torch.mm(normalized, normalized.T)
            distances = 1 - similarity
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

        return distances

    def _batch_all_mining(
        self,
        distances: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Mine all valid triplets in the batch."""
        batch_size = labels.size(0)

        # Create masks for positive and negative pairs
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        labels_not_equal = ~labels_equal

        # Remove diagonal (self-comparisons)
        mask_pos = labels_equal.float() - torch.eye(batch_size, device=labels.device)
        mask_neg = labels_not_equal.float()

        # Compute triplet losses
        # For each anchor, find all valid (positive, negative) pairs
        anchor_pos_dist = distances.unsqueeze(2)  # (batch, batch, 1)
        anchor_neg_dist = distances.unsqueeze(1)  # (batch, 1, batch)

        # Triplet loss: max(d(a,p) - d(a,n) + margin, 0)
        triplet_loss = anchor_pos_dist - anchor_neg_dist + self.margin

        # Apply masks
        mask = mask_pos.unsqueeze(2) * mask_neg.unsqueeze(1)
        triplet_loss = mask * triplet_loss

        # Keep only positive losses
        triplet_loss = torch.clamp(triplet_loss, min=0)

        return triplet_loss

    def _batch_hard_mining(
        self,
        distances: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Mine hardest positive and negative for each anchor."""
        batch_size = labels.size(0)

        # Create masks
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask_pos = labels_equal.float() - torch.eye(batch_size, device=labels.device)
        mask_neg = (~labels_equal).float()

        # Hardest positive: max distance among positives
        pos_distances = distances.clone()
        pos_distances[mask_pos == 0] = -1
        hardest_pos = torch.max(pos_distances, dim=1)[0]

        # Hardest negative: min distance among negatives
        neg_distances = distances.clone()
        neg_distances[mask_neg == 0] = float('inf')
        hardest_neg = torch.min(neg_distances, dim=1)[0]

        # Triplet loss
        triplet_loss = torch.clamp(
            hardest_pos - hardest_neg + self.margin,
            min=0
        )

        return triplet_loss

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet loss.

        Args:
            features: Feature embeddings of shape (batch_size, feat_dim)
            labels: Ground truth labels of shape (batch_size,)

        Returns:
            Triplet loss value
        """
        distances = self._compute_distances(features)

        if self.mining_strategy == 'batch_all':
            triplet_loss = self._batch_all_mining(distances, labels)
            # Count valid triplets
            num_positive_triplets = (triplet_loss > 1e-16).float().sum()
            loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)
        elif self.mining_strategy == 'batch_hard':
            loss = self._batch_hard_mining(distances, labels).mean()
        else:
            raise ValueError(f"Unknown mining strategy: {self.mining_strategy}")

        return loss


class HybridLoss(nn.Module):
    """
    Hybrid Loss combining Cross-Entropy, Center Loss, and Triplet Loss.

    L_total = L_CE + λ_center * L_center + λ_triplet * L_triplet

    Args:
        num_classes: Number of classes
        feat_dim: Feature dimension for center loss
        lambda_center: Weight for center loss
        lambda_triplet: Weight for triplet loss
        triplet_margin: Margin for triplet loss
        label_smoothing: Label smoothing for cross-entropy
    """

    def __init__(
        self,
        num_classes: int = 3,
        feat_dim: int = 128,
        lambda_center: float = 0.01,
        lambda_triplet: float = 0.001,
        triplet_margin: float = 0.2,
        label_smoothing: float = 0.1
    ):
        super(HybridLoss, self).__init__()

        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.lambda_center = lambda_center
        self.lambda_triplet = lambda_triplet

        # Cross-entropy loss
        self.ce_loss = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing
        )

        # Center loss
        self.center_loss = CenterLoss(
            num_classes=num_classes,
            feat_dim=feat_dim
        )

        # Triplet loss
        self.triplet_loss = TripletLoss(
            margin=triplet_margin,
            mining_strategy='batch_all'
        )

    def forward(
        self,
        logits: torch.Tensor,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> tuple:
        """
        Compute hybrid loss.

        Args:
            logits: Model output logits of shape (batch_size, num_classes)
            features: Feature embeddings of shape (batch_size, feat_dim)
            labels: Ground truth labels of shape (batch_size,)

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Cross-entropy loss
        loss_ce = self.ce_loss(logits, labels)

        # Center loss
        loss_center = self.center_loss(features, labels)

        # Triplet loss
        loss_triplet = self.triplet_loss(features, labels)

        # Total loss
        loss_total = loss_ce + \
                     self.lambda_center * loss_center + \
                     self.lambda_triplet * loss_triplet

        loss_dict = {
            'total': loss_total.item(),
            'ce': loss_ce.item(),
            'center': loss_center.item(),
            'triplet': loss_triplet.item()
        }

        return loss_total, loss_dict


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    Down-weights easy examples and focuses on hard examples.

    Reference:
        Lin et al., "Focal Loss for Dense Object Detection"

    Args:
        alpha: Weighting factor (optional)
        gamma: Focusing parameter
        reduction: Reduction method
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super(FocalLoss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Model output logits
            labels: Ground truth labels

        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(
            logits, labels,
            weight=self.alpha,
            reduction='none'
        )

        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


if __name__ == "__main__":
    # Test losses
    batch_size = 32
    feat_dim = 128
    num_classes = 3

    # Create sample data
    features = torch.randn(batch_size, feat_dim)
    labels = torch.randint(0, num_classes, (batch_size,))

    print("Testing Center Loss...")
    center_loss = CenterLoss(num_classes=num_classes, feat_dim=feat_dim)
    loss_c = center_loss(features, labels)
    print(f"Center loss: {loss_c.item():.4f}")

    print("\nTesting Triplet Loss...")
    triplet_loss = TripletLoss(margin=0.2, mining_strategy='batch_all')
    loss_t = triplet_loss(features, labels)
    print(f"Triplet loss: {loss_t.item():.4f}")

    print("\nTesting Hybrid Loss...")
    logits = torch.randn(batch_size, num_classes)
    hybrid_loss = HybridLoss(
        num_classes=num_classes,
        feat_dim=feat_dim,
        lambda_center=0.01,
        lambda_triplet=0.001
    )
    loss_h, loss_dict = hybrid_loss(logits, features, labels)
    print(f"Hybrid loss components: {loss_dict}")
