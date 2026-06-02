"""
Visualization utilities for MSTS-AN.

Generates publication-quality figures for:
- Confusion matrices
- ROC curves
- t-SNE embeddings
- Training curves
- Attention maps
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.2)
sns.set_palette("husl")


def plot_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: List[str],
    normalize: bool = True,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix"
) -> plt.Figure:
    """
    Plot confusion matrix.

    Args:
        labels: Ground truth labels
        predictions: Model predictions
        class_names: List of class names
        normalize: Whether to normalize values
        figsize: Figure size
        save_path: Path to save figure
        title: Plot title

    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(labels, predictions)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm, annot=True, fmt=fmt, cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )

    ax.set_xlabel('Predicted Label', fontweight='bold')
    ax.set_ylabel('True Label', fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")

    return fig


def plot_roc_curves(
    labels: np.ndarray,
    probabilities: np.ndarray,
    class_names: List[str],
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    title: str = "ROC Curves"
) -> plt.Figure:
    """
    Plot ROC curves for multi-class classification.

    Args:
        labels: Ground truth labels
        probabilities: Predicted probabilities (n_samples, n_classes)
        class_names: List of class names
        figsize: Figure size
        save_path: Path to save figure
        title: Plot title

    Returns:
        Matplotlib figure
    """
    n_classes = len(class_names)

    # Compute ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(n_classes):
        binary_labels = (labels == i).astype(int)
        fpr[i], tpr[i], _ = roc_curve(binary_labels, probabilities[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    for i, color in enumerate(colors):
        ax.plot(
            fpr[i], tpr[i], color=color, lw=2,
            label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})'
        )

    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved ROC curves to {save_path}")

    return fig


def plot_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    perplexity: int = 30,
    n_iter: int = 1000,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    title: str = "t-SNE Visualization"
) -> plt.Figure:
    """
    Plot t-SNE visualization of feature embeddings.

    Args:
        features: Feature embeddings (n_samples, n_features)
        labels: Ground truth labels
        class_names: List of class names
        perplexity: t-SNE perplexity
        n_iter: Number of iterations
        figsize: Figure size
        save_path: Path to save figure
        title: Plot title

    Returns:
        Matplotlib figure
    """
    # Apply t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=42
    )
    embeddings_2d = tsne.fit_transform(features)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))

    for i, class_name in enumerate(class_names):
        mask = labels == i
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[i]],
            label=class_name,
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )

    ax.set_xlabel('t-SNE Component 1', fontweight='bold')
    ax.set_ylabel('t-SNE Component 2', fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(frameon=True, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved t-SNE plot to {save_path}")

    return fig


def plot_training_curves(
    history: Dict[str, List[float]],
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training curves (loss and accuracy).

    Args:
        history: Dictionary containing 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    epochs = range(1, len(history.get('train_loss', [])) + 1)

    # Loss plot
    if 'train_loss' in history:
        ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    if 'val_loss' in history:
        ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Loss', fontweight='bold')
    ax1.set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
    ax1.legend(frameon=True)
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    if 'train_acc' in history:
        ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    if 'val_acc' in history:
        ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontweight='bold')
    ax2.set_title('Training and Validation Accuracy', fontsize=12, fontweight='bold')
    ax2.legend(frameon=True)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")

    return fig


def plot_band_attention(
    band_weights: np.ndarray,
    class_labels: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot band attention weights.

    Args:
        band_weights: Attention weights (n_samples, n_bands)
        class_labels: Optional class labels for grouping
        class_names: List of class names
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    band_names = ['Delta', 'Theta', 'Alpha', 'Beta']

    fig, ax = plt.subplots(figsize=figsize)

    if class_labels is not None and class_names is not None:
        # Group by class
        for i, class_name in enumerate(class_names):
            mask = class_labels == i
            class_weights = band_weights[mask].mean(axis=0)
            ax.bar(
                np.arange(len(band_names)) + i * 0.2,
                class_weights,
                width=0.2,
                label=class_name,
                alpha=0.8
            )
        ax.legend(frameon=True)
    else:
        # Plot mean and std
        mean_weights = band_weights.mean(axis=0)
        std_weights = band_weights.std(axis=0)

        ax.bar(
            range(len(band_names)),
            mean_weights,
            yerr=std_weights,
            capsize=5,
            alpha=0.8,
            color='steelblue'
        )

    ax.set_xticks(range(len(band_names)))
    ax.set_xticklabels(band_names)
    ax.set_ylabel('Attention Weight', fontweight='bold')
    ax.set_xlabel('Frequency Band', fontweight='bold')
    ax.set_title('Band-Specific Attention Weights', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved band attention plot to {save_path}")

    return fig


def plot_attention_maps(
    attention_weights: np.ndarray,
    channel_names: Optional[List[str]] = None,
    band_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot attention maps for all bands.

    Args:
        attention_weights: Attention weights (n_bands, n_channels)
        channel_names: List of channel names
        band_names: List of band names
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    if band_names is None:
        band_names = ['Delta', 'Theta', 'Alpha', 'Beta']

    if channel_names is None:
        n_channels = attention_weights.shape[1]
        channel_names = [f'Ch{i+1}' for i in range(n_channels)]

    n_bands = len(band_names)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for i, (band_name, ax) in enumerate(zip(band_names, axes)):
        weights = attention_weights[i]

        # Create heatmap
        im = ax.imshow(weights.reshape(1, -1), cmap='viridis', aspect='auto')
        ax.set_xticks(range(len(channel_names)))
        ax.set_xticklabels(channel_names, rotation=90, fontsize=8)
        ax.set_yticks([])
        ax.set_title(f'{band_name} Band', fontsize=12, fontweight='bold')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
        cbar.set_label('Attention', fontsize=10)

    plt.suptitle('Band-Specific Attention Maps', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved attention maps to {save_path}")

    return fig


if __name__ == "__main__":
    # Test visualizations
    n_samples = 100
    n_classes = 3
    n_features = 128

    labels = np.random.randint(0, n_classes, n_samples)
    predictions = np.random.randint(0, n_classes, n_samples)
    probabilities = np.random.rand(n_samples, n_classes)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    features = np.random.randn(n_samples, n_features)

    class_names = ['HC', 'MCI', 'AD']

    # Test confusion matrix
    fig = plot_confusion_matrix(labels, predictions, class_names)
    plt.close()

    # Test ROC curves
    fig = plot_roc_curves(labels, probabilities, class_names)
    plt.close()

    # Test t-SNE
    fig = plot_tsne(features, labels, class_names)
    plt.close()

    print("All visualization tests passed!")
