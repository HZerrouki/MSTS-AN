"""
Evaluation Script for MSTS-AN.

Evaluates trained models and generates:
- Classification metrics (accuracy, precision, recall, F1, AUC)
- Confusion matrices
- ROC curves
- t-SNE visualizations
- Attention maps for explainability
"""

import os
import sys
import argparse
import yaml
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import MSTSAN
from data import EEGDataset, EEGGraphBuilder, collate_fn
from utils import (
    plot_confusion_matrix, plot_roc_curves, plot_tsne,
    plot_training_curves, compute_metrics, save_metrics_json
)
from torch.utils.data import DataLoader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate MSTS-AN model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to test data')
    parser.add_argument('--output_dir', type=str, default='results/evaluation',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--folds', type=int, nargs='+', default=None,
                       help='Specific folds to evaluate')
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path: str, config: Dict, device: str) -> MSTSAN:
    """Load trained model from checkpoint."""
    model_config = config['model']

    model = MSTSAN(
        n_channels=config['data'].get('n_channels', 19),
        seq_length=config['data']['segment_samples'],
        n_bands=len(config['data']['bands']),
        n_classes=config['data']['num_classes'],
        gcn_hidden_dims=model_config['gcn']['hidden_dims'],
        gcn_out_dim=model_config['gcn']['hidden_dims'][-1],
        vit_embed_dim=model_config['vit']['embed_dim'],
        vit_num_heads=model_config['vit']['num_heads'],
        vit_num_layers=model_config['vit']['num_layers'],
        fusion_hidden_dim=model_config['attention_fusion']['hidden_dim'],
        classifier_hidden_dims=model_config['classifier']['hidden_dims'],
        dropout=model_config['gcn']['dropout']
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model


def evaluate_model(
    model: MSTSAN,
    test_loader: DataLoader,
    edge_index: torch.Tensor,
    device: str
) -> Dict:
    """
    Evaluate model on test set.

    Returns:
        Dictionary containing predictions, labels, probabilities, and features
    """
    all_preds = []
    all_labels = []
    all_probs = []
    all_features = []
    all_band_weights = []

    with torch.no_grad():
        for batch_data, labels in tqdm(test_loader, desc='Evaluating'):
            # Move data to device
            for band in batch_data:
                batch_data[band] = batch_data[band].to(device)
            labels = labels.to(device)

            batch_size = labels.size(0)

            # Repeat edge_index for batch
            edge_index_batch = []
            n_nodes = batch_data['delta'].size(1)
            for b in range(batch_size):
                offset = b * n_nodes
                edge_index_batch.append(edge_index + offset)
            edge_index_batch = torch.cat(edge_index_batch, dim=1).to(device)

            # Forward pass
            logits, features, band_weights = model(
                batch_data, edge_index_batch,
                return_features=True, return_attention=True
            )

            # Get predictions
            probs = F.softmax(logits, dim=1)
            _, preds = torch.max(logits, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_features.append(features.cpu().numpy())
            all_band_weights.append(band_weights.cpu().numpy())

    return {
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'probabilities': np.array(all_probs),
        'features': np.concatenate(all_features, axis=0),
        'band_weights': np.concatenate(all_band_weights, axis=0)
    }


def compute_classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    class_names: List[str]
) -> Dict:
    """Compute comprehensive classification metrics."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report,
        roc_auc_score, roc_curve
    )

    metrics = {}

    # Basic metrics
    metrics['accuracy'] = accuracy_score(labels, predictions) * 100
    metrics['precision_macro'] = precision_score(labels, predictions, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(labels, predictions, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(labels, predictions, average='macro', zero_division=0)

    metrics['precision_weighted'] = precision_score(labels, predictions, average='weighted', zero_division=0)
    metrics['recall_weighted'] = recall_score(labels, predictions, average='weighted', zero_division=0)
    metrics['f1_weighted'] = f1_score(labels, predictions, average='weighted', zero_division=0)

    # Per-class metrics
    precision_per_class = precision_score(labels, predictions, average=None, zero_division=0)
    recall_per_class = recall_score(labels, predictions, average=None, zero_division=0)
    f1_per_class = f1_score(labels, predictions, average=None, zero_division=0)

    for i, name in enumerate(class_names):
        metrics[f'precision_{name}'] = precision_per_class[i]
        metrics[f'recall_{name}'] = recall_per_class[i]
        metrics[f'f1_{name}'] = f1_per_class[i]
        metrics[f'sensitivity_{name}'] = recall_per_class[i]

    # Specificity per class
    cm = confusion_matrix(labels, predictions)
    for i, name in enumerate(class_names):
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        fp = np.sum(cm[:, i]) - cm[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics[f'specificity_{name}'] = specificity

    # ROC-AUC
    try:
        metrics['auc_macro'] = roc_auc_score(labels, probabilities, multi_class='ovr', average='macro')
        metrics['auc_weighted'] = roc_auc_score(labels, probabilities, multi_class='ovr', average='weighted')

        # Per-class AUC
        for i, name in enumerate(class_names):
            binary_labels = (labels == i).astype(int)
            if len(np.unique(binary_labels)) > 1:
                metrics[f'auc_{name}'] = roc_auc_score(binary_labels, probabilities[:, i])
    except ValueError:
        metrics['auc_macro'] = 0.0
        metrics['auc_weighted'] = 0.0

    # Confusion matrix
    metrics['confusion_matrix'] = cm.tolist()

    # Classification report
    metrics['classification_report'] = classification_report(
        labels, predictions, target_names=class_names, zero_division=0
    )

    return metrics


def analyze_band_importance(band_weights: np.ndarray) -> Dict:
    """Analyze band attention weights."""
    band_names = ['delta', 'theta', 'alpha', 'beta']

    analysis = {}
    analysis['mean_weights'] = {
        band: float(np.mean(band_weights[:, i]))
        for i, band in enumerate(band_names)
    }
    analysis['std_weights'] = {
        band: float(np.std(band_weights[:, i]))
        for i, band in enumerate(band_names)
    }

    # Rank bands by importance
    mean_weights = [analysis['mean_weights'][band] for band in band_names]
    ranked_indices = np.argsort(mean_weights)[::-1]
    analysis['ranking'] = [band_names[i] for i in ranked_indices]

    return analysis


def main():
    """Main evaluation function."""
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create graph structure
    n_channels = config['data'].get('n_channels', 19)
    graph_builder = EEGGraphBuilder(n_channels=n_channels)
    edge_index = graph_builder.build_edge_index()

    # For demonstration, create dummy test data
    # In practice, load actual test data here
    n_samples = 52  # Test set size
    segment_samples = config['data']['segment_samples']

    dummy_test_data = {
        'delta': {
            'data': [torch.randn(n_channels, segment_samples) for _ in range(n_samples)],
            'labels': np.random.randint(0, 3, n_samples).tolist()
        },
        'theta': {
            'data': [torch.randn(n_channels, segment_samples) for _ in range(n_samples)],
            'labels': np.random.randint(0, 3, n_samples).tolist()
        },
        'alpha': {
            'data': [torch.randn(n_channels, segment_samples) for _ in range(n_samples)],
            'labels': np.random.randint(0, 3, n_samples).tolist()
        },
        'beta': {
            'data': [torch.randn(n_channels, segment_samples) for _ in range(n_samples)],
            'labels': np.random.randint(0, 3, n_samples).tolist()
        }
    }

    # Determine which folds to evaluate
    folds_to_evaluate = args.folds if args.folds else range(config['training']['n_splits'])

    all_results = []

    for fold in folds_to_evaluate:
        checkpoint_path = args.checkpoint.replace('{fold}', str(fold))
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            continue

        print(f"\n{'='*50}")
        print(f"Evaluating Fold {fold}")
        print(f"{'='*50}")

        # Load model
        model = load_model(checkpoint_path, config, device)
        print(f"Loaded model from {checkpoint_path}")

        # Create test dataset and loader
        test_dataset = EEGDataset(dummy_test_data)
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )

        # Evaluate
        results = evaluate_model(model, test_loader, edge_index, device)

        # Compute metrics
        class_names = config['data']['class_names']
        metrics = compute_classification_metrics(
            results['labels'],
            results['predictions'],
            results['probabilities'],
            class_names
        )

        # Analyze band importance
        band_analysis = analyze_band_importance(results['band_weights'])
        metrics['band_importance'] = band_analysis

        # Print results
        print("\nClassification Metrics:")
        print(f"  Accuracy: {metrics['accuracy']:.2f}%")
        print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
        print(f"  Recall (macro): {metrics['recall_macro']:.4f}")
        print(f"  F1-Score (macro): {metrics['f1_macro']:.4f}")
        print(f"  AUC (macro): {metrics['auc_macro']:.4f}")

        print("\nBand Importance Ranking:")
        for i, band in enumerate(band_analysis['ranking'], 1):
            weight = band_analysis['mean_weights'][band]
            print(f"  {i}. {band}: {weight:.4f}")

        # Generate visualizations
        fold_output_dir = os.path.join(args.output_dir, f'fold_{fold}')
        os.makedirs(fold_output_dir, exist_ok=True)

        # Confusion matrix
        plot_confusion_matrix(
            results['labels'],
            results['predictions'],
            class_names,
            save_path=os.path.join(fold_output_dir, 'confusion_matrix.png')
        )

        # ROC curves
        plot_roc_curves(
            results['labels'],
            results['probabilities'],
            class_names,
            save_path=os.path.join(fold_output_dir, 'roc_curves.png')
        )

        # t-SNE visualization
        plot_tsne(
            results['features'],
            results['labels'],
            class_names,
            save_path=os.path.join(fold_output_dir, 'tsne.png')
        )

        # Save metrics
        save_metrics_json(
            metrics,
            os.path.join(fold_output_dir, 'metrics.json')
        )

        all_results.append(metrics)

        print(f"\nResults saved to: {fold_output_dir}")

    # Compute and save average metrics across folds
    if len(all_results) > 1:
        avg_metrics = {}
        for key in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'auc_macro']:
            values = [m[key] for m in all_results if key in m]
            avg_metrics[f'{key}_mean'] = np.mean(values)
            avg_metrics[f'{key}_std'] = np.std(values)

        print("\n" + "="*50)
        print("Average Metrics Across Folds")
        print("="*50)
        print(f"Accuracy: {avg_metrics['accuracy_mean']:.2f}% ± {avg_metrics['accuracy_std']:.2f}%")
        print(f"F1-Score: {avg_metrics['f1_macro_mean']:.4f} ± {avg_metrics['f1_macro_std']:.4f}")
        print(f"AUC: {avg_metrics['auc_macro_mean']:.4f} ± {avg_metrics['auc_macro_std']:.4f}")

        save_metrics_json(
            avg_metrics,
            os.path.join(args.output_dir, 'average_metrics.json')
        )


if __name__ == "__main__":
    main()
