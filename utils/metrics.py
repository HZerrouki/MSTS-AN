"""
Metrics computation utilities for MSTS-AN.

Implements classification metrics and tracking for model evaluation.
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score as sklearn_f1,
    confusion_matrix, roc_auc_score, roc_curve,
    classification_report
)


class MetricsTracker:
    """
    Track and aggregate metrics during training and evaluation.

    Attributes:
        metrics_history: Dictionary storing metric values over time
    """

    def __init__(self):
        self.metrics_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'val_auc': []
        }
        self.best_metrics = {}

    def update(self, split: str, metrics: Dict[str, float]):
        """Update metrics for a given split (train/val/test)."""
        for key, value in metrics.items():
            metric_key = f"{split}_{key}"
            if metric_key not in self.metrics_history:
                self.metrics_history[metric_key] = []
            self.metrics_history[metric_key].append(value)

    def get_best_metric(self, metric_name: str, mode: str = 'max') -> Tuple[float, int]:
        """
        Get best value and epoch for a metric.

        Args:
            metric_name: Name of the metric
            mode: 'max' or 'min'

        Returns:
            Tuple of (best_value, best_epoch)
        """
        values = self.metrics_history.get(metric_name, [])
        if not values:
            return 0.0, 0

        if mode == 'max':
            best_value = max(values)
        else:
            best_value = min(values)

        best_epoch = values.index(best_value)
        return best_value, best_epoch

    def get_latest(self, metric_name: str) -> float:
        """Get latest value of a metric."""
        values = self.metrics_history.get(metric_name, [])
        return values[-1] if values else 0.0

    def summary(self) -> Dict:
        """Get summary of all tracked metrics."""
        summary = {}
        for key, values in self.metrics_history.items():
            if values:
                summary[key] = {
                    'latest': values[-1],
                    'best': max(values) if 'loss' not in key else min(values),
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
        return summary


def accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Compute accuracy."""
    return accuracy_score(labels, predictions) * 100


def sensitivity(labels: np.ndarray, predictions: np.ndarray, average: str = 'macro') -> float:
    """
    Compute sensitivity (recall).

    Sensitivity = TP / (TP + FN)
    """
    return recall_score(labels, predictions, average=average, zero_division=0)


def specificity(labels: np.ndarray, predictions: np.ndarray) -> Dict[int, float]:
    """
    Compute specificity for each class.

    Specificity = TN / (TN + FP)
    """
    cm = confusion_matrix(labels, predictions)
    specificities = {}

    for i in range(cm.shape[0]):
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        fp = np.sum(cm[:, i]) - cm[i, i]
        specificities[i] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return specificities


def f1_score(labels: np.ndarray, predictions: np.ndarray, average: str = 'macro') -> float:
    """Compute F1 score."""
    return sklearn_f1(labels, predictions, average=average, zero_division=0)


def auc_score(
    labels: np.ndarray,
    probabilities: np.ndarray,
    multi_class: str = 'ovr',
    average: str = 'macro'
) -> float:
    """Compute AUC-ROC score."""
    try:
        return roc_auc_score(labels, probabilities, multi_class=multi_class, average=average)
    except ValueError:
        return 0.0


def compute_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    Compute comprehensive classification metrics.

    Args:
        labels: Ground truth labels
        predictions: Model predictions
        probabilities: Class probabilities (optional, for AUC)
        class_names: List of class names

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Basic metrics
    metrics['accuracy'] = accuracy(labels, predictions)
    metrics['precision_macro'] = precision_score(labels, predictions, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(labels, predictions, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(labels, predictions, average='macro')

    metrics['precision_weighted'] = precision_score(labels, predictions, average='weighted', zero_division=0)
    metrics['recall_weighted'] = recall_score(labels, predictions, average='weighted', zero_division=0)
    metrics['f1_weighted'] = f1_score(labels, predictions, average='weighted')

    # Per-class metrics
    n_classes = len(np.unique(labels))
    precision_per_class = precision_score(labels, predictions, average=None, zero_division=0)
    recall_per_class = recall_score(labels, predictions, average=None, zero_division=0)
    f1_per_class = sklearn_f1(labels, predictions, average=None, zero_division=0)

    for i in range(n_classes):
        class_name = class_names[i] if class_names else f'class_{i}'
        metrics[f'precision_{class_name}'] = precision_per_class[i]
        metrics[f'recall_{class_name}'] = recall_per_class[i]
        metrics[f'f1_{class_name}'] = f1_per_class[i]
        metrics[f'sensitivity_{class_name}'] = recall_per_class[i]

    # Specificity per class
    specificities = specificity(labels, predictions)
    for i in range(n_classes):
        class_name = class_names[i] if class_names else f'class_{i}'
        metrics[f'specificity_{class_name}'] = specificities[i]

    # AUC
    if probabilities is not None:
        metrics['auc_macro'] = auc_score(labels, probabilities, 'ovr', 'macro')
        metrics['auc_weighted'] = auc_score(labels, probabilities, 'ovr', 'weighted')

        # Per-class AUC
        for i in range(n_classes):
            class_name = class_names[i] if class_names else f'class_{i}'
            binary_labels = (labels == i).astype(int)
            if len(np.unique(binary_labels)) > 1:
                metrics[f'auc_{class_name}'] = roc_auc_score(binary_labels, probabilities[:, i])

    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(labels, predictions).tolist()

    return metrics


def save_metrics_json(metrics: Dict, filepath: str):
    """Save metrics to JSON file."""
    # Convert numpy types to native Python types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(item) for item in obj]
        return obj

    metrics = convert(metrics)

    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)


def load_metrics_json(filepath: str) -> Dict:
    """Load metrics from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    # Test metrics
    n_samples = 100
    n_classes = 3

    labels = np.random.randint(0, n_classes, n_samples)
    predictions = np.random.randint(0, n_classes, n_samples)
    probabilities = np.random.rand(n_samples, n_classes)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)

    class_names = ['HC', 'MCI', 'AD']

    metrics = compute_metrics(labels, predictions, probabilities, class_names)

    print("Classification Metrics:")
    for key, value in metrics.items():
        if key != 'confusion_matrix':
            print(f"  {key}: {value}")
