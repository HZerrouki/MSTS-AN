"""
Training Script for MSTS-AN.

Implements the training pipeline with:
- 5-fold cross-validation
- Early stopping
- Model checkpointing
- Hyperparameter tuning with Optuna (optional)
"""

import os
import sys
import yaml
import argparse
import logging
from datetime import datetime
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import MSTSAN, HybridLoss
from data import EEGPreprocessor, EEGDataset, get_data_loaders, EEGGraphBuilder
from utils import MetricsTracker, Logger, set_seed, save_checkpoint, load_checkpoint


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train MSTS-AN model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--data_path', type=str, default='data/processed',
                       help='Path to processed data')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--fold', type=int, default=None,
                       help='Specific fold to train (for parallel training)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--tune', action='store_true',
                       help='Run hyperparameter tuning with Optuna')
    parser.add_argument('--n_trials', type=int, default=100,
                       help='Number of Optuna trials')
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: Dict, device: str) -> MSTSAN:
    """Create MSTS-AN model from config."""
    model_config = config['model']

    model = MSTSAN(
        n_channels=config['data']['n_channels'] if 'n_channels' in config['data'] else 19,
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

    return model.to(device)


def create_optimizer(model: nn.Module, config: Dict):
    """Create optimizer from config."""
    train_config = config['training']

    optimizer = Adam(
        model.parameters(),
        lr=train_config['learning_rate'],
        weight_decay=train_config['weight_decay']
    )

    return optimizer


def create_scheduler(optimizer, config: Dict):
    """Create learning rate scheduler."""
    train_config = config['training']

    if train_config.get('use_scheduler', False):
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=train_config.get('scheduler_factor', 0.5),
            patience=train_config.get('scheduler_patience', 10),
            verbose=True
        )
        return scheduler
    return None


def create_criterion(config: Dict, device: str):
    """Create loss function from config."""
    loss_config = config['training']['loss']

    criterion = HybridLoss(
        num_classes=config['data']['num_classes'],
        feat_dim=config['model']['attention_fusion']['hidden_dim'],
        lambda_center=loss_config['lambda_center'],
        lambda_triplet=loss_config['lambda_triplet'],
        triplet_margin=loss_config['triplet_margin'],
        label_smoothing=config['training'].get('label_smoothing', 0.0)
    )

    return criterion.to(device)


def train_epoch(
    model: MSTSAN,
    train_loader,
    criterion,
    optimizer,
    edge_index,
    device: str
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_center_loss = 0
    total_triplet_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc='Training')
    for batch_data, labels in pbar:
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
        optimizer.zero_grad()
        logits, features = model(
            batch_data, edge_index_batch,
            return_features=True
        )

        # Compute loss
        loss, loss_dict = criterion(logits, features, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item() * batch_size
        total_ce_loss += loss_dict['ce'] * batch_size
        total_center_loss += loss_dict['center'] * batch_size
        total_triplet_loss += loss_dict['triplet'] * batch_size

        _, predicted = torch.max(logits, 1)
        total += batch_size
        correct += (predicted == labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{100*correct/total:.2f}%"
        })

    metrics = {
        'loss': total_loss / total,
        'ce_loss': total_ce_loss / total,
        'center_loss': total_center_loss / total,
        'triplet_loss': total_triplet_loss / total,
        'accuracy': 100 * correct / total
    }

    return metrics


def validate(
    model: MSTSAN,
    val_loader,
    criterion,
    edge_index,
    device: str
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    total_loss = 0
    total_ce_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_data, labels in tqdm(val_loader, desc='Validation'):
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
            logits, features = model(
                batch_data, edge_index_batch,
                return_features=True
            )

            # Compute loss
            loss, loss_dict = criterion(logits, features, labels)

            # Track metrics
            total_loss += loss.item() * batch_size
            total_ce_loss += loss_dict['ce'] * batch_size

            probs = torch.softmax(logits, dim=1)
            _, predicted = torch.max(logits, 1)

            total += batch_size
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Compute additional metrics
    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        confusion_matrix, roc_auc_score
    )

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    metrics = {
        'loss': total_loss / total,
        'ce_loss': total_ce_loss / total,
        'accuracy': 100 * correct / total,
        'precision': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }

    return metrics


def train_fold(
    model: MSTSAN,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    edge_index,
    config: Dict,
    fold: int,
    output_dir: str,
    device: str,
    logger: logging.Logger
) -> Dict[str, float]:
    """Train for one fold."""
    best_val_loss = float('inf')
    patience_counter = 0
    patience = config['training']['early_stopping_patience']

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(config['training']['num_epochs']):
        logger.info(f"Epoch {epoch+1}/{config['training']['num_epochs']}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer,
            edge_index, device
        )

        # Validate
        val_metrics = validate(
            model, val_loader, criterion, edge_index, device
        )

        # Update scheduler
        if scheduler is not None:
            scheduler.step(val_metrics['loss'])

        # Log metrics
        logger.info(
            f"Train - Loss: {train_metrics['loss']:.4f}, "
            f"Acc: {train_metrics['accuracy']:.2f}%"
        )
        logger.info(
            f"Val - Loss: {val_metrics['loss']:.4f}, "
            f"Acc: {val_metrics['accuracy']:.2f}%, "
            f"F1: {val_metrics['f1']:.4f}"
        )

        # Track history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0

            save_checkpoint(
                model, optimizer, epoch,
                os.path.join(output_dir, f'best_model_fold{fold}.pth')
            )
            logger.info("Saved best model")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model
    load_checkpoint(
        os.path.join(output_dir, f'best_model_fold{fold}.pth'),
        model, optimizer
    )

    # Final evaluation
    final_metrics = validate(
        model, val_loader, criterion, edge_index, device
    )

    # Save history
    np.save(
        os.path.join(output_dir, f'training_history_fold{fold}.npy'),
        history
    )

    return final_metrics


def main():
    """Main training function."""
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Set seed for reproducibility
    set_seed(config['reproducibility']['seed'])

    # Setup output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    # Setup logging
    logger = Logger(os.path.join(output_dir, 'train.log')).get_logger()
    logger.info("Starting MSTS-AN training")
    logger.info(f"Config: {config}")

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load data (placeholder - user needs to implement data loading)
    logger.info("Loading data...")
    # data = load_data(args.data_path)

    # For demonstration, create dummy data
    from data.dataset import EEGDataset, collate_fn
    from torch.utils.data import DataLoader

    n_samples = 156
    n_channels = 19
    segment_samples = 1024

    dummy_data = {
        'delta': {
            'data': [torch.randn(n_channels, segment_samples) for _ in range(n_samples)],
            'labels': torch.randint(0, 3, (n_samples,)).tolist()
        },
        'theta': {
            'data': [torch.randn(n_channels, segment_samples) for _ in range(n_samples)],
            'labels': torch.randint(0, 3, (n_samples,)).tolist()
        },
        'alpha': {
            'data': [torch.randn(n_channels, segment_samples) for _ in range(n_samples)],
            'labels': torch.randint(0, 3, (n_samples,)).tolist()
        },
        'beta': {
            'data': [torch.randn(n_channels, segment_samples) for _ in range(n_samples)],
            'labels': torch.randint(0, 3, (n_samples,)).tolist()
        }
    }

    # Create graph
    graph_builder = EEGGraphBuilder(n_channels=n_channels)
    edge_index = graph_builder.build_edge_index()

    labels = np.array(dummy_data['delta']['labels'])

    # 5-fold cross-validation
    skf = StratifiedKFold(
        n_splits=config['training']['n_splits'],
        shuffle=True,
        random_state=config['reproducibility']['seed']
    )

    all_fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        if args.fold is not None and fold != args.fold:
            continue

        logger.info(f"\n{'='*50}")
        logger.info(f"Fold {fold + 1}/{config['training']['n_splits']}")
        logger.info(f"{'='*50}")

        # Create datasets for this fold
        def create_fold_data(indices):
            return {
                band: {
                    'data': [dummy_data[band]['data'][i] for i in indices],
                    'labels': [dummy_data[band]['labels'][i] for i in indices]
                }
                for band in ['delta', 'theta', 'alpha', 'beta']
            }

        train_data = create_fold_data(train_idx)
        val_data = create_fold_data(val_idx)

        # Create data loaders
        train_dataset = EEGDataset(train_data)
        val_dataset = EEGDataset(val_data)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )

        # Create model
        model = create_model(config, device)
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Create optimizer, scheduler, criterion
        optimizer = create_optimizer(model, config)
        scheduler = create_scheduler(optimizer, config)
        criterion = create_criterion(config, device)

        # Train fold
        fold_metrics = train_fold(
            model, train_loader, val_loader,
            criterion, optimizer, scheduler,
            edge_index, config, fold, output_dir,
            device, logger
        )

        all_fold_metrics.append(fold_metrics)

        logger.info(f"Fold {fold+1} - Accuracy: {fold_metrics['accuracy']:.2f}%, F1: {fold_metrics['f1']:.4f}")

    # Compute average metrics
    logger.info("\n" + "="*50)
    logger.info("Cross-Validation Results")
    logger.info("="*50)

    avg_acc = np.mean([m['accuracy'] for m in all_fold_metrics])
    std_acc = np.std([m['accuracy'] for m in all_fold_metrics])
    avg_f1 = np.mean([m['f1'] for m in all_fold_metrics])

    logger.info(f"Average Accuracy: {avg_acc:.2f}% ± {std_acc:.2f}%")
    logger.info(f"Average F1-Score: {avg_f1:.4f}")

    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
