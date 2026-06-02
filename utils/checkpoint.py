"""
Model checkpointing utilities for MSTS-AN.

Provides functions for saving and loading model checkpoints.
"""

import os
import torch
from typing import Optional, Dict


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    filepath: str,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    metrics: Optional[Dict] = None,
    config: Optional[Dict] = None
):
    """
    Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        filepath: Path to save checkpoint
        scheduler: Optional learning rate scheduler
        metrics: Optional metrics dictionary
        config: Optional configuration dictionary
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    if metrics is not None:
        checkpoint['metrics'] = metrics

    if config is not None:
        checkpoint['config'] = config

    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = 'cpu'
) -> Dict:
    """
    Load model checkpoint.

    Args:
        filepath: Path to checkpoint file
        model: Optional model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load checkpoint on

    Returns:
        Checkpoint dictionary
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=device)

    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint


def save_model_only(
    model: torch.nn.Module,
    filepath: str,
    metadata: Optional[Dict] = None
):
    """
    Save only the model weights (not training state).

    Args:
        model: Model to save
        filepath: Path to save model
        metadata: Optional metadata dictionary
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    save_dict = {
        'model_state_dict': model.state_dict()
    }

    if metadata is not None:
        save_dict['metadata'] = metadata

    torch.save(save_dict, filepath)


def load_model_only(
    filepath: str,
    model: torch.nn.Module,
    device: str = 'cpu'
) -> Dict:
    """
    Load only model weights.

    Args:
        filepath: Path to model file
        model: Model to load weights into
        device: Device to load on

    Returns:
        Dictionary containing metadata if available
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    return checkpoint.get('metadata', {})


class CheckpointManager:
    """
    Manages multiple checkpoints, keeping only the best ones.

    Args:
        checkpoint_dir: Directory to save checkpoints
        max_checkpoints: Maximum number of checkpoints to keep
        metric_name: Name of metric to track for best checkpoint
        mode: 'min' or 'max' for metric
    """

    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 5,
        metric_name: str = 'val_loss',
        mode: str = 'min'
    ):
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.metric_name = metric_name
        self.mode = mode

        os.makedirs(checkpoint_dir, exist_ok=True)

        self.checkpoints = []
        self.best_metric = float('inf') if mode == 'min' else float('-inf')

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ):
        """Save checkpoint and manage checkpoint files."""
        metric_value = metrics.get(self.metric_name, 0)

        # Determine if this is the best checkpoint
        is_best = False
        if self.mode == 'min':
            is_best = metric_value < self.best_metric
        else:
            is_best = metric_value > self.best_metric

        if is_best:
            self.best_metric = metric_value

            # Save best checkpoint
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, best_path, scheduler, metrics)

        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'checkpoint_epoch_{epoch}.pth'
        )
        save_checkpoint(model, optimizer, epoch, checkpoint_path, scheduler, metrics)

        self.checkpoints.append({
            'epoch': epoch,
            'path': checkpoint_path,
            'metric': metric_value
        })

        # Remove old checkpoints if exceeding max
        if len(self.checkpoints) > self.max_checkpoints:
            oldest = self.checkpoints.pop(0)
            if os.path.exists(oldest['path']):
                os.remove(oldest['path'])

    def load_best(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = 'cpu'
    ) -> Dict:
        """Load the best checkpoint."""
        best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
        return load_checkpoint(best_path, model, optimizer, device=device)

    def load_latest(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = 'cpu'
    ) -> Dict:
        """Load the most recent checkpoint."""
        if not self.checkpoints:
            raise ValueError("No checkpoints available")

        latest = self.checkpoints[-1]
        return load_checkpoint(latest['path'], model, optimizer, device=device)


if __name__ == "__main__":
    import tempfile
    import shutil

    # Create temporary directory for testing
    temp_dir = tempfile.mkdtemp()

    try:
        # Create dummy model
        model = torch.nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters())

        # Test saving
        checkpoint_path = os.path.join(temp_dir, 'checkpoint.pth')
        save_checkpoint(model, optimizer, 10, checkpoint_path, metrics={'accuracy': 0.95})
        print(f"Checkpoint saved to {checkpoint_path}")

        # Test loading
        checkpoint = load_checkpoint(checkpoint_path)
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"Metrics: {checkpoint.get('metrics', {})}")

        # Test checkpoint manager
        manager = CheckpointManager(temp_dir, max_checkpoints=3)
        for epoch in range(5):
            manager.save(model, optimizer, epoch, {'val_loss': 1.0 / (epoch + 1)})

        print("Checkpoint manager test passed!")

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
