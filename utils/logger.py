"""
Logging utilities for MSTS-AN.

Provides consistent logging across the project and utilities for
ensuring reproducibility.
"""

import os
import sys
import logging
import random
from datetime import datetime
from typing import Optional

import numpy as np
import torch


class Logger:
    """
    Custom logger for MSTS-AN.

    Creates both file and console handlers with consistent formatting.

    Args:
        log_file: Path to log file
        log_level: Logging level (default: INFO)
        name: Logger name
    """

    def __init__(
        self,
        log_file: Optional[str] = None,
        log_level: int = logging.INFO,
        name: str = "MSTSAN"
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)

        # Clear existing handlers
        self.logger.handlers = []

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def get_logger(self) -> logging.Logger:
        """Get the configured logger."""
        return self.logger


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make PyTorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"Random seed set to {seed}")


def log_system_info(logger: logging.Logger):
    """Log system information."""
    import platform

    logger.info("="*50)
    logger.info("System Information")
    logger.info("="*50)
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    logger.info("="*50)


def log_config(logger: logging.Logger, config: dict):
    """Log configuration dictionary."""
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")


def log_model_info(logger: logging.Logger, model: torch.nn.Module):
    """Log model architecture and parameter count."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info("Model Information:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")


class TensorBoardLogger:
    """
    TensorBoard logger wrapper.

    Args:
        log_dir: Directory for tensorboard logs
    """

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.writer = None

        try:
            from torch.utils.tensorboard import SummaryWriter
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
        except ImportError:
            print("TensorBoard not available. Install with: pip install tensorboard")

    def log_scalar(self, tag: str, value: float, step: int):
        """Log scalar value."""
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int):
        """Log multiple scalars."""
        if self.writer:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_histogram(self, tag: str, values, step: int):
        """Log histogram of values."""
        if self.writer:
            self.writer.add_histogram(tag, values, step)

    def log_model_graph(self, model, input_sample):
        """Log model graph."""
        if self.writer:
            self.writer.add_graph(model, input_sample)

    def close(self):
        """Close the writer."""
        if self.writer:
            self.writer.close()


if __name__ == "__main__":
    # Test logger
    logger = Logger("test.log").get_logger()
    logger.info("Test info message")
    logger.warning("Test warning message")
    logger.error("Test error message")

    # Test seed setting
    set_seed(42)

    # Test system info
    log_system_info(logger)
