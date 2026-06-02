"""
PyTorch Dataset classes for MSTS-AN.

Implements EEG dataset handling with multi-band frequency representations
and graph structure construction for GCN processing.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Callable
import os
import pickle


class EEGDataset(Dataset):
    """
    EEG Dataset for multi-band frequency analysis.

    Each sample consists of EEG segments from multiple frequency bands
    (delta, theta, alpha, beta) along with a graph adjacency matrix.

    Args:
        data_dict: Dictionary with keys 'delta', 'theta', 'alpha', 'beta',
                   each containing {'data': List[np.ndarray], 'labels': List[int]}
        adjacency_matrix: Graph adjacency matrix for GCN
        transform: Optional transform to apply to samples
    """

    def __init__(
        self,
        data_dict: Dict[str, Dict[str, List]],
        adjacency_matrix: Optional[np.ndarray] = None,
        transform: Optional[Callable] = None
    ):
        self.bands = ['delta', 'theta', 'alpha', 'beta']

        # Verify all bands have same number of samples
        n_samples = len(data_dict[self.bands[0]]['data'])
        for band in self.bands:
            assert len(data_dict[band]['data']) == n_samples, \
                f"Band {band} has different number of samples"

        self.data_dict = data_dict
        self.n_samples = n_samples

        # Store adjacency matrix (will be used by DataLoader collate_fn)
        self.adjacency_matrix = adjacency_matrix

        # Extract labels (assume same across all bands)
        if 'labels' in data_dict[self.bands[0]] and data_dict[self.bands[0]]['labels']:
            self.labels = np.array(data_dict[self.bands[0]]['labels'])
        else:
            self.labels = np.zeros(n_samples, dtype=np.int64)

        self.transform = transform

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get a sample from the dataset.

        Returns:
            Tuple of (band_data_dict, label)
            band_data_dict: Dictionary mapping band names to tensors of shape (n_channels, n_samples)
            label: Class label (0: HC, 1: MCI, 2: AD)
        """
        band_data = {}

        for band in self.bands:
            # Get data for this band
            data = self.data_dict[band]['data'][idx]

            # Convert to tensor
            data_tensor = torch.FloatTensor(data)

            # Apply transform if provided
            if self.transform:
                data_tensor = self.transform(data_tensor)

            band_data[band] = data_tensor

        label = torch.LongTensor([self.labels[idx]])[0]

        return band_data, label

    def get_adjacency_matrix(self) -> Optional[torch.Tensor]:
        """Get the graph adjacency matrix as a PyTorch tensor."""
        if self.adjacency_matrix is not None:
            return torch.FloatTensor(self.adjacency_matrix)
        return None

    def get_class_distribution(self) -> Dict[int, int]:
        """Get the distribution of classes in the dataset."""
        unique, counts = np.unique(self.labels, return_counts=True)
        return {int(k): int(v) for k, v in zip(unique, counts)}

    def get_band_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for each frequency band."""
        stats = {}
        for band in self.bands:
            data_list = self.data_dict[band]['data']
            all_data = np.stack([d.flatten() for d in data_list])
            stats[band] = {
                'mean': float(np.mean(all_data)),
                'std': float(np.std(all_data)),
                'min': float(np.min(all_data)),
                'max': float(np.max(all_data))
            }
        return stats


class EEGGraphDataset(Dataset):
    """
    EEG Dataset with graph structure for PyTorch Geometric.

    Each sample is a graph where nodes are EEG channels and edges represent
    spatial relationships between electrodes.
    """

    def __init__(
        self,
        data_dict: Dict[str, Dict[str, List]],
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        transform: Optional[Callable] = None
    ):
        self.bands = ['delta', 'theta', 'alpha', 'beta']
        self.data_dict = data_dict
        self.n_samples = len(data_dict[self.bands[0]]['data'])
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.transform = transform

        # Extract labels
        if 'labels' in data_dict[self.bands[0]] and data_dict[self.bands[0]]['labels']:
            self.labels = np.array(data_dict[self.bands[0]]['labels'])
        else:
            self.labels = np.zeros(self.n_samples, dtype=np.int64)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        """Get a graph sample."""
        from torch_geometric.data import Data

        # Combine all bands as node features
        # Shape: (n_channels, n_bands * n_samples_per_band)
        features = []
        for band in self.bands:
            data = self.data_dict[band]['data'][idx]  # (n_channels, n_samples)
            features.append(data)

        # Stack features: (n_channels, n_bands * n_samples)
        x = torch.FloatTensor(np.concatenate(features, axis=1))

        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            y=torch.LongTensor([self.labels[idx]])[0]
        )

        if self.transform:
            data = self.transform(data)

        return data


def collate_fn(batch: List[Tuple]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Custom collate function for batching EEG data.

    Args:
        batch: List of (band_data_dict, label) tuples

    Returns:
        Tuple of (batched_band_data, labels, adjacency_matrix)
    """
    band_data_list = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])

    # Stack data for each band
    batched_data = {}
    bands = band_data_list[0].keys()

    for band in bands:
        band_tensors = [sample[band] for sample in band_data_list]
        batched_data[band] = torch.stack(band_tensors)

    return batched_data, labels


def get_data_loaders(
    train_data: Dict[str, Dict[str, List]],
    val_data: Optional[Dict[str, Dict[str, List]]] = None,
    test_data: Optional[Dict[str, Dict[str, List]]] = None,
    adjacency_matrix: Optional[np.ndarray] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Create PyTorch DataLoaders for training, validation, and testing.

    Args:
        train_data: Training data dictionary
        val_data: Validation data dictionary (optional)
        test_data: Test data dictionary (optional)
        adjacency_matrix: Graph adjacency matrix
        batch_size: Batch size for training
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for GPU transfer

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = EEGDataset(train_data, adjacency_matrix)

    # Store adjacency matrix for retrieval during training
    if adjacency_matrix is not None:
        train_dataset.adjacency_matrix = torch.FloatTensor(adjacency_matrix)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=True
    )

    val_loader = None
    if val_data is not None:
        val_dataset = EEGDataset(val_data, adjacency_matrix)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn
        )

    test_loader = None
    if test_data is not None:
        test_dataset = EEGDataset(test_data, adjacency_matrix)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn
        )

    return train_loader, val_loader, test_loader


def save_dataset(dataset: EEGDataset, filepath: str):
    """Save dataset to disk."""
    with open(filepath, 'wb') as f:
        pickle.dump({
            'data_dict': dataset.data_dict,
            'adjacency_matrix': dataset.adjacency_matrix,
            'labels': dataset.labels
        }, f)


def load_dataset(filepath: str) -> EEGDataset:
    """Load dataset from disk."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    return EEGDataset(
        data_dict=data['data_dict'],
        adjacency_matrix=data['adjacency_matrix']
    )


if __name__ == "__main__":
    # Test dataset creation
    n_samples = 100
    n_channels = 19
    segment_length = 1024

    # Create synthetic data
    data_dict = {
        'delta': {
            'data': [np.random.randn(n_channels, segment_length) for _ in range(n_samples)],
            'labels': np.random.randint(0, 3, n_samples)
        },
        'theta': {
            'data': [np.random.randn(n_channels, segment_length) for _ in range(n_samples)],
            'labels': np.random.randint(0, 3, n_samples)
        },
        'alpha': {
            'data': [np.random.randn(n_channels, segment_length) for _ in range(n_samples)],
            'labels': np.random.randint(0, 3, n_samples)
        },
        'beta': {
            'data': [np.random.randn(n_channels, segment_length) for _ in range(n_samples)],
            'labels': np.random.randint(0, 3, n_samples)
        }
    }

    # Create dataset
    dataset = EEGDataset(data_dict)

    print(f"Dataset size: {len(dataset)}")
    print(f"Class distribution: {dataset.get_class_distribution()}")
    print(f"Band statistics: {dataset.get_band_statistics()}")

    # Test DataLoader
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

    for batch_data, batch_labels in loader:
        print(f"Batch shapes: {[(k, v.shape) for k, v in batch_data.items()]}")
        print(f"Label shape: {batch_labels.shape}")
        break
