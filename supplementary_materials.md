# Supplementary Materials: MSTS-AN Code Documentation

## Overview

This document provides detailed documentation of the MSTS-AN source code repository to accompany the paper *"MSTS-AN: A Hybrid GCN-Transformer Approach for Early Alzheimer's Detection from EEG Signals"*.

**Repository URL**: https://github.com/HZerrouki/MSTS-AN

---

## 1. Code Repository Structure

The complete source code is organized as follows:

```
MSTS-AN/
├── README.md                      # User documentation and usage guide
├── requirements.txt               # Python package dependencies
├── setup.py                       # Package installation configuration
├── configs/
│   └── config.yaml               # Centralized hyperparameter configuration
├── data/                          # Data processing modules
│   ├── __init__.py
│   ├── preprocessor.py           # EEG preprocessing pipeline (FIR, ICA, DWT)
│   ├── dataset.py                # PyTorch Dataset and DataLoader classes
│   └── graph_builder.py          # GCN adjacency matrix construction
├── models/                        # Neural network architectures
│   ├── __init__.py
│   ├── msts_an.py                # Main MSTS-AN architecture (Section III.C)
│   ├── gcn_module.py             # Graph Convolutional Network (Eq. 4)
│   ├── vit_module.py             # Vision Transformer (Eqs. 6-8)
│   ├── attention_fusion.py       # Hybrid attention fusion (Eqs. 9-10)
│   └── loss_functions.py         # Hybrid loss function (Eqs. 11-15)
├── train.py                      # Training script with 5-fold CV
├── evaluate.py                   # Evaluation and visualization script
├── utils/                        # Utility functions
│   ├── metrics.py                # Classification metrics computation
│   ├── visualization.py          # t-SNE, Grad-CAM, attention maps
│   ├── logger.py                 # Training logger
│   └── checkpoint.py             # Model checkpoint management
├── scripts/                      # Bash scripts for experiments
│   ├── train_5fold.sh            # 5-fold cross-validation execution
│   └── run_experiments.sh        # Complete experimental pipeline
├── notebooks/                    # Interactive tutorials
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_visualization.ipynb
│   └── 03_biomarker_analysis.ipynb
└── pretrained/                   # Pretrained model weights (to be added)
    └── README.md
```

---

## 2. Implementation Details

### 2.1 Multi-Scale Feature Extraction (Section III.C.1)

**File**: `data/preprocessor.py`

Implements wavelet decomposition using Daubechies 4 (db4) wavelet:

```python
# Wavelet decomposition parameters
WAVELET = 'db4'
DECOMPOSITION_LEVEL = 4

# Frequency bands (Hz)
BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),      # Primary biomarker band
    'alpha': (8, 13),     # Primary biomarker band
    'beta': (13, 30)
}
```

**Key Functions**:
- `bandpass_filter()`: FIR filter with Hamming window (0.5-45 Hz, 512 taps)
- `apply_ica()`: Extended Infomax ICA with ICLabel artifact removal
- `wavelet_decompose()`: 4-level DWT yielding band-specific coefficients
- `segment_eeg()`: 4-second non-overlapping windows (1024 samples @ 256 Hz)

### 2.2 GCN Spatial Encoding (Section III.C.2, Equation 4)

**File**: `models/gcn_module.py`

Implements normalized graph Laplacian and graph convolution:

```python
class GraphConvolution(nn.Module):
    """
    Graph Convolution Layer: H^(l+1) = σ(D̃^(-1/2) Ã D̃^(-1/2) H^(l) W^(l))

    Where:
    - Ã = A + I (adjacency matrix with self-connections)
    - D̃ is the degree matrix
    - σ is ReLU activation
    """
```

**Graph Construction**:
- Adjacency matrix based on 10-20 EEG electrode system
- Spatial neighbors within 10cm on scalp surface
- Self-connections added (Ã = A + I)

**Architecture**:
- Input: 19 channels (10-20 system)
- Hidden dimensions: [64, 128, 256]
- Dropout: 0.3
- Batch normalization applied

### 2.3 Vision Transformer Temporal Encoding (Section III.C.3, Equations 6-8)

**File**: `models/vit_module.py`

Implements Vision Transformer for temporal feature extraction:

```python
class VisionTransformer(nn.Module):
    """
    Vision Transformer with:
    - Patch embedding (patch_size=16, embed_dim=256)
    - Multi-head self-attention (num_heads=8)
    - 4 transformer encoder blocks
    - Feed-forward network (MLP ratio=4)
    """
```

**Configuration**:
```python
PATCH_SIZE = 16
EMBED_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 4
DROPOUT = 0.1
ATTENTION_DROPOUT = 0.1
```

### 2.4 Hybrid Attention Fusion (Section III.C.4, Equations 9-10)

**File**: `models/attention_fusion.py`

Implements band-specific attention mechanism:

```python
class BandAttentionFusion(nn.Module):
    """
    Band-specific attention: α_b = softmax(w^T tanh(W_h h_b + b_h) + b)
    Fused representation: z_fused = Σ α_b · h_b
    """
```

**Features**:
- Learnable attention weights per frequency band
- Channel-wise attention within each band
- Adaptive feature fusion

### 2.5 Hybrid Loss Function (Section III.E, Equations 11-15)

**File**: `models/loss_functions.py`

Implements composite loss with three components:

```python
class HybridLoss(nn.Module):
    """
    L_total = L_CE + λ₁·L_center + λ₂·L_triplet

    Where:
    - L_CE: Cross-entropy loss
    - L_center: Center loss for intra-class compactness
    - L_triplet: Triplet loss for inter-class separability
    """
```

**Hyperparameters** (from `configs/config.yaml`):
```yaml
loss:
  lambda_center: 0.01
  lambda_triplet: 0.001
  triplet_margin: 0.2
```

---

## 3. Training Configuration

### 3.1 Cross-Validation Protocol (Section III.F.2)

**File**: `train.py`

Implements stratified 5-fold cross-validation:

```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Data split per fold:
# - Training: 60% (3 folds)
# - Validation: 20% (1 fold)
# - Testing: 20% (1 fold)
```

### 3.2 Optimization Settings

**Configuration**:
```yaml
training:
  optimizer: Adam
  learning_rate: 0.0001
  weight_decay: 0.0001  # L2 regularization
  batch_size: 32
  num_epochs: 100
  early_stopping_patience: 20
```

**Regularization**:
- Dropout: 0.3 (GCN), 0.1 (ViT)
- Gradient clipping: max norm = 1.0
- Batch normalization in GCN layers

### 3.3 Hyperparameter Tuning

**File**: `train.py` (Optuna integration)

Search space for 100 trials:
```python
search_space = {
    'learning_rate': (1e-5, 1e-3),
    'batch_size': [16, 32, 64],
    'gcn_hidden_dim': [32, 64, 128, 256],
    'vit_num_heads': [4, 8, 16],
    'dropout': (0.1, 0.5),
    'lambda_center': (0.001, 0.1),
    'lambda_triplet': (0.0001, 0.01)
}
```

---

## 4. Evaluation and Visualization

### 4.1 Classification Metrics

**File**: `utils/metrics.py`

Computes the following metrics:
- Accuracy
- Sensitivity (Recall)
- Specificity
- Precision
- F1-Score
- AUC-ROC (per class)
- Confusion Matrix

### 4.2 Explainability Visualization

**File**: `utils/visualization.py`

Implements biomarker identification methods:

**Grad-CAM**:
```python
compute_gradcam(model, input_tensor, target_class)
# Returns: Channel importance heatmap
```

**Attention Rollout**:
```python
compute_attention_rollout(model, input_tensor)
# Returns: Attention flow across layers
```

**t-SNE Visualization**:
```python
plot_tsne_embeddings(features, labels)
# Projects learned features to 2D space
```

---

## 5. Reproducibility

### 5.1 Random Seed Control

All random processes are seeded for reproducibility:
```python
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
```

### 5.2 Configuration Management

All hyperparameters are stored in `configs/config.yaml` ensuring:
- Experiment reproducibility
- Easy parameter modification
- Version control of configurations

### 5.3 Logging

Training logs include:
- Loss curves (train/validation)
- Metrics per epoch
- Learning rate schedule
- Model checkpoints
- TensorBoard integration

---

## 6. Usage Examples

### 6.1 Training from Scratch

```bash
# 5-fold cross-validation
bash scripts/train_5fold.sh

# Single fold with specific GPU
python train.py --fold 0 --device cuda:0
```

### 6.2 Evaluation

```bash
python evaluate.py \
    --model_path checkpoints/best_model.pth \
    --data_path data/processed \
    --compute_gradcam \
    --compute_attention_rollout
```

### 6.3 Hyperparameter Tuning

```bash
python train.py --config configs/config.yaml --tune
```

---

## 7. Software Dependencies

### Core Frameworks
- **PyTorch ≥ 1.12**: Deep learning framework
- **PyTorch Geometric ≥ 2.1**: Graph neural network operations
- **MNE-Python ≥ 1.2**: EEG processing and analysis

### Scientific Computing
- **NumPy ≥ 1.21**: Numerical computations
- **SciPy ≥ 1.7**: Signal processing
- **PyWavelets ≥ 1.3**: Wavelet decomposition
- **scikit-learn ≥ 1.0**: Machine learning utilities

### Visualization
- **Matplotlib ≥ 3.5**: Static plotting
- **Seaborn ≥ 0.11**: Statistical visualization

### Utilities
- **Optuna ≥ 3.0**: Hyperparameter optimization
- **PyYAML ≥ 6.0**: Configuration management
- **tqdm ≥ 4.62**: Progress bars

---

## 8. Hardware Requirements

### Minimum Requirements
- CPU: 4 cores
- RAM: 16 GB
- Storage: 10 GB

### Recommended for Training
- GPU: NVIDIA GPU with 8GB+ VRAM
- RAM: 32 GB
- CUDA: Version 11.3+

### Inference Only
- CPU sufficient for single-sample inference
- GPU recommended for batch processing

---

## 9. Dataset Information

### OpenNeuro Dataset (ds004504)

**Subjects**: 156 (52 HC, 52 MCI, 52 AD)
**Age**: 72.3 ± 6.8 years
**Education**: 14.2 ± 3.1 years
**Female**: 48%

**Recording Parameters**:
- Sampling rate: 256 Hz
- Duration: 5-10 minutes resting-state
- Channels: 19 (10-20 international system)
- Reference: Linked mastoids

**Preprocessing**:
1. FIR bandpass filter (0.5-45 Hz)
2. ICA artifact removal
3. Wavelet decomposition (db4, 4 levels)
4. Segmentation (4-second windows)

---

## 10. Expected Performance

Results from 5-fold cross-validation:

| Metric | Mean ± Std |
|--------|------------|
| Accuracy | 94.2% ± 0.8% |
| Sensitivity | 93.8% ± 0.9% |
| Specificity | 94.6% ± 0.7% |
| F1-Score | 94.0% ± 0.8% |

Per-class AUC:
- HC: 0.972
- MCI: 0.958
- AD: 0.981

---

## 11. Limitations and Future Work

### Current Limitations
1. Dataset size (156 subjects) - validation on larger datasets needed
2. Cross-sectional data - longitudinal tracking would strengthen evidence
3. Single dataset - multi-site validation recommended

### Planned Extensions
- Multi-modal fusion (EEG + MRI)
- Real-time deployment optimization
- Extended to other dementias (FTD, LBD)

---

## 12. Contact and Support

**Repository**: https://github.com/HZerrouki/MSTS-AN
**Issues**: https://github.com/HZerrouki/MSTS-AN/issues

**Authors**:
- Hadj Zerrouki (zerrouki.hadj@gmail.com)
- Salima Azzaz-Rahmani

**Institution**:
Department of Telecommunication, Faculty of Electrical Engineering
Djillali Liabes University of Sidi Bel Abbes, Algeria

---

## 13. References

[1] OpenNeuro Dataset: https://openneuro.org/datasets/ds004504

[2] PyTorch: https://pytorch.org/

[3] PyTorch Geometric: https://pytorch-geometric.readthedocs.io/

[4] MNE-Python: https://mne.tools/

---

**Last Updated**: May 31, 2026
**Version**: 1.0.0
**License**: MIT
