# MSTS-AN: Multi-Scale Temporal-Spatial Attention Network

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Official implementation of **"MSTS-AN: A Hybrid GCN-Transformer Approach for Early Alzheimer's Detection from EEG Signals"** 
## Overview

MSTS-AN is a novel deep learning architecture that combines **Graph Convolutional Networks (GCN)** for spatial feature extraction and **Vision Transformers (ViT)** for temporal dependency modeling to detect early-stage Alzheimer's Disease (AD) and Mild Cognitive Impairment (MCI) from EEG signals.

### Key Features

- **Hybrid Architecture**: Unified GCN-ViT framework for simultaneous spatial-temporal EEG analysis
- **Multi-Scale Analysis**: Wavelet decomposition into delta, theta, alpha, and beta frequency bands
- **Band-Specific Attention**: Adaptive weighting of frequency bands based on discriminative power
- **Hybrid Loss Function**: Combines Cross-Entropy, Center Loss, and Triplet Loss for improved feature discrimination
- **Explainability**: Integrated Grad-CAM and attention rollout for clinical interpretability
- **State-of-the-Art Performance**: 94.2% accuracy on three-class classification (HC/MCI/AD)

## Architecture

```
Input EEG (C, T)
    ↓
Wavelet Decomposition (4 bands: δ, θ, α, β)
    ↓
Parallel GCN-ViT Branches (per band)
    ├── GCN: Spatial encoding (electrode topology)
    └── ViT: Temporal encoding (long-range dependencies)
    ↓
Band-Specific Attention Fusion
    ↓
Hybrid Loss (CE + Center + Triplet)
    ↓
Classification (HC / MCI / AD)
```

### Model Components

| Component | Parameters | Description |
|-----------|------------|-------------|
| **Multi-Scale Feature Extraction** | - | Daubechies-4 wavelet decomposition |
| **GCN Encoder** | 86,912 | 3-layer graph convolution with normalized Laplacian |
| **Vision Transformer** | 2,197,376 | 4 blocks, 8 heads, 256-dim embeddings |
| **Attention Fusion** | 3,591 | Band-specific + channel-wise attention |
| **Total** | **~2.4M** | Trainable parameters |

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.3+ (for GPU support)
- 16GB+ RAM recommended
- NVIDIA GPU with 8GB+ VRAM (for training)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/HZerrouki/MSTS-AN.git
cd MSTS-AN

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Dependencies

Core dependencies:
```
torch>=1.12.0
torch-geometric>=2.1.0
pywavelets>=1.3.0
mne>=1.2.0
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
pandas>=1.3.0
pyyaml>=6.0
tqdm>=4.62.0
optuna>=3.0.0
jupyter>=1.0.0
```

## Dataset

### OpenNeuro Dataset

This implementation uses the publicly available OpenNeuro dataset (accession: [ds004504](https://openneuro.org/datasets/ds004504)):

- **156 subjects**: 52 HC, 52 MCI, 52 AD
- **Age**: 72.3 ± 6.8 years
- **Recording**: 5-10 minutes resting-state EEG
- **Sampling Rate**: 256 Hz
- **Channels**: 19 channels (10-20 international system)

### Preprocessing Pipeline

1. **Bandpass Filtering**: FIR filter (0.5-45 Hz, Hamming window, 512 taps)
2. **Artifact Removal**: Extended Infomax ICA with ICLabel classifier
3. **Wavelet Decomposition**: Daubechies-4, 4 levels
4. **Segmentation**: 4-second non-overlapping windows (1024 samples)

### Data Directory Structure

```
data/
├── raw/                    # Original OpenNeuro data
│   └── ds004504/
├── processed/              # Preprocessed segments
│   ├── train/
│   ├── val/
│   └── test/
└── graphs/                 # Adjacency matrices
    └── 10_20_adjacency.npy
```

## Usage

### 1. Data Preparation

```bash
# Download and preprocess data
python -c "from data.preprocessor import EEGPreprocessor; EEGPreprocessor.run_all()"
```

Or use the provided script:
```bash
bash scripts/download_data.sh  # Requires OpenNeuro credentials
```

### 2. Training

#### Single Fold Training
```bash
python train.py --config configs/config.yaml \
                --data_path data/processed \
                --output_dir results \
                --device cuda
```

#### 5-Fold Cross-Validation
```bash
# Train all folds sequentially
bash scripts/train_5fold.sh

# Or train specific fold
python train.py --fold 0 --device cuda
```

#### With Hyperparameter Tuning (Optuna)
```bash
python train.py --config configs/config.yaml --tune
```

### 3. Evaluation

```bash
python evaluate.py --model_path checkpoints/best_model.pth \
                   --data_path data/processed \
                   --output_dir results \
                   --compute_gradcam \
                   --compute_attention_rollout
```

### 4. Inference on New Data

```python
import torch
from models import MSTSAN
from data import EEGPreprocessor

# Load pretrained model
model = MSTSAN.from_pretrained('pretrained/msts_an_best.pth')
model.eval()

# Preprocess new EEG data
preprocessor = EEGPreprocessor()
eeg_segment = preprocessor.process('path/to/eeg.edf')  # Shape: (19, 1024)

# Predict
with torch.no_grad():
    output = model(eeg_segment.unsqueeze(0))
    prediction = output.argmax(dim=1)

class_names = ['HC', 'MCI', 'AD']
print(f"Predicted class: {class_names[prediction.item()]}")
```

## Configuration

All hyperparameters are configurable via `configs/config.yaml`:

### Key Parameters

```yaml
# Model Architecture
model:
  gcn:
    hidden_dims: [64, 128, 256]
    dropout: 0.3
  vit:
    patch_size: 16
    embed_dim: 256
    num_heads: 8
    num_layers: 4

# Training
training:
  batch_size: 32
  learning_rate: 0.0001
  num_epochs: 100
  early_stopping_patience: 20

# Loss Function
loss:
  lambda_center: 0.01
  lambda_triplet: 0.001
  triplet_margin: 0.2
```

## Expected Results

Performance on OpenNeuro 3-class classification (HC/MCI/AD):

| Metric | Value |
|--------|-------|
| **Accuracy** | 94.2% |
| **Sensitivity** | 93.8% |
| **Specificity** | 94.6% |
| **F1-Score** | 94.0% |
| **AUC (HC)** | 0.972 |
| **AUC (MCI)** | 0.958 |
| **AUC (AD)** | 0.981 |

### Ablation Study Results

| Configuration | Accuracy | Δ from Full |
|--------------|----------|-------------|
| Full MSTS-AN | 94.2% | - |
| w/o GCN Module | 91.4% | -2.8% |
| w/o ViT Module | 92.1% | -2.1% |
| w/o Band Attention | 93.0% | -1.2% |
| w/o Triplet Loss | 93.5% | -0.7% |

## Notebooks

Interactive tutorials are provided in `notebooks/`:

1. **[01_data_exploration.ipynb](notebooks/01_data_exploration.ipynb)** - Data loading and visualization
2. **[02_model_visualization.ipynb](notebooks/02_model_visualization.ipynb)** - Architecture visualization and attention maps
3. **[03_biomarker_analysis.ipynb](notebooks/03_biomarker_analysis.ipynb)** - Grad-CAM and clinical interpretation

## Project Structure

```
MSTS-AN/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── setup.py                       # Package installation
├── configs/
│   └── config.yaml               # Model and training hyperparameters
├── data/
│   ├── __init__.py
│   ├── preprocessor.py           # EEG preprocessing pipeline
│   ├── dataset.py                # PyTorch Dataset classes
│   └── graph_builder.py          # GCN adjacency matrix construction
├── models/
│   ├── __init__.py
│   ├── msts_an.py                # Main MSTS-AN architecture
│   ├── gcn_module.py             # Graph Convolutional Network
│   ├── vit_module.py             # Vision Transformer temporal module
│   ├── attention_fusion.py       # Hybrid attention fusion
│   └── loss_functions.py         # Hybrid loss (CE + Center + Triplet)
├── train.py                      # Training script
├── evaluate.py                   # Evaluation script
├── utils/
│   ├── metrics.py                # Classification metrics
│   ├── visualization.py          # t-SNE, Grad-CAM, attention maps
│   ├── logger.py                 # Training logger
│   └── checkpoint.py             # Model checkpointing
├── scripts/
│   ├── train_5fold.sh            # 5-fold cross-validation script
│   └── run_experiments.sh        # Full pipeline
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_visualization.ipynb
│   └── 03_biomarker_analysis.ipynb
└── pretrained/
    └── README.md                 # Pretrained model weights
```

## Pretrained Models

Pretrained model weights will be available upon paper acceptance:

```bash
# Download pretrained weights (after acceptance)
wget https://github.com/HZerrouki/MSTS-AN/releases/download/v1.0.0/msts_an_best.pth
mv msts_an_best.pth pretrained/
```

## Citation

If you use this code or find our work helpful, please cite:

```bibtex
@article{zerrouki2026mstsan,
  title={MSTS-AN: A Hybrid GCN-Transformer Approach for Early Alzheimer's Detection from EEG Signals},
  author={Zerrouki, Hadj and Azzaz-Rahmani, Salima},
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black models/ data/ utils/
isort models/ data/ utils/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenNeuro for providing the EEG dataset
- PyTorch and PyTorch Geometric teams for the excellent frameworks
- The Alzheimer's research community for valuable insights

## Contact

For questions or issues, please:

1. Open an issue on GitHub
2. Contact the authors:
   - Hadj Zerrouki: zerrouki.hadj@gmail.com
   - Salima Azzaz-Rahmani

## FAQ

**Q: Can I use this code with my own EEG dataset?**
A: Yes! The code is designed to be modular. You'll need to:
1. Format your data to match the expected input shape (channels, samples)
2. Update the adjacency matrix in `data/graph_builder.py` if using a different electrode montage
3. Adjust preprocessing parameters in `configs/config.yaml`

**Q: What hardware do I need?**
A: For training: NVIDIA GPU with 8GB+ VRAM. For inference: CPU is sufficient but GPU recommended for batch processing.

**Q: How long does training take?**
A: Approximately 2-3 hours per fold on an NVIDIA A100 GPU for the full 100 epochs.

**Q: Can I fine-tune on my data?**
A: Yes, use the pretrained weights and set a lower learning rate (e.g., 1e-5) for fine-tuning.

## Troubleshooting

### Common Issues

**CUDA out of memory**
```bash
# Reduce batch size in config.yaml
training:
  batch_size: 16  # or 8
```

**Import errors**
```bash
# Ensure package is installed
pip install -e .
```

**Data loading errors**
```bash
# Verify data structure
python -c "from data.dataset import verify_data; verify_data('data/processed')"
```

## Changelog

### v1.0.0 (2026-05-31)
- Initial release
- Complete MSTS-AN implementation
- 5-fold cross-validation support
- Grad-CAM and attention rollout visualization
- Comprehensive documentation

---

**Note**: This repository is under active development. The pretrained models will be released upon paper acceptance.
