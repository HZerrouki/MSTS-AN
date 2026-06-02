# Quick Start Guide

This guide will help you get started with MSTS-AN in under 10 minutes.

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training)
- 8GB+ RAM

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/HZerrouki/MSTS-AN.git
cd MSTS-AN
```

### 2. Create Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Example

### Option 1: Train on Sample Data

```bash
# Generate synthetic data for testing
python -c "
from data.dataset import create_synthetic_data
create_synthetic_data(output_dir='data/test', n_samples=100)
"

# Train model
python train.py --config configs/config.yaml \
                --data_path data/test \
                --output_dir results/test \
                --device cpu
```

### Option 2: Run Pre-configured Experiment

```bash
# This runs a minimal experiment with reduced epochs
bash scripts/run_experiments.sh --test-mode
```

### Option 3: Use Notebooks

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

## Expected Output

After training, you'll find:

```
results/
└── run_YYYYMMDD_HHMMSS/
    ├── checkpoints/
    │   └── best_model.pth
    ├── logs/
    │   └── training.log
    ├── figures/
    │   ├── confusion_matrix.png
    │   ├── roc_curves.png
    │   └── training_curves.png
    └── metrics.json
```

## Next Steps

1. **Prepare Real Data**: See [Data Preparation Guide](docs/data_preparation.md)
2. **Customize Training**: Edit `configs/config.yaml`
3. **Evaluate Model**: Run `python evaluate.py --help`
4. **Visualize Results**: Check the notebooks in `notebooks/`

## Common Issues

### CUDA Out of Memory

```bash
# Reduce batch size in configs/config.yaml
training:
  batch_size: 8  # or even 4
```

### Import Errors

```bash
# Ensure package is installed
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Slow Training

```bash
# Use GPU if available
python train.py --device cuda

# Or reduce model size in config
model:
  vit:
    num_layers: 2  # instead of 4
```

## Help

For detailed documentation, see [README.md](README.md).

For issues, open a GitHub issue: https://github.com/HZerrouki/MSTS-AN/issues
