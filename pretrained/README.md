# Pre-trained Models

This directory contains pre-trained MSTS-AN model weights.

## Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 94.2% |
| Sensitivity | 93.8% |
| Specificity | 94.6% |
| F1-Score | 94.0% |
| AUC (HC) | 0.972 |
| AUC (MCI) | 0.958 |
| AUC (AD) | 0.981 |

## Model Files

- `msts_an_best.pth` - Best model checkpoint (5-fold cross-validation)
- `config.yaml` - Model configuration used for training

## Usage

```python
import torch
from models import MSTSAN

# Load model
model = MSTSAN(n_channels=19, seq_length=1024, n_bands=4, n_classes=3)
checkpoint = torch.load('pretrained/msts_an_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Make predictions
predictions = model.predict(band_data, edge_index)
```

## Training Details

- Dataset: OpenNeuro EEG (156 subjects)
- Classes: HC (52), MCI (52), AD (52)
- Cross-validation: 5-fold
- Optimizer: Adam (lr=1e-4)
- Loss: Hybrid (CE + Center + Triplet)

## Download

Model weights will be available for download from the GitHub releases page.
