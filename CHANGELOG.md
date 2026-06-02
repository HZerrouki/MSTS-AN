# Changelog

All notable changes to the MSTS-AN project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-05-31

### Added
- Initial release of MSTS-AN implementation
- Complete GCN-ViT hybrid architecture for EEG classification
- Multi-scale wavelet decomposition with Daubechies-4
- Band-specific attention fusion mechanism
- Hybrid loss function (Cross-Entropy + Center Loss + Triplet Loss)
- 5-fold stratified cross-validation implementation
- Grad-CAM and attention rollout for explainability
- Comprehensive data preprocessing pipeline (FIR, ICA, DWT)
- Hyperparameter tuning with Optuna integration
- Interactive Jupyter notebooks for tutorials
- Complete documentation (README, supplementary materials)
- Training and evaluation scripts
- Pretrained model placeholder structure

### Features
- **Models**:
  - `MSTSAN`: Main architecture combining GCN and ViT
  - `GraphConvolution`: Spatial encoding with normalized Laplacian
  - `VisionTransformer`: Temporal encoding with multi-head attention
  - `BandAttentionFusion`: Adaptive band-wise feature fusion
  - `HybridLoss`: Combined classification and metric learning loss

- **Data Processing**:
  - `EEGPreprocessor`: Complete preprocessing pipeline
  - `EEGDataset`: PyTorch dataset with augmentation
  - `EEGGraphBuilder`: 10-20 electrode adjacency matrix

- **Utilities**:
  - Classification metrics (accuracy, sensitivity, specificity, AUC)
  - Visualization tools (t-SNE, confusion matrix, ROC curves)
  - Model checkpointing and logging
  - Training logger with TensorBoard support

### Configuration
- YAML-based configuration system
- Adjustable hyperparameters for all components
- Support for hyperparameter tuning with Optuna

### Documentation
- Comprehensive README with installation and usage instructions
- Supplementary materials for paper submission
- API documentation in docstrings
- Interactive Jupyter notebooks

## [Unreleased]

### Planned
- Pretrained model weights (pending paper acceptance)
- Additional dataset support
- Real-time inference optimization
- Docker container for easy deployment
- Web interface for demo

## Release Notes

### v1.0.0
This is the initial release accompanying the paper submission to AECE Journal. The code implements the complete MSTS-AN architecture as described in the paper, including:

1. **Multi-Scale Temporal-Spatial Attention Network** with GCN and ViT components
2. **Complete training pipeline** with 5-fold cross-validation
3. **Evaluation framework** with biomarker visualization
4. **Comprehensive documentation** for reproducibility

The repository is ready for:
- Training on OpenNeuro EEG dataset
- Reproducing paper results
- Extension to custom datasets
- Research and development

---

For detailed usage instructions, see [README.md](README.md).
For implementation details, see [supplementary_materials.md](supplementary_materials.md).
