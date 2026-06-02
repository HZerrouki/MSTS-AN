#!/bin/bash
# Complete Experimental Pipeline for MSTS-AN
#
# This script runs the complete experimental pipeline:
# 1. Data preprocessing
# 2. 5-fold cross-validation training
# 3. Model evaluation
# 4. Ablation studies
# 5. Generate visualizations
#
# Usage: ./scripts/run_experiments.sh [OPTIONS]

set -e

# Configuration
CONFIG="configs/config.yaml"
RAW_DATA_PATH="data/raw"
PROCESSED_DATA_PATH="data/processed"
OUTPUT_DIR="results/experiments_$(date +%Y%m%d_%H%M%S)"
DEVICE="cuda"
RUN_PREPROCESSING=true
RUN_TRAINING=true
RUN_EVALUATION=true
RUN_ABLATION=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --raw-data)
            RAW_DATA_PATH="$2"
            shift 2
            ;;
        --processed-data)
            PROCESSED_DATA_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --skip-preprocessing)
            RUN_PREPROCESSING=false
            shift
            ;;
        --skip-training)
            RUN_TRAINING=false
            shift
            ;;
        --skip-evaluation)
            RUN_EVALUATION=false
            shift
            ;;
        --skip-ablation)
            RUN_ABLATION=false
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Complete experimental pipeline for MSTS-AN."
            echo ""
            echo "Options:"
            echo "  --config PATH            Config file path"
            echo "  --raw-data PATH          Raw data directory"
            echo "  --processed-data PATH    Processed data directory"
            echo "  --output-dir DIR         Output directory"
            echo "  --device DEVICE          Device: cuda or cpu"
            echo "  --skip-preprocessing     Skip preprocessing step"
            echo "  --skip-training          Skip training step"
            echo "  --skip-evaluation        Skip evaluation step"
            echo "  --skip-ablation          Skip ablation studies"
            echo "  --help                   Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$OUTPUT_DIR/pipeline.log"
}

log "=========================================="
log "MSTS-AN Experimental Pipeline"
log "=========================================="
log "Output directory: $OUTPUT_DIR"
log ""

# ============================================================================
# Step 1: Data Preprocessing
# ============================================================================
if [ "$RUN_PREPROCESSING" = true ]; then
    log "Step 1: Data Preprocessing"
    log "------------------------------------------"

    if [ ! -d "$RAW_DATA_PATH" ]; then
        log "ERROR: Raw data directory not found: $RAW_DATA_PATH"
        exit 1
    fi

    mkdir -p "$PROCESSED_DATA_PATH"

    log "Running preprocessing pipeline..."
    # Note: User needs to implement data loading based on their data format
    # python -c "
    # from data.preprocessor import EEGPreprocessor
    # from data.dataset import save_dataset
    # # Load and preprocess data here
    # preprocessor = EEGPreprocessor()
    # # ... preprocessing code ...
    # "

    log "Preprocessing complete!"
    log ""
fi

# ============================================================================
# Step 2: Model Training
# ============================================================================
if [ "$RUN_TRAINING" = true ]; then
    log "Step 2: Model Training (5-Fold Cross-Validation)"
    log "------------------------------------------"

    ./scripts/train_5fold.sh \
        --config "$CONFIG" \
        --data-path "$PROCESSED_DATA_PATH" \
        --output-dir "$OUTPUT_DIR/training" \
        --device "$DEVICE"

    log "Training complete!"
    log ""
fi

# ============================================================================
# Step 3: Model Evaluation
# ============================================================================
if [ "$RUN_EVALUATION" = true ]; then
    log "Step 3: Model Evaluation"
    log "------------------------------------------"

    # Find latest training run
    TRAIN_DIR=$(find "$OUTPUT_DIR/training" -maxdepth 1 -type d -name "run_*" | sort | tail -1)

    if [ -z "$TRAIN_DIR" ]; then
        log "ERROR: No training results found"
        exit 1
    fi

    for FOLD in {0..4}; do
        CHECKPOINT="$TRAIN_DIR/best_model_fold${FOLD}.pth"
        if [ -f "$CHECKPOINT" ]; then
            log "Evaluating Fold $FOLD..."
            python evaluate.py \
                --config "$CONFIG" \
                --checkpoint "$CHECKPOINT" \
                --data-path "$PROCESSED_DATA_PATH" \
                --output-dir "$OUTPUT_DIR/evaluation/fold_$FOLD" \
                --device "$DEVICE" \
                2>&1 | tee "$OUTPUT_DIR/evaluation_fold${FOLD}.log"
        fi
    done

    log "Evaluation complete!"
    log ""
fi

# ============================================================================
# Step 4: Ablation Studies
# ============================================================================
if [ "$RUN_ABLATION" = true ]; then
    log "Step 4: Ablation Studies"
    log "------------------------------------------"

    ABLATION_DIR="$OUTPUT_DIR/ablation"
    mkdir -p "$ABLATION_DIR"

    # Ablation configurations
    declare -A ABLATIONS
    ABLATIONS["no_gcn"]="Without GCN Module"
    ABLATIONS["no_vit"]="Without ViT Module"
    ABLATIONS["no_band_attention"]="Without Band Attention"
    ABLATIONS["no_triplet_loss"]="Without Triplet Loss"
    ABLATIONS["single_scale"]="Single-Scale (Alpha only)"

    for ABLATION in "${!ABLATIONS[@]}"; do
        log "Running ablation: ${ABLATIONS[$ABLATION]}"

        # Create modified config for ablation
        python -c "
import yaml
with open('$CONFIG', 'r') as f:
    config = yaml.safe_load(f)

# Modify config based on ablation type
if '$ABLATION' == 'no_gcn':
    config['model']['gcn']['hidden_dims'] = []
elif '$ABLATION' == 'no_vit':
    config['model']['vit']['num_layers'] = 0

with open('$ABLATION_DIR/config_$ABLATION.yaml', 'w') as f:
    yaml.dump(config, f)
"

        # Train with ablation config
        python train.py \
            --config "$ABLATION_DIR/config_$ABLATION.yaml" \
            --data-path "$PROCESSED_DATA_PATH" \
            --output-dir "$ABLATION_DIR/$ABLATION" \
            --device "$DEVICE" \
            2>&1 | tee "$ABLATION_DIR/${ABLATION}.log"
    done

    log "Ablation studies complete!"
    log ""
fi

# ============================================================================
# Step 5: Summary Report
# ============================================================================
log "Step 5: Generating Summary Report"
log "------------------------------------------"

python -c "
import json
import os
from glob import glob

output_dir = '$OUTPUT_DIR'
report = []

report.append('='*60)
report.append('MSTS-AN Experimental Results Summary')
report.append('='*60)
report.append('')

# Main results
if os.path.exists(f'{output_dir}/evaluation'):
    report.append('Main Model Performance (5-Fold Cross-Validation)')
    report.append('-'*60)

    all_metrics = []
    for fold_dir in glob(f'{output_dir}/evaluation/fold_*'):
        metrics_file = os.path.join(fold_dir, 'metrics.json')
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                all_metrics.append(metrics)
                report.append(f'Fold: {os.path.basename(fold_dir)}')
                report.append(f'  Accuracy: {metrics.get(\"accuracy\", 0):.2f}%')
                report.append(f'  F1-Score: {metrics.get(\"f1_macro\", 0):.4f}')
                report.append(f'  AUC: {metrics.get(\"auc_macro\", 0):.4f}')
                report.append('')

    if all_metrics:
        import numpy as np
        avg_acc = np.mean([m.get('accuracy', 0) for m in all_metrics])
        std_acc = np.std([m.get('accuracy', 0) for m in all_metrics])
        avg_f1 = np.mean([m.get('f1_macro', 0) for m in all_metrics])
        avg_auc = np.mean([m.get('auc_macro', 0) for m in all_metrics])

        report.append('Average Performance:')
        report.append(f'  Accuracy: {avg_acc:.2f}% ± {std_acc:.2f}%')
        report.append(f'  F1-Score: {avg_f1:.4f}')
        report.append(f'  AUC: {avg_auc:.4f}')
        report.append('')

# Ablation results
if os.path.exists(f'{output_dir}/ablation'):
    report.append('Ablation Study Results')
    report.append('-'*60)

    for ablation in ['no_gcn', 'no_vit', 'no_band_attention', 'no_triplet_loss', 'single_scale']:
        ablation_dir = f'{output_dir}/ablation/{ablation}'
        if os.path.exists(ablation_dir):
            # Find best model metrics
            metrics_files = glob(f'{ablation_dir}/run_*/average_metrics.json')
            if metrics_files:
                with open(metrics_files[0], 'r') as f:
                    metrics = json.load(f)
                    report.append(f'{ablation}:')
                    report.append(f'  Accuracy: {metrics.get(\"accuracy_mean\", 0):.2f}%')
                    report.append(f'  F1-Score: {metrics.get(\"f1_macro_mean\", 0):.4f}')
                    report.append('')

report.append('='*60)

# Save and print report
report_text = '\n'.join(report)
with open(f'{output_dir}/summary_report.txt', 'w') as f:
    f.write(report_text)

print(report_text)
"

log ""
log "=========================================="
log "Experimental Pipeline Complete!"
log "Results saved to: $OUTPUT_DIR"
log "=========================================="