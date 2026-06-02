#!/bin/bash
# 5-Fold Cross-Validation Training Script for MSTS-AN
#
# This script trains the MSTS-AN model using 5-fold cross-validation.
# Each fold can be trained in parallel for faster execution.
#
# Usage: ./scripts/train_5fold.sh [OPTIONS]

set -e  # Exit on error

# Default configuration
CONFIG="configs/config.yaml"
DATA_PATH="data/processed"
OUTPUT_DIR="results"
DEVICE="cuda"
PARALLEL=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --data-path)
            DATA_PATH="$2"
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
        --no-parallel)
            PARALLEL=false
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config PATH       Path to config file (default: configs/config.yaml)"
            echo "  --data-path PATH    Path to processed data (default: data/processed)"
            echo "  --output-dir DIR    Output directory (default: results)"
            echo "  --device DEVICE     Device to use: cuda or cpu (default: cuda)"
            echo "  --no-parallel       Run folds sequentially instead of parallel"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "MSTS-AN 5-Fold Cross-Validation Training"
echo "=========================================="
echo "Config: $CONFIG"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo "Parallel: $PARALLEL"
echo ""

# Check if data exists
if [ ! -d "$DATA_PATH" ]; then
    echo "Error: Data directory not found: $DATA_PATH"
    echo "Please run preprocessing first or specify correct path."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to train a single fold
train_fold() {
    local FOLD=$1
    echo "Starting Fold $FOLD..."
    python train.py \
        --config "$CONFIG" \
        --data-path "$DATA_PATH" \
        --output-dir "$OUTPUT_DIR" \
        --device "$DEVICE" \
        --fold $FOLD \
        2>&1 | tee "$OUTPUT_DIR/train_fold${FOLD}.log"
    echo "Fold $FOLD completed!"
}

export -f train_fold
export CONFIG DATA_PATH OUTPUT_DIR DEVICE

if [ "$PARALLEL" = true ] && [ "$DEVICE" = "cuda" ]; then
    # Check number of GPUs
    NUM_GPUS=$(nvidia-smi -L | wc -l)
    echo "Detected $NUM_GPUS GPU(s)"

    if [ $NUM_GPUS -ge 5 ]; then
        echo "Training all 5 folds in parallel..."
        for FOLD in {0..4}; do
            CUDA_VISIBLE_DEVICES=$FOLD train_fold $FOLD &
        done
        wait
    else
        echo "Limited GPUs. Training folds sequentially..."
        for FOLD in {0..4}; do
            train_fold $FOLD
        done
    fi
else
    # Sequential training
    echo "Training folds sequentially..."
    for FOLD in {0..4}; do
        train_fold $FOLD
    done
fi

echo ""
echo "=========================================="
echo "5-Fold Cross-Validation Complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="

# Run evaluation on all folds
echo ""
echo "Running evaluation on all folds..."
python evaluate.py \
    --config "$CONFIG" \
    --checkpoint "$OUTPUT_DIR/run_*/best_model_fold*.pth" \
    --data-path "$DATA_PATH" \
    --output-dir "$OUTPUT_DIR/evaluation" \
    --device "$DEVICE"

echo ""
echo "Training and evaluation complete!"
echo "Check $OUTPUT_DIR for results."