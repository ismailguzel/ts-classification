#!/bin/bash
# Training Pipeline — Flat classifier
#
# Usage:
#   bash run-training.sh shape                 # Study A, TSFresh features
#   bash run-training.sh shape topo            # Study A, topology features only
#   bash run-training.sh shape hybrid          # Study A, TSFresh + topology merged

set -e

PYTHON="${PYTHON:-python}"

MODE=${1:-shape}        # see modes.sh / 01-data-generation/*-config.json
FEATURES=${2:-tsfresh}  # tsfresh | topo | hybrid

BASE_DIR=$(pwd)
source "$BASE_DIR/modes.sh"

OUTPUT_DIR="$MODEL_DIR"

case "$FEATURES" in
    topo)
        DATA_SELECTED="$DATA_TOPO_SEL"
        EXTRA_PATH=""
        PREREQ="bash run-preprocessing.sh $MODE topo"
        ;;
    hybrid)
        # DATA_SELECTED stays as the TSFresh selection from modes.sh; topology rides along
        EXTRA_PATH="$DATA_TOPO_SEL"
        PREREQ="bash run-preprocessing.sh $MODE hybrid"
        ;;
    *)  # tsfresh (default)
        EXTRA_PATH=""
        PREREQ="bash run-preprocessing.sh $MODE"
        ;;
esac

[ -d "$DATA_SELECTED" ] || {
    echo "Error: $DATA_SELECTED not found."
    echo "  Run: $PREREQ"
    exit 1
}

if [ -n "$EXTRA_PATH" ]; then
    [ -d "$EXTRA_PATH" ] || {
        echo "Error: $EXTRA_PATH not found."
        echo "  Run: $PREREQ"
        exit 1
    }
fi

echo "============================================================"
echo "Training — $MODE  |  features: $FEATURES"
echo "============================================================"
echo "Input : $DATA_SELECTED"
[ -n "$EXTRA_PATH" ] && echo "Extra : $EXTRA_PATH"
echo "Output: $OUTPUT_DIR"
echo "Start : $(date)"

mkdir -p "$OUTPUT_DIR"

TRAIN_CMD="$PYTHON 03-models/flat_classifier/train.py \
    --features-path $DATA_SELECTED \
    --save-dir      $OUTPUT_DIR \
    --n-jobs        -1"

[ -n "$EXTRA_PATH" ] && TRAIN_CMD="$TRAIN_CMD --extra-features-path $EXTRA_PATH"

$TRAIN_CMD

[ -f "$OUTPUT_DIR/classifier.pkl" ] || { echo "Error: Training failed."; exit 1; }

echo "============================================================"
echo "Training Complete!  features=$FEATURES"
echo "Output: $OUTPUT_DIR"
echo "End   : $(date)"
echo "============================================================"
