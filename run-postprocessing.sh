#!/bin/bash
# Post-Processing Pipeline
#
# Usage:
#   bash run-postprocessing.sh                       # TSFresh output (default)
#   bash run-postprocessing.sh test                  # test, TSFresh
#   bash run-postprocessing.sh topo-test             # topo-test (10 classes), TSFresh
#   bash run-postprocessing.sh test topo             # test, topology
#   bash run-postprocessing.sh topo-test topo        # topo-test, topology
#   bash run-postprocessing.sh test hybrid           # test, hybrid

set -e

PYTHON="${PYTHON:-python}"

MODE=${1:-full}         # full | test | topo-test
FEATURES=${2:-tsfresh}  # tsfresh | topo | hybrid

BASE_DIR=$(pwd)
POST_DIR="$BASE_DIR/04-postprocessing"
FIGURES_DIR="$POST_DIR/figures/${MODE}_${FEATURES}"

case "$FEATURES" in
    topo)    MODEL_DIR="$BASE_DIR/03-models/flat_classifier/output_topo" ;;
    hybrid)  MODEL_DIR="$BASE_DIR/03-models/flat_classifier/output_hybrid" ;;
    *)       MODEL_DIR="$BASE_DIR/03-models/flat_classifier/output" ;;
esac

[ -d "$MODEL_DIR" ] || {
    echo "Warning: $MODEL_DIR not found. Run run-training.sh $MODE $FEATURES first."
}

echo "============================================================"
echo "Post-Processing — $MODE  |  features: $FEATURES"
echo "============================================================"
echo "Model dir: $MODEL_DIR"
echo "Start    : $(date)"

mkdir -p "$FIGURES_DIR"
cd "$POST_DIR"

echo "Step 1: Feature Importance"
$PYTHON visualize_feature_importance.py \
    --model model1 \
    --saved-models-dir "$MODEL_DIR" \
    --figures-dir "$FIGURES_DIR" \
    --top 30 \
    --save-fig \
    --normalize sum

echo "Step 2: Error Analysis"
$PYTHON visualize_errors_simple.py \
    --model model1 \
    --saved-models-dir "$MODEL_DIR" \
    --figures-dir "$FIGURES_DIR" \
    --save-fig

echo "Step 3: Confusion Matrices"
$PYTHON visualize_confusion_matrices.py \
    --model model1 \
    --saved-models-dir "$MODEL_DIR" \
    --figures-dir "$FIGURES_DIR" \
    --save-fig

cd "$BASE_DIR"

echo "============================================================"
echo "Post-Processing Complete!  features=$FEATURES"
echo "Figures: $FIGURES_DIR"
echo "End: $(date)"
echo "============================================================"
