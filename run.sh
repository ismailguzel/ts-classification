#!/bin/bash
# ============================================================================
# Master Pipeline Script
# ============================================================================
# Usage:
#   bash run.sh [step] [mode] [features]
#
# Steps:
#   all           : Run full pipeline (Generation -> Preprocessing -> Training -> Postprocessing)
#   generation    : Generate synthetic dataset
#   preprocessing : Run feature extraction & selection
#   training      : Train flat classifier
#   postprocessing: Analyze results & generate figures
#
# Mode    : shape (default) | shape-full | season-structure | season-anomaly
#           One per 01-data-generation/<mode>-config.json — see modes.sh
# Features: tsfresh (default) | topo | hybrid
#
# Examples:
#   bash run.sh                               # full pipeline, shape, tsfresh
#   bash run.sh all shape hybrid              # full pipeline, shape, TSFresh + topology
#   bash run.sh all season-anomaly topo       # full pipeline, Study B2, topology
#   bash run.sh training shape topo           # training only, shape, topology
# ============================================================================

set -e

LOG_DIR="logs"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
mkdir -p "$LOG_DIR"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

STEP=${1:-all}      # all | generation | preprocessing | training | postprocessing
MODE=${2:-shape}    # shape | shape-full | season-structure | season-anomaly
FEATURES=${3:-tsfresh}  # tsfresh | topo | hybrid

VALID_STEPS=("all" "generation" "preprocessing" "training" "postprocessing")

if [[ ! " ${VALID_STEPS[@]} " =~ " ${STEP} " ]]; then
    echo -e "${RED}Error: Invalid step '$STEP'${NC}"
    echo ""
    echo "Usage: bash run.sh [step] [mode] [features]"
    echo ""
    echo "Steps   : all | generation | preprocessing | training | postprocessing"
    echo "Mode    : shape (default) | shape-full | season-structure | season-anomaly"
    echo "Features: tsfresh (default) | topo | hybrid"
    exit 1
fi

function log_run() {
    local script_name=$1
    shift
    local args="$*"
    local label="${script_name%.*}"
    [[ -n "$args" ]] && label="${label}_$(echo "$args" | tr ' ' '_')"
    local log_file="$LOG_DIR/${TIMESTAMP}_${label}.log"

    echo -e "${YELLOW}>>> Running $script_name $args${NC}"
    echo "    Log: $log_file"

    bash "$script_name" $args 2>&1 | tee "$log_file"

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo -e "${GREEN}✓ $script_name completed successfully.${NC}\n"
    else
        echo -e "${RED}✗ $script_name failed. Check: $log_file${NC}\n"
        exit 1
    fi
}

echo -e "${GREEN}============================================================================${NC}"
echo -e "${GREEN}MASTER PIPELINE STARTED${NC}"
echo -e "${GREEN}============================================================================${NC}"
echo "Step    : $STEP"
echo "Mode    : $MODE"
echo "Features: $FEATURES"
echo "Start   : $(date)"
echo "Logs    : $LOG_DIR/"
echo ""

if [[ "$STEP" == "all" || "$STEP" == "generation" ]]; then
    log_run "run-generation.sh" "$MODE"
fi

if [[ "$STEP" == "all" || "$STEP" == "preprocessing" ]]; then
    log_run "run-preprocessing.sh" "$MODE" "$FEATURES"
fi

if [[ "$STEP" == "all" || "$STEP" == "training" ]]; then
    log_run "run-training.sh" "$MODE" "$FEATURES"
fi

if [[ "$STEP" == "all" || "$STEP" == "postprocessing" ]]; then
    log_run "run-postprocessing.sh" "$MODE" "$FEATURES"
fi

echo -e "${GREEN}============================================================================${NC}"
echo -e "${GREEN}MASTER PIPELINE COMPLETED${NC}"
echo -e "${GREEN}============================================================================${NC}"
echo "End: $(date)"

