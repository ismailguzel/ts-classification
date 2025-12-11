#!/bin/bash
# ============================================================================
# Master Pipeline Script
# ============================================================================
# Usage:
#   bash run.sh [step]
#
# Steps:
#   all           : Run full pipeline (Generation -> Preprocessing -> Training -> Baseline)
#   generation    : Generate synthetic dataset
#   preprocessing : Run feature extraction & selection
#   training      : Train hierarchical models
#   postprocessing: Analyze results & generate figures
#   baseline      : Run traditional baseline tests
# ============================================================================

set -e

# Configuration
LOG_DIR="logs"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
mkdir -p "$LOG_DIR"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

STEP=${1:-all}  # Default to 'all'

# Valid steps
VALID_STEPS=("all" "generation" "preprocessing" "training" "postprocessing" "baseline")

# Check if step is valid
if [[ ! " ${VALID_STEPS[@]} " =~ " ${STEP} " ]]; then
    echo -e "${RED}Error: Invalid step '$STEP'${NC}"
    echo ""
    echo "Valid steps:"
    echo "  all           - Run full pipeline"
    echo "  generation    - Generate synthetic dataset"
    echo "  preprocessing - Run feature extraction & selection"
    echo "  training      - Train hierarchical models"
    echo "  postprocessing - Analyze results & generate figures"
    echo "  baseline      - Run traditional baseline tests"
    echo ""
    echo "Usage: bash run.sh [step]"
    exit 1
fi

function log_run() {
    local script_name=$1
    local log_file="$LOG_DIR/${TIMESTAMP}_${script_name%.*}.log"
    
    echo -e "${YELLOW}>>> Running $script_name...${NC}"
    echo "    Log: $log_file"
    
    # Run script and pipe output to both stdout and log file
    # 2>&1 redirects stderr to stdout so both are captured
    bash "$script_name" 2>&1 | tee "$log_file"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo -e "${GREEN}✓ $script_name completed successfully.${NC}\n"
    else
        echo -e "${RED}✗ $script_name failed. Check log for details.${NC}\n"
        exit 1
    fi
}

echo -e "${GREEN}============================================================================${NC}"
echo -e "${GREEN}MASTER PIPELINE STARTED ($STEP)${NC}"
echo -e "${GREEN}============================================================================${NC}"
echo "Start time: $(date)"
echo "Logs directory: $LOG_DIR"
echo ""

# 1. Generation
if [[ "$STEP" == "all" || "$STEP" == "generation" ]]; then
    log_run "run-generation.sh"
fi

# 2. Preprocessing
if [[ "$STEP" == "all" || "$STEP" == "preprocessing" ]]; then
    log_run "run-preprocessing.sh"
fi

# 3. Training
if [[ "$STEP" == "all" || "$STEP" == "training" ]]; then
    log_run "run-training.sh"
fi

# 4. Post-Processing
if [[ "$STEP" == "all" || "$STEP" == "postprocessing" ]]; then
    log_run "run-postprocessing.sh"
fi

# 5. Baseline Comparison
if [[ "$STEP" == "all" || "$STEP" == "baseline" ]]; then
    log_run "run-baseline.sh"
fi

# Check if any step was executed
STEPS_EXECUTED=false
if [[ "$STEP" == "all" || "$STEP" == "generation" || "$STEP" == "preprocessing" || "$STEP" == "training" || "$STEP" == "postprocessing" || "$STEP" == "baseline" ]]; then
    STEPS_EXECUTED=true
fi

if [ "$STEPS_EXECUTED" = false ]; then
    echo -e "${RED}Error: No valid steps were executed.${NC}"
    echo "Please check the step name and try again."
    exit 1
fi

echo -e "${GREEN}============================================================================${NC}"
echo -e "${GREEN}MASTER PIPELINE COMPLETED${NC}"
echo -e "${GREEN}============================================================================${NC}"
echo "End time: $(date)"

