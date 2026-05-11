#!/bin/bash
# ============================================================================
# Baseline Comparison Script
# ============================================================================
# Computes accuracy of traditional stationarity tests:
#   - ADF (Augmented Dickey-Fuller) Test
#   - KPSS (Kwiatkowski-Phillips-Schmidt-Shin) Test
#   - Phillips-Perron (PP) Test
#
# Compares against our Model 1 (Binary Classification) results.
# ============================================================================

set -e  # Exit on error

# Load required module (if on cluster)
if command -v module &> /dev/null; then
    module load apps/truba-ai/gpu-2024.0
fi

# Activate environment
conda activate ts-sktime

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}============================================================================${NC}"
echo -e "${GREEN}BASELINE COMPARISON: Traditional Tests vs ML Model${NC}"
echo -e "${GREEN}============================================================================${NC}"
echo "Start time: $(date)"
echo ""

# Paths
BASE_DIR=$(pwd)
DATA_PATH="$BASE_DIR/data/raw/unified-20k"
OUTPUT_DIR="$BASE_DIR/05-baseline-comparison/results"
SCRIPT_DIR="$BASE_DIR/05-baseline-comparison"

# Configuration
# SAMPLE_SIZE: Number of series to sample per pattern
#   - 1000: Fast (~47 min with 100 cores, 13K total samples)
#   - 0: Full dataset (~2-3 hours with 100 cores, 20K samples)
SAMPLE_SIZE=0

# Validate Input
if [ ! -d "$DATA_PATH" ]; then
    echo -e "${RED}✗ Error: Data directory not found: $DATA_PATH${NC}"
    echo "  Please run 'run-generation.sh' first."
    exit 1
fi

# Create results directory
mkdir -p "$OUTPUT_DIR"

echo -e "${YELLOW}Configuration:${NC}"
echo "  Data path: $DATA_PATH"
echo "  Output dir: $OUTPUT_DIR"
if [ "$SAMPLE_SIZE" -eq 0 ]; then
    echo "  Sample size: ALL (full dataset)"
    ESTIMATED_TIME="2-3 hours"
    TOTAL_SAMPLES="~20K"
else
    echo "  Sample size: $SAMPLE_SIZE series per pattern"
    ESTIMATED_TIME="~45 min"
    TOTAL_SAMPLES="~13K"
fi
echo "  Alpha: 0.05"
echo "  Parallel workers: 100"
echo ""

echo -e "${BLUE}Running baseline tests...${NC}"
echo "Estimated time: $ESTIMATED_TIME ($TOTAL_SAMPLES samples, 100 cores)."
echo ""

# Navigate to script directory to run python script
cd "$SCRIPT_DIR"

# Run baseline comparison
python -u compute_baselines.py \
    --data-path "$DATA_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --n-jobs 100 \
    --alpha 0.05 \
    --sample-size "$SAMPLE_SIZE"

cd "$BASE_DIR"

echo ""
echo -e "${GREEN}============================================================================${NC}"
echo -e "${GREEN}BASELINE COMPARISON COMPLETE!${NC}"
echo -e "${GREEN}============================================================================${NC}"
echo "End time: $(date)"
echo ""

echo -e "${YELLOW}Results saved in: $OUTPUT_DIR${NC}"
ls -lh "$OUTPUT_DIR"

echo ""
echo -e "${YELLOW}Summary:${NC}"
if [ -f "$OUTPUT_DIR/baseline_summary.csv" ]; then
    cat "$OUTPUT_DIR/baseline_summary.csv" | column -t -s,
fi

echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "  1. Review results: cat $OUTPUT_DIR/baseline_summary.csv"
echo "  2. View figures: $OUTPUT_DIR/baseline_comparison.png"
echo "  3. Update Technical Report"
echo ""
echo -e "${GREEN}============================================================================${NC}"

