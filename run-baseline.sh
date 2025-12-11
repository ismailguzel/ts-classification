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
echo "  Sample size: 1000 series per pattern"
echo "  Alpha: 0.05"
echo "  Parallel workers: 100"
echo ""

echo -e "${BLUE}Running baseline tests...${NC}"
echo "This will take approximately 5-10 minutes (~10K sampled series, 100 cores)."
echo ""

# Navigate to script directory to run python script
cd "$SCRIPT_DIR"

# Run baseline comparison
python -u compute_baselines.py \
    --data-path "$DATA_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --n-jobs 100 \
    --alpha 0.05 \
    --sample-size 1000

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

