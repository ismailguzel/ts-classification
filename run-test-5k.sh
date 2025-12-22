#!/bin/bash
# ============================================================================
# Quick Test Pipeline with 5K Dataset
# ============================================================================
# Fast end-to-end test with smaller dataset for development/testing.
# This is a wrapper around run.sh for 5k dataset with all feature types.
#
# Usage:
#   bash run-test-5k.sh [step]
#
# Steps:
#   all           : Run full pipeline for all 3 feature types
#   generation    : Generate synthetic 5k dataset
#   preprocessing : Run feature extraction & selection (all types)
#   training      : Train with all 3 feature types
#   single        : Run for single feature type (uses FEATURE_TYPE from run.sh)
# ============================================================================

set -e

STEP=${1:-all}

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}============================================================================${NC}"
echo -e "${GREEN}5K TEST PIPELINE${NC}"
echo -e "${GREEN}============================================================================${NC}"
echo ""

if [ "$STEP" == "single" ]; then
    # Single feature type (whatever is set in run.sh)
    echo -e "${BLUE}Running single pipeline with current configuration...${NC}"
    bash run.sh all 5k
    
elif [ "$STEP" == "generation" ]; then
    # Only generation
    bash run.sh generation 5k
    
elif [ "$STEP" == "preprocessing" ]; then
    # Only preprocessing (all feature types)
    bash run.sh preprocessing 5k
    
elif [ "$STEP" == "training" ]; then
    # Training for all 3 feature types
    for FEATURE_TYPE in "statistical" "topological" "combined"; do
        echo ""
        echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
        echo -e "${BLUE}Training: ${FEATURE_TYPE} features (5k)${NC}"
        echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
        bash run.sh training 5k "$FEATURE_TYPE"
        bash run.sh postprocessing 5k "$FEATURE_TYPE"
    done
    
else
    # Full pipeline for all 3 feature types
    echo -e "${BLUE}Running complete pipeline for all feature types...${NC}"
    echo ""
    
    # Generation and preprocessing (once for all)
    bash run.sh generation 5k
    bash run.sh preprocessing 5k
    
    # Training + postprocessing for each feature type
    for FEATURE_TYPE in "statistical" "topological" "combined"; do
        echo ""
        echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
        echo -e "${BLUE}Pipeline: ${FEATURE_TYPE} features (5k)${NC}"
        echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
        bash run.sh training 5k "$FEATURE_TYPE"
        bash run.sh postprocessing 5k "$FEATURE_TYPE"
    done
fi

# ============================================================================
# Summary
# ============================================================================

echo ""
echo -e "${GREEN}============================================================================${NC}"
echo -e "${GREEN}5K PIPELINE COMPLETE${NC}"
echo -e "${GREEN}============================================================================${NC}"
echo ""
echo -e "${BLUE}Results:${NC}"
echo -e "  Data:     data/raw/unified-5k/"
echo -e "  Features: data/features/unified-5k/"
echo -e "  Models:   03-models/hierarchical/*/output/*_5k/"
echo -e "  Figures:  04-postprocessing/figures/*_5k/"
echo ""
echo -e "${BLUE}Compare results:${NC}"
echo "  Statistical: cat 03-models/hierarchical/model1_binary/output/model1_statistical_5k/results_summary.txt"
echo "  Topological: cat 03-models/hierarchical/model1_binary/output/model1_topological_5k/results_summary.txt"
echo "  Combined:    cat 03-models/hierarchical/model1_binary/output/model1_combined_5k/results_summary.txt"
echo ""
echo -e "${BLUE}Visualizations:${NC}"
echo "  ls -lh 04-postprocessing/figures/statistical_5k/"
echo "  ls -lh 04-postprocessing/figures/topological_5k/"
echo "  ls -lh 04-postprocessing/figures/combined_5k/"
echo ""
echo -e "${YELLOW}Next: Run full 20k pipeline${NC}"
echo "  Statistical: bash run.sh all 20k statistical"
echo "  Topological: bash run.sh all 20k topological"
echo "  Combined:    bash run.sh all 20k combined"
echo ""
