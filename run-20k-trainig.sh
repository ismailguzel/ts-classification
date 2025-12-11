#!/bin/bash
# ============================================================================
# Hierarchical Time Series Classification - 20K Training Pipeline
# ============================================================================
# This script trains all models (Model 1 & Model 2) using:
#   - FEATURES mode: Uses TSFresh-extracted features with sklearn classifiers
#     (XGBoost, RandomForest, CatBoost, SVM)
#
# Output:
#   saved_models/
#   ├── model1_binary_features/              # Binary classification + feature_importance CSVs
#   └── model2_nonstationary_features/       # 5-class classification + feature_importance CSVs
# ============================================================================

set -e  # Exit on error
# Note: set -u removed - causes issues with conda activation scripts

# Load required module
module load apps/truba-ai/gpu-2024.0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}============================================================================${NC}"
echo -e "${GREEN}Starting 20K Training Pipeline${NC}"
echo -e "${GREEN}============================================================================${NC}"
echo "Start time: $(date)"
echo ""

### Model 1 (Binary Classification)
echo -e "\n${YELLOW}=== Starting Model 1 Training ===${NC}\n"
cd /arf/home/iguzel/ts-stationary/hierarchical-ts-classification/03-models/hierarchical/model1_binary

# Activate sktime environment
echo -e "${GREEN}[1/2] Activating ts-sktime environment${NC}"
conda activate ts-sktime

# FEATURES mode (selected features) - with feature importance
echo -e "\n${GREEN}[2/2] Training Model 1 - Selected Features (with feature importance)${NC}"
mkdir -p ./out
python -u train_model1.py \
    --mode features \
    --features-path ../../../data/features/unified-20k/selected \
    2>&1 | tee ./out/train_selected-20k.out
echo -e "${GREEN}✓ Selected features completed${NC}"
echo -e "${YELLOW}  → Feature importance CSVs saved in saved_models/model1_binary_features/${NC}"

echo -e "\n${GREEN}Model 1 training completed!${NC}"
echo -e "Results saved in: saved_models/model1_binary_*/"
echo ""

### Model 2 (5-Class Classification)
echo -e "\n${YELLOW}=== Starting Model 2 Training ===${NC}\n"
cd /arf/home/iguzel/ts-stationary/hierarchical-ts-classification/03-models/hierarchical/model2_nonstationary

# Activate sktime environment
echo -e "${GREEN}[1/2] Activating ts-sktime environment${NC}"
conda activate ts-sktime

# RAW mode (trains all classifiers: rocket, arsenal, tsforest)
# ⚠️ COMMENTED OUT - RAW mode training is very slow, use FEATURES mode instead
# echo -e "\n${GREEN}[2/5] Training Model 2 - RAW mode${NC}"
# mkdir -p ./out
# python -u train_model2.py \
#     --mode raw \
#     --data-path ../../../data/raw/unified-20k \
#     2>&1 | tee ./out/train2_raw-20k.out
# echo -e "${GREEN}✓ RAW mode completed${NC}"

# FEATURES mode (selected features) - with feature importance
echo -e "\n${GREEN}[2/2] Training Model 2 - Selected Features (with feature importance)${NC}"
mkdir -p ./out
python -u train_model2.py \
    --mode features \
    --features-path ../../../data/features/unified-20k/selected \
    2>&1 | tee ./out/train2_selected-20k.out
echo -e "${GREEN}✓ Selected features completed${NC}"
echo -e "${YELLOW}  → Feature importance CSVs saved in saved_models/model2_nonstationary_features/${NC}"

echo -e "\n${GREEN}Model 2 training completed!${NC}"
echo -e "Results saved in: saved_models/model2_nonstationary_*/"

echo -e "\n${GREEN}============================================================================${NC}"
echo -e "${GREEN}ALL TRAINING COMPLETED!${NC}"
echo -e "${GREEN}============================================================================${NC}"
echo "End time: $(date)"
echo ""
echo -e "${YELLOW}Summary:${NC}"
echo "  Model 1 (Binary): saved_models/model1_binary_features/"
echo "  Model 2 (5-Class): saved_models/model2_nonstationary_features/"
echo ""
echo -e "${YELLOW}Feature Importance:${NC}"
echo "  ✓ Feature importance CSVs available in:"
echo "    - model1_binary_features/feature_importance_*.csv"
echo "    - model2_nonstationary_features/feature_importance_*.csv"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "  1. Review metrics in saved_models/*/*.json"
echo "  2. Run post-processing: bash run-postprocessing.sh"
echo "  3. Analyze feature importance: cd 04-postprocessing && python visualize_feature_importance.py"
echo "  4. Analyze errors: python visualize_errors_simple.py"
echo ""
