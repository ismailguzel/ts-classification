#!/bin/bash
# ============================================================================
# Hierarchical Time Series Classification - Post-Processing Pipeline
# ============================================================================
# This script runs all post-processing and visualization tools after training
# to generate comprehensive analysis figures and reports.
#
# Prerequisites:
#   - Training must be completed (saved_models/ directories exist)
#   - Feature importance CSVs available
#   - Predictions and misclassified CSVs generated
#
# Output:
#   04-postprocessing/figures/
#   ├── Error Analysis (Model 1 & Model 2)
#   │   ├── error_analysis_model1.png
#   │   └── error_analysis_model2.png
#   ├── Feature Importance (Model 1 & Model 2)
#   │   ├── feature_importance_top20_comparison_model1.png
#   │   ├── feature_importance_categories_model1.png
#   │   ├── feature_importance_top20_comparison_model2.png
#   │   └── feature_importance_categories_model2.png
#   └── Individual Misclassified Samples
#       ├── misclassified_<ID>_<Classifier>.png
#       └── ...
# ============================================================================

set -e  # Exit on error

# Load required module
module load apps/truba-ai/gpu-2024.0

# Activate environment
conda activate ts-sktime

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Navigate to post-processing directory
cd "$(dirname "$0")/04-postprocessing"

echo -e "${GREEN}============================================================================${NC}"
echo -e "${GREEN}POST-PROCESSING & VISUALIZATION PIPELINE${NC}"
echo -e "${GREEN}============================================================================${NC}"
echo "Start time: $(date)"
echo ""

# Create figures directory
mkdir -p figures

# ============================================================================
# Step 1: Error Analysis
# ============================================================================
echo -e "\n${YELLOW}[Step 1/4] Error Analysis - Batch Processing${NC}"
echo -e "${BLUE}────────────────────────────────────────────────────────────────────────────${NC}\n"

echo -e "${GREEN}[1.1] Analyzing Model 1 (Binary Classification) Errors...${NC}"
python visualize_errors_simple.py \
    --model model1 \
    --top 10 \
    --save-fig

echo -e "\n${GREEN}[1.2] Analyzing Model 2 (5-Class Classification) Errors...${NC}"
python visualize_errors_simple.py \
    --model model2 \
    --top 10 \
    --save-fig

echo -e "\n${GREEN}✓ Error analysis complete${NC}"
echo -e "  Output: figures/error_analysis_model1.png"
echo -e "  Output: figures/error_analysis_model2.png"

# ============================================================================
# Step 2: Feature Importance Analysis
# ============================================================================
echo -e "\n${YELLOW}[Step 2/4] Feature Importance Analysis${NC}"
echo -e "${BLUE}────────────────────────────────────────────────────────────────────────────${NC}\n"

echo -e "${GREEN}[2.1] Analyzing Model 1 Feature Importance...${NC}"
python visualize_feature_importance.py \
    --model model1 \
    --top 20 \
    --save-fig \
    --no-plot

echo -e "\n${GREEN}[2.2] Analyzing Model 2 Feature Importance...${NC}"
python visualize_feature_importance.py \
    --model model2 \
    --top 20 \
    --save-fig \
    --no-plot

echo -e "\n${GREEN}✓ Feature importance analysis complete${NC}"
echo -e "  Output: figures/feature_importance_top20_comparison_model1.png"
echo -e "  Output: figures/feature_importance_categories_model1.png"
echo -e "  Output: figures/feature_importance_top20_comparison_model2.png"
echo -e "  Output: figures/feature_importance_categories_model2.png"

# ============================================================================
# Step 3: Individual Misclassified Samples (Optional)
# ============================================================================
echo -e "\n${YELLOW}[Step 3/4] Visualizing Worst Misclassified Samples (Optional)${NC}"
echo -e "${BLUE}────────────────────────────────────────────────────────────────────────────${NC}\n"

# Get worst cases from Model 1
echo -e "${GREEN}[3.1] Extracting worst cases from Model 1...${NC}"
MODEL1_WORST=$(python visualize_errors_simple.py --model model1 --top 3 --no-plot 2>&1 | \
    grep -oP 'ID:\s+\K\d+' | head -3)

if [ ! -z "$MODEL1_WORST" ]; then
    echo -e "  Found IDs: $(echo $MODEL1_WORST | tr '\n' ' ')"
    for ID in $MODEL1_WORST; do
        echo -e "  Visualizing ID: $ID"
        python visualize_misclassified.py --id $ID --model model1 --save-fig > /dev/null 2>&1 || true
    done
else
    echo -e "${YELLOW}  ⚠ Could not extract worst case IDs (skipping individual visualizations)${NC}"
fi

# Get worst cases from Model 2
echo -e "\n${GREEN}[3.2] Extracting worst cases from Model 2...${NC}"
MODEL2_WORST=$(python visualize_errors_simple.py --model model2 --top 3 --no-plot 2>&1 | \
    grep -oP 'ID:\s+\K\d+' | head -3)

if [ ! -z "$MODEL2_WORST" ]; then
    echo -e "  Found IDs: $(echo $MODEL2_WORST | tr '\n' ' ')"
    for ID in $MODEL2_WORST; do
        echo -e "  Visualizing ID: $ID"
        python visualize_misclassified.py --id $ID --model model2 --save-fig > /dev/null 2>&1 || true
    done
else
    echo -e "${YELLOW}  ⚠ Could not extract worst case IDs (skipping individual visualizations)${NC}"
fi

echo -e "\n${GREEN}✓ Individual sample visualization complete${NC}"

# ============================================================================
# Step 4: Summary Report
# ============================================================================
echo -e "\n${YELLOW}[Step 4/4] Generating Summary Report${NC}"
echo -e "${BLUE}────────────────────────────────────────────────────────────────────────────${NC}\n"

# Count figures
TOTAL_FIGURES=$(find 04-postprocessing/figures/ -name "*.png" -type f 2>/dev/null | wc -l)
ERROR_FIGS=$(find 04-postprocessing/figures/ -name "error_analysis*.png" -type f 2>/dev/null | wc -l)
FEATURE_FIGS=$(find 04-postprocessing/figures/ -name "feature_importance*.png" -type f 2>/dev/null | wc -l)
MISCLASS_FIGS=$(find 04-postprocessing/figures/ -name "misclassified_*.png" -type f 2>/dev/null | wc -l)

# Get metrics from JSON files (Model 1)
MODEL1_BEST=$(find 03-models/hierarchical/model1_binary/saved_models/model1_binary_features/ \
    -name "*_summary.json" -type f 2>/dev/null | head -1)
if [ -f "$MODEL1_BEST" ]; then
    MODEL1_ACC=$(python3 -c "import json; print(f\"{json.load(open('$MODEL1_BEST'))['best_model']['test_accuracy']*100:.2f}%\")" 2>/dev/null || echo "N/A")
    MODEL1_NAME=$(python3 -c "import json; print(json.load(open('$MODEL1_BEST'))['best_model']['model_name'])" 2>/dev/null || echo "N/A")
else
    MODEL1_ACC="N/A"
    MODEL1_NAME="N/A"
fi

# Get metrics from JSON files (Model 2)
MODEL2_BEST=$(find 03-models/hierarchical/model2_nonstationary/saved_models/model2_nonstationary_features/ \
    -name "*_summary.json" -type f 2>/dev/null | head -1)
if [ -f "$MODEL2_BEST" ]; then
    MODEL2_ACC=$(python3 -c "import json; print(f\"{json.load(open('$MODEL2_BEST'))['best_model']['test_accuracy']*100:.2f}%\")" 2>/dev/null || echo "N/A")
    MODEL2_NAME=$(python3 -c "import json; print(json.load(open('$MODEL2_BEST'))['best_model']['model_name'])" 2>/dev/null || echo "N/A")
else
    MODEL2_ACC="N/A"
    MODEL2_NAME="N/A"
fi

echo -e "${GREEN}============================================================================${NC}"
echo -e "${GREEN}POST-PROCESSING COMPLETE!${NC}"
echo -e "${GREEN}============================================================================${NC}"
echo "End time: $(date)"
echo ""
echo -e "${YELLOW}📊 SUMMARY STATISTICS${NC}"
echo -e "${BLUE}────────────────────────────────────────────────────────────────────────────${NC}"
echo ""
echo -e "  Model Performance:"
echo -e "    Model 1 (Binary):      $MODEL1_ACC ($MODEL1_NAME)"
echo -e "    Model 2 (5-Class):     $MODEL2_ACC ($MODEL2_NAME)"
echo ""
echo -e "  Generated Figures:       $TOTAL_FIGURES total"
echo -e "    Error Analysis:        $ERROR_FIGS figures"
echo -e "    Feature Importance:    $FEATURE_FIGS figures"
echo -e "    Misclassified Samples: $MISCLASS_FIGS figures"
echo ""
echo -e "${YELLOW}📁 OUTPUT DIRECTORY${NC}"
echo -e "${BLUE}────────────────────────────────────────────────────────────────────────────${NC}"
echo "  04-postprocessing/figures/"
echo ""
ls -1 figures/*.png 2>/dev/null | sed 's/^/    /' || echo "    (no figures found)"
echo ""
echo -e "${YELLOW}📖 NEXT STEPS${NC}"
echo -e "${BLUE}────────────────────────────────────────────────────────────────────────────${NC}"
echo "  1. Review figures in: 04-postprocessing/figures/"
echo "  2. Check detailed metrics: 03-models/hierarchical/*/saved_models/*/*.json"
echo "  3. Generate paper-ready report: See REPORT_GUIDE.md"
echo "  4. Visualize specific misclassified samples:"
echo "     python visualize_misclassified.py --id <ID> --model <model1/model2> --save-fig"
echo ""
echo -e "${GREEN}============================================================================${NC}"
