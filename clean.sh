#!/bin/bash
# ============================================================================
# Clean All Generated Files
# ============================================================================
# This script removes all generated data, features, models, and logs
# to start fresh. Use with caution!
#
# Usage:
#   bash clean.sh [option]
#
# Options:
#   all       : Clean everything (data, features, models, logs, figures)
#   data      : Clean only raw data
#   features  : Clean only extracted features
#   models    : Clean only trained models
#   logs      : Clean only log files
#   figures   : Clean only figures
#   cache     : Clean only Python cache files
# ============================================================================

set -e

OPTION=${1:-all}

# Colors
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${RED}============================================================================${NC}"
echo -e "${RED}CLEANUP SCRIPT${NC}"
echo -e "${RED}============================================================================${NC}"
echo ""

clean_data() {
    echo -e "${YELLOW}Cleaning raw data...${NC}"
    if [ -d "data/raw/unified-5k" ]; then
        rm -rf data/raw/unified-5k
        echo "  ✓ Removed data/raw/unified-5k/"
    fi
    if [ -d "data/raw/unified-20k" ]; then
        rm -rf data/raw/unified-20k
        echo "  ✓ Removed data/raw/unified-20k/"
    fi
}

clean_features() {
    echo -e "${YELLOW}Cleaning extracted features...${NC}"
    if [ -d "data/features/unified-5k" ]; then
        rm -rf data/features/unified-5k
        echo "  ✓ Removed data/features/unified-5k/"
    fi
    if [ -d "data/features/unified-20k" ]; then
        rm -rf data/features/unified-20k
        echo "  ✓ Removed data/features/unified-20k/"
    fi
}

clean_models() {
    echo -e "${YELLOW}Cleaning trained models...${NC}"
    if [ -d "03-models/hierarchical/model1_binary/output" ]; then
        rm -rf 03-models/hierarchical/model1_binary/output/*
        echo "  ✓ Removed 03-models/hierarchical/model1_binary/output/*"
    fi
    if [ -d "03-models/hierarchical/model1_binary/saved_models" ]; then
        rm -rf 03-models/hierarchical/model1_binary/saved_models/*
        echo "  ✓ Removed 03-models/hierarchical/model1_binary/saved_models/*"
    fi
    if [ -d "03-models/hierarchical/model2_nonstationary/output" ]; then
        rm -rf 03-models/hierarchical/model2_nonstationary/output/*
        echo "  ✓ Removed 03-models/hierarchical/model2_nonstationary/output/*"
    fi
    if [ -d "03-models/hierarchical/model2_nonstationary/saved_models" ]; then
        rm -rf 03-models/hierarchical/model2_nonstationary/saved_models/*
        echo "  ✓ Removed 03-models/hierarchical/model2_nonstationary/saved_models/*"
    fi
}

clean_logs() {
    echo -e "${YELLOW}Cleaning log files...${NC}"
    if [ -d "logs" ]; then
        rm -rf logs/*
        echo "  ✓ Removed logs/*"
    fi
}

clean_figures() {
    echo -e "${YELLOW}Cleaning figures...${NC}"
    if [ -d "04-postprocessing/figures" ]; then
        # Keep the directory but remove all subdirectories
        find 04-postprocessing/figures -mindepth 1 -type d -exec rm -rf {} + 2>/dev/null || true
        find 04-postprocessing/figures -type f -name "*.png" -delete 2>/dev/null || true
        echo "  ✓ Removed 04-postprocessing/figures/*"
    fi
}

clean_cache() {
    echo -e "${YELLOW}Cleaning Python cache files...${NC}"
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    echo "  ✓ Removed Python cache files"
}

# Execute based on option
case "$OPTION" in
    all)
        echo -e "${RED}Cleaning ALL generated files...${NC}"
        echo ""
        clean_data
        clean_features
        clean_models
        clean_logs
        clean_figures
        clean_cache
        # Recreate necessary directories
        mkdir -p 03-models/hierarchical/model1_binary/output
        mkdir -p 03-models/hierarchical/model2_nonstationary/output
        mkdir -p logs
        mkdir -p 04-postprocessing/figures
        echo "  ✓ Recreated necessary directories"
        ;;
    data)
        clean_data
        ;;
    features)
        clean_features
        ;;
    models)
        clean_models
        ;;
    logs)
        clean_logs
        ;;
    figures)
        clean_figures
        ;;
    cache)
        clean_cache
        ;;
    *)
        echo -e "${RED}Invalid option: $OPTION${NC}"
        echo ""
        echo "Valid options:"
        echo "  all       : Clean everything"
        echo "  data      : Clean only raw data"
        echo "  features  : Clean only extracted features"
        echo "  models    : Clean only trained models"
        echo "  logs      : Clean only log files"
        echo "  figures   : Clean only figures"
        echo "  cache     : Clean only Python cache files"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}============================================================================${NC}"
echo -e "${GREEN}CLEANUP COMPLETE${NC}"
echo -e "${GREEN}============================================================================${NC}"
echo ""
echo -e "${BLUE}Ready to start fresh pipeline:${NC}"
echo "  5k test:  bash run-test-5k.sh"
echo "  20k full: bash run.sh all 20k statistical"
echo ""
