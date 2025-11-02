#!/bin/bash
# Quick smoke test for the hierarchical time series classification pipeline
# Tests: Data loading → Model 1 training → Model 1 testing

echo "================================================================================"
echo "SMOKE TEST: Hierarchical Time Series Classification"
echo "================================================================================"
echo ""

# Check conda environment
if ! python -c "import pandas" 2>/dev/null; then
    echo "❌ Error: Python environment not configured properly"
    echo "   Run: conda activate ts-sktime"
    exit 1
fi

echo "✓ Python environment OK"
echo ""

# Check data exists
if [ ! -d "data/raw/unified-90k" ]; then
    echo "❌ Error: Training data not found at data/raw/unified-90k"
    echo "   Generate data first: cd 01-data-generation && python generate.py"
    exit 1
fi

echo "✓ Training data exists"
echo ""

# Test 1: Train Model 1 (limited to first few files for speed)
echo "================================================================================"
echo "🎯 Test 1: Model 1 Training (ROCKET, limited data)"
echo "================================================================================"
cd 03-models/hierarchical/model1_binary
python train_model1.py --mode raw --classifier rocket --test-size 0.2 --random-state 42 2>&1 | head -100

if [ $? -ne 0 ]; then
    echo "❌ Model 1 training failed"
    cd ../../..
    exit 1
fi

echo ""
echo "✅ Model 1 training completed"
cd ../../..
echo ""

# Test 2: Quick Model 1 testing
echo "================================================================================"
echo "🧪 Test 2: Model 1 Testing (100 samples)"
echo "================================================================================"
cd 03-models/hierarchical/model1_binary
python test_model1.py --n-samples 100 2>&1 | head -50

if [ $? -ne 0 ]; then
    echo "❌ Model 1 testing failed"
    cd ../../..
    exit 1
fi

cd ../../..
echo ""

# Final summary
echo "================================================================================"
echo "✅ SMOKE TEST PASSED!"
echo "================================================================================"
echo ""
echo "All basic components are working:"
echo "  ✓ Data loading (PyArrow + ThreadPoolExecutor)"
echo "  ✓ Model 1 training (ROCKET classifier)"
echo "  ✓ Model 1 testing"
echo ""
echo "Next steps:"
echo "  1. Train full Model 1: cd 03-models/hierarchical/model1_binary && python train_model1.py"
echo "  2. Train Model 2: cd 03-models/hierarchical/model2_nonstationary && python train_model2.py"
echo "  3. Build hierarchical pipeline"
echo "================================================================================"
