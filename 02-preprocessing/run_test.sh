#!/bin/bash

# Test script for topological features with ts-top environment

echo "======================================================================="
echo "Testing Topological Feature Extraction (with ts-top environment)"
echo "======================================================================="

# Activate environment
echo ""
echo "Activating ts-top environment..."
module load apps/truba-ai/gpu-2024.0
source activate ts-top

# Run the test
echo ""
echo "Running test..."
python test_topological.py

echo ""
echo "======================================================================="
echo "Test completed!"
echo "======================================================================="
