#!/bin/bash
# ============================================================================
# Data Generation Pipeline
# ============================================================================
# Generates synthetic time series dataset (20k scale by default).
# Output: data/raw/unified-20k/
# ============================================================================

set -e  # Exit on error

# Load required module (if on cluster)
if command -v module &> /dev/null; then
    module load apps/truba-ai/gpu-2024.0
fi

# Activate environment
conda activate ts-generation

# Configuration
SCALE="20k"  # Default scale

echo "============================================================"
echo "Starting Data Generation Pipeline ($SCALE)"
echo "============================================================"
echo "Start time: $(date)"
echo ""

# Navigate to generation directory
cd 01-data-generation

# Print configuration summary
python config.py "$SCALE"

# Run generation
python generate.py --scale "$SCALE"

cd ..

# Check output
OUTPUT_DIR="data/raw/unified-$SCALE"
if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A $OUTPUT_DIR)" ]; then
   echo "✓ Data generation successful: $OUTPUT_DIR"
else
   echo "✗ Error: Output directory not created or empty: $OUTPUT_DIR"
   exit 1
fi

echo ""
echo "============================================================"
echo "Data Generation Complete!"
echo "Output: data/raw/unified-$SCALE"
echo "============================================================"
echo "End time: $(date)"

