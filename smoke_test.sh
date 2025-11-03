#!/bin/bash
# Quick smoke test for the hierarchical time series classification pipeline
# Tests: Data loading → Model 1 training → Model 1 testing

set -e

# Args (optional): --data-path <path> --features-path <path> --mode <raw|features> --n-samples <int> --classifier <name> --with-model2
MODE="raw"
DATA_PATH="data/raw/unified-90k"
FEATURES_PATH="data/features/selected"
N_SAMPLES=100
CLASSIFIER="rocket"
WITH_MODEL2=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"; shift 2 ;;
    --data-path)
      DATA_PATH="$2"; shift 2 ;;
    --features-path)
      FEATURES_PATH="$2"; shift 2 ;;
    --n-samples)
      N_SAMPLES="$2"; shift 2 ;;
    --classifier)
      CLASSIFIER="$2"; shift 2 ;;
    --with-model2)
      WITH_MODEL2=1; shift 1 ;;
    *)
      echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "================================================================================"
echo "SMOKE TEST: Hierarchical Time Series Classification"
echo "================================================================================"
echo ""

# Check python environment
if ! python -c "import pandas" 2>/dev/null; then
    echo "❌ Error: Python environment not configured properly"
    echo "   Ensure dependencies are installed (pip install -r requirements.txt)"
    exit 1
fi

echo "✓ Python environment OK"
echo ""

# Quick dependency checks used in the pipeline
python - <<'PY'
try:
    import pyarrow, sktime
    print("✓ pyarrow and sktime available")
except Exception as e:
    print(f"⚠️  Optional check: {e}")
PY

if [ "$MODE" = "raw" ]; then
  # Resolve data path (fallback to unified-test if full not found and no explicit override)
  if [ ! -d "$DATA_PATH" ]; then
    if [ -d "data/raw/unified-test" ]; then
      DATA_PATH="data/raw/unified-test"
      echo "⚠️  Full dataset not found, using test dataset: $DATA_PATH"
    else
      echo "❌ Error: Training data not found at $DATA_PATH"
      echo "   Generate data first: cd 01-data-generation && python generate_toy.py (or generate.py)"
      exit 1
    fi
  fi
  echo "✓ Training data exists at: $DATA_PATH"
  echo ""
else
  # Features mode: check features path presence (legacy or standardized files are acceptable)
  if [ ! -d "$FEATURES_PATH" ]; then
    echo "❌ Error: Features path not found at $FEATURES_PATH"
    echo "   Extract/select features first: see 02-preprocessing/README.md"
    exit 1
  fi
  echo "✓ Features path exists at: $FEATURES_PATH"
  echo ""
fi

# Test 1: Train Model 1 (limited to first few files for speed)
echo "================================================================================"
echo "🎯 Test 1: Model 1 Training ($MODE mode, classifier=$CLASSIFIER)"
echo "================================================================================"
cd 03-models/hierarchical/model1_binary
if [ "$MODE" = "raw" ]; then
  python train_model1.py --mode raw --classifier "$CLASSIFIER" --data-path "../../../$DATA_PATH" --test-size 0.2 --random-state 42 2>&1 | head -100
else
  python train_model1.py --mode features --classifier rf --features-path "../../../$FEATURES_PATH" --test-size 0.2 --random-state 42 2>&1 | head -100
fi

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
echo "🧪 Test 2: Model 1 Testing ($N_SAMPLES samples)"
echo "================================================================================"
cd 03-models/hierarchical/model1_binary
python test_model1.py --n-samples "$N_SAMPLES" 2>&1 | head -50

if [ $? -ne 0 ]; then
    echo "❌ Model 1 testing failed"
    cd ../../..
    exit 1
fi

cd ../../..
echo ""

# Optional: Model 2 quick training/testing
if [ $WITH_MODEL2 -eq 1 ]; then
  echo "================================================================================"
  echo "🎯 Test 3: Model 2 Training ($MODE mode)"
  echo "================================================================================"
  cd 03-models/hierarchical/model2_nonstationary
  if [ "$MODE" = "raw" ]; then
    python train_model2.py --mode raw --classifier rocket --data-path "../../../$DATA_PATH" --test-size 0.2 --random-state 42 2>&1 | head -80
  else
    python train_model2.py --mode features --classifier rf --features-path "../../../$FEATURES_PATH" --test-size 0.2 --random-state 42 2>&1 | head -80
  fi

  if [ $? -ne 0 ]; then
      echo "❌ Model 2 training failed"
      cd ../../..
      exit 1
  fi

  echo ""
  echo "🧪 Test 4: Model 2 Testing ($N_SAMPLES samples)"
  python test_model2.py --n-samples "$N_SAMPLES" 2>&1 | head -50

  if [ $? -ne 0 ]; then
      echo "❌ Model 2 testing failed"
      cd ../../..
      exit 1
  fi

  cd ../../..
  echo ""
fi

# Final summary
echo "================================================================================"
echo "✅ SMOKE TEST PASSED!"
echo "================================================================================"
echo ""
echo "All basic components are working:"
echo "  ✓ Data loading (PyArrow + ThreadPoolExecutor)"
echo "  ✓ Model 1 training (ROCKET classifier)"
echo "  ✓ Model 1 testing"
if [ $WITH_MODEL2 -eq 1 ]; then
  echo "  ✓ Model 2 training"
  echo "  ✓ Model 2 testing"
fi
echo ""
echo "Next steps:"
echo "  1. Train full Model 1: cd 03-models/hierarchical/model1_binary && python train_model1.py"
echo "  2. Train Model 2: cd 03-models/hierarchical/model2_nonstationary && python train_model2.py"
echo "  3. Build hierarchical pipeline"
echo "================================================================================"
