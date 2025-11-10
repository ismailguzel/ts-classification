#!/bin/bash
# Unified smoke test runner covering raw, full features, and selected features scenarios.
# Usage examples:
#   bash smoke_test.sh                     # run raw + features scenarios
#   bash smoke_test.sh --mode raw          # run only the raw pipeline
#   bash smoke_test.sh --scenarios raw,selected --with-model2
#   bash smoke_test.sh --data-path data/raw/unified-test --features-path data/features/demo

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATA_PATH="data/raw/unified-20k"
FEATURES_PATH="data/features/unified-20k/allfeatures"
SELECTED_FEATURES_PATH="data/features/unified-20k/selected"
FALLBACK_DATA_PATH="data/raw/unified-20k"

SCENARIOS=("raw" "features")
N_SAMPLES=100
CLASSIFIER="rocket"
WITH_MODEL2=0
TRAIN_PREVIEW_LINES=120
TEST_PREVIEW_LINES=60

normalize_scenarios() {
  declare -ga SCENARIOS=()
  declare -A seen=()
  for entry in "$@"; do
    local lowered trimmed
    lowered=$(echo "$entry" | tr '[:upper:]' '[:lower:]')
    trimmed=${lowered// /}
    [[ -z "$trimmed" ]] && continue
    if [[ -z "${seen[$trimmed]+x}" ]]; then
      SCENARIOS+=("$trimmed")
      seen[$trimmed]=1
    fi
  done
  if [[ ${#SCENARIOS[@]} -eq 0 ]]; then
    echo "❌ No scenarios specified" >&2
    exit 1
  fi
}

run_and_preview() {
  local preview_lines="$1"
  shift
  local tmp_file
  tmp_file=$(mktemp)

  set +e
  "$@" &>"$tmp_file"
  local status=$?
  set -e

  head -n "$preview_lines" "$tmp_file"
  rm -f "$tmp_file"
  return $status
}

run_pipeline() {
  local scenario="$1"
  local mode="$2"
  local data_path="$3"
  local features_path="$4"
  local label="$5"

  local resolved_data=""
  local resolved_features=""

  printf '\n================================================================================\n'
  echo "SMOKE TEST SCENARIO: $label"
  echo "Mode: $mode"
  printf '================================================================================\n'

  if [[ "$mode" == "raw" ]]; then
    resolved_data="$data_path"
    if [[ ! -d "$resolved_data" ]]; then
      if [[ -d "$FALLBACK_DATA_PATH" ]]; then
        echo "⚠️  Training data not found at $resolved_data, falling back to $FALLBACK_DATA_PATH"
        resolved_data="$FALLBACK_DATA_PATH"
      else
        echo "❌ Error: Training data not found at $resolved_data"
        echo "   Generate data first via 01-data-generation/generate.py"
        return 1
      fi
    fi
    echo "✓ Training data available: $resolved_data"
  else
    resolved_features="$features_path"
    if [[ -z "$resolved_features" ]]; then
      echo "❌ Error: Features path not provided for scenario '$scenario'"
      return 1
    fi
    if [[ ! -d "$resolved_features" ]]; then
      echo "❌ Error: Features path not found at $resolved_features"
      echo "   Extract or adjust features before running this scenario."
      return 1
    fi
    echo "✓ Features path available: $resolved_features"
  fi

  pushd "$SCRIPT_DIR/03-models/hierarchical/model1_binary" >/dev/null
  if [[ "$mode" == "raw" ]]; then
    if ! run_and_preview "$TRAIN_PREVIEW_LINES" python train_model1.py \
        --mode raw \
        --classifier "$CLASSIFIER" \
        --data-path "../../../$resolved_data" \
        --test-size 0.2 \
        --random-state 42; then
      echo "❌ Model 1 training failed for scenario '$label'"
      popd >/dev/null
      return 1
    fi
  else
    if ! run_and_preview "$TRAIN_PREVIEW_LINES" python train_model1.py \
        --mode features \
        --classifier "$CLASSIFIER" \
        --features-path "../../../$resolved_features" \
        --test-size 0.2 \
        --random-state 42; then
      echo "❌ Model 1 training failed for scenario '$label'"
      popd >/dev/null
      return 1
    fi
  fi
  popd >/dev/null
  echo "✅ Model 1 training completed"

  pushd "$SCRIPT_DIR/03-models/hierarchical/model1_binary" >/dev/null
  if ! run_and_preview "$TEST_PREVIEW_LINES" python test_model1.py --n-samples "$N_SAMPLES"; then
    echo "❌ Model 1 testing failed for scenario '$label'"
    popd >/dev/null
    return 1
  fi
  popd >/dev/null
  echo "✅ Model 1 testing completed"

  if [[ $WITH_MODEL2 -eq 1 ]]; then
    pushd "$SCRIPT_DIR/03-models/hierarchical/model2_nonstationary" >/dev/null
    local model2_classifier="rocket"
    if [[ "$mode" == "features" ]]; then
      model2_classifier="rf"
    fi

    if [[ "$mode" == "raw" ]]; then
      if ! run_and_preview "$TRAIN_PREVIEW_LINES" python train_model2.py \
          --mode raw \
          --classifier "$model2_classifier" \
          --data-path "../../../$resolved_data" \
          --test-size 0.2 \
          --random-state 42; then
        echo "❌ Model 2 training failed for scenario '$label'"
        popd >/dev/null
        return 1
      fi
    else
      if ! run_and_preview "$TRAIN_PREVIEW_LINES" python train_model2.py \
          --mode features \
          --classifier "$model2_classifier" \
          --features-path "../../../$resolved_features" \
          --test-size 0.2 \
          --random-state 42; then
        echo "❌ Model 2 training failed for scenario '$label'"
        popd >/dev/null
        return 1
      fi
    fi

    if ! run_and_preview "$TEST_PREVIEW_LINES" python test_model2.py --n-samples "$N_SAMPLES"; then
      echo "❌ Model 2 testing failed for scenario '$label'"
      popd >/dev/null
      return 1
    fi
    popd >/dev/null
    echo "✅ Model 2 validation completed"
  fi

  echo "✅ Scenario '$label' passed"
  return 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE_ARG=$(echo "$2" | tr '[:upper:]' '[:lower:]')
      normalize_scenarios "$MODE_ARG"
      shift 2
      ;;
    --scenarios)
      IFS=',' read -ra requested <<< "$2"
      normalize_scenarios "${requested[@]}"
      shift 2
      ;;
    --data-path)
      DATA_PATH="$2"; shift 2 ;;
    --features-path)
      FEATURES_PATH="$2"; shift 2 ;;
    --selected-features-path)
      SELECTED_FEATURES_PATH="$2"; shift 2 ;;
    --n-samples)
      N_SAMPLES="$2"; shift 2 ;;
    --classifier)
      CLASSIFIER="$2"; shift 2 ;;
    --with-model2)
      WITH_MODEL2=1; shift 1 ;;
    --help|-h)
      echo "Usage: bash smoke_test.sh [options]"
      echo "  --mode <raw|features|selected>      Run only the specified scenario"
      echo "  --scenarios a,b,c                  Comma-separated list (raw, features, selected)"
      echo "  --data-path PATH                   Raw data directory"
      echo "  --features-path PATH               Full feature directory"
      echo "  --selected-features-path PATH      Selected feature directory"
      echo "  --n-samples INT                    Samples for quick testing (default: $N_SAMPLES)"
      echo "  --classifier NAME                  Model 1 classifier (default: $CLASSIFIER)"
      echo "  --with-model2                      Include Model 2 smoke tests"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

# If normalize_scenarios wasn't triggered, ensure defaults are normalized
if [[ ${#SCENARIOS[@]} -eq 0 ]]; then
  normalize_scenarios "raw" "features"
fi

echo "================================================================================"
echo "SMOKE TEST: Hierarchical Time Series Classification"
echo "Scenarios: ${SCENARIOS[*]}"
echo "Classifier: $CLASSIFIER"
echo "Model 2 enabled: ${WITH_MODEL2}" 
echo "================================================================================"

if ! python -c "import pandas" 2>/dev/null; then
  echo "❌ Error: Python environment not configured properly"
  echo "   Ensure dependencies are installed (pip install -r requirements.txt)"
  exit 1
fi
echo "✓ Python environment OK"

python - <<'PY'
try:
    import pyarrow, sktime
    print("✓ pyarrow and sktime available")
except Exception as exc:
    print(f"⚠️  Optional check: {exc}")
PY

TOTAL=${#SCENARIOS[@]}
COUNT=0
for scenario in "${SCENARIOS[@]}"; do
  case "$scenario" in
    raw)
      run_pipeline "raw" "raw" "$DATA_PATH" "" "Raw time series" || exit 1
      ;;
    features|full|fullfeatures)
      run_pipeline "$scenario" "features" "" "$FEATURES_PATH" "Full feature set" || exit 1
      ;;
    selected|selectedfeatures|selected-features)
      path_to_use="$SELECTED_FEATURES_PATH"
      if [[ -z "$path_to_use" ]]; then
        path_to_use="$FEATURES_PATH"
      fi
      run_pipeline "$scenario" "features" "" "$path_to_use" "Selected feature set" || exit 1
      unset path_to_use
      ;;
    *)
      echo "❌ Unsupported scenario: $scenario" >&2
      exit 1
      ;;
  esac
  COUNT=$((COUNT + 1))
done

echo "================================================================================"
echo "✅ Smoke test completed successfully ($COUNT/$TOTAL scenarios)"
if [[ $WITH_MODEL2 -eq 1 ]]; then
  echo "  • Model 2 checks were included"
fi
echo "================================================================================"
