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
FEATURES_PATH="data/features/smoke-dask"
SELECTED_FEATURES_PATH="data/features/smoke-dask"
FALLBACK_DATA_PATH="data/raw/unified-20k"

RUN_FEATURE_EXTRACTION=1
FEATURE_MAX_FILES=20
FEATURE_SET="efficient"
FEATURE_INPUT_PATH=""
FEATURE_NO_PROGRESS=1
FEATURE_USE_CLIENT=0
FEATURE_GATHER_STATS=0
FEATURE_EXTRACTION_SCRIPT="$SCRIPT_DIR/02-preprocessing/extract_dask.py"

SCENARIOS=("raw" "features")
N_SAMPLES=100
CLASSIFIER="rocket"
WITH_MODEL2=0
TRAIN_PREVIEW_LINES=120
TEST_PREVIEW_LINES=60

declare -A FEATURE_DONE=()

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

run_feature_extraction() {
  local input_path="$1"
  local output_path="$2"

  if [[ $RUN_FEATURE_EXTRACTION -eq 0 ]]; then
    echo "↷ Skipping feature extraction (disabled)"
    return 0
  fi

  if [[ -z "$input_path" ]]; then
    echo "❌ Feature extraction input path not provided" >&2
    return 1
  fi

  if [[ ! -d "$input_path" ]]; then
    echo "❌ Feature extraction input path not found: $input_path" >&2
    return 1
  fi

  if [[ ! -f "$FEATURE_EXTRACTION_SCRIPT" ]]; then
    echo "❌ Feature extraction script not found: $FEATURE_EXTRACTION_SCRIPT" >&2
    return 1
  fi

  local input_abs
  input_abs=$(cd "$input_path" && pwd)

  mkdir -p "$output_path"
  local output_abs
  output_abs=$(cd "$output_path" && pwd)

  if [[ -n "${FEATURE_DONE[$output_abs]+x}" ]]; then
    echo "↷ Feature extraction already executed for $output_abs (skipping)"
    return 0
  fi

  rm -f "$output_abs/features.parquet" "$output_abs/labels.parquet" "$output_abs/feature_names.txt"
  rm -rf "$output_abs/temp_chunks"

  echo "▶ Running feature extraction (input: $input_abs → output: $output_abs)"

  local cmd=(
    python "$FEATURE_EXTRACTION_SCRIPT"
    --input "$input_abs"
    --output "$output_abs"
    --feature-set "$FEATURE_SET"
  )

  if [[ -n "$FEATURE_MAX_FILES" && "$FEATURE_MAX_FILES" != "all" && "$FEATURE_MAX_FILES" != "none" ]]; then
    cmd+=(--max-files "$FEATURE_MAX_FILES")
  fi

  if [[ $FEATURE_NO_PROGRESS -eq 1 ]]; then
    cmd+=(--no-progress)
  fi

  if [[ $FEATURE_USE_CLIENT -eq 0 ]]; then
    cmd+=(--no-client)
  fi

  if [[ $FEATURE_GATHER_STATS -eq 1 ]]; then
    cmd+=(--gather-statistics)
  fi

  if ! run_and_preview "$TRAIN_PREVIEW_LINES" "${cmd[@]}"; then
    echo "❌ Feature extraction failed" >&2
    return 1
  fi

  FEATURE_DONE[$output_abs]=1
  echo "✅ Feature extraction completed for $output_abs"
  return 0
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
    resolved_data="$(cd "$resolved_data" && pwd)"
    echo "✓ Training data available: $resolved_data"
  else
    resolved_features="$features_path"
    if [[ -z "$resolved_features" ]]; then
      echo "❌ Error: Features path not provided for scenario '$scenario'"
      return 1
    fi
    local extraction_input="$data_path"
    if [[ -n "$FEATURE_INPUT_PATH" ]]; then
      extraction_input="$FEATURE_INPUT_PATH"
    fi
    if [[ -z "$extraction_input" ]]; then
      extraction_input="$DATA_PATH"
    fi
    if [[ -z "$extraction_input" ]]; then
      extraction_input="$FALLBACK_DATA_PATH"
    fi
    if [[ -z "$extraction_input" ]]; then
      echo "❌ Unable to determine raw data path for feature extraction" >&2
      return 1
    fi

    if ! run_feature_extraction "$extraction_input" "$resolved_features"; then
      return 1
    fi

    if [[ ! -d "$resolved_features" ]]; then
      echo "❌ Error: Features path not found at $resolved_features"
      echo "   Feature extraction step did not produce the expected directory."
      return 1
    fi
    resolved_features="$(cd "$resolved_features" && pwd)"
    if [[ -d "$extraction_input" ]]; then
      echo "✓ Features generated from: $(cd "$extraction_input" && pwd)"
    else
      echo "✓ Features generated from: $extraction_input"
    fi
    echo "✓ Features path ready: $resolved_features"
  fi

  local data_arg=""
  local features_arg=""
  if [[ "$mode" == "raw" ]]; then
    data_arg="$resolved_data"
  else
    features_arg="$resolved_features"
  fi

  pushd "$SCRIPT_DIR/03-models/hierarchical/model1_binary" >/dev/null
  if [[ "$mode" == "raw" ]]; then
    if ! run_and_preview "$TRAIN_PREVIEW_LINES" python train_model1.py \
        --mode raw \
        --classifier "$CLASSIFIER" \
        --data-path "$data_arg" \
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
        --features-path "$features_arg" \
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
  local test_cmd=(python test_model1.py --n-samples "$N_SAMPLES")
  if [[ -n "$data_arg" ]]; then
    test_cmd+=(--data-path "$data_arg")
  fi
  if [[ -n "$features_arg" ]]; then
    test_cmd+=(--features-path "$features_arg")
  fi
  if ! run_and_preview "$TEST_PREVIEW_LINES" "${test_cmd[@]}"; then
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
          --data-path "$data_arg" \
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
          --features-path "$features_arg" \
          --test-size 0.2 \
          --random-state 42; then
        echo "❌ Model 2 training failed for scenario '$label'"
        popd >/dev/null
        return 1
      fi
    fi

  local test2_cmd=(python test_model2.py --n-samples "$N_SAMPLES")
    if [[ -n "$data_arg" ]]; then
      test2_cmd+=(--data-path "$data_arg")
    fi
    if [[ -n "$features_arg" ]]; then
      test2_cmd+=(--features-path "$features_arg")
    fi

    if ! run_and_preview "$TEST_PREVIEW_LINES" "${test2_cmd[@]}"; then
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
    --feature-input-path)
      FEATURE_INPUT_PATH="$2"; shift 2 ;;
    --feature-max-files)
      FEATURE_MAX_FILES="$2"; shift 2 ;;
    --feature-set)
      FEATURE_SET="$2"; shift 2 ;;
    --skip-feature-extraction)
      RUN_FEATURE_EXTRACTION=0; shift 1 ;;
    --feature-use-client)
      FEATURE_USE_CLIENT=1; shift 1 ;;
    --feature-no-client)
      FEATURE_USE_CLIENT=0; shift 1 ;;
    --feature-progress)
      FEATURE_NO_PROGRESS=0; shift 1 ;;
    --feature-no-progress)
      FEATURE_NO_PROGRESS=1; shift 1 ;;
    --feature-gather-statistics)
      FEATURE_GATHER_STATS=1; shift 1 ;;
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
  echo "  --feature-input-path PATH           Raw data path for feature extraction"
  echo "  --feature-max-files INT            Limit files during feature extraction (default: $FEATURE_MAX_FILES)"
  echo "  --feature-set NAME                 TSFresh feature set (default: $FEATURE_SET)"
  echo "  --skip-feature-extraction          Skip feature engineering step"
  echo "  --feature-use-client               Launch a local Dask client"
  echo "  --feature-no-client                Disable Dask client (default)"
  echo "  --feature-progress                 Show TSFresh progress bars"
  echo "  --feature-no-progress              Hide TSFresh progress bars (default)"
  echo "  --feature-gather-statistics        Gather parquet metadata statistics"
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
if [[ $RUN_FEATURE_EXTRACTION -eq 1 ]]; then
  max_desc="$FEATURE_MAX_FILES"
  if [[ -z "$max_desc" || "$max_desc" == "none" ]]; then
    max_desc="auto"
  fi
  echo "Feature extraction: enabled (max_files=$max_desc, set=$FEATURE_SET)"
else
  echo "Feature extraction: skipped"
fi
if [[ -n "$FEATURE_INPUT_PATH" ]]; then
  feature_input_display="$FEATURE_INPUT_PATH"
else
  feature_input_display="$DATA_PATH"
fi
if [[ -d "$feature_input_display" ]]; then
  feature_input_display="$(cd "$feature_input_display" && pwd)"
fi
echo "Feature input path: $feature_input_display"

feature_output_display="$FEATURES_PATH"
if [[ -d "$feature_output_display" ]]; then
  feature_output_display="$(cd "$feature_output_display" && pwd)"
fi
echo "Feature output path: $feature_output_display"
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
      run_pipeline "$scenario" "features" "$DATA_PATH" "$FEATURES_PATH" "Full feature set" || exit 1
      ;;
    selected|selectedfeatures|selected-features)
      path_to_use="$SELECTED_FEATURES_PATH"
      if [[ -z "$path_to_use" ]]; then
        path_to_use="$FEATURES_PATH"
      fi
      run_pipeline "$scenario" "features" "$DATA_PATH" "$path_to_use" "Selected feature set" || exit 1
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
