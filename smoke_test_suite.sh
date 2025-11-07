#!/bin/bash
# Run smoke_test.sh across multiple data representations and capture logs.
# Scenarios:
#   1. Raw time series
#   2. Full feature set
#   3. Selected feature set
#
# Usage examples:
#   bash smoke_test_suite.sh
#   bash smoke_test_suite.sh --raw-path data/raw/unified-test --log-root logs/smoke/custom
#   bash smoke_test_suite.sh -- --n-samples 50 --classifier rocket

set -euo pipefail

RAW_PATH="data/raw/unified-90k"
FEATURES_PATH="data/features"
SELECTED_FEATURES_PATH="data/features/selected"
LOG_ROOT="logs/smoke/$(date +%Y%m%d_%H%M%S)"
WITH_MODEL2=true
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --raw-path)
      RAW_PATH="$2"
      shift 2
      ;;
    --features-path)
      FEATURES_PATH="$2"
      shift 2
      ;;
    --selected-features-path)
      SELECTED_FEATURES_PATH="$2"
      shift 2
      ;;
    --log-root)
      LOG_ROOT="$2"
      shift 2
      ;;
    --skip-model2)
      WITH_MODEL2=false
      shift 1
      ;;
    --)
      shift 1
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift 1
      ;;
  esac
done

mkdir -p "$LOG_ROOT"

echo "🗂️  Logs will be written under: $LOG_ROOT"

declare -a SCENARIOS
SCENARIOS=(
  "raw::--mode raw --data-path $RAW_PATH"
  "features::--mode features --features-path $FEATURES_PATH"
  "selected::--mode features --features-path $SELECTED_FEATURES_PATH"
)

COUNTER=1
TOTAL=${#SCENARIOS[@]}

for ENTRY in "${SCENARIOS[@]}"; do
  NAME=${ENTRY%%::*}
  ARGS_STRING=${ENTRY##*::}
  read -r -a MODE_ARGS <<< "$ARGS_STRING"

  LOG_FILE="$LOG_ROOT/${COUNTER}_${NAME}.log"
  printf '\n================================================================================\n'
  echo "Scenario $COUNTER/$TOTAL: $NAME"
  echo "Command args: ${MODE_ARGS[*]} ${EXTRA_ARGS[*]}"
  echo "Logs: $LOG_FILE"
  printf '================================================================================\n'

  CMD=("bash" "smoke_test.sh" "${MODE_ARGS[@]}")
  if $WITH_MODEL2; then
    CMD+=("--with-model2")
  fi
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_ARGS[@]}")
  fi

  # shellcheck disable=SC2068
  "${CMD[@]}" 2>&1 | tee "$LOG_FILE"

  ((COUNTER++))
  echo "Finished scenario: $NAME"
  printf '================================================================================\n\n'

done

echo "All scenarios completed. Logs stored in $LOG_ROOT"
