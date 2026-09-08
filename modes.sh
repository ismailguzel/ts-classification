#!/bin/bash
# ============================================================================
# Shared mode -> path mapping. Sourced by every run-*.sh; not run directly.
# ============================================================================
# Every path a mode needs is derived mechanically from its name, so adding a
# pipeline mode means adding exactly one file:
#
#     01-data-generation/<mode>-config.json
#
# and nothing else. Two consequences worth keeping:
#
#   1. No if/elif chain to update in four places (and to forget in one of them).
#   2. MODEL_DIR and FIGURES_DIR both carry $MODE, so Study A and Study B runs
#      can never overwrite each other's results. The pre-2026-09 layout used a
#      bare output/ + output_topo/ + output_hybrid/ and would have silently
#      clobbered across studies.
#
# Expects MODE and (optionally) FEATURES to be set by the caller.
# ============================================================================

BASE_DIR="${BASE_DIR:-$(pwd)}"
MODE="${MODE:-shape}"
FEATURES="${FEATURES:-tsfresh}"

CONFIG_NAME="${MODE}-config.json"
CONFIG_FILE="$BASE_DIR/01-data-generation/$CONFIG_NAME"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: unknown mode '$MODE' — no such config: $CONFIG_FILE"
    echo ""
    echo "Available modes:"
    for f in "$BASE_DIR"/01-data-generation/*-config.json; do
        [ -e "$f" ] || continue
        b=$(basename "$f")
        echo "  ${b%-config.json}"
    done
    exit 1
fi

DATA_RAW="$BASE_DIR/data/raw/$MODE"
RAW_PARQUET="$DATA_RAW/${MODE}.parquet"

DATA_FEATURES="$BASE_DIR/data/features/$MODE/allfeatures"
DATA_SELECTED="$BASE_DIR/data/features/$MODE/selected"
DATA_TOPO="$BASE_DIR/data/features/$MODE/topological"
DATA_TOPO_SEL="$BASE_DIR/data/features/$MODE/topological_selected"

MODEL_DIR="$BASE_DIR/03-models/flat_classifier/output/${MODE}_${FEATURES}"
FIGURES_DIR="$BASE_DIR/04-postprocessing/figures/${MODE}_${FEATURES}"

# Per-series z-normalisation before the filtration and the Takens embedding.
# Derived from the mode rather than passed by hand, because getting it wrong is
# silent: sub/superlevel persistence is measured in the units of the signal.
#
#   season-*  ON.  Measured on the generated data, per-series sigma is 8.45 for
#                  sarima against 0.28-0.86 for the other three seasonal bases.
#                  Raw amplitude alone would separate sarima and tell us nothing
#                  about the shape of its loop.
#   shape     OFF. variance_shift and volatility are genuinely about amplitude;
#                  standardising it away would delete the signal.
#
# Override with ZSCORE=1 / ZSCORE=0 to run a mode both ways — that comparison is
# how you measure how much of a diagram family's power is shape and how much is
# scale.
case "$MODE" in
    season-*) ZSCORE_DEFAULT=1 ;;
    *)        ZSCORE_DEFAULT=0 ;;
esac
ZSCORE="${ZSCORE:-$ZSCORE_DEFAULT}"
