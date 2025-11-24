#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------------------------------------
# Defaults (can be overridden via CLI flags below)
# ----------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/Results}"
SUMMARY_OUT="${SUMMARY_OUT:-}"
PLOTS_DIR="${PLOTS_DIR:-}"
VALUE_KEY="iter_count"

COMPARE_ARGS=()
PLOT_DATASET_ARGS=()
DATASETS=()
SUMMARY_SET=false
PLOTS_SET=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python-bin)
            PYTHON_BIN="$2"
            shift 2
            ;;
        --results-root)
            RESULTS_ROOT="$(realpath "$2")"
            shift 2
            ;;
        --summary-out)
            SUMMARY_OUT="$(realpath "$2")"
            SUMMARY_SET=true
            shift 2
            ;;
        --plots-dir)
            PLOTS_DIR="$(realpath "$2")"
            PLOTS_SET=true
            shift 2
            ;;
        --dataset)
            COMPARE_ARGS+=("$1" "$2")
            PLOT_DATASET_ARGS+=("$1" "$2")
            DATASETS+=("$2")
            shift 2
            ;;
        --num-bins)
            COMPARE_ARGS+=("$1" "$2")
            shift 2
            ;;
        --value-key)
            VALUE_KEY="$2"
            COMPARE_ARGS+=("$1" "$2")
            shift 2
            ;;
        *)
            COMPARE_ARGS+=("$1")
            shift 1
            ;;
    esac
done

if ! $SUMMARY_SET; then
    suffix="${VALUE_KEY}"
    if [[ ${#DATASETS[@]} -eq 1 ]]; then
        suffix="${DATASETS[0]}_${suffix}"
    fi
    SUMMARY_OUT="${RESULTS_ROOT}/difficulty_summary_${suffix}.json"
fi

if ! $PLOTS_SET; then
    suffix="${VALUE_KEY}"
    if [[ ${#DATASETS[@]} -eq 1 ]]; then
        suffix="${DATASETS[0]}_${suffix}"
    fi
    PLOTS_DIR="${RESULTS_ROOT}/difficulty_plots_${suffix}"
fi

mkdir -p "${RESULTS_ROOT}"

echo "[INFO] Running difficulty aggregation..."
"${PYTHON_BIN}" "${REPO_ROOT}/SearchDifficulty/compare_query_difficulty.py" \
    --results-root "${RESULTS_ROOT}" \
    --output "${SUMMARY_OUT}" \
    "${COMPARE_ARGS[@]}"

echo "[INFO] Rendering improvement plots..."
"${PYTHON_BIN}" "${REPO_ROOT}/SearchDifficulty/plot_difficulty_improvements.py" \
    --summary "${SUMMARY_OUT}" \
    --output-dir "${PLOTS_DIR}" \
    "${PLOT_DATASET_ARGS[@]}"

echo "[OK] Pipeline finished."


