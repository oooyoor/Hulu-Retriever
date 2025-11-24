#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/Results}"
SUMMARY_PATH="${SUMMARY_PATH:-${RESULTS_ROOT}/difficulty_summary_iter_count.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${RESULTS_ROOT}/difficulty_iter_recall_plots}"
FORMAT_LIST="${FORMAT_LIST:-png,pdf}"

DATASET_ARGS=()

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
        --summary)
            SUMMARY_PATH="$(realpath "$2")"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$(realpath "$2")"
            shift 2
            ;;
        --format)
            FORMAT_LIST="$2"
            shift 2
            ;;
        --dataset)
            DATASET_ARGS+=("$1" "$2")
            shift 2
            ;;
        *)
            echo "[WARN] Unknown arg: $1" >&2
            shift 1
            ;;
    esac
done

if [[ ! -f "${SUMMARY_PATH}" ]]; then
    echo "[ERR] Summary JSON not found: ${SUMMARY_PATH}" >&2
    exit 1
fi

# build repeated --format flags
IFS=',' read -r -a __formats <<< "${FORMAT_LIST}"
FORMAT_ARGS=()
for fmt in "${__formats[@]}"; do
    fmt_trimmed="$(echo "$fmt" | xargs)"
    [[ -z "${fmt_trimmed}" ]] && continue
    FORMAT_ARGS+=("--format" "${fmt_trimmed}")
done
if [[ ${#FORMAT_ARGS[@]} -eq 0 ]]; then
    FORMAT_ARGS=("--format" "png" "--format" "pdf")
fi

echo "[INFO] Rendering iteration + recall trade-off plots..."
"${PYTHON_BIN}" "${SCRIPT_DIR}/plot_iter_recall_tradeoffs.py" \
    --summary "${SUMMARY_PATH}" \
    --results-root "${RESULTS_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    "${FORMAT_ARGS[@]}" \
    "${DATASET_ARGS[@]}"

echo "[OK] Trade-off plots ready at ${OUTPUT_DIR}"

