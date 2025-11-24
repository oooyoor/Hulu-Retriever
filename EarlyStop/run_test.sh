#!/bin/bash
set -euo pipefail

EXEC_NAME=${1:-}
NUM_THREADS=${2:-}

WORKSPACE_ROOT="/home/zqf/Hulu-Retriever"
CONFIG_PATH="${WORKSPACE_ROOT}/configs/Config.json"
EXEC_DIR="./execs"
WARMUP_SCRIPT="${WORKSPACE_ROOT}/WarmUp/run_warmup.sh"
WARMUP_BIN="${WORKSPACE_ROOT}/WarmUp/execs/warmup"

if [[ -z "$EXEC_NAME" ]]; then
    echo "[run_test] 用法: $0 <exec_name> [num_threads]" >&2
    exit 1
fi

if [[ ! -x "${EXEC_DIR}/${EXEC_NAME}" ]]; then
    echo "[run_test] 可执行文件不存在: ${EXEC_DIR}/${EXEC_NAME}" >&2
    exit 1
fi

mapfile -t DATASETS < <(jq -r '.dataset_list[].dataset_name' "$CONFIG_PATH")
REPEATS=1

log()   { echo -e "\033[0;32m[RUN_TEST]\033[0m $1"; }
warn()  { echo -e "\033[1;33m[RUN_TEST]\033[0m $1"; }
err()   { echo -e "\033[0;31m[RUN_TEST]\033[0m $1" >&2; }

clear_cache() {
    log "清理系统缓存"
    sudo sync
    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
}

wait_for_exit() {
    local pattern="$1"
    local label="$2"
    local timeout="${3:-120}"
    local waited=0

    while pgrep -f "$pattern" >/dev/null 2>&1; do
        if (( waited == 0 )); then
            log "等待 ${label} 彻底退出..."
        fi
        sleep 1
        waited=$(( waited + 1 ))
        if (( waited >= timeout )); then
            warn "${label} 超过 ${timeout}s 仍在运行，尝试强制结束"
            pkill -9 -f "$pattern" || true
            break
        fi
    done
}

ensure_clean_environment() {
    local patterns=(
        "${EXEC_DIR}/offset"
        "${EXEC_DIR}/single"
        "${EXEC_DIR}/multi"
        "${EXEC_DIR}/${EXEC_NAME}"
        "${WARMUP_BIN}"
    )

    for pattern in "${patterns[@]}"; do
        if pgrep -f "$pattern" >/dev/null 2>&1; then
            warn "检测到残留进程 (${pattern})，尝试 kill"
            pkill -9 -f "$pattern" || true
        fi
        wait_for_exit "$pattern" "$pattern" 30
    done
}

run_warmup() {
    log "执行 warmup"
    ensure_clean_environment
    "$WARMUP_SCRIPT"
    wait_for_exit "$WARMUP_BIN" "warmup" 90
    clear_cache
}

ensure_clean_environment

for DATASET_NAME in "${DATASETS[@]}"; do
    for ((i = 1; i <= REPEATS; i++)); do
        log "====================================="
        log "数据集: ${DATASET_NAME} | Repeat: ${i}"
        log "====================================="

        run_warmup
        clear_cache

        if [[ -n "$NUM_THREADS" ]]; then
            "${EXEC_DIR}/${EXEC_NAME}" "$CONFIG_PATH" "$DATASET_NAME" "$i" "$NUM_THREADS"
        else
            "${EXEC_DIR}/${EXEC_NAME}" "$CONFIG_PATH" "$DATASET_NAME" "$i"
        fi

        wait_for_exit "${EXEC_DIR}/${EXEC_NAME}" "${EXEC_NAME}" 180

        sleep 5
    done
done
