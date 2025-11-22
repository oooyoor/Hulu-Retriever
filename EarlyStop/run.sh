#!/bin/bash

# ====== 日志函数 ======
log()   { echo -e "\033[0;32m[RUN_TEST]\033[0m $1"; }
warn()  { echo -e "\033[1;33m[RUN_TEST]\033[0m $1"; }
err()   { echo -e "\033[0;31m[RUN_TEST]\033[0m $1" >&2; }

# ====== 清缓存 ======
clear_cache() {
    log "清理系统缓存"
    sudo sync
    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
}

# ====== Warmup ======
WORKSPACE_ROOT="/home/zqf/Hulu-Retriever"
WARMUP_SCRIPT="${WORKSPACE_ROOT}/WarmUp/run_warmup.sh"
WARMUP_BIN="${WORKSPACE_ROOT}/WarmUp/execs/warmup"

run_warmup() {
    log "执行 warmup"
    ensure_clean_environment
    "$WARMUP_SCRIPT"
    wait_for_exit "$WARMUP_BIN" "warmup" 60
    clear_cache
}

# ====== Config 与 Exec 一一对应 ======
CONFIG_PATH_LIST=(
    "${WORKSPACE_ROOT}/configs/Config.json"
    # "${WORKSPACE_ROOT}/configs/NewRecallConfig.json"
)

EXEC_LIST=(
    # "${WORKSPACE_ROOT}/EarlyStop/execs/offset"
    "${WORKSPACE_ROOT}/EarlyStop/execs/recall"
)

# ====== 默认重复次数 ======
REPEATS="${REPEATS:-1}"

# ====== 遍历每个 config / exec 配对 ======
for idx in "${!CONFIG_PATH_LIST[@]}"; do
    CONFIG_PATH="${CONFIG_PATH_LIST[$idx]}"
    EXEC_PATH="${EXEC_LIST[$idx]}"
    EXEC_NAME=$(basename "$EXEC_PATH")

    if [[ ! -x "$EXEC_PATH" ]]; then
        err "可执行文件不存在: $EXEC_PATH"
        continue
    fi

    log "---------------------------------------------"
    log "执行配对组 index=$idx"
    log "Config: $CONFIG_PATH"
    log "Exec:   $EXEC_PATH"
    log "---------------------------------------------"

    # ====== 从 JSON 中读取 dataset 名称 ======
    mapfile -t DATASETS < <(jq -r '.dataset_list[].dataset_name' "$CONFIG_PATH")

    if [[ ${#DATASETS[@]} -eq 0 ]]; then
        err "Config 中 dataset_list 为空！路径: $CONFIG_PATH"
        continue
    fi

    log "解析到 ${#DATASETS[@]} 个数据集: ${DATASETS[*]}"

    # ====== 遍历每个 dataset ======
    for DATASET_NAME in "${DATASETS[@]}"; do
        for ((i = 1; i <= REPEATS; i++)); do
            log "====================================="
            log "Config: $(basename "$CONFIG_PATH")"
            log "Exec:   $EXEC_NAME"
            log "Dataset: ${DATASET_NAME} | Repeat: ${i}"
            log "====================================="

            run_warmup
            clear_cache

            if [[ -n "$NUM_THREADS" ]]; then
                "$EXEC_PATH" "$CONFIG_PATH" "$DATASET_NAME" "$i" "$NUM_THREADS"
            else
                "$EXEC_PATH" "$CONFIG_PATH" "$DATASET_NAME" "$i"
            fi

            wait_for_exit "$EXEC_PATH" "$EXEC_NAME" 60

            sleep 3
        done
    done
done
