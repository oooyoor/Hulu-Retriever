#!/bin/bash
set -e

# 主测试脚本：按顺序执行所有测试，支持多线程数对比
# 用法: sudo ./run_all_tests.sh [线程数列表，用空格分隔，默认: 2 4 8 16]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

log() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
info() { echo -e "${BLUE}[INFO]${NC} $1"; }
err() { echo -e "${RED}[ERROR]${NC} $1" >&2; }

# 参数解析：区分线程数与标志(--quiet/-q)
THREAD_COUNTS_DEFAULT=(2 4 8 16)
THREAD_COUNTS=()
QUIET=0

for arg in "$@"; do
    case "$arg" in
        --quiet|-q)
            QUIET=1
            ;;
        '' ) ;; # skip empty
        *)
            if [[ "$arg" =~ ^[0-9]+$ ]]; then
                THREAD_COUNTS+=("$arg")
            else
                warn "忽略未知参数: $arg"
            fi
            ;;
    esac
done

if [ ${#THREAD_COUNTS[@]} -eq 0 ]; then
    THREAD_COUNTS=("${THREAD_COUNTS_DEFAULT[@]}")
fi

QUIET_ARGS=()
if [[ $QUIET -eq 1 ]]; then
    QUIET_ARGS+=("--quiet")
fi

log "===== 开始完整测试流程 ====="
log "测试线程数: ${THREAD_COUNTS[*]}"
log "当前目录: $SCRIPT_DIR"

# 检查必要文件是否存在
if [ ! -f "./raw_nvme_experiment.sh" ]; then
    err "raw_nvme_experiment.sh 不存在！"
    exit 1
fi

if [ ! -f "./run_test.sh" ]; then
    err "run_test.sh 不存在！"
    exit 1
fi

chmod +x ./raw_nvme_experiment.sh ./run_test.sh 2>/dev/null || true

# ================================
# 1. Raw NVMe 测试（对所有线程数）
# ================================
# log "===== 步骤 1: Raw NVMe 测试 ====="

CONFIG_PATH="/home/zqf/Hulu-Retriever/configs/Config.json"
BACKUP_CONFIG="${CONFIG_PATH}.backup"

# 备份原配置
if [ ! -f "$BACKUP_CONFIG" ]; then
    cp "$CONFIG_PATH" "$BACKUP_CONFIG"
    log "已备份配置文件到: $BACKUP_CONFIG"
fi

# for NUM_THREADS in "${THREAD_COUNTS[@]}"; do
#     log "--- Raw 测试，线程数: $NUM_THREADS ---"
    
#     # 临时修改线程数（使用 jq）
#     jq ".num_threads = $NUM_THREADS" "$BACKUP_CONFIG" > "${CONFIG_PATH}.tmp" && mv "${CONFIG_PATH}.tmp" "$CONFIG_PATH"
    
#     # 每次都运行完整的 raw_nvme_experiment.sh（包括磁盘准备）
#     # 这样可以确保每次测试都有相同的初始条件
#     log "执行完整的 raw_nvme_experiment.sh（包括磁盘准备）..."
#     sudo ./raw_nvme_experiment.sh "$NUM_THREADS"
    
#     sleep 10  # 测试间隔
# done

# ================================
# 2. Single Layer FS 测试
# ================================
# if [ ! -f "./new_setup_fs.sh" ]; then
#     err "new_setup_fs.sh 不存在！"
#     exit 1
# fi
# chmod +x ./new_setup_fs.sh 2>/dev/null || true
# sudo ./new_setup_fs.sh "${QUIET_ARGS[@]}"

SINGLE_FS_PATH="/mnt/nvme0n1/lab1"
MULTI_FS_PATH="/mnt/nvme0n1/tree"

log "===== 步骤 1: Single Layer FS 测试 (全部线程) ====="

log "准备单层文件系统结构"

for NUM_THREADS in "${THREAD_COUNTS[@]}"; do
    log "--- Single 测试: 线程数=$NUM_THREADS ---"
    jq ".num_threads = $NUM_THREADS | .fs_data_dir_path = \"$SINGLE_FS_PATH\"" "$BACKUP_CONFIG" > "${CONFIG_PATH}.tmp" && mv "${CONFIG_PATH}.tmp" "$CONFIG_PATH"
    if [[ $QUIET -eq 1 ]]; then
        mkdir -p logs
        sudo ./run_test.sh single "$NUM_THREADS" > "logs/single_${NUM_THREADS}.log" 2>&1 || err "single 测试失败: $NUM_THREADS"
    else
        sudo ./run_test.sh single "$NUM_THREADS"
    fi
    sleep 5
done

# ================================
# 3. Multi Layer FS 测试
# ================================
log "===== 步骤 2: Multi Layer FS 测试 (全部线程) ====="

log "准备多层文件系统结构 (--tree) ..."

for NUM_THREADS in "${THREAD_COUNTS[@]}"; do
    log "--- Multi 测试: 线程数=$NUM_THREADS ---"
    jq ".num_threads = $NUM_THREADS | .fs_data_dir_path = \"$MULTI_FS_PATH\"" "$BACKUP_CONFIG" > "${CONFIG_PATH}.tmp" && mv "${CONFIG_PATH}.tmp" "$CONFIG_PATH"
    if [[ $QUIET -eq 1 ]]; then
        mkdir -p logs
        sudo ./run_test.sh multi "$NUM_THREADS" > "logs/multi_${NUM_THREADS}.log" 2>&1 || err "multi 测试失败: $NUM_THREADS"
    else
        sudo ./run_test.sh multi "$NUM_THREADS"
    fi
    sleep 5
done

# ================================
# 恢复配置文件
# ================================
if [ -f "$BACKUP_CONFIG" ]; then
    log "恢复原始配置文件..."
    mv "$BACKUP_CONFIG" "$CONFIG_PATH"
fi

log "===== 所有测试完成 ====="
log "结果位置："
log "  - Single: ./results/Fs_results/"
log "  - Multi: ./results/MultiLayer_Fs_results/"
if [[ $QUIET -eq 1 ]]; then
    log "日志: ./MakeFsWorkSpace/logs/*.log"
fi

# 生成测试摘要
log ""
log "===== 测试摘要 ====="
log "测试线程数: ${THREAD_COUNTS[*]}"
log "测试完成时间: $(date)"

