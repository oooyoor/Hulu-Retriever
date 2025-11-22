#!/bin/bash
set -e

RAW_SCRIPT="./raw_nvme_experiment.sh"
OFFSET_BIN="./execs/offset"
FS_SINGLE_SCRIPT="./ext4_single_dir_experiment.sh"
FS_TREE_SCRIPT="./ext4_uniform_tree_experiment.sh"


CONFIG_PATH="/home/zqf/Hulu-Retriever/configs/Config.json"

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

log() { echo -e "${GREEN}[STEP]${NC} $1"; }
err() { echo -e "${RED}[ERROR]${NC} $1"; }

# =======================
# 检查必要文件是否存在
# =======================
check_files() {
    log "检查必要文件..."
    [ ! -f "$RAW_SCRIPT" ] && err "$RAW_SCRIPT 不存在" && exit 1
    [ ! -f "$FS_SINGLE_SCRIPT" ] && err "$FS_SINGLE_SCRIPT 不存在" && exit 1
    [ ! -f "$FS_TREE_SCRIPT" ] && err "$FS_TREE_SCRIPT 不存在" && exit 1
    [ ! -f "$OFFSET_BIN" ] && err "offset 可执行文件找不到" && exit 1
    [ ! -f "$RUN_TEST" ] && err "run_test.sh 不存在" && exit 1

    chmod +x $RAW_SCRIPT
    chmod +x $FS_SINGLE_SCRIPT
    chmod +x $FS_TREE_SCRIPT
    chmod +x $RUN_TEST
}

# =======================
# 0. 创建整体结果目录
# =======================
setup_result_dirs() {
    mkdir -p results/raw
    mkdir -p results/fs_single
    mkdir -p results/fs_tree
    mkdir -p results/offset
}

# =======================
# 1. RAW区块实验
# =======================
run_raw_test() {
    log "运行 RAW NVMe 实验 ..."
    sudo bash $RAW_SCRIPT
    cp -r ./results/offset_results/* ./results/raw/ 2>/dev/null || true
}

# =======================
# 2. ext4 单层200万文件实验
# =======================
run_fs_single() {
    log "运行 ext4 单层目录 200万文件 实验 ..."
    sudo bash $FS_SINGLE_SCRIPT
    cp -r ./results/fs_single_layer/* ./results/fs_single/ 2>/dev/null || true
}

# =======================
# 3. ext4 100×100×100 树状结构实验
# =======================
run_fs_tree() {
    log "运行 ext4 三层 100×100×100 树实验 ..."
    sudo bash $FS_TREE_SCRIPT
    cp -r ./results/fs_tree_layer/* ./results/fs_tree/ 2>/dev/null || true
}

# =======================
# 4. Offset 全局读取实验
# =======================
run_offset_read() {
    log "运行 offset 读取实验 (run_test.sh) ..."
    ./run_test.sh offset
    cp -r ./results/offset_results/* ./results/offset/
}

# =======================
# 主流程
# =======================
main() {
    log "===== NVMe 全流程性能测试 PIPELINE 启动 ====="
    check_files
    setup_result_dirs

    run_raw_test
    run_fs_single
    run_fs_tree
    run_offset_read

    log "===== PIPELINE 全部完成，可以开始绘图 ====="
    log "结果已全部保存在 ./results 目录中。"
}

main
