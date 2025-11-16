#!/usr/bin/env bash
set -euo pipefail

# ===== 参数 =====
DISK="/dev/nvme0n1"
MOUNT_POINT="/mnt/nvme0n1"

NUM_FILES=1000001
FILE_SIZE=4096
PARALLEL_JOBS=64

# C++ 可执行文件路径
PREPARE_BIN="./prepare_files"
# 如果不想建 tree，可以加上 --no-tree
PREPARE_EXTRA_ARGS=""   # 比如 "--no-tree"
# =================

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info(){  echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn(){  echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error(){ echo -e "${RED}[ERROR]${NC} $*"; }

check_root() {
  if [[ $EUID -ne 0 ]]; then
    log_error "请用 root 运行本脚本（涉及 mkfs/mount）"
    exit 1
  fi
}

check_disk() {
  if [[ ! -b "$DISK" ]]; then
    log_error "块设备 $DISK 不存在"
    exit 1
  fi
  log_info "找到磁盘: $DISK"
}

check_and_format() {
  log_info "检查挂载点: $MOUNT_POINT"
  if mountpoint -q "$MOUNT_POINT"; then
    log_info "挂载点已挂载，跳过 mkfs+mount"
    return
  fi

  log_warn "挂载点未挂载，将对 $DISK 执行 mkfs.ext4 并挂载到 $MOUNT_POINT"
  mkdir -p "$MOUNT_POINT"
  # 如果你只想第一次 mkfs，后面不重新格式化，可以加一点额外判断 fs type。
  mkfs.ext4 -F "$DISK"
  mount "$DISK" "$MOUNT_POINT"
  log_info "格式化并挂载完成"
}

run_prepare() {
  if [[ ! -x "$PREPARE_BIN" ]]; then
    log_error "找不到可执行文件: $PREPARE_BIN (请先 g++ 编译)"
    exit 1
  fi

  log_info "开始运行多线程文件生成程序..."
  log_info "线程数: $PARALLEL_JOBS, 文件数: $NUM_FILES, 文件大小: $FILE_SIZE"

  "$PREPARE_BIN" \
    --mount-point "$MOUNT_POINT" \
    --num-files "$NUM_FILES" \
    --file-size "$FILE_SIZE" \
    --threads "$PARALLEL_JOBS" \
    $PREPARE_EXTRA_ARGS

  log_info "prepare_files 执行完成"
}

main() {
  check_root
  check_disk
  check_and_format
  run_prepare
}

main "$@"
