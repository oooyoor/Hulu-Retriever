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

# 运行选项（默认构建树结构, 非静默）
BUILD_TREE=1
QUIET=0
EXTRA_PREPARE_ARGS=()

usage(){
  cat <<EOF
用法: sudo ./new_setup_fs.sh [选项]
选项:
  --no-tree        生成扁平(单层)文件结构
  --tree           生成多层树结构(默认)
  --num-files N    设置生成文件总数(默认: $NUM_FILES)
  --file-size B    设置每个文件大小字节(默认: $FILE_SIZE)
  --threads N      设置并发线程数(默认: $PARALLEL_JOBS)
  --quiet|-q       降低脚本输出, prepare_files 输出重定向到日志
  --help|-h        显示本帮助
示例:
  sudo ./new_setup_fs.sh --no-tree --quiet --num-files 10000 --threads 8
EOF
}

for arg in "$@"; do
  case "$arg" in
    --no-tree)
      BUILD_TREE=0
      ;;
    --tree)
      BUILD_TREE=1
      ;;
    --num-files)
      shift || { echo "缺少 --num-files 参数值" >&2; exit 1; }
      NUM_FILES="$1"
      ;;
    --file-size)
      shift || { echo "缺少 --file-size 参数值" >&2; exit 1; }
      FILE_SIZE="$1"
      ;;
    --threads)
      shift || { echo "缺少 --threads 参数值" >&2; exit 1; }
      PARALLEL_JOBS="$1"
      ;;
    --quiet|-q)
      QUIET=1
      ;;
    --help|-h)
      usage; exit 0
      ;;
    *)
      echo "[WARN] 未知参数: $arg" >&2
      ;;
  esac
done

if [[ $BUILD_TREE -eq 0 ]]; then
  EXTRA_PREPARE_ARGS+=("--no-tree")
fi
# =================

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info(){  [[ $QUIET -eq 1 ]] && echo -e "${GREEN}[INFO]${NC} $*" || echo -e "${GREEN}[INFO]${NC} $*"; }
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
  
  # 如果已挂载，先卸载（确保可以重新格式化）
  if mountpoint -q "$MOUNT_POINT"; then
    log_info "挂载点已挂载，先卸载..."
    umount "$MOUNT_POINT" || true
  fi
  
  # 检查是否有其他挂载点
  if mount | grep -q "^$DISK "; then
    log_info "发现磁盘在其他位置挂载，卸载..."
    umount "$DISK" || true
  fi

  log_info "将对 $DISK 执行 mkfs.ext4 并挂载到 $MOUNT_POINT"
  mkdir -p "$MOUNT_POINT"
  # 强制格式化（-F 参数）, 静默时加 -q
  if [[ $QUIET -eq 1 ]]; then
    mkfs.ext4 -F -q "$DISK"
  else
    mkfs.ext4 -F "$DISK"
  fi
  mount "$DISK" "$MOUNT_POINT"
  log_info "格式化并挂载完成"
}

run_prepare() {
  if [[ ! -x "$PREPARE_BIN" ]]; then
    log_error "找不到可执行文件: $PREPARE_BIN (请先 g++ 编译)"
    exit 1
  fi

  log_info "开始运行多线程文件生成程序..."
  log_info "线程数: $PARALLEL_JOBS, 文件数: $NUM_FILES, 文件大小: $FILE_SIZE, 树结构: $([[ $BUILD_TREE -eq 1 ]] && echo yes || echo no)"

  if [[ $QUIET -eq 1 ]]; then
    LOG_DIR="./logs"
    mkdir -p "$LOG_DIR"
    LOG_FILE="$LOG_DIR/prepare_$(date +%Y%m%d_%H%M%S)_$([[ $BUILD_TREE -eq 1 ]] && echo tree || echo flat).log"
    log_info "静默模式: 输出重定向到 $LOG_FILE"
    set +e
    "$PREPARE_BIN" \
      --mount-point "$MOUNT_POINT" \
      --num-files "$NUM_FILES" \
      --file-size "$FILE_SIZE" \
      --threads "$PARALLEL_JOBS" \
      "${EXTRA_PREPARE_ARGS[@]}" \
      > "$LOG_FILE" 2>&1
    STATUS=$?
    set -e
    if [[ $STATUS -ne 0 ]]; then
      log_error "prepare_files 失败 (退出码: $STATUS), 查看日志: $LOG_FILE"
      exit $STATUS
    fi
  else
    "$PREPARE_BIN" \
      --mount-point "$MOUNT_POINT" \
      --num-files "$NUM_FILES" \
      --file-size "$FILE_SIZE" \
      --threads "$PARALLEL_JOBS" \
      "${EXTRA_PREPARE_ARGS[@]}"
  fi

  log_info "prepare_files 执行完成"
}

main() {
  check_root
  check_disk
  check_and_format
  run_prepare
}

main "$@"
