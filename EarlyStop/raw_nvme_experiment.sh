#!/bin/bash
set -e

DISK="/dev/nvme0n1"
DISK_SYS="/sys/block/nvme0n1"
COOL_TIME=30
EXEC_NAME=$1
CONFIG_PATH="/home/zqf/Hulu-Retriever/configs/Config.json"
NUM_THREADS="$2"  # 可选的线程数参数

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

log() { echo -e "${GREEN}[INFO]${NC} $1"; }
err() { echo -e "${RED}[ERROR]${NC} $1" >&2; }

# ================================
# 识别磁盘状态
# ================================
detect_state() {
    log "检测磁盘当前状态..."

    # 检查整盘是否挂载
    WHOLE_MOUNT=$(mount | grep "^$DISK " || true)
    if [ -n "$WHOLE_MOUNT" ]; then
        log "发现整盘被挂载到: $(echo $WHOLE_MOUNT | awk '{print $3}')"
        WHOLE_MOUNT_POINT=$(echo $WHOLE_MOUNT | awk '{print $3}')
    else
        WHOLE_MOUNT_POINT=""
    fi

    # 检查磁盘是否被格式化为文件系统（ext4/xfs）
    FS_TYPE=$(lsblk -no FSTYPE $DISK)
    if [ -n "$FS_TYPE" ]; then
        log "整盘包含文件系统: $FS_TYPE"
    fi

    # 检查是否存在分区
    PARTS=$(lsblk -ln -o NAME $DISK | tail -n +2)
    if [ -n "$PARTS" ]; then
        log "发现分区：$(echo $PARTS)"
    fi
}

# ================================
# 卸载所有挂载点（整盘 + 分区）
# ================================
unmount_all() {
    log "卸载磁盘的全部挂载点..."

    # 卸载整盘挂载
    if [ -n "$WHOLE_MOUNT_POINT" ]; then
        log "卸载整盘挂载: $WHOLE_MOUNT_POINT"
        sudo umount -l "$WHOLE_MOUNT_POINT" || true
    fi

    # 卸载分区挂载
    for p in $(lsblk -ln -o NAME $DISK | tail -n +2); do
        if mount | grep -q "/dev/$p"; then
            log "卸载 /dev/$p ..."
            sudo umount -l "/dev/$p" || true
        fi
    done
}

# ================================
# 清理分区表 + 刷新内核
# ================================
cleanup_partition_table() {
    log "清空磁盘 GPT/MBR 分区表..."

    sudo wipefs -a $DISK || true
    sudo sgdisk --zap-all $DISK || true

    log "尝试让内核刷新分区表..."
    sudo partprobe $DISK || true

    # 内核不刷新 → 使用 hdparm 强制刷新
    log "使用 hdparm -z 强制刷新内核分区表缓存..."
    sudo hdparm -z $DISK || true
    
    # 再次通知内核（如果文件存在）
    if [ -w "$DISK_SYS/device/rescan" ]; then
        echo 1 | sudo tee "$DISK_SYS/device/rescan" >/dev/null 2>&1 || true
    fi
}


# ================================
# 强制 detach + PCI rescan
# ================================
force_detach_rescan() {
    log "执行 NVMe detach + PCI rescan..."

    if [ -e "$DISK_SYS/device/delete" ]; then
        echo 1 | sudo tee "$DISK_SYS/device/delete" >/dev/null || true
        sleep 1
    fi

    echo 1 | sudo tee /sys/bus/pci/rescan >/dev/null
    sleep 1
}

# ================================
# 清空 FTL
# ================================
reset_ftl() {
    log "尝试 blkdiscard 清空 FTL..."

    if sudo blkdiscard $DISK 2>/dev/null; then
        log "blkdiscard 成功"
        return
    fi

    err "blkdiscard 失败，尝试 nvme format..."
    if sudo nvme format $DISK -s 1; then
        log "nvme format 成功"
        return
    fi

    err "format failed，执行 detach + retry"
    force_detach_rescan

    sudo blkdiscard $DISK || sudo nvme format $DISK -s 1
}

# ================================
# 磁盘识别检查
# ================================
check_ssd() {
    log "检查 NVMe 是否被识别为 SSD..."

    if [ "$(cat /sys/block/nvme0n1/queue/rotational)" = "0" ]; then
        log "OK: rotational = 0，识别为 SSD"
    else
        err "错误：rotational=1，系统误识别为 HDD！"
        exit 1
    fi

    SCHED=$(cat /sys/block/nvme0n1/queue/scheduler)
    log "I/O scheduler: $SCHED"
}

# ================================
# FTL warmup
# ================================
ftl_warmup() {
    log "FTL 预热（顺序写入 1GB）..."
    sudo dd if=/dev/zero of=$DISK bs=4M count=256 oflag=direct status=none

    log "等待 SSD 控制器冷却 ${COOL_TIME}s..."
    sleep $COOL_TIME
}

# ================================
# offset 测试
# ================================
run_tests() {
    log "运行 offset 读取测试..."

    if [ ! -f "./run_test.sh" ]; then
        err "run_test.sh 不存在！"
        exit 1
    fi

    chmod +x ./run_test.sh 2>/dev/null || true

    # 支持可选的线程数参数
    if [ -n "$NUM_THREADS" ]; then
        ./run_test.sh $EXEC_NAME $NUM_THREADS
    else
        ./run_test.sh $EXEC_NAME
    fi
}

# ================================
# 主流程
# ================================
main() {
    log "===== NVMe Raw 实验开始 ====="

    detect_state             # ★ 自动识别当前状态
    unmount_all              # ★ 自动卸载
    cleanup_partition_table  # ★ 删除分区表
    reset_ftl                # 清空 FTL
    check_ssd                # 确认识别为 SSD
    ftl_warmup               # 预热
    run_tests                # offset 测试

    log "===== NVMe Raw 实验完成 ====="
}

main
