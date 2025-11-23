#!/bin/bash
set -e

# === Path Setting ===
SUBMODULE_PATH="external/hnswlib"
MODULE_NAME=$1
if [ -z "$MODULE_NAME" ]; then
    echo "Usage: $0 <module_name>"
    exit 1
fi
PATCH_DIR="${MODULE_NAME}/patches"
PATCH_NAME="${MODULE_NAME}.patch"   

echo "[SAVE PATCH] Exporting ${MODULE_NAME} changes into patch..."

# 确保补丁目录存在
mkdir -p "$PATCH_DIR"

# 导出补丁（只导出 submodule 内的 diff）
git -C "$SUBMODULE_PATH" diff > "$PATCH_DIR/$PATCH_NAME"

echo "[SAVE PATCH] Saved to: $PATCH_DIR/$PATCH_NAME"

# === restore submodule to a clean state ===
echo "[SAVE PATCH] Resetting submodule to clean state..."
git submodule update --init --recursive --force

echo "[SAVE PATCH] Done."
