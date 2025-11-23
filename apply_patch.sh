#!/bin/bash
set -e

# === Path Setting ===
MODULE_NAME=$1
if [ -z "$MODULE_NAME" ]; then
    echo "Usage: $0 <module_name>"
    exit 1
fi
SUBMODULE_PATH="external/hnswlib"
PATCH_DIR="${MODULE_NAME}/patches"
PATCH_NAME="${MODULE_NAME}.patch"
PATCH_PATH="$PATCH_DIR/$PATCH_NAME"

echo "[APPLY PATCH] Resetting submodule to clean version..."
git submodule update --init --recursive --force

if [ ! -f "$PATCH_PATH" ]; then
    echo "[APPLY PATCH] No patch found: $PATCH_PATH"
    exit 0
fi

echo "[APPLY PATCH] Applying patch: $PATCH_PATH"

# Convert patch path to absolute path since git -C changes directory
PATCH_ABS_PATH="$(realpath "$PATCH_PATH")"
git -C "$SUBMODULE_PATH" apply "$PATCH_ABS_PATH"

echo "[APPLY PATCH] Patch applied successfully."