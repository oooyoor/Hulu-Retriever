#!/bin/bash
set -e

BASE_MODULE_NAME="BaseCppFiles"
MODULE_NAME=$1

if [ -z "$MODULE_NAME" ]; then
    echo "Usage: ./init_module.sh <module_name>"
    exit 1
fi

echo "=== Initializing new module: $MODULE_NAME ==="

# 创建模块目录
if [ -d "$MODULE_NAME" ]; then
    echo "Module '$MODULE_NAME' already exists."
    exit 1
fi

mkdir -p "$MODULE_NAME"
cp -r "$BASE_MODULE_NAME/"* "$MODULE_NAME/"

# 创建构建和输出目录
mkdir -p "$MODULE_NAME/build"
mkdir -p "$MODULE_NAME/execs"

echo "Module '$MODULE_NAME' initialized successfully."
echo ""
echo "Next steps:"
echo "  1) Edit $MODULE_NAME/CMakeLists.txt according to your module logic"
echo "  2) Run: cd $MODULE_NAME && ./build_clean.sh"
echo ""
