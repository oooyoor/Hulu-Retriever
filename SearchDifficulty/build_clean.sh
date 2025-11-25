#!/bin/bash
set -e

echo "=== Building Clean HNSW Benchmark ==="

# 清理并创建构建目录
if [ -d "build" ]; then
    rm -rf build
fi
mkdir -p build
cd build

# 编译配置
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-Ofast -march=native" ..
make -j$(nproc)
cd ..

# 确保 execs 存在
mkdir -p execs

# 检查可执行文件是否生成
if [ ! -f "execs/lab" ]; then
    echo "Error: execs/lab does not exist. Build output missing?"
    exit 1
fi

# 重命名输出
if [ -z "$1" ]; then
    echo "No output name provided. Keeping 'execs/lab'."
else
    echo "Renaming execs/lab -> execs/$1"
    mv execs/lab "execs/$1"
fi

# ===========================================================
# 关键：保留 compile_commands.json 并自动链接到项目根目录
# ===========================================================

ROOT_DIR=$(git rev-parse --show-toplevel)
SRC_COMPILE="$MODULE_DIR/build/compile_commands.json"
DST_COMPILE="$ROOT_DIR/compile_commands.json"

if [ -f "$SRC_COMPILE" ]; then
    ln -sf "$SRC_COMPILE" "$DST_COMPILE"
    echo "Linked compile_commands.json → $DST_COMPILE"
else
    echo "Warning: compile_commands.json not found in $MODULE_NAME/build"
fi

echo "=== Build Done ==="
echo "clangd updated to module: $MODULE_NAME"
