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
cp compile_commands.json ../compile_commands.json
# 修复 compile_commands.json 中的 directory 路径，使其指向当前目录而不是 build
sed -i "s|\"directory\": \"$(pwd)\"|\"directory\": \"$(cd .. && pwd)\"|g" ../compile_commands.json
# 修复 include 路径，将相对路径改为绝对路径
sed -i "s|BaseCppFiles/\.\./external|/home/zqf/Hulu-Retriever/external|g" ../compile_commands.json
sed -i "s|BaseCppFiles/\.\./include|/home/zqf/Hulu-Retriever/include|g" ../compile_commands.json
cd ..

# 确保 execs 文件夹存在
mkdir -p execs

# 如果 execs/lab 不存在，提示错误
if [ ! -f "execs/lab" ]; then
    echo "Error: execs/lab does not exist. Build output missing?"
    exit 1
fi

# 判断是否提供了 $1
if [ -z "$1" ]; then
    echo "No output name provided. 'execs/lab' will be kept."
else
    echo "Renaming execs/lab to execs/$1 ..."
    mv execs/lab "execs/$1"
fi

# 清理构建文件
rm -rf build

echo "=== Build Done ==="
