#!/bin/bash
# 恢复 hnswlib 子模块到原始状态的脚本
# 使用方法: ./EarlyStop/restore_hnswlib.sh

set -e

echo "正在恢复 hnswlib 子模块到原始状态..."

# 进入子模块目录
cd "$(dirname "$0")/../external/hnswlib"

# 检查是否有未提交的更改
if [ -n "$(git status --porcelain)" ]; then
    echo "检测到未提交的更改，正在恢复..."
    
    # 方法1: 恢复所有修改的文件到 HEAD 状态
    git restore .
    
    # 或者方法2: 硬重置到当前提交（更彻底，会丢弃所有本地修改）
    # git reset --hard HEAD
    
    echo "✓ hnswlib 子模块已恢复到原始状态"
else
    echo "✓ hnswlib 子模块已经是干净状态"
fi

# 返回主仓库目录
cd "$(dirname "$0")/.."

# 检查主仓库中的子模块状态
echo ""
echo "检查主仓库中的子模块状态:"
git status external/hnswlib

echo ""
echo "完成！现在可以开始新的实验修改了。"

