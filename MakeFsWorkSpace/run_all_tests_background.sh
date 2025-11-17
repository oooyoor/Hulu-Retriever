#!/bin/bash
# 后台运行所有测试的包装脚本
# 用法: sudo ./run_all_tests_background.sh [线程数列表]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_FILE="${SCRIPT_DIR}/test_run_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${SCRIPT_DIR}/test_run.pid"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# ========================
# Guard：防止重复启动
# ========================
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE" 2>/dev/null || echo "")
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo -e "${YELLOW}已有测试在运行，PID: $OLD_PID${NC}"
        echo "如需重启，请先执行:"
        echo "  sudo kill $OLD_PID"
        echo "  rm -f $PID_FILE"
        exit 1
    fi
fi

echo -e "${GREEN}启动后台测试...${NC}"
echo "日志文件: $LOG_FILE"
echo "PID 文件: $PID_FILE"

# 在后台运行主测试脚本
nohup sudo "$SCRIPT_DIR/run_all_tests.sh" "$@" > "$LOG_FILE" 2>&1 &
PID=$!

# 保存 PID
echo $PID > "$PID_FILE"

echo -e "${GREEN}测试已在后台启动，PID: $PID${NC}"
echo "查看日志: tail -f $LOG_FILE"
echo "停止测试: sudo kill $PID && rm -f $PID_FILE"
echo ""
echo "使用以下命令监控进度:"
echo "  tail -f $LOG_FILE"
echo "  ps aux | grep run_all_tests"
