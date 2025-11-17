#!/bin/bash

# Usage:
# ./run_warmup.sh /path/to/config.json

CONFIG_PATH="/home/zqf/Hulu-Retriever/configs/Config.json"
if [ -z "$CONFIG_PATH" ]; then
    echo "Usage: $0 <config_json_path>"
    exit 1
fi

# 运行 warmup
/home/zqf/Hulu-Retriever/WarmUp/execs/warmup --config "$CONFIG_PATH"
