#!/bin/bash
EXEC_NAME=$1
CONFIG_PATH="/home/zqf/Hulu-Retriever/configs/Config.json"
DATASETS=($(jq -r '.dataset_list[].dataset_name' $CONFIG_PATH))
REPEATS=3
clear_cache() {
    echo "Clearing memory cache..."
    sudo sync
    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
}

for DATASET_NAME in "${DATASETS[@]}"; do
    for ((i = 1; i <= REPEATS; i++)); do
        echo "Running test $i for dataset: $DATASET_NAME"
        
        # 清除内存缓存
        clear_cache
        
        # 执行测试命令
        ./execs/$EXEC_NAME $CONFIG_PATH $DATASET_NAME $i
        
        # 在每次测试后可加入延迟，避免过快的执行
        sleep 5
        
    done
done