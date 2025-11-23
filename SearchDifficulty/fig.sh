#!/bin/bash
PYTHON_PATH="/home/zqf/miniconda3/bin/python"
$PYTHON_PATH analyze_search_difficulty.py /home/zqf/Hulu-Retriever/SearchDifficulty/search_difficulty_results/fs_results/Offset_results  --num-bins 5     --plot fs_difficulty.png     --txt fs_difficulty.txt     --md fs_difficulty.md