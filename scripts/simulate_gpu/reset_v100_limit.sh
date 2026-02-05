#!/bin/bash

PID_FILE="./tmp/v100_limit.pid"

echo "=== Resetting GPU to Default A100 State ==="

# 1. Reset Clocks
nvidia-smi -rgc
echo "Graphics clocks reset to default."

# 2. Kill Occupier
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    # 杀掉 Python 进程及其所有并行的子进程
    pkill -P $PID 
    kill $PID
    rm -f "$PID_FILE"
    echo "Memory occupier stopped."
else
    # 如果文件找不到了，尝试直接按名字杀掉
    pkill -f mem_occupier_v100.py
    echo "Forced pkill for mem_occupier_v100.py"
fi

echo "=== Reset Complete ==="
sleep 3
nvidia-smi --query-gpu=index,gpu_name,memory.total,memory.used,clocks.gr,clocks.sm --format=csv