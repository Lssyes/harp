#!/bin/bash

# Configuration
TARGET_FREQ=585  # Target frequency in MHz
mkdir -p ./tmp
PID_FILE="./tmp/v100_limit.pid"
LOG_FILE="./tmp/v100_limit.log"

echo "=== Starting V100-32GB Simulation Mode ==="

# 1. Enable Persistence Mode
# Note: Requires sudo/root
nvidia-smi -pm 1

# 2. Lock Clocks (Min, Max)
echo "Locking graphics clock to ${TARGET_FREQ} MHz..."
nvidia-smi -lgc ${TARGET_FREQ},${TARGET_FREQ}

# 3. Start Memory Occupier in Background
# Added: Improved check to see if the process is TRULY running
if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
    echo "Warning: A simulation process is already running (PID: $(cat $PID_FILE))."
else
    # Remove stale PID file if the process crashed previously
    rm -f "$PID_FILE"
    
    # Ensure the python script exists before running
    if [ ! -f "mem_occupier_v100.py" ]; then
        echo "Error: mem_occupier_v100.py not found in current directory!"
        exit 1
    fi

    nohup python3 mem_occupier_v100.py > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    echo "Memory occupier started (PID: $!). Check $LOG_FILE for details."
    
    # Brief sleep to allow Python to allocate memory before the final status check
    sleep 10 
fi

echo "=== Setup Complete! Current Status: ==="
nvidia-smi --query-gpu=index,gpu_name,memory.total,memory.used,clocks.gr,clocks.sm --format=csv