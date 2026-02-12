#!/bin/bash
# set -euo pipefail # Uncomment to enable strict mode, but be careful with error handling

#===============================================================================
#                           Configuration and Colors
#===============================================================================
# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

#===============================================================================
#                           Cluster Configuration
#===============================================================================
# The first IP is the head node (script assumes it is running on the head node)
# Use IP for the first node because Ray requires it at startup
# Nodes with same GPU type must contiguously follow each other in the list
# Heterogeneous cluster configuration
declare -a ip_list=("192.168.0.2" "192.168.0.5" "192.168.0.4")
# declare -a gpu_list=("gpu-type-V100:8" "gpu-type-A100:2" "gpu-type-A100:2")
declare -a gpu_list=("gpu-type-V100:8" "gpu-type-A101:2" "gpu-type-A100:2")

# Homogeneous cluster configuration for A100
# declare -a ip_list=("192.168.0.5" "192.168.0.4")
# declare -a gpu_list=("gpu-type-A100:2" "gpu-type-A100:2")

# Homogeneous cluster configuration for V100
# declare -a ip_list=("192.168.0.2")
# declare -a gpu_list=("gpu-type-V100:8")

# Heterogeneous Homo cluster configuration
# declare -a ip_list=("192.168.0.2" "192.168.0.5")
# declare -a gpu_list=("gpu-type-A100:2" "gpu-type-A100:2")






# SSH, Ray paths, ports, etc.
PORT=$HARP_SSH_PORT
SSH_KEY="/root/.ssh/id_rsa"
RAY_ROOT="/usr/local/bin/ray"

# Head node info
HEAD_IP="${ip_list[0]}"
HEAD_GPU="${gpu_list[0]}"
HEAD_ADDRESS="${HEAD_IP}:6379"

#===============================================================================
#                       Logging and Divider Functions
#===============================================================================
info() {
    echo -e "${CYAN}[INFO]${RESET} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${RESET} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${RESET} $1"
}

error() {
    echo -e "${RED}[ERROR]${RESET} $1"
}

divider() {
    # If parameter is passed, display it on the divider line
    echo -e "\n${BLUE}----------------------------------------${RESET}"
    if [ -n "$1" ]; then
        echo -e "${BLUE}-- $1 --${RESET}"
    fi
    echo -e "${BLUE}----------------------------------------${RESET}\n"
}

#===============================================================================
#                          Functional Definitions
#===============================================================================

#--------------------------
# 1. Check and Install rsync
#--------------------------
install_rsync() {
    divider "install_rsync: START"
    info "[Local] Checking/Installing rsync (if not installed)..."
    command -v rsync >/dev/null 2>&1 || (apt-get update && apt-get install rsync -y)

    # Remote check/install for Worker nodes
    for ((i = 1; i < ${#ip_list[@]}; i++)); do
        local ip="${ip_list[$i]}"
        info "[Worker: $ip] Checking/Installing rsync..."
        ssh -i "$SSH_KEY" root@"$ip" -p "$PORT" "command -v rsync >/dev/null 2>&1 || ( apt-get update && apt-get install rsync -y)"
    done

    success "rsync check/install completed."
}

#--------------------------
# 2. Start Ray Cluster
#--------------------------
start_cluster() {
    divider "start_cluster: START"

    # 2.1 Head node (Local): Start Head
    info "[Local] Starting Ray Head Node (IP = ${HEAD_IP}), Resources: ${HEAD_GPU}"
    
    local head_gpu_json
    head_gpu_json=$(echo "${HEAD_GPU}" | awk -F: '{print "{\""$1"\": "$2"}"}')
    
    local head_gpu_num
    head_gpu_num=$(echo "${HEAD_GPU}" | cut -d':' -f2)
    
    "$RAY_ROOT" start --head --resources="$head_gpu_json" --port=6379 --num-gpus "$head_gpu_num" --object-store-memory 12147483648

    # 2.2 Worker nodes: Start Worker
    for ((i = 1; i < ${#ip_list[@]}; i++)); do
        local ip="${ip_list[$i]}"
        local gpu="${gpu_list[$i]}"
        
        local gpu_json
        gpu_json=$(echo "${gpu}" | awk -F: '{print "{\""$1"\": "$2"}"}')
        
        local gpu_num
        gpu_num=$(echo "$gpu" | cut -d':' -f2)

        info "[Worker: $ip] Starting Ray Worker Node, Resources: ${gpu}"
        ssh -i "$SSH_KEY" root@"$ip" -p "$PORT" \
            "export PATH=$PATH; $RAY_ROOT start --address='${HEAD_ADDRESS}' --resources='${gpu_json}' --num-gpus ${gpu_num} --object-store-memory 12147483648"
    done

    success "Ray cluster started successfully!"
}

#--------------------------
# 3. Stop Ray Cluster
#--------------------------
stop_cluster() {
    divider "stop_cluster: START"

    # Stop Head Node (Local)
    info "[Local] Stopping Ray Head Node..."
    "$RAY_ROOT" stop

    # Stop Worker Nodes
    for ((i = 1; i < ${#ip_list[@]}; i++)); do
        local ip="${ip_list[$i]}"
        info "[Worker: $ip] Stopping Ray..."
        ssh -i "$SSH_KEY" root@"$ip" -p "$PORT" "$RAY_ROOT stop"
    done

    success "Ray cluster stopped successfully!"
}

#--------------------------
# 4. Kill all Python processes
#--------------------------
kill_python_processes() {
    divider "kill_python_processes: START"

    info "[Local] Killing all Python processes..."
    pkill -9 python || true # Don't fail if no python process

    for ((i = 1; i < ${#ip_list[@]}; i++)); do
        local ip="${ip_list[$i]}"
        info "[Worker: $ip] Killing all Python processes..."
        ssh -i "$SSH_KEY" root@"$ip" -p "$PORT" "pkill -9 python || true"
    done

    success "All Python processes killed."
}

#--------------------------
# 5. Sync code to all nodes
#--------------------------
sync_code() {
    divider "sync_code: START"

    local src_dir="/workspace/harp"
    # local src_dir="/workspace/build_jaxlib"
    local dst_dir="/workspace/"

    info "[Local] Ensuring local rsync is installed, starting sync to workers..."

    for ((i = 1; i < ${#ip_list[@]}; i++)); do
        local ip="${ip_list[$i]}"
        info "[Worker: $ip] Syncing directory: ${src_dir} ..."
        rsync -avz --exclude='*.pkl' -e "ssh -p $PORT -i $SSH_KEY" --times --perms "$src_dir" root@"$ip":"$dst_dir"
    done
    success "Code sync completed."
}

#--------------------------
# 6. Full Workflow: Kill -> Stop -> Sync -> Start
#--------------------------
full_restart() {
    divider "full_restart: START"
    install_rsync
    kill_python_processes
    stop_cluster
    sync_code
    start_cluster
    echo -e "${MAGENTA}[COMPLETE]${RESET} Full restart sequence finished!"
}

#--------------------------
# 6.5 Workflow: Stop -> Sync -> Start (No Kill)
#--------------------------
full_restart_nokillpython() {
    divider "full_restart_nokillpython: START"
    install_rsync
    stop_cluster
    sync_code
    start_cluster
    echo -e "${MAGENTA}[COMPLETE]${RESET} Restart sequence finished!"
}

#--------------------------
# 7. Install JAXLib from wheel
#--------------------------
install_jaxlib_from_wheel() {
    divider "install_jaxlib_from_wheel: START"
    info "Installing JAXLib on all nodes..."

    info "[Local] Compiling JAXLib..."
    cd /workspace/build_jaxlib/ || exit
    export SM="80"
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH
    python build/build.py \
        --python_bin_path=/usr/bin/python3.8 \
        --enable_cuda \
        --cuda_path=/usr/local/cuda-11.3 \
        --cudnn_path=/usr/lib/x86_64-linux-gnu \
        --cuda_version=11.3 \
        --cudnn_version=8 \
        --cuda_compute_capabilities="sm_${SM},compute_${SM}"\
        --dev_install --bazel_options=--config=linux \
        --bazel_options=--override_repository=org_tensorflow=/workspace/tensorflow-alpa \
         --bazel_options=--action_env=TF_CUDA_PATHS=/usr/local/cuda-11.3,/usr/lib/x86_64-linux-gnu,/usr
    info "Distributing jaxlib wheel..."
    sync_code

    # Head Node Operation
    info "[Local] Installing JAXLib..."
    cd /workspace/build_jaxlib/dist || exit
    /usr/bin/pip install -e . || error "Local JAXLib installation failed!"

    # Worker Node Operation
    for ((i = 1; i < ${#ip_list[@]}; i++)); do
        local ip="${ip_list[$i]}"
        info "[Worker: $ip] Installing JAXLib..."
        ssh -i "$SSH_KEY" root@"$ip" -p "$PORT" \
            "cd /workspace/build_jaxlib/dist && /usr/bin/pip install -e . || echo 'JAXLib installation failed!'"
    done

    success "JAXLib installation completed on all nodes!"
}

#===============================================================================
#                    Print Cluster Information (IP & GPU)
#===============================================================================
print_cluster_info() {
    divider "Cluster Information"
    echo -e "${BOLD}${GREEN}Cluster Node Information:${RESET}"

    # Table Header
    echo -e "${BLUE}${BOLD}-------------------------------------------${RESET}"
    printf "${BLUE}${BOLD}| %-15s | %-15s |\n${RESET}" "Host IP" "GPU Info"
    echo -e "${BLUE}${BOLD}|-----------------|-----------------|${RESET}"

    # Table Content
    for ((i = 0; i < ${#ip_list[@]}; i++)); do
        printf "${CYAN}${BOLD}| %-15s | %-15s |\n${RESET}" "${ip_list[$i]}" "${gpu_list[$i]}"
    done
    echo -e "${BLUE}${BOLD}-------------------------------------------${RESET}\n"
}

#===============================================================================
#                                 Main Menu
#===============================================================================
print_cluster_info

echo -e "${CYAN}Please select an operation:${RESET}"
echo -e "${YELLOW}1.${RESET} Start Ray Cluster"
echo -e "${YELLOW}2.${RESET} Stop Ray Cluster"
echo -e "${YELLOW}3.${RESET} Kill Python Processes"
echo -e "${YELLOW}4.${RESET} Sync Code"
echo -e "${YELLOW}5.${RESET} Full Restart (Kill -> Stop -> Sync -> Start)"
echo -e "${YELLOW}6.${RESET} Restart (Stop -> Sync -> Start)"
echo -e "${YELLOW}7.${RESET} Install JAXLib from wheel"
echo -e "${YELLOW}8.${RESET} Exit Script"
read -p "Enter your choice [1-8]: " choice

case "$choice" in
1)
    start_cluster
    ;;
2)
    stop_cluster
    ;;
3)
    kill_python_processes
    ;;
4)
    sync_code
    ;;
5)
    full_restart
    ;;
6)
    full_restart_nokillpython
    ;;
7)
    install_jaxlib_from_wheel
    ;;
8)
    echo -e "${GREEN}Exiting script...${RESET}"
    exit 0
    ;;
*)
    error "Invalid input!"
    ;;
esac
