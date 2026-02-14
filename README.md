# HARP: Orchestrating Automated Parallel Training on Heterogeneous GPU Clusters
HARP is a system for automated parallel training on heterogeneous GPU clusters. Built on Alpa, JAX, and Ray, HARP extends Alpa’s automatic parallelization framework to handle clusters composed of mixed GPU types connected by asymmetric or low-bandwidth interconnects.

It introduces a heterogeneity-aware planner and adaptive runtime scheduler that jointly optimize computation placement and communication overlap, enabling efficient, large-scale distributed training on heterogeneous GPU clusters with only a few lines of code.

## Key Features
+ 💻 Heterogeneity-Aware Automatic Parallelization: Extends Alpa to intelligently group devices with similar performance. It plans intra-operator parallelism within homogeneous submeshes and coordinates inter-operator (pipeline) parallelism across asymmetric links.

+ 🚀 Adaptive Runtime Scheduling: A new H-1F1B scheduler adaptively adjusts microbatch launches and overlaps computation with communication based on observed device and network speeds.

+ ✨ Seamless Integration: Built on top of Alpa, JAX, XLA, and Ray. Preserves developer-friendly APIs while enabling large-model training on mixed GPU environments.


## 1. Prerequisites
Before running HARP, ensure your cluster meets the following requirements:

* **Hardware:** A minimum of **2 nodes** is required, with **at least 1 NVIDIA GPU** per node. HARP is specifically designed for mixed-hardware environments and has been validated on:
    * **Enterprise setups:** e.g., A100 and V100 nodes.
    * **Consumer-grade setups:** e.g., RTX 4080 and RTX 4090 nodes.


* **Network:** Nodes must be able to reach each other via TCP.


* **(Recommended)** A shared NFS directory mounted on all nodes is recommended for sharing profiling databases


## 2. Installation & Build
### 2.1 Clone the Repository
Clone the repository recursively to include all submodules (Alpa, Ray extensions, etc.).

```shell
Bash
git clone --recursive https://github.com/Lssyes/harp.git
cd harp
```

### 2.2 Setup SSH Keys for Cluster Automation
HARP uses SSH to coordinate between heterogeneous nodes. Before builing docker image, you need to generate a key pair that will be shared inside the Docker containers.

```Bash
mkdir -p docker/ssh_key
ssh-keygen -t rsa -b 4096 -f ./docker/ssh_key/id_rsa -N ""
cat ./docker/ssh_key/id_rsa.pub >> ./docker/ssh_key/id_rsa/authorized_keys
```

### 2.3 Build the Docker Image
Select the configuration matching your GPU architecture. Run this build command on every node type you plan to use.

| GPU | Type | Architecture | SM (Compute Capability) | Base Image |
| :--- | :--- | :--- | :--- | :--- |
| V100 | Volta | 70 | 7.0 | nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04 |
| A100 | Ampere | 80 | 8.0 | nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04 |
| RTX40 | Ada Lovelace | 89 | 8.9 | nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04 |
| H100 | Hopper | 90 | 9.0 | nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04 |

**Example Build Command:** Replace `SM`, `CUDNN_VERSION` and `CUDA_VERSION` with the values refer to https://en.wikipedia.org/wiki/CUDA to select the correct architecture and base image.

```Bash
# # V100(Volta)
SM=70
CUDA_VERSION=11.3.1
CUDNN_VERSION=8
UBUNTU_VERSION=20.04

# # A100(Ampere)
SM=80
CUDA_VERSION=11.3.1
CUDNN_VERSION=8
UBUNTU_VERSION=20.04

# # RTX40(Ada Lovelace)
SM=89
CUDA_VERSION=11.8.0
CUDNN_VERSION=8
UBUNTU_VERSION=20.04

# # H100(Hopper)
SM=90
CUDA_VERSION=11.8.0
CUDNN_VERSION=8
UBUNTU_VERSION=20.04


docker build \
  --build-arg SM="${SM}" \
  --build-arg CUDA_VERSION="${CUDA_VERSION}" \
  --build-arg CUDNN_VERSION="${CUDNN_VERSION}" \
  --build-arg UBUNTU_VERSION="${UBUNTU_VERSION}" \
  -f docker/build_harp.Dockerfile \
  -t "harp:sm$GPU_TYPE" .

```



## 3. Running HARP
### 3.1 Start the Docker Container
Run the following command on each node in your cluster. Note: We recommend mounting a shared NFS directory to simplify file exchange.

```Bash
# Set your GPU SM version (e.g., 80 for A100, 70 for V100)
export GPU_TYPE=80 

docker run -it \
    --rm \
    --network host \
    --shm-size 24G \
    --privileged \
    --ulimit memlock=-1 \
    --name harp \
    --gpus all \
    -v /path/to/your/nfs:/workspace/nfs_share \
    harp:sm$GPU_TYPE /bin/bash
# (Make sure to adjust -v to mount your actual NFS path if available)
```

### 3.2 Pre-profiling
HARP needs to profile the compute capability of each heterogeneous node type. Action: Run the profiling script on one representative node of each GPU type.

**Example Scenario:** 1 Node of A100 + 1 Node of V100.

On the A100 Node:

```Bash
cd /workspace/benchmark/harp
mkdir -p tmp/A100

# Generate profile for A100
ray start --head
python3 gen_prof_database.py \
    --cluster-key A100 \
    --filename /nfs_share/prof_database_A100.pkl \
    --cache-filename ./tmp/A100/hlo_op_cost_dict.pkl \
    --max-comm-size-intra-node 33 \
    --max-comm-size-inter-node 30 \
    --max-fail-retry 4
ray stop
```
On the V100 Node:

```Bash
cd /workspace/benchmark/harp
mkdir -p tmp/V100

# Generate profile for V100
ray start --head
python3 gen_prof_database.py \
    --cluster-key V100 \
    --filename /nfs_share/prof_database_V100.pkl \
    --cache-filename ./tmp/V100/hlo_op_cost_dict.pkl \
    --max-comm-size-intra-node 32 \
    --max-comm-size-inter-node 29 \
    --max-fail-retry 4
ray stop
```
**Important:** If you do not have NFS, you must manually SCP the generated prof_database_*.pkl files to all nodes (place them in the same path on every node).


### 3.3 Launch Ray Cluster
HARP relies on a Ray cluster to manage distributed resources.

Edit the configuration script: Open /workspace/harp/scripts/ray_cluster.sh and update the ip_list and gpu_list variables to match your actual cluster IPs.

```Bash

# declare -a ip_list=("<V100_NODE_IP>", "<A100_NODE_IP>")
# declare -a gpu_list=("gpu-type-<GPU-Type1>>:<GPU-Num1>" "gpu-type-<GPU-Type2>>:<GPU-Num2>")

# Example in ray_cluster.sh for 1x2 V100 + 1x1 A100 setup:
# declare -a ip_list=("192.168.0.1, "192.168.0.2")
# declare -a gpu_list=("gpu-type-V100:2" "gpu-type-A100:1")

bash /workspace/harp/scripts/ray_cluster.sh
# Follow the on-screen prompts (usually select option '1' to start)
```
**Note:** This command should be executed on the first node (the one specified as ip_list[0]).



## 4. Evaluation (Minimal Working Example)
### 4.1 Benchmark Search & Execution
We provide a script to run a grid search for the optimal heterogeneous parallel strategy and H-1F1B schedule.

The following command runs a GPT model training benchmark on a heterogeneous setup (e.g., 2 nodes, 2 V100s + 1 A100).

```Bash
# Run this on the head node (usually the first node in your IP list)
python /workspace/benchmark/harp/benchmark.py \
    --suite gpt.grid_search_auto_heterogeneous \
    --niter 2 \
    --enable-hetero \
    --num-hetero-clusters 2 \
    --num-hosts 1,1 \
    --num-devices-per-host 2,1 \
    --gpu-info V100,A100 \
    --gin-file /workspace/harp/alpa/config/hetero_V100_A100.gin 
```
The major parameters related to the paper are controlled by the gin-file. We provide pre-configured files for several common cluster setups in the config folder.

### 4.2 Interpreting Results
The script will output the best-found strategy (Parallelism + Scheduling). Logs will show the throughput (Tokens/GPU/sec)/

**Run Output**

```bash
============================================================
Heterogeneous Pipeline Schedule (Cost: 14.3811)
Stage  | Cluster  | Layers     | Launch MB  | Comp Cost  | Comm Cost 
------------------------------------------------------------
0      | 0        | [0,22)     | 3          | 0.2143     | 0.0760    
1      | 1        | [22,50)    | 1          | 0.2202     | 0.0000    
------------------------------------------------------------
Path: (C0:L[0,22), MB=3) -> (C1:L[22,50), MB=1)
============================================================

Result forward_stage_layer_ids: [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21], [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]]
Result mesh_shapes: [(0, (1, 2)), (1, (1, 1))]
Result logical_mesh_shapes: [(2, 1), (1, 1)]
Result autosharding_option_dicts: [{'force_batch_dim_to_mesh_dim': 0}, {'force_batch_dim_to_mesh_dim': 0}]
Result mesh_shapes(readable) [(V100, (1, 2)), (A100, (1, 1))]
compile_pipeshard_executable::stage construction: 164.87 s
compile_pipeshard_executable::apply grad: 0.79 s
compile_pipeshard_executable::shard stages: 12.90 s
compile_pipeshard_executable::launch meshes: 0.46 s
Task ( 0 -> 1 ) use IB: False
Task ( 1 -> 0 ) use IB: False
compile_pipeshard_executable::runtime emitter: 13.33 s
compile_pipeshard_executable::driver executable: 1.07 s
 - Compile (driver): 231.71 s
compilation time breakdown: {'stage-construction': '164.33', 'stage-construction-dp': '1.79', 'stage-construction-compilation': '58.57', 'stage-construction-profiling': '70.65'}
 - Compile (worker): 8.84 s
Iteration 0 ...
Iteration 1 ...
 - Benchmark: 68.12 s
Type: gpt  Model Config: GPTModelConfig(seq_len=1024, hidden_size=1536, num_layers=24, num_heads=16, vocab_size=51200)  #Microbatch: 64  #GPU: (2, 1)  Parallel Config: SearchParallelArgs(prefer_reduce_scatter=True, use_remat=True, num_auto_layers=2, auto_stage_option={'submesh_physical_shape_space': 'single_node_power_of_two', 'submesh_logical_shape_space': 'single_node_model_parallel_and_safememory_dp', 'stage_imbalance_tolerance': 0.3, 'use_hlo_cost_model': True, 'profiling_database_filename': ['/workspace/benchmark/harp/prof_database_V100.pkl', '/workspace/benchmark/harp/prof_database_A100.pkl'], 'gpu_flops': [125, 312], 'cluster_key': ['V100', 'A100'], 'layer_profile_mode': 'individual', 'cached_compute_communication_cost_load_path': None, 'cached_compute_communication_cost_store_path_prefix': '/workspace/_tmp/2.6B-1x8V100_2x2A100/', 'cached_profile_result': None})  Mean Time (s): 31.000  Std Time (s): 0.000  #Params (Billion): 0.759B  TFLOPs: 220.26  Peak Mem (GB): [5.934,14.367] (max: 14.367GB)  Metadata: {'compilation_times': "{'stage-construction': '164.33', 'stage-construction-dp': '1.79', 'stage-construction-compilation': '58.57', 'stage-construction-profiling': '70.65'}", 'compute_cost_file_name': 'None', 'forward_stage_layer_ids': '[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21], [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]]', 'submesh_shapes': '[[0, [1, 2]], [1, [1, 1]]]', 'logical_mesh_shapes': '[[2, 1], [1, 1]]', 'autosharding_option_dicts': "[{'force_batch_dim_to_mesh_dim': '0'}, {'force_batch_dim_to_mesh_dim': '0'}]"}  
root@a100x8-ib-ip2:/workspace# 
```


