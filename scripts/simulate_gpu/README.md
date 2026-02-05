## For simulate V100 on A100

```shell
bash set_v100_limit.sh
```

You should see output similar to:

```shell
root@a100x8-ib-ip2:/workspace/harp/scripts/simulate_gpu# bash set_v100_limit.sh 
=== Starting V100-32GB Simulation Mode ===
Persistence mode is already Enabled for GPU 00000000:00:10.0.
Persistence mode is already Enabled for GPU 00000000:00:11.0.
Persistence mode is already Enabled for GPU 00000000:00:12.0.
Persistence mode is already Enabled for GPU 00000000:00:13.0.
Persistence mode is already Enabled for GPU 00000000:00:14.0.
Persistence mode is already Enabled for GPU 00000000:00:15.0.
Persistence mode is already Enabled for GPU 00000000:00:16.0.
Persistence mode is already Enabled for GPU 00000000:00:17.0.
All done.
Locking graphics clock to 585 MHz...
GPU clocks set to "(gpuClkMin 585, gpuClkMax 585)" for GPU 00000000:00:10.0
GPU clocks set to "(gpuClkMin 585, gpuClkMax 585)" for GPU 00000000:00:11.0
GPU clocks set to "(gpuClkMin 585, gpuClkMax 585)" for GPU 00000000:00:12.0
GPU clocks set to "(gpuClkMin 585, gpuClkMax 585)" for GPU 00000000:00:13.0
GPU clocks set to "(gpuClkMin 585, gpuClkMax 585)" for GPU 00000000:00:14.0
GPU clocks set to "(gpuClkMin 585, gpuClkMax 585)" for GPU 00000000:00:15.0
GPU clocks set to "(gpuClkMin 585, gpuClkMax 585)" for GPU 00000000:00:16.0
GPU clocks set to "(gpuClkMin 585, gpuClkMax 585)" for GPU 00000000:00:17.0
All done.
Memory occupier started (PID: 620343). Check ./tmp/v100_limit.log for details.
=== Setup Complete! Current Status: ===
index, name, memory.total [MiB], memory.used [MiB], clocks.current.graphics [MHz], clocks.current.sm [MHz]
0, NVIDIA A100-SXM4-40GB, 40960 MiB, 7683 MiB, 585 MHz, 585 MHz
1, NVIDIA A100-SXM4-40GB, 40960 MiB, 7683 MiB, 585 MHz, 585 MHz
2, NVIDIA A100-SXM4-40GB, 40960 MiB, 7683 MiB, 585 MHz, 585 MHz
3, NVIDIA A100-SXM4-40GB, 40960 MiB, 7683 MiB, 585 MHz, 585 MHz
4, NVIDIA A100-SXM4-40GB, 40960 MiB, 7683 MiB, 585 MHz, 585 MHz
5, NVIDIA A100-SXM4-40GB, 40960 MiB, 7683 MiB, 585 MHz, 585 MHz
6, NVIDIA A100-SXM4-40GB, 40960 MiB, 7683 MiB, 585 MHz, 585 MHz
7, NVIDIA A100-SXM4-40GB, 40960 MiB, 7683 MiB, 585 MHz, 585 MHz
```

You can benchmark the emulated V100 performance with:
torchrun --nproc_per_node=8 benchmark_v100_emulation.py

You will see results similar to the following. Note that the numbers are slightly higher than a real V100, so the MFU reported in the paper’s evaluation may look a bit optimistic. This is still a fair comparison because the same setting is applied to all baselines.
```shell
--- GPU Benchmark (Emulating V100 on A100) ---
World Size: 8 GPUs
[Rank 4] Tensor Core: 126.95 TFLOPS | NVLink Bandwidth: 113.62 GB/s
[Rank 5] Tensor Core: 126.95 TFLOPS | NVLink Bandwidth: 113.62 GB/s
[Rank 3] Tensor Core: 126.95 TFLOPS | NVLink Bandwidth: 113.62 GB/s
[Rank 6] Tensor Core: 126.95 TFLOPS | NVLink Bandwidth: 113.62 GB/s
[Rank 7] Tensor Core: 126.95 TFLOPS | NVLink Bandwidth: 113.63 GB/s
[Rank 2] Tensor Core: 126.94 TFLOPS | NVLink Bandwidth: 113.62 GB/s
[Rank 1] Tensor Core: 126.95 TFLOPS | NVLink Bandwidth: 113.62 GB/s
[Rank 0] Tensor Core: 126.94 TFLOPS | NVLink Bandwidth: 113.62 GB/s
```



## For reset

```shell
bash reset_v100_limit.sh
```

You should see output similar to:
```shell
root@a100x8-ib-ip2:/workspace/harp/scripts/simulate_gpu# bash reset_v100_limit.sh 
=== Resetting GPU to Default A100 State ===
All done.
Graphics clocks reset to default.
Memory occupier stopped.
=== Reset Complete ===
index, name, memory.total [MiB], memory.used [MiB], clocks.current.graphics [MHz], clocks.current.sm [MHz]
0, NVIDIA A100-SXM4-40GB, 40960 MiB, 1 MiB, 240 MHz, 240 MHz
1, NVIDIA A100-SXM4-40GB, 40960 MiB, 1 MiB, 225 MHz, 225 MHz
2, NVIDIA A100-SXM4-40GB, 40960 MiB, 1 MiB, 240 MHz, 240 MHz
3, NVIDIA A100-SXM4-40GB, 40960 MiB, 1 MiB, 270 MHz, 270 MHz
4, NVIDIA A100-SXM4-40GB, 40960 MiB, 1 MiB, 240 MHz, 240 MHz
5, NVIDIA A100-SXM4-40GB, 40960 MiB, 1 MiB, 285 MHz, 285 MHz
6, NVIDIA A100-SXM4-40GB, 40960 MiB, 1 MiB, 285 MHz, 285 MHz
7, NVIDIA A100-SXM4-40GB, 40960 MiB, 1 MiB, 1410 MHz, 1410 MHz
```
