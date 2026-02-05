import torch
import torch.distributed as dist
import time
import os

def benchmark_tensor_core(device, iterations=100):
    # 矩阵大小：16384 x 16384
    size = 16384
    a = torch.randn(size, size, device=device, dtype=torch.float16)
    b = torch.randn(size, size, device=device, dtype=torch.float16)
    
    # Warm up
    for _ in range(10):
        torch.matmul(a, b)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(iterations):
        torch.matmul(a, b)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / iterations
    # TFLOPS 计算公式: (2 * M * N * K) / (time * 10^12)
    tflops = (2 * size**3) / (avg_time * 1e12)
    return tflops

def benchmark_nvlink(device, rank, world_size, iterations=50):
    # 传输 512MB 的数据
    tensor_size = 512 * 1024 * 1024 // 2 # float16 占 2 bytes
    data = torch.randn(tensor_size, device=device, dtype=torch.float16)
    
    # Warm up
    for _ in range(5):
        dist.all_reduce(data, op=dist.ReduceOp.SUM)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(iterations):
        dist.all_reduce(data, op=dist.ReduceOp.SUM)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / iterations
    # 算法带宽 (Algorithm Bandwidth) GB/s
    size_gb = (tensor_size * 2) / 1e9
    bus_bandwidth = size_gb / avg_time * (2 * (world_size - 1) / world_size) # 转化为总线带宽
    
    return bus_bandwidth

def main():
    # 初始化分布式环境
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    if rank == 0:
        print(f"--- GPU Benchmark (Emulating V100 on A100) ---")
        print(f"World Size: {world_size} GPUs")

    # 1. 测试 Tensor Core 性能
    tflops = benchmark_tensor_core(device)
    
    # 2. 测试 NVLink 性能
    nvlink_gbps = benchmark_nvlink(device, rank, world_size)

    # 打印结果
    print(f"[Rank {rank}] Tensor Core: {tflops:.2f} TFLOPS | NVLink Bandwidth: {nvlink_gbps:.2f} GB/s")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()