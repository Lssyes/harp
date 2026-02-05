import torch
import time

# 配置参数（根据显存决定）
M = N = K = 8192  # 8192 适合 16 GB 显存；可调节大小
dtype = torch.float16
device = torch.device("cuda")

# 创建输入张量
a = torch.randn((M, K), device=device, dtype=dtype)
b = torch.randn((K, N), device=device, dtype=dtype)
c = torch.empty((M, N), device=device, dtype=dtype)

# 预热 CUDA（第一次有编译/加载延迟）
for _ in range(3):
    torch.matmul(a, b)

# 计时运行
torch.cuda.synchronize()

# 实际乘法  
for _ in range(10):
    start = time.time()
    for __ in range(10):
        # 计算矩阵乘法
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    end = time.time()

    # 理论计算量：2*M*N*K 次浮点操作（GEMM）
    tflops = 10 * 2 * M * N * K / (end - start) / 1e12

    print(f"FP16 GEMM TFLOPs: {tflops:.2f}")
