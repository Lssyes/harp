import torch
import time
import multiprocessing as mp
import os

def occupy_gpu(device_id, target_free_gb):
    try:
        # Set the device for this specific process
        torch.cuda.set_device(device_id)
        
        total_mem = torch.cuda.get_device_properties(device_id).total_memory
        target_mem_bytes = target_free_gb * 1024**3
        occupy_bytes = total_mem - target_mem_bytes
        
        if occupy_bytes > 0:
            # 1. Use empty() instead of zeros() for instant allocation
            # 2. float16 = 2 bytes per element
            num_elements = int(occupy_bytes / 2)
            # Reserve a small buffer for context overhead
            num_elements -= int(300 * 1024**2 / 2) 
            
            # Allocation happens here
            holder = torch.empty(num_elements, dtype=torch.float16, device=f'cuda:{device_id}')
            
            print(f"[Process {os.getpid()}] Device {device_id}: Locked {occupy_bytes / 1024**3:.2f} GB")
            
            # Keep the process alive to hold the memory
            while True:
                time.sleep(100)
        else:
            print(f"Device {device_id}: Already below {target_free_gb}GB. Idle.")
            
    except Exception as e:
        print(f"Error on Device {device_id}: {e}")

def main():
    target_free_gb = 32.0
    device_count = torch.cuda.device_count()
    
    print(f"Starting parallel occupation on {device_count} GPUs...")
    
    processes = []
    # Start a separate process for each GPU
    for i in range(device_count):
        p = mp.Process(target=occupy_gpu, args=(i, target_free_gb))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

if __name__ == "__main__":
    # Use 'spawn' to ensure clean CUDA context in sub-processes
    mp.set_start_method('spawn', force=True)
    main()