# Multi-GPU Single Node Guide

## How to Run with Multiple GPUs on a Single Node

When you have multiple GPUs available on a single node, you should use `torchrun` to launch distributed processes. Here's how:

### 1. **Check Available GPUs**

First, check how many GPUs you have:
```bash
python -c "import torch; print(f'GPUs available: {torch.cuda.device_count()}')"
nvidia-smi  # Shows detailed GPU information
```

### 2. **Launch Commands for Different GPU Configurations**

#### **2 GPUs Available**
```bash
torchrun --nproc_per_node=2 example_ddp.py
```

#### **4 GPUs Available**
```bash
torchrun --nproc_per_node=4 example_ddp.py
```

#### **8 GPUs Available**
```bash
torchrun --nproc_per_node=8 example_ddp.py
```

#### **Use Subset of Available GPUs**
If you have 8 GPUs but only want to use 4:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 example_ddp.py
```

### 3. **What Happens Behind the Scenes**

When you run `torchrun --nproc_per_node=N example_ddp.py`:

1. **Process Creation**: torchrun spawns N processes (one per GPU)
2. **Environment Variables**: Each process gets:
   - `RANK`: Global process rank (0, 1, 2, ..., N-1)
   - `LOCAL_RANK`: Local rank within the node (0, 1, 2, ..., N-1)
   - `WORLD_SIZE`: Total number of processes (N)
   - `MASTER_ADDR`: Communication address (localhost for single node)
   - `MASTER_PORT`: Communication port (auto-assigned)

3. **GPU Assignment**: Each process automatically gets assigned to its corresponding GPU:
   - Process 0 → GPU 0 (LOCAL_RANK=0)
   - Process 1 → GPU 1 (LOCAL_RANK=1)
   - Process 2 → GPU 2 (LOCAL_RANK=2)
   - etc.

4. **Work Distribution**: The image windows are automatically distributed across processes using `DistributedSampler`

5. **Result Gathering**: All results are gathered back to rank 0 (the first process)

### 4. **Example Output for 4 GPUs**

```bash
$ torchrun --nproc_per_node=4 example_ddp.py

# Process 0 (Rank 0, GPU 0):
Setup: LOCAL_RANK=0, RANK=0, WORLD_SIZE=4
Using CUDA device: cuda:0
Rank 0: Processing 12 batches...  # Processes 1/4 of windows

# Process 1 (Rank 1, GPU 1):
Setup: LOCAL_RANK=1, RANK=1, WORLD_SIZE=4
Using CUDA device: cuda:1  
Rank 1: Processing 12 batches...  # Processes 1/4 of windows

# Process 2 (Rank 2, GPU 2):
Setup: LOCAL_RANK=2, RANK=2, WORLD_SIZE=4
Using CUDA device: cuda:2
Rank 2: Processing 12 batches...  # Processes 1/4 of windows

# Process 3 (Rank 3, GPU 3):
Setup: LOCAL_RANK=3, RANK=3, WORLD_SIZE=4  
Using CUDA device: cuda:3
Rank 3: Processing 12 batches...  # Processes 1/4 of windows

# Final result from Rank 0:
SUCCESS: Got 48 outputs. Example shape[0]: (3, 192, 192)
```

### 5. **Performance Benefits**

**Speed-up Example (48 windows, 4 GPUs):**
- Single GPU: Processes all 48 windows sequentially
- 4 GPUs: Each GPU processes 12 windows in parallel
- **Theoretical speedup: ~4x faster**

**Memory Benefits:**
- Each GPU only holds 1/N of the total data
- Allows processing larger datasets that wouldn't fit on a single GPU

### 6. **Configuration Recommendations**

#### **For Image Processing Tasks:**
```bash
# Optimal batch size per GPU (adjust based on GPU memory)
batch_size = 4-8   # per GPU

# Number of workers (adjust based on CPU cores)
num_workers = 2-4  # per GPU

# Enable mixed precision for memory efficiency
use_amp = True
```

#### **Example Launch with Custom Settings:**
```bash
# 4 GPUs with optimized settings
torchrun --nproc_per_node=4 example_ddp.py --batch_size 8 --num_workers 4
```

### 7. **Troubleshooting Multi-GPU Issues**

#### **GPU Memory Issues:**
```bash
# Reduce batch size
torchrun --nproc_per_node=4 example_ddp.py  # batch_size will be 4 by default

# Monitor GPU memory
watch -n 1 nvidia-smi
```

#### **Communication Issues:**
```bash
# Enable debugging
TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --nproc_per_node=4 example_ddp.py

# Specify network interface (if needed)
NCCL_SOCKET_IFNAME=eth0 torchrun --nproc_per_node=4 example_ddp.py
```

#### **Process Synchronization Issues:**
```bash
# Increase timeouts for slow systems
NCCL_TIMEOUT_MS=600000 torchrun --nproc_per_node=4 example_ddp.py
```

### 8. **Advanced Multi-GPU Configurations**

#### **Use Specific GPUs:**
```bash
# Use only GPUs 0, 2, 4, 6 out of 8 available
CUDA_VISIBLE_DEVICES=0,2,4,6 torchrun --nproc_per_node=4 example_ddp.py
```

#### **Mixed GPU Types:**
```bash
# If you have different GPU types, use nvidia-smi to check compatibility
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
```

#### **CPU + GPU Hybrid:**
```bash
# Force some processes to use CPU (for testing)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=4 example_ddp.py
# Processes 0,1 use GPU, processes 2,3 fall back to CPU
```

### 9. **Performance Monitoring**

#### **Monitor GPU Utilization:**
```bash
# In another terminal while script is running
watch -n 0.5 nvidia-smi

# Log GPU usage to file
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total --format=csv -l 1 > gpu_usage.log
```

#### **Profile Performance:**
```bash
# Add profiling to the script
TORCH_PROFILER_ENABLED=1 torchrun --nproc_per_node=4 example_ddp.py
```

### 10. **Expected Performance Scaling**

| GPUs | Theoretical Speedup | Realistic Speedup* | Memory Per GPU |
|------|-------------------|-------------------|----------------|
| 1    | 1x                | 1x                | 100%          |
| 2    | 2x                | 1.8x              | 50%           |
| 4    | 4x                | 3.5x              | 25%           |
| 8    | 8x                | 6.5x              | 12.5%         |

*Realistic speedup accounts for communication overhead and synchronization

## Summary

**For multi-GPU single node setups:**
1. Use `torchrun --nproc_per_node=N` where N = number of GPUs
2. Each process automatically gets assigned to a GPU
3. Work is distributed automatically via `DistributedSampler`
4. Results are gathered back to rank 0
5. Monitor GPU usage with `nvidia-smi`
6. Adjust batch_size and num_workers based on your GPU memory and CPU cores

The script handles all the complexity automatically - you just need to specify how many GPUs to use!
