# SLURM Distributed Processing Setup

## Overview
This document explains how to run the distributed tiled image processing on SLURM clusters using the updated `slurm_example.sh` script.

## Key Improvements Made

### 1. **Enhanced Resource Allocation**
- **Before**: `--cpus-per-task=1` 
- **After**: `--cpus-per-task=4` (better for I/O and data loading)
- **Before**: `--time=00:05:00`
- **After**: `--time=00:10:00` (more realistic for image processing)

### 2. **Better Environment Variable Setup**
- Added comprehensive debugging output for distributed training variables
- Proper SLURM-specific environment variable mapping:
  - `RANK` = `$SLURM_PROCID` (global process rank across all nodes)
  - `LOCAL_RANK` = `$SLURM_LOCALID` (local rank within each node)
  - `WORLD_SIZE` = `$SLURM_NTASKS` (total number of processes)

### 3. **Enhanced NCCL Configuration**
- Added multi-node NCCL optimizations:
  - `NCCL_IB_DISABLE=0` (enable InfiniBand if available)
  - `NCCL_NET_GDR_LEVEL=2` (GPU Direct RDMA optimization)
  - `NCCL_P2P_DISABLE=0` (enable peer-to-peer GPU communication)

### 4. **Improved Debugging and Monitoring**
- Added GPU device verification for each process
- Comprehensive logging of job information
- Process-specific debugging output
- Better error handling and fault tolerance

### 5. **Resource Optimization**
- `--gpu-bind=closest` ensures optimal GPU-CPU affinity
- `--cpu-bind=cores` provides dedicated CPU cores per task
- `--kill-on-bad-exit=1` ensures clean job termination on failures

## Usage Instructions

### Submit the Job
```bash
sbatch slurm_example.sh
```

### Monitor the Job
```bash
# Check job status
squeue -u $USER

# View output logs
tail -f $WORK/logs/deepinv-distributed-<JOB_ID>.out
tail -f $WORK/logs/deepinv-distributed-<JOB_ID>.err
```

### Cancel if Needed
```bash
scancel <JOB_ID>
```

## Environment Variables Set by SLURM Script

| Variable | Source | Purpose |
|----------|--------|---------|
| `MASTER_ADDR` | First node from `$SLURM_NODELIST` | Communication endpoint |
| `MASTER_PORT` | Random port 20000-65000 | Communication port |
| `WORLD_SIZE` | `$SLURM_NTASKS` | Total number of processes |
| `RANK` | `$SLURM_PROCID` | Global process rank |
| `LOCAL_RANK` | `$SLURM_LOCALID` | Local rank within node |

## Configuration for Different Cluster Sizes

### Small Scale (1 node, 2 GPUs)
```bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
```

### Medium Scale (2 nodes, 4 GPUs)
```bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
```

### Large Scale (4 nodes, 8 GPUs)
```bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
```

## Troubleshooting

### Common Issues and Solutions

1. **NCCL Errors**: Check that `--gpus-per-task=1` matches your GPU allocation
2. **Module Loading**: Ensure `pytorch-gpu/py3/2.4.0` is available on your cluster
3. **Path Issues**: Update the Python script path in the srun command
4. **Memory Issues**: Increase `--cpus-per-task` or reduce `batch_size` in the Python script

### Debug Mode
To enable maximum debugging, add to the SLURM script:
```bash
export NCCL_DEBUG=TRACE
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_LAUNCH_BLOCKING=1
```

## Performance Tips

1. **Optimal Batch Size**: Start with `batch_size=4` and adjust based on GPU memory
2. **Worker Threads**: Use `num_workers=2-4` for good I/O performance
3. **Mixed Precision**: Keep `use_amp=True` for better memory efficiency
4. **NCCL Backend**: Will be automatically selected for CUDA devices

## Compatibility

- **PyTorch**: 2.4.0+
- **CUDA**: 12.2+
- **NCCL**: 2.19.3+
- **SLURM**: 20.02+

The script is designed to work with both `torchrun` (for local/interactive use) and SLURM (for cluster deployment) seamlessly.
