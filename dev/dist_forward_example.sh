#!/bin/bash
#
# Distributed Radio Interferometry Processing Script
# 
# This script runs distributed_forward.py across multiple GPUs/nodes using SLURM.
# 
# Required data files in the script directory:
#   - uv_coordinates.npy (shape: 208171, 2)
#   - 3c353_gdth.npy (shape: 512, 512) 
#   - briggs_weight.npy (shape: 208171, 1)
#
# Usage:
#   sbatch dist_forward_example.sh
#
# Configuration:
#   - 2 nodes, 2 tasks per node = 4 total GPUs
#   - Each task gets 1 GPU and 4 CPU cores
#   - 30 minutes runtime limit
#
#SBATCH --job-name=deepinv-radio-distributed
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --account=fio@v100
#SBATCH --hint=nomultithread

set -euo pipefail

# Create log directory
LOGDIR="$WORK/logs"
mkdir -p "$LOGDIR"

# Redirect output to log files
exec >"$LOGDIR/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" \
     2>"$LOGDIR/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err"

# Load required modules
module purge
module load pytorch-gpu/py3/2.7.0

echo "SLURM Job Information:"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Nodes: $SLURM_NNODES"
echo "  Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "  Total tasks: $SLURM_NTASKS"
echo "  Node list: $SLURM_NODELIST"

# Verify PyTorch installation
python -c "print('hello')"

# Set up distributed training environment variables
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=${MASTER_PORT:-$(shuf -i 20000-65000 -n1)}
export WORLD_SIZE=$SLURM_NTASKS

echo "Distributed Training Setup:"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  WORLD_SIZE: $WORLD_SIZE"

# Threading and NCCL optimization
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_NCCL_BLOCKING_WAIT=1

# Additional NCCL optimizations for multi-node
# export NCCL_IB_DISABLE=0
# export NCCL_NET_GDR_LEVEL=2
# export NCCL_P2P_DISABLE=0

# Radio interferometry specific optimizations
# export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=300

# Debugging and fault tolerance
export PYTHONFAULTHANDLER=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_LAUNCH_BLOCKING=0

echo "Starting distributed radio interferometry processing..."

# Execute the distributed script
srun \
  --mpi=none \
  --gpu-bind=closest \
  --cpu-bind=cores \
  --kill-on-bad-exit=1 \
  --label \
  bash -lc '
    # Set process-specific environment variables
    export RANK=$SLURM_PROCID
    export LOCAL_RANK=$SLURM_LOCALID
    
    echo "Process $SLURM_PROCID: Starting on node $(hostname) with LOCAL_RANK=$LOCAL_RANK"
    
    # Verify CUDA device assignment
    if command -v nvidia-smi &> /dev/null; then
        echo "Process $SLURM_PROCID: Available GPUs:"
        nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits | head -n 1
    fi
    
    
    # Execute the distributed radio interferometry script
    exec python -u "/lustre/fswork/projects/rech/fio/ulx23va/projects/deepinv_PR/repos/deepinv/dev/distributed_forward.py"
  '

echo "Distributed radio interferometry processing completed."
