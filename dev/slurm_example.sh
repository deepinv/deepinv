#!/bin/bash
#SBATCH --job-name=deepinv-distributed
#SBATCH --nodes=2
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

echo "SLURM Job Information:"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Nodes: $SLURM_NNODES"
echo "  Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "  Total tasks: $SLURM_NTASKS"
echo "  Node list: $SLURM_NODELIST"

# Load required modules
module purge
module load pytorch-gpu/py3/2.7.0

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

# Debugging and fault tolerance
export PYTHONFAULTHANDLER=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_LAUNCH_BLOCKING=0

echo "Starting distributed processing..."

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
    
    # Execute the Python script
    exec python -u "/lustre/fswork/projects/rech/fio/ulx23va/projects/deepinv_PR/repos/deepinv/dev/example_ddp.py"
  '

echo "Distributed processing completed."
