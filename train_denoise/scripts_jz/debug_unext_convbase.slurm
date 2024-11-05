#!/bin/bash

#### select resources
#SBATCH --job-name=convbase
#SBATCH --time=02:00:00
#SBATCH --nodes=1                    # works with 1 node // set to 2 for 2 nodes
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node // set to 4 for 2 nodes
#SBATCH --gres=gpu:1                 # number of GPUs per node  // set to 4 for 2 nodes
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --output=logs/convbase.out     # output file
#SBATCH --error=logs/convbase.err      # error file  23-11-18-12:49:48
#SBATCH --qos=qos_gpu-dev  # 100h+ = gpu-t4
#SBATCH --account=tbo@v100

module purge
module load cuda/11.8.0

cd $WORK/experiments/ram_project

export PATH=$WORK/miniconda3/bin:$PATH
eval "$($WORK/miniconda3/bin/conda shell.bash hook)"
conda activate deepinv_dev

srun python train_denoise.py --model_name 'unext' --gpu_num 1 --seed 0 --conv_type 'base' --pool_type 'next'
