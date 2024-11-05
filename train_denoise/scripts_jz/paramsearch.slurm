#!/bin/bash

#### select resources
#SBATCH --job-name=sweep_0.1_0.1_ortho
#SBATCH --time=02:00:00
#SBATCH --nodes=1                    # works with 1 node // set to 2 for 2 nodes
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node // set to 4 for 2 nodes
#SBATCH --gres=gpu:1                 # number of GPUs per node  // set to 4 for 2 nodes
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --output=logs/%x.out     # output file
#SBATCH --error=logs/%x.err      # error file
#SBATCH --qos=qos_gpu-dev  # 100h+ = gpu-t4
#SBATCH --account=tbo@v100

module purge
module load cuda/11.8.0

cd $WORK/experiments/ram_project

export PATH=$WORK/miniconda3/bin:$PATH
eval "$($WORK/miniconda3/bin/conda shell.bash hook)"
conda activate deepinv_dev

srun python train_denoise.py --model_name 'unext' --gpu_num 1 --seed 0 --conv_type 'next' --pool_type 'base' --epochs 20 --gain_init_linear 0.1 --gain_init_conv 0.1 --init_type 'ortho'