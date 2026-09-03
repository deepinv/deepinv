#!/bin/bash

## JOB INFO
#SBATCH --output=slurm_logs/%x.out
#SBATCH --error=slurm_logs/%x.err

## NODE CONFIGURATION
#SBATCH --constraint=v100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread

## JOB ACCOUNTABILITY
#SBATCH --account=nyd@v100
#SBATCH --qos=qos_gpu-dev
#SBATCH --time=02:00:00

cd $WORK/Repos/ram-experiments
export PATH=$WORK/miniconda3/bin:$PATH
eval "$($WORK/miniconda3/bin/conda shell.bash hook)"
conda activate experiments

python examples/external-libraries/demo_astra_2detect.py