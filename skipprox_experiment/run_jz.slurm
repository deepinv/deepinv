#!/bin/bash
# Usage (from $WORK/skipprox_exp/skipprox_experiment):
#   sbatch --job-name=sp-checks run_jz.slurm checks
#   sbatch --job-name=sp-img0  run_jz.slurm sweep --image 0 --tag main
#SBATCH --account=kkn@v100
#SBATCH -C v100-32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -e
module purge
module load pytorch-gpu/py3/2.8.0
source $WORK/deepinv/venv/bin/activate
export PYTHONNOUSERSITE=1 MPLBACKEND=Agg OMP_NUM_THREADS=8
export PYTHONPATH=$WORK/skipprox_exp/src          # deepinv working tree (incl. GSDRUNet "download_prox" support)
CKPT=$WORK/skipprox_exp/ckpt/Prox-DRUNet_torch.pth # sha256 7e9f1c58...fa97, == official Prox_DRUNet.ckpt weights
cd $SLURM_SUBMIT_DIR
nvidia-smi --query-gpu=name,memory.total --format=csv
python -u skipprox.py "$@" --ckpt $CKPT --device cuda
