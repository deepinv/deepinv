#!/bin/bash

#### select resources
#SBATCH --job-name=copy_file
#SBATCH --partition=archive
#SBATCH --time=20:00:00
#SBATCH --nodes=1                    # works with 1 node // set to 2 for 2 nodes
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node // set to 4 for 2 nodes
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --output=logs/%x.out     # output file
#SBATCH --error=logs/%x.err      # error file  23-11-18-12:49:48
#SBATCH --account=tbo@v100

cp -r /gpfsssd/scratch/rech/nyd/commun/datasets /lustre/fsn1/projects/rech/nyd/commun/datasets
chmod g+rx -R /lustre/fsn1/projects/rech/nyd/commun/datasets
