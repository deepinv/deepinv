#!/bin/bash

# Define the range of hyperparameter values
#gains=(1e-3 1e-2 1e-1 1)
gainslinear=(0)
gainsconv=(1e-3 1e-2 1e-1 1)
drop_probs=(0 0.1 0.2 0.3 0.4 0.5)
learning_rates=(1e-3 1e-4 1e-5)
init_type='ortho'


# Loop over the combinations of gains
for gain_init_linear in "${gainslinear[@]}"; do
  for gain_init_conv in "${gainsconv[@]}"; do
    for drop_prob in "${drop_probs[@]}"; do
      for lr in "${learning_rates[@]}"; do
        # Create a unique job name
        job_name="sweep_${gain_init_linear}_${gain_init_conv}_${init_type}_${drop_prob}_${lr}"

        # Create the SLURM job script using a here document
        cat <<EOF > ${job_name}.slurm
#!/bin/bash

#### select resources
#SBATCH --job-name=${job_name}
#SBATCH --time=02:00:00
#SBATCH --nodes=1                    # works with 1 node // set to 2 for 2 nodes
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node // set to 4 for 2 nodes
#SBATCH --gres=gpu:1                 # number of GPUs per node  // set to 4 for 2 nodes
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --output=logs/%x.out     # output file
#SBATCH --error=logs/%x.err      # error file
#SBATCH --account=tbo@v100

module purge
module load cuda/11.8.0

cd \$WORK/experiments/ram_project

export PATH=\$WORK/miniconda3/bin:\$PATH
eval "\$($WORK/miniconda3/bin/conda shell.bash hook)"
conda activate deepinv_dev

srun python train_denoise.py --model_name 'unext' --gain_init_linear ${gain_init_linear} --gain_init_conv ${gain_init_conv} --init_type '${init_type}' --drop_prob '${drop_prob}' --lr ${lr}
EOF

        # Submit the job
        sbatch ${job_name}.slurm
      done
    done
  done
done