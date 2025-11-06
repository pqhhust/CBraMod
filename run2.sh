#!/bin/bash
#SBATCH --job-name=preprocess_tuab      # Job name
#SBATCH --output=log_slurm/result_pqhung_2.txt      # Output file
#SBATCH --error=log_slurm/error_pqhung_2.txt        # Error file
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --gpus=0                 # Number of GPUs per node
#SBATCH --cpus-per-task=8                              # Number of CPU cores per task
sh shell/bash2.sh
