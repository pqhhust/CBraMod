#!/bin/bash
#SBATCH --job-name=test_pearl      # Job name
#SBATCH --output=log_slurm/result_pqhung3.txt      # Output file
#SBATCH --error=log_slurm/error_pqhung3.txt        # Error file
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --gpus=1                 # Number of GPUs per node
#SBATCH --cpus-per-task=20                 # Number of CPU cores per task
sh shell/bash.sh
