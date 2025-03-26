#!/usr/bin/env bash
#
#SBATCH --job-name=int

#SBATCH --mail-type=NONE
#SBATCH --mail-user=fverdian@sissa.it
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72

#SBATCH --time=40:00:00  
#
#SBATCH --partition=batch
#SBATCH --output=Slurm-output/int
export OMP_NUM_THREADS=$((${SLURM_CPUS_PER_TASK}/2))

python script.py