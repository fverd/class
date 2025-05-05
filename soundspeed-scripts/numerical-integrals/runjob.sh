#!/usr/bin/env bash
#
#SBATCH --job-name=int22

#SBATCH --mail-type=END
#SBATCH --mail-user=fverdian@sissa.it
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

#SBATCH --time=40:00:00  
#
#SBATCH --partition=batch
#SBATCH --output=Slurm-output/%x.o%j
export OMP_NUM_THREADS=$((${SLURM_CPUS_PER_TASK}/2))

python /home/fverdian/class/soundspeed-scripts/numerical-integrals/integrate_22.py -rtol 0.001 -N 1000
