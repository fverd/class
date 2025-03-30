#!/usr/bin/env bash
#
#SBATCH --job-name=int13-lotk

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

python /home/fverdian/class/soundspeed-scripts/numerical-integrals/integrate_13.py -rtol 0.00001 -N 5000
