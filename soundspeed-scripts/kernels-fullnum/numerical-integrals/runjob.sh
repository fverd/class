#!/usr/bin/env bash
#
#SBATCH --job-name=4diegodT22

#SBATCH --mail-type=END
#SBATCH --mail-user=fverdian@sissa.it
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36

#SBATCH --time=40:00:00  
#
#SBATCH --partition=batch
#SBATCH --output=Slurm-output/%x.o%j
export OMP_NUM_THREADS=$((${SLURM_CPUS_PER_TASK}/2))

python /home/fverdian/class/soundspeed-scripts/kernels-fullnum/numerical-integrals/4diego-theta-z4/break_dT_22.py -fx 0.1 -rtol 0.001 -N 1000 -kref 0.45
