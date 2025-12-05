#!/usr/bin/env bash
#
#SBATCH --job-name=13_m26_again

#SBATCH --mail-type=END
#SBATCH --mail-user=fverdian@sissa.it
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

#SBATCH --time=52:00:00  
#
#SBATCH --partition=batch
#SBATCH --output=Slurm-output/%x.o%j
export OMP_NUM_THREADS=$((${SLURM_CPUS_PER_TASK}/2))

# python /home/fverdian/class/soundspeed-scripts/numerical-integrals/neutrinos/integrate_13_nu.py -rtol 0.01 -N 16000 -Mnu 1.0 -p 0 

python /home/fverdian/class/soundspeed-scripts/numerical-integrals/integrate_13.py -rtol 0.01 -N 5000 -fx 0.2 -ma 1.e-26 -p 2