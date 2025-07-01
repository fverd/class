#!/usr/bin/env bash
#
#SBATCH --job-name=22_smallf

#SBATCH --mail-type=END
#SBATCH --mail-user=fverdian@sissa.it
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

#SBATCH --time=24:00:00  
#
#SBATCH --partition=batch
#SBATCH --output=Slurm-output/%x.o%j
export OMP_NUM_THREADS=$((${SLURM_CPUS_PER_TASK}/2))

# python /home/fverdian/class/soundspeed-scripts/numerical-integrals/neutrinos/integrate_13_nu.py -rtol 0.01 -N 16000 -Mnu 1.0 -p 0 

python /home/fverdian/class/soundspeed-scripts/numerical-integrals/integrate_22.py -rtol 0.01 -N 6000 -fx 0.02 -ma 1.e-27 -p 2