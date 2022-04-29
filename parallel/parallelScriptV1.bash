#!/bin/bash
###
###
#SBATCH --time=06:00:00
#SBATCH -N 19
#SBATCH -n 512
#SBATCH --partition=shortq
#SBATCH --job-name=brownian
#SBATCH --output=pbrownian.o%j

module load gcc mpich slurm
#source ~/.bashrc

mpirun -np 512 ./runBrownianParallel
