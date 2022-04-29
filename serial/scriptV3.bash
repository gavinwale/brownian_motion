#!/bin/bash
###
###
#SBATCH --time=06:00:00
#SBATCH --tasks=1
#SBATCH --partition=shortq
#SBATCH --job-name=471final
#SBATCH --output=sbrownian.o%j

module load slurm gcc mpich

#source ~/.bashrc

for i in {1..10};

	do mpirun -np 1 ./runBrownian;
	sleep 1;
	done
