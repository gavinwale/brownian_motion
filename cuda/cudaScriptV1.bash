#!/bin/bash
###
###
#SBATCH --time=06:00:00
#SBATCH --tasks=1
#SBATCH --job-name=brown
#SBATCH --output=gbrownian.o%j
#SBATCH -p gpuq
#SBATCH --gres=gpu:1

module load cuda10.0 slurm gcc/7.2.0

./runBrownianCuda

status=$?
if [ $status -ne 0 ]; then
	exit $status
fi

