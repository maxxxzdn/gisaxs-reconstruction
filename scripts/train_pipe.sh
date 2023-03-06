#!/bin/bash -l
#SBATCH -p hlab
#SBATCH -A hlab
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH -o logs/hostname_%j.out
#SBATCH --gres=gpu:1

module load cuda
module load anaconda
module load gcc

source activate pyg

python train_pipe.py
