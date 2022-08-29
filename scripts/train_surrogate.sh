#!/bin/bash -l
#SBATCH -p hlab
#SBATCH -A hlab
#SBATCH -t 48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -o ../logs/hostname_%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=0

module load cuda
module load anaconda
module load gcc/5.5.0
module load openmpi/3.1.2
module load python/3.8

source activate py38

mpirun -np 1 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python train_surrogate.py --start_id 0 --end_id 50000 --n_layers 3 --to_augment 0 --drop_prob 0. --n_epochs 100 --loss 'l2' --batch_size 32 --lr 5e-4 --hidden_dim 32
 