#!/bin/bash -l
#SBATCH -p hlab
#SBATCH -A hlab
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -o logs/hostname_%j.out
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
    python train_surrogate.py --augmentation 1 --n_samples 50000 --n_epochs 10000 --log 1 --minmax 1 --equalize 0 --data_path '/bigdata/hplsim/aipp/Maksim/BA_simulation/layer_3/' --n_layers 3 --model 'fcnet' --in_shape 128 16 --learning_rate 5e-4 --distance 'l1' --window_size 15 5 --drop_y 0
