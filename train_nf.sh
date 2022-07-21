#!/bin/bash -l
#SBATCH -p fwkt_v100
#SBATCH -A fwkt_v100
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
module load python

source activate py38

mpirun -np 1 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python train_nf.py --in_shape 128 16 --n_samples 200000 --n_layers 3 --train 1 --n_channels 32 --n_epochs 1000 --data_path '/bigdata/hplsim/aipp/Maksim/BA_simulation/layer_3/'
