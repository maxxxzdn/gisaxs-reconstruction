#!/bin/bash -l
#SBATCH -p hlab
#SBATCH -A hlab
#SBATCH -t 30:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -o logs/hostname_%j.out
#SBATCH --gres=gpu:0
#SBATCH --mem=0

module load gcc/7.3.0
module load cuda/9.0
module load lapack/3.8.0 fftw/3.3.8 gnuplot/5.2.4 octave/5.1.0 ghostscript/9.27
module load root/6.16.00
module load boost gsl/2.5 boost/1.78.0 
module load openmpi
module load python

source ../BornAgain/bin/thisbornagain.sh

mpirun -np 1 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python make_samples.py --a 0 --b 10000 --path /bigdata/hplsim/aipp/Maksim/BA_simulation/layer_3_hard/ --n_layers 3 --config './dataset/config_hard.json'
