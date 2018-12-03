#!/bin/bash
#SBATCH --job-name=DeepLidar   # Job name
#SBATCH --mail-type=END               # Mail events (NONE, BEGIN, END, FAIL, AL$
#SBATCH --mail-user=benweinstein2010@gmail.com  # Where to send mail
#SBATCH --account=ewhite
#SBATCH --nodes=1                 # Number of MPI ranks
#SBATCH --ntasks=1                 # Number of MPI ranks
#SBATCH --cpus-per-task=5            # Number of cores per MPI rank
#SBATCH --mem-per-cpu=10GB
#SBATCH --time=48:00:00       #Time limit hrs:min:sec
#SBATCH --output=/home/b.weinstein/logs/DeepLidar.out   # Standard output and error log
#SBATCH --error=/home/b.weinstein/logs/DeepLidar.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1

ml git
ml geos/3.6.2
ml tensorflow/1.10.1
export PATH=${PATH}:/home/b.weinstein/miniconda/envs/DeepLidar/bin/
export PYTHONPATH=${PYTHONPATH}:/home/b.weinstein/miniconda/envs/DeepLidar/lib/python3.6/site-packages/
echo $PYTHONPATH
sleep 2

cd /home/b.weinstein/DeepLidar
which python

python train.py --mode train --dir $stamp
