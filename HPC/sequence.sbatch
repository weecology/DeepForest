#!/bin/bash
#SBATCH --job-name=retrain_sequence   # Job name
#SBATCH --mail-type=END               # Mail events (NONE, BEGIN, END, FAIL, AL$
#SBATCH --mail-user=benweinstein2010@gmail.com  # Where to send mail
#SBATCH --account=ewhite
#SBATCH --nodes=1                 # Number of MPI ranks
#SBATCH --ntasks=1                 # Number of MPI ranks
#SBATCH --cpus-per-task=4            # Number of cores per MPI rank
#SBATCH --mem-per-cpu=20GB
#SBATCH --time=48:00:00       #Time limit hrs:min:sec
#SBATCH --output=/home/b.weinstein/logs/retrain_sequence.out   # Standard output and error log
#SBATCH --error=/home/b.weinstein/logs/retrain_sequence.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1

ml git
ml gcc
ml geos
ml tensorflow
export PATH=${PATH}:/home/b.weinstein/miniconda/envs/DeepLidar/bin/
export PYTHONPATH=${PYTHONPATH}:/home/b.weinstein/miniconda/envs/DeepLidar/lib/python3.6/site-packages/
    
cd /home/b.weinstein/DeepLidar

python retrain_sequence.py

cat analysis/pretraining_size.csv
