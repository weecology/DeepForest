#!/bin/bash
#SBATCH --job-name=DeepForest_cpu   # Job name
#SBATCH --mail-type=END               # Mail events (NONE, BEGIN, END, FAIL, AL$
#SBATCH --mail-user=benweinstein2010@gmail.com  # Where to send mail
#SBATCH --account=ewhite
#SBATCH --nodes=1                 # Number of MPI ranks
#SBATCH --ntasks=1                # Number of MPI ranks
#SBATCH --cpus-per-task=5            # Number of cores per MPI rank
#SBATCH --mem-per-cpu=5GB
#SBATCH --time=24:00:00       #Time limit hrs:min:sec
#SBATCH --output=/home/b.weinstein/logs/DeepForest_cpu.out   # Standard output and error log
#SBATCH --error=/home/b.weinstein/logs/DeepForest_cpu.err

ml git
ml gcc
ml gdal
ml tensorflow/1.7.0
export PYTHONPATH=${PYTHONPATH}:/home/b.weinstein/miniconda3/envs/DeepForest/lib/python3.6/site-packages/
echo $PYTHONPATH


cd /home/b.weinstein/DeepForest

python train.py
