#!/bin/bash
#SBATCH --job-name=detection   # Job name
#SBATCH --mail-type=END               # Mail events (NONE, BEGIN, END, FAIL, AL$
#SBATCH --mail-user=ben.weinstein@weecology.org   # Where to send mail
#SBATCH --account=ewhite
#SBATCH --qos=ewhite-b
#SBATCH --ntasks=1                 # Number of MPI ranks
#SBATCH --cpus-per-task=1            # Number of cores per MPI rank
#SBATCH --mem-per-cpu=20GB
#SBATCH --time=72:00:00       #Time limit hrs:min:sec
#SBATCH --output=/home/b.weinstein/logs/pretraining_detection.out   # Standard output and error log
#SBATCH --error=/home/b.weinstein/logs/pretraining_detection.err

#This is a generic R submission script
#conda activate DeepForest
#python create_lists.py
module load R
Rscript /home/b.weinstein/DeepForest/model/pretraining/loop_pretraining.R