#!/bin/bash
#SBATCH --job-name=train_birds   # Job name
#SBATCH --mail-type=END               # Mail events
#SBATCH --mail-user=benweinstein2010@gmail.com  # Where to send mail
#SBATCH --account=ewhite
#SBATCH --nodes=1                 # Number of MPI ran
#SBATCH --cpus-per-task=10
#SBATCH --mem=200GB
#SBATCH --time=48:00:00       #Time limit hrs:min:sec
#SBATCH --output=/home/b.weinstein/logs/train_birds%j.out   # Standard output and error log
#SBATCH --error=/home/b.weinstein/logs/train_birds%j.err
#SBATCH --partition=hpg-b200
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1

# Example usage:
# First prepare the data:
#uv run python src/deepforest/scripts/prepare_birds.py --output_dir /blue/ewhite/b.weinstein/bird_detector_retrain/data/

srun uv run python src/deepforest/scripts/train_birds.py \
    --data_dir /blue/ewhite/b.weinstein/bird_detector_retrain/data/ \
    --batch_size 32 \
    --workers 10 \
    --epochs 40 