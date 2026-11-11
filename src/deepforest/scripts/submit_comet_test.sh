#!/bin/bash
#SBATCH --job-name=train_birds   # Job name
#SBATCH --mail-type=END               # Mail events
#SBATCH --mail-user=benweinstein2010@gmail.com  # Where to send mail
#SBATCH --account=ewhite
#SBATCH --nodes=2                # Number of nodes
#SBATCH --cpus-per-task=10
#SBATCH --mem=60GB
#SBATCH --time=48:00:00       #Time limit hrs:min:sec
#SBATCH --output=/home/b.weinstein/logs/train_birds%j.out   # Standard output and error log
#SBATCH --error=/home/b.weinstein/logs/train_birds%j.err
#SBATCH --partition=hpg-b200
#SBATCH --ntasks-per-node=2      # One task per GPU
#SBATCH --gpus-per-node=2

# One task per GPU is required by Lightning's DDP launcher: ntasks-per-node must
# equal the GPUs per node, and --num_nodes below must equal --nodes.
#
# Prepare the data once beforehand (script lives on the bird_training branch):
#   uv run python src/deepforest/scripts/prepare_birds.py --output_dir /blue/ewhite/b.weinstein/bird_detector_retrain/data/

cd /blue/ewhite/b.weinstein/src/DeepForest

# Sync the environment once, so tasks do not race to modify it
uv sync --all-extras --dev

srun uv run --no-sync python scripts/train_multi_node_comettest.py \
    --data_dir /blue/ewhite/b.weinstein/bird_detector_retrain/data/ \
    --num_nodes "$SLURM_JOB_NUM_NODES" \
    --batch_size 32 \
    --workers 10 \
    --epochs 40
