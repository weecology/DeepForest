#!/bin/bash
#SBATCH --job-name=marine-mammal-detector
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

# Update these paths for your HPC project layout.
DATA_DIR="/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data/crops/marine_mammal"
IMAGE_ROOT="/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data/crops"
LOG_ROOT="/blue/ewhite/b.weinstein/BOEM/marine_mammal_training"

module purge
module load cuda

cd /blue/ewhite/b.weinstein/DeepForest

# Activate your environment here.
# source ~/.bashrc
# conda activate deepforest

srun uv run python reproducibility/marine_mammal_detector/train_marine_mammal_detector.py \
  --train-csv "${DATA_DIR}/train.csv" \
  --test-csv "${DATA_DIR}/test.csv" \
  --zero-shot-csv "${DATA_DIR}/zero_shot.csv" \
  --root-dir "${IMAGE_ROOT}" \
  --log-root "${LOG_ROOT}" \
  --experiment-name "marine-mammal-v1" \
  --strategy "ddp" \
  --tensorboard
