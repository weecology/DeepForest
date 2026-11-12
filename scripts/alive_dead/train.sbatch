#!/bin/bash
#SBATCH --job-name=alive_dead
#SBATCH --account=ewhite
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=benweinstein2010@gmail.com
#SBATCH --output=/home/b.weinstein/logs/alive_dead_%j.out
#SBATCH --error=/home/b.weinstein/logs/alive_dead_%j.err

# Train an Alive/Dead CropModel. Usage:
#   sbatch train.sbatch <split_name> <run_name> [extra train_cropmodel.py args]
set -euo pipefail
ROOT=/blue/ewhite/b.weinstein/DeepForest/alive_dead
REPO=/blue/ewhite/b.weinstein/src/DeepForest-alive-dead
SPLIT=$1; RUN=$2; shift 2

cd "$REPO/scripts/alive_dead"
"$REPO/.venv/bin/python" train_cropmodel.py \
    --split-dir "$ROOT/splits/$SPLIT" --out "$ROOT/runs/$RUN" "$@"
