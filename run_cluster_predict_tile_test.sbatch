#!/usr/bin/env bash
#SBATCH --job-name=deepforest-predict-tile-test
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=05:30:00

set -euo pipefail

REPO_ROOT="/blue/ewhite/everglades/cluster_deepforest/DeepForest"
DEFAULT_TILE_DIR="/blue/ewhite/everglades/projected_mosaics/2023/6thBridge"
CONDA_MODULE="${CONDA_MODULE:-conda}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-predict}"

export RUN_CLUSTER_TESTS="${RUN_CLUSTER_TESTS:-1}"
export DEEPFOREST_HPC_NNODES="${DEEPFOREST_HPC_NNODES:-${SLURM_NNODES:-2}}"
export DEEPFOREST_HPC_GPUS_PER_NODE="${DEEPFOREST_HPC_GPUS_PER_NODE:-2}"
export DEEPFOREST_HPC_OUTPUT_DIR="${DEEPFOREST_HPC_OUTPUT_DIR:-$REPO_ROOT/cluster_tile_outputs}"
export DEEPFOREST_HPC_TILE_DIR="${DEEPFOREST_HPC_TILE_DIR:-$DEFAULT_TILE_DIR}"
export DEEPFOREST_HPC_TILE_MODEL_NAME="${DEEPFOREST_HPC_TILE_MODEL_NAME:-weecology/everglades-bird-species-detector}"
export DEEPFOREST_HPC_TILE_PATCH_SIZE="${DEEPFOREST_HPC_TILE_PATCH_SIZE:-1500}"
export DEEPFOREST_HPC_TILE_PATCH_OVERLAP="${DEEPFOREST_HPC_TILE_PATCH_OVERLAP:-0}"
export DEEPFOREST_HPC_TILE_IOU_THRESHOLD="${DEEPFOREST_HPC_TILE_IOU_THRESHOLD:-0.15}"
export DEEPFOREST_HPC_TILE_DATALOADER_STRATEGY="${DEEPFOREST_HPC_TILE_DATALOADER_STRATEGY:-window}"

mkdir -p "$DEEPFOREST_HPC_OUTPUT_DIR"
cd "$REPO_ROOT"

ml "$CONDA_MODULE"
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"

if [[ -z "${DEEPFOREST_HPC_TILE_PATH:-}" ]]; then
    shopt -s globstar nullglob nocaseglob
    tile_candidates=("$DEEPFOREST_HPC_TILE_DIR"/**/*.tif "$DEEPFOREST_HPC_TILE_DIR"/*.tif)
    shopt -u globstar nullglob nocaseglob

    if (( ${#tile_candidates[@]} == 0 )); then
        echo "No TIFF files found under: $DEEPFOREST_HPC_TILE_DIR"
        exit 1
    fi

    export DEEPFOREST_HPC_TILE_PATH="${tile_candidates[0]}"
fi

echo "Running cluster tiled predict integration test"
echo "TILE_PATH: $DEEPFOREST_HPC_TILE_PATH"
echo "OUTPUT_DIR: $DEEPFOREST_HPC_OUTPUT_DIR"
echo "MODEL_NAME: $DEEPFOREST_HPC_TILE_MODEL_NAME"
echo "NNODES: $DEEPFOREST_HPC_NNODES"
echo "GPUS_PER_NODE: $DEEPFOREST_HPC_GPUS_PER_NODE"
echo "PATCH_SIZE: $DEEPFOREST_HPC_TILE_PATCH_SIZE"
echo "PATCH_OVERLAP: $DEEPFOREST_HPC_TILE_PATCH_OVERLAP"
echo "DATALOADER_STRATEGY: $DEEPFOREST_HPC_TILE_DATALOADER_STRATEGY"
echo "CONDA_MODULE: $CONDA_MODULE"
echo "CONDA_ENV_NAME: $CONDA_ENV_NAME"

if [[ ! -f "$DEEPFOREST_HPC_TILE_PATH" ]]; then
    echo "Tile path not found: $DEEPFOREST_HPC_TILE_PATH"
    exit 1
fi

uv run pytest -m cluster tests/test_cluster_predict_tile.py -s
