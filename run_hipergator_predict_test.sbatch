#!/usr/bin/env bash
#SBATCH --job-name=deepforest-predict-test
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=05:30:00

set -euo pipefail

REPO_ROOT="/blue/ewhite/everglades/Henry_GPU/DeepForest"
DEFAULT_BIRD_ROOT="/blue/ewhite/b.weinstein/bird_detector_retrain/zero_shot/avian_images_annotated/test_splits/patch_600"
DEFAULT_BIRD_CSV="$DEFAULT_BIRD_ROOT/test_split_patch_600.csv"
CONDA_MODULE="${CONDA_MODULE:-conda}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-predict}"

export DEEPFOREST_HPC_PREDICT_CSV="${DEEPFOREST_HPC_PREDICT_CSV:-$DEFAULT_BIRD_CSV}"
export DEEPFOREST_HPC_ROOT_DIR="${DEEPFOREST_HPC_ROOT_DIR:-$DEFAULT_BIRD_ROOT}"

export RUN_HIPERGATOR_TESTS="${RUN_HIPERGATOR_TESTS:-1}"
export DEEPFOREST_HPC_NNODES="${DEEPFOREST_HPC_NNODES:-${SLURM_NNODES:-2}}"
export DEEPFOREST_HPC_GPUS_PER_NODE="${DEEPFOREST_HPC_GPUS_PER_NODE:-2}"
export DEEPFOREST_HPC_OUTPUT_DIR="${DEEPFOREST_HPC_OUTPUT_DIR:-$REPO_ROOT/hipergator_test_outputs}"

mkdir -p "$DEEPFOREST_HPC_OUTPUT_DIR"
cd "$REPO_ROOT"

ml "$CONDA_MODULE"
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"

echo "Running Hipergator predict integration test"
echo "CSV: $DEEPFOREST_HPC_PREDICT_CSV"
echo "ROOT_DIR: $DEEPFOREST_HPC_ROOT_DIR"
echo "OUTPUT_DIR: $DEEPFOREST_HPC_OUTPUT_DIR"
echo "NNODES: $DEEPFOREST_HPC_NNODES"
echo "GPUS_PER_NODE: $DEEPFOREST_HPC_GPUS_PER_NODE"
echo "CONDA_MODULE: $CONDA_MODULE"
echo "CONDA_ENV_NAME: $CONDA_ENV_NAME"

if [[ ! -f "$DEEPFOREST_HPC_PREDICT_CSV" ]]; then
    echo "Prediction CSV not found: $DEEPFOREST_HPC_PREDICT_CSV"
    exit 1
fi

if [[ ! -d "$DEEPFOREST_HPC_ROOT_DIR" ]]; then
    echo "Prediction root directory not found: $DEEPFOREST_HPC_ROOT_DIR"
    exit 1
fi

uv run pytest -m hipergator tests/test_hipergator_predict.py -s
