#!/usr/bin/env bash
# Tree detector + dead/alive crop model on treeA.jpg and treeB.jpg (2 GPUs).
#
#   cd /path/to/DeepForest-crop-ddp
#   sbatch HPC/run_cluster_predict_crop_alive_dead.sbatch
#
#SBATCH --job-name=deepforest-crop-alive-dead
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --partition=hpg-b200
#SBATCH --account=ewhite

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set}"
REPO_ROOT="$(pwd)"

DEFAULT_ROOT="/blue/ewhite/everglades/cluster_deepforest/hpc_test_data"
CONDA_MODULE="${CONDA_MODULE:-conda}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-mypredict}"

export RUN_CLUSTER_TESTS="${RUN_CLUSTER_TESTS:-1}"
export DEEPFOREST_HPC_PREDICT_CSV="${DEEPFOREST_HPC_PREDICT_CSV:-$DEFAULT_ROOT/TreeAB.csv}"
export DEEPFOREST_HPC_ROOT_DIR="${DEEPFOREST_HPC_ROOT_DIR:-$DEFAULT_ROOT}"
export DEEPFOREST_HPC_GPUS_PER_NODE="${DEEPFOREST_HPC_GPUS_PER_NODE:-2}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/blue/ewhite/henrysenyondo/.cache/huggingface}"
export GDAL_ERROR_ON_LIBJPEG_WARNING=FALSE

ml "$CONDA_MODULE"
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"

echo "Running tree + dead/alive crop cluster test"
echo "CSV: $DEEPFOREST_HPC_PREDICT_CSV"
echo "ROOT_DIR: $DEEPFOREST_HPC_ROOT_DIR"
echo "CONDA_ENV_NAME: $CONDA_ENV_NAME"

if [[ ! -f "$DEEPFOREST_HPC_PREDICT_CSV" ]]; then
    echo "Predict CSV not found: $DEEPFOREST_HPC_PREDICT_CSV"
    exit 1
fi

srun --ntasks-per-node="$DEEPFOREST_HPC_GPUS_PER_NODE" \
    python -m pytest -o addopts="" -m cluster \
    tests/test_cluster_predict_crop_alive_dead.py -s
