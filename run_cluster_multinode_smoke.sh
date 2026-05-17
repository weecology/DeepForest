#!/usr/bin/env bash
# Distributed smoke test: submit with Slurm, launch with srun (Lightning reads the job env).
#
# Example:
#   sbatch --nodes=2 --ntasks-per-node=2 --gpus-per-node=2 --cpus-per-task=4 --mem=80G \
#     --time=00:30:00 run_cluster_multinode_smoke.sh
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DATA_ROOT="$REPO_ROOT/src/deepforest/data"
TRAIN_CSV="${TRAIN_CSV:-$DATA_ROOT/OSBS_029.csv}"
TRAIN_ROOT="${TRAIN_ROOT:-$DATA_ROOT}"
VAL_CSV="${VAL_CSV:-$DATA_ROOT/OSBS_029.csv}"
VAL_ROOT="${VAL_ROOT:-$DATA_ROOT}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/lightning_logs_multinode_smoke}"

GPUS_PER_NODE="${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-1}}}"
NNODES="${NNODES:-${SLURM_NNODES:-1}}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "SLURM_JOB_ID is not set. Submit with sbatch or run inside salloc."
    exit 1
fi

echo "HOSTNAME=$(hostname)"
echo "SLURM_NNODES=$NNODES"
echo "GPUS_PER_NODE=$GPUS_PER_NODE"
echo "SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE:-?}"

cd "$REPO_ROOT"

srun --kill-on-bad-exit=1 uv run deepforest train \
  --disable-checkpoint \
  --strategy ddp \
  train.fast_dev_run=true \
  workers=0 \
  accelerator=gpu \
  devices="$GPUS_PER_NODE" \
  num_nodes="$NNODES" \
  train.csv_file="$TRAIN_CSV" \
  train.root_dir="$TRAIN_ROOT" \
  validation.csv_file="$VAL_CSV" \
  validation.root_dir="$VAL_ROOT" \
  log_root="$LOG_ROOT"
