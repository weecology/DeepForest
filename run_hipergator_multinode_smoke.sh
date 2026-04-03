#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/blue/ewhite/everglades/Henry_GPU/DeepForest"
DATA_ROOT="$REPO_ROOT/src/deepforest/data"
TRAIN_CSV="${TRAIN_CSV:-$DATA_ROOT/OSBS_029.csv}"
TRAIN_ROOT="${TRAIN_ROOT:-$DATA_ROOT}"
VAL_CSV="${VAL_CSV:-$DATA_ROOT/OSBS_029.csv}"
VAL_ROOT="${VAL_ROOT:-$DATA_ROOT}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/lightning_logs_multinode_smoke}"
MASTER_PORT="${MASTER_PORT:-29500}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"

if [[ -z "${SLURM_JOB_NODELIST:-}" ]]; then
    echo "SLURM_JOB_NODELIST is not set. Run this script inside a Slurm allocation."
    exit 1
fi

if [[ -z "${SLURM_NODEID:-}" ]]; then
    echo "SLURM_NODEID is not set. Launch this script via srun so each node gets a rank."
    exit 1
fi

MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)}"
NNODES="${NNODES:-${SLURM_NNODES:-1}}"
NODE_RANK="${NODE_RANK:-$SLURM_NODEID}"

echo "HOSTNAME=$(hostname)"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "NNODES=$NNODES"
echo "NODE_RANK=$NODE_RANK"
echo "GPUS_PER_NODE=$GPUS_PER_NODE"

cd "$REPO_ROOT"

uv run torchrun \
  --nnodes="$NNODES" \
  --nproc_per_node="$GPUS_PER_NODE" \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  -m deepforest.scripts.cli train \
  --disable-checkpoint \
  train.fast_dev_run=true \
  workers=0 \
  accelerator=gpu \
  devices="$GPUS_PER_NODE" \
  strategy=ddp \
  num_nodes="$NNODES" \
  train.csv_file="$TRAIN_CSV" \
  train.root_dir="$TRAIN_ROOT" \
  validation.csv_file="$VAL_CSV" \
  validation.root_dir="$VAL_ROOT" \
  log_root="$LOG_ROOT"
