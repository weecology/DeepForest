#!/usr/bin/env bash
# Cluster training launcher for Slurm (HiPerGator and similar).
#
# TRAIN_MODE:
#   train  - full training (default CONFIG_NAME=bird)
#   smoke  - 1-epoch OSBS smoke test (default CONFIG_NAME=smoke)
#
# SCENARIO (smoke, or inferred for train when unset):
#   1gpu       - single GPU
#   multigpu   - DDP on one node
#   multinode  - DDP across nodes
#
# Override at submit time, e.g.:
#   sbatch src/deepforest/scripts/HPC/run_cluster_train.sbatch train.lr=0.0005
#   TRAIN_MODE=smoke SCENARIO=multigpu sbatch --nodes=1 --ntasks-per-node=2 --gpus-per-node=2 \
#     src/deepforest/scripts/HPC/run_cluster_train.sbatch
set -euo pipefail

TRAIN_MODE="${TRAIN_MODE:-train}"
CONFIG_NAME="${CONFIG_NAME:-}"
SCENARIO="${SCENARIO:-}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-1}}}"
NNODES="${NNODES:-${SLURM_NNODES:-1}}"

CONDA_MODULE="${CONDA_MODULE:-conda}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-predict}"
USE_COMET="${USE_COMET:-1}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "SLURM_JOB_ID is not set. Submit with sbatch."
  exit 1
fi

mkdir -p "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set}/slurm_logs"
cd "${SLURM_SUBMIT_DIR}"
REPO_ROOT="$(pwd)"

ml "$CONDA_MODULE"
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"

if command -v uv &>/dev/null; then
  DEEPFOREST_CMD=(uv run deepforest)
elif command -v deepforest &>/dev/null; then
  DEEPFOREST_CMD=(deepforest)
else
  echo "Neither 'uv' nor 'deepforest' found after conda activate $CONDA_ENV_NAME"
  exit 1
fi

STRATEGY_ARGS=()
EXTRA_TRAIN_ARGS=()
COMET_ARGS=()
DEVICES=1
NODES=1

if [[ "$TRAIN_MODE" == "smoke" ]]; then
  CONFIG_NAME="${CONFIG_NAME:-smoke}"
  SCENARIO="${SCENARIO:-1gpu}"
  DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/src/deepforest/data}"
  TRAIN_CSV="${TRAIN_CSV:-$DATA_ROOT/OSBS_029.csv}"
  TRAIN_ROOT="${TRAIN_ROOT:-$DATA_ROOT}"
  VAL_CSV="${VAL_CSV:-$DATA_ROOT/OSBS_029.csv}"
  VAL_ROOT="${VAL_ROOT:-$DATA_ROOT}"
  LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/lightning_logs_smoke}"
  EXTRA_TRAIN_ARGS+=(--disable-checkpoint)
  COMET_TAGS=(smoke)
else
  CONFIG_NAME="${CONFIG_NAME:-bird}"
  LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/lightning_logs}"
  COMET_TAGS=(bird)
  if [[ -z "$SCENARIO" ]]; then
    if [[ "$NNODES" -gt 1 ]]; then
      SCENARIO=multinode
    elif [[ "$GPUS_PER_NODE" -gt 1 ]]; then
      SCENARIO=multigpu
    else
      SCENARIO=1gpu
    fi
  fi
fi

case "$SCENARIO" in
  1gpu)
    DEVICES=1
    NODES=1
    ;;
  multigpu)
    DEVICES="${GPUS_PER_NODE}"
    NODES=1
    STRATEGY_ARGS=(--strategy ddp)
    ;;
  multinode)
    DEVICES="${GPUS_PER_NODE}"
    NODES="${NNODES}"
    STRATEGY_ARGS=(--strategy ddp)
    ;;
  *)
    echo "Unknown SCENARIO=$SCENARIO (use 1gpu, multigpu, or multinode)"
    exit 1
    ;;
esac

if [[ "$USE_COMET" == "1" ]]; then
  export COMET_WORKSPACE="${COMET_WORKSPACE:-henrykironde}"
  export COMET_PROJECT="${COMET_PROJECT:-bird-detector}"
  if [[ -z "${COMET_API_KEY:-}" && -f "$HOME/.comet_api_key" ]]; then
    export COMET_API_KEY="$(cat "$HOME/.comet_api_key")"
  fi
  if [[ -z "${COMET_API_KEY:-}" ]]; then
    echo "WARNING: COMET_API_KEY not set; running without --comet"
  else
    COMET_ARGS=(--comet)
    for tag in "${COMET_TAGS[@]}"; do
      COMET_ARGS+=(--tag "$tag")
    done
    if [[ "$TRAIN_MODE" == "smoke" ]]; then
      COMET_ARGS+=(--tag "$SCENARIO")
    fi
    if [[ -n "${COMET_EXPERIMENT_NAME:-}" ]]; then
      COMET_ARGS+=(--experiment-name "$COMET_EXPERIMENT_NAME")
    elif [[ "$TRAIN_MODE" == "smoke" ]]; then
      COMET_ARGS+=(--experiment-name "smoke-${SCENARIO}-job${SLURM_JOB_ID}")
    fi
  fi
fi

if [[ "$TRAIN_MODE" == "smoke" ]]; then
  if [[ ! -f "$TRAIN_CSV" ]]; then
    echo "Training CSV not found: $TRAIN_CSV"
    exit 1
  fi
fi

if [[ -f "$HOME/.secrets/hf_token" ]]; then
  export HF_TOKEN="$(cat "$HOME/.secrets/hf_token")"
fi

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"

RESUME_CKPT="${RESUME_CKPT:-}"
RESUME_ARGS=()
if [[ -n "$RESUME_CKPT" ]]; then
  RESUME_ARGS+=(--resume "$RESUME_CKPT")
fi

echo "=== DeepForest cluster train ==="
echo "TRAIN_MODE=$TRAIN_MODE"
echo "CONFIG_NAME=$CONFIG_NAME"
echo "SCENARIO=$SCENARIO"
echo "HOSTNAME=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_NNODES=$NODES"
echo "GPUS_PER_NODE=$DEVICES"
echo "SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE:-?}"
echo "USE_COMET=$USE_COMET"
echo "================================"

TRAIN_ARGS=(
  --config-name="$CONFIG_NAME"
  train
  "${STRATEGY_ARGS[@]}"
  "${COMET_ARGS[@]}"
  "${EXTRA_TRAIN_ARGS[@]}"
  accelerator=gpu
  "devices=$DEVICES"
  "num_nodes=$NODES"
  "log_root=$LOG_ROOT"
  "${RESUME_ARGS[@]}"
  "$@"
)

if [[ "$TRAIN_MODE" == "smoke" ]]; then
  TRAIN_ARGS+=(
    workers=0
    "train.csv_file=$TRAIN_CSV"
    "train.root_dir=$TRAIN_ROOT"
    "validation.csv_file=$VAL_CSV"
    "validation.root_dir=$VAL_ROOT"
  )
fi

if [[ "$SCENARIO" == "1gpu" ]]; then
  echo "Launching training (single GPU)"
  "${DEEPFOREST_CMD[@]}" "${TRAIN_ARGS[@]}"
else
  echo "Launching training via srun (distributed)"
  srun --kill-on-bad-exit=1 --export=ALL --cpu-bind=none \
    "${DEEPFOREST_CMD[@]}" "${TRAIN_ARGS[@]}"
fi

echo "Cluster training finished: mode=$TRAIN_MODE scenario=$SCENARIO"
