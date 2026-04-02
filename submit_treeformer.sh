#!/bin/bash
# Submit a linked pretrain + finetune job pair under a shared experiment name.
#
# Usage:
#   ./submit_treeformer.sh <experiment-name> [pretrain_overrides...] [-- finetune_overrides...]
#
# Examples:
#   ./submit_treeformer.sh run_003
#   ./submit_treeformer.sh run_003 -- keypoint.count_cls_weight=0.1
#   ./submit_treeformer.sh run_003 train.epochs=30 -- keypoint.log_count_loss=false

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <experiment-name> [pretrain_overrides...] [-- finetune_overrides...]" >&2
    exit 1
fi

NAME="$1"; shift

PRETRAIN_ARGS=()
FINETUNE_ARGS=()
IN_FINETUNE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --) IN_FINETUNE=1; shift ;;
        *)
            if [[ "${IN_FINETUNE}" -eq 1 ]]; then
                FINETUNE_ARGS+=("$1")
            else
                PRETRAIN_ARGS+=("$1")
            fi
            shift ;;
    esac
done

HF_MODEL="logs/treeformer/${NAME}_pretrain/hf_model"

echo "Submitting pretrain: ${NAME}_pretrain"
PRETRAIN_JOB=$(sbatch --parsable \
    treeformer_pretrain.slurm \
    --experiment-name "${NAME}_pretrain" \
    "${PRETRAIN_ARGS[@]+"${PRETRAIN_ARGS[@]}"}")
echo "  Job ID: ${PRETRAIN_JOB}"

echo "Submitting finetune: ${NAME}_finetune (depends on ${PRETRAIN_JOB})"
FINETUNE_JOB=$(sbatch --parsable \
    --dependency=afterok:"${PRETRAIN_JOB}" \
    treeformer_finetune.slurm \
    --experiment-name "${NAME}_finetune" \
    model.name="${HF_MODEL}" \
    "${FINETUNE_ARGS[@]+"${FINETUNE_ARGS[@]}"}")
echo "  Job ID: ${FINETUNE_JOB}"

echo ""
echo "  slurm_logs/${PRETRAIN_JOB}.out  (pretrain)"
echo "  slurm_logs/${FINETUNE_JOB}.out  (finetune)"
echo "  HF model: ${HF_MODEL}"
