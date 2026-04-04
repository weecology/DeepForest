#!/bin/bash
#SBATCH --job-name=Treeformer-pretrain
#SBATCH --nodes=1
#SBATCH --partition=hpg-b200
#SBATCH --cpus-per-task=8
#SBATCH --mem=196GB
#SBATCH --time=12:00:00
#SBATCH --gpus=4
#SBATCH --output=./slurm_logs/%A.out
#SBATCH --error=./slurm_logs/%A.err
#SBATCH --ntasks-per-node=4

printenv | grep -i slurm | sort

export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO

CONFIG_NAME=treeformer_pretrain
export COMET_PROJECT="treeformer-deepforest"

srun uv run deepforest --config-name="${CONFIG_NAME}" train --comet --verbose --export-hf --tag treeformer --tag pretrain --strategy ddp_find_unused_parameters_true "$@"