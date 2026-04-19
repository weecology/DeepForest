#!/bin/bash
#SBATCH --job-name=london-replicate
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=hpg-b200
#SBATCH --mem=128GB
#SBATCH --time=8:00:00
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./slurm_logs/%A.out
#SBATCH --error=./slurm_logs/%A.err

printenv | grep -i slurm | sort

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export COMET_PROJECT="treeformer-kcl"

srun --kill-on-bad-exit=1 uv run deepforest --config-name=treeformer_kcl train --comet --tag treeformer-replicate "$@"
