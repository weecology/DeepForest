#!/bin/bash
#SBATCH --job-name=Treeformer-finetune
#SBATCH --nodes=1
#SBATCH --partition=hpg-b200
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=00:30:00
#SBATCH --gpus=1
#SBATCH --output=./slurm_logs/%A.out
#SBATCH --error=./slurm_logs/%A.err
#SBATCH --ntasks-per-node=1

printenv | grep -i slurm | sort

CONFIG_NAME=treeformer_finetune
export COMET_PROJECT="treeformer-deepforest"

srun uv run deepforest --config-name="${CONFIG_NAME}" train --comet --verbose --tag treeformer --tag finetune "$@"
