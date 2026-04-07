#!/bin/bash
#SBATCH --job-name=london-replicate
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=4:00:00
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./slurm_logs/%A.out
#SBATCH --error=./slurm_logs/%A.err

printenv | grep -i slurm | sort

export COMET_PROJECT="treeformer-kcl"

srun uv run deepforest --config-name=london train --comet --tag treeformer-replicate "$@"
