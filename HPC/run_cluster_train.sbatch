#!/usr/bin/env bash
#SBATCH --job-name=deepforest-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --time=15:00:00
#SBATCH --output=./slurm_logs/train-%j.out
#SBATCH --error=./slurm_logs/train-%j.err
#SBATCH --signal=SIGUSR1@360

set -euo pipefail

# Slurm copies this .sbatch into the spool dir, so BASH_SOURCE is not the repo.
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set}"
bash "${SLURM_SUBMIT_DIR}/HPC/run_cluster_train.sh" "$@"
