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

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set}"
HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$HPC_DIR/run_cluster_train.sh" "$@"
