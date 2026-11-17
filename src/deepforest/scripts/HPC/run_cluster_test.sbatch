#!/usr/bin/env bash
# Shared Slurm entry for cluster pytest cases.
#
# Default resources fit small 2-GPU validation cases. Override at submit time:
#   sbatch --export=ALL,CASE=validation_empty_rank \
#     src/deepforest/scripts/HPC/run_cluster_test.sbatch
#   sbatch --time=00:20:00 --export=ALL,CASE=validation_empty_rank \
#     src/deepforest/scripts/HPC/run_cluster_test.sbatch
#
#SBATCH --job-name=deepforest-cluster-test
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00

set -euo pipefail

HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASE="${CASE:-validation_empty_rank}"
export CASE

bash "${HPC_DIR}/run_cluster_test.sh"
