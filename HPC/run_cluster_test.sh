#!/usr/bin/env bash
# Shared launcher for DeepForest cluster pytest cases.
#
# Usage (batch):
#   sbatch --export=ALL,CASE=validation_empty_rank HPC/run_cluster_test.sbatch
#
# Usage (interactive allocation):
#   CASE=validation_empty_rank bash HPC/run_cluster_test.sh
#
# Add a new case by extending the CASE table below and adding a matching
# tests/test_cluster_*.py file. Keep site paths and GPU counts in this
# launcher; keep asserts in pytest.
set -euo pipefail

HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${HPC_DIR}/_cluster_common.sh"

CASE="${CASE:-validation_empty_rank}"
REPO_ROOT="$(cluster_repo_root)"
cd "${REPO_ROOT}"

export RUN_CLUSTER_TESTS="${RUN_CLUSTER_TESTS:-1}"
cluster_export_hf_home
cluster_export_pythonpath "${REPO_ROOT}"

DEFAULT_HPC_ROOT="/blue/ewhite/everglades/cluster_deepforest/hpc_test_data"
GPUS_PER_NODE="${DEEPFOREST_HPC_GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-2}}}"
export DEEPFOREST_HPC_GPUS_PER_NODE="${GPUS_PER_NODE}"

PYTEST_TARGET=""
JOB_DESC=""
REQUIRED_FILES=()
REQUIRED_DIRS=()

case "${CASE}" in
    validation_empty_rank)
        JOB_DESC="empty-rank validation"
        CONDA_ENV_NAME="${CONDA_ENV_NAME:-mypredict}"
        export DEEPFOREST_HPC_VALIDATION_CSV="${DEEPFOREST_HPC_VALIDATION_CSV:-${DEFAULT_HPC_ROOT}/treeA_blank.csv}"
        export DEEPFOREST_HPC_ROOT_DIR="${DEEPFOREST_HPC_ROOT_DIR:-${DEFAULT_HPC_ROOT}}"
        PYTEST_TARGET="tests/test_cluster_validation_empty_rank.py"
        REQUIRED_FILES+=("${DEEPFOREST_HPC_VALIDATION_CSV}")
        REQUIRED_DIRS+=("${DEEPFOREST_HPC_ROOT_DIR}")
        ;;
    *)
        echo "Unknown CASE: ${CASE}"
        echo "Known cases: validation_empty_rank"
        exit 1
        ;;
esac

cluster_activate_conda

echo "Running cluster case: ${CASE} (${JOB_DESC})"
echo "REPO_ROOT: ${REPO_ROOT}"
echo "PYTEST_TARGET: ${PYTEST_TARGET}"
echo "GPUS_PER_NODE: ${DEEPFOREST_HPC_GPUS_PER_NODE}"
if [[ -n "${DEEPFOREST_HPC_VALIDATION_CSV:-}" ]]; then
    echo "VALIDATION_CSV: ${DEEPFOREST_HPC_VALIDATION_CSV}"
fi
if [[ -n "${DEEPFOREST_HPC_ROOT_DIR:-}" ]]; then
    echo "ROOT_DIR: ${DEEPFOREST_HPC_ROOT_DIR}"
fi

for path in "${REQUIRED_FILES[@]}"; do
    cluster_require_file "${path}"
done
for path in "${REQUIRED_DIRS[@]}"; do
    cluster_require_dir "${path}"
done

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "SLURM_JOB_ID is not set. Submit with sbatch or run inside an allocation."
    exit 1
fi

srun --ntasks-per-node="${DEEPFOREST_HPC_GPUS_PER_NODE}" \
    python -m pytest -o addopts="" -m cluster \
    "${PYTEST_TARGET}" -s
