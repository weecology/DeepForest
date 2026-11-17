#!/usr/bin/env bash
# Shared helpers for DeepForest Slurm cluster integration tests.
# shellcheck shell=bash

cluster_repo_root() {
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        cd "${SLURM_SUBMIT_DIR}"
        pwd
        return
    fi
    # This file lives at src/deepforest/scripts/HPC/_cluster_common.sh
    local common_dir
    common_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "${common_dir}/../../../.."
    pwd
}

cluster_activate_conda() {
    local module_name="${CONDA_MODULE:-conda}"
    local env_name="${CONDA_ENV_NAME:-mypredict}"

    ml "${module_name}"
    eval "$(conda shell.bash hook)"
    conda activate "${env_name}"
    echo "CONDA_MODULE: ${module_name}"
    echo "CONDA_ENV_NAME: ${env_name}"
}

cluster_export_pythonpath() {
    local repo_root="$1"
    export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:$PYTHONPATH}"
}

cluster_export_hf_home() {
    export HF_HOME="${HF_HOME:-/blue/ewhite/henrysenyondo/.cache/huggingface}"
}

cluster_require_file() {
    local path="$1"
    local label="${2:-File}"
    if [[ ! -f "${path}" ]]; then
        echo "${label} not found: ${path}"
        exit 1
    fi
}

cluster_require_dir() {
    local path="$1"
    local label="${2:-Directory}"
    if [[ ! -d "${path}" ]]; then
        echo "${label} not found: ${path}"
        exit 1
    fi
}
