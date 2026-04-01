#!/bin/bash
#SBATCH --job-name=Treeformer
#SBATCH --nodes=1
#SBATCH --partition=hpg-b200
#SBATCH --cpus-per-task=8
#SBATCH --mem=196GB
#SBATCH --time=24:00:00
#SBATCH --gpus=4
#SBATCH --output=./slurm_logs/%A.out
#SBATCH --error=./slurm_logs/%A.err
#SBATCH --ntasks-per-node=4

printenv | grep -i slurm | sort

CONFIG_NAME=treeformer_pretrain
export COMET_PROJECT="treeformer-deepforest"

srun uv run deepforest --config-name="${CONFIG_NAME}" train --comet --verbose --tag treeformer --tag pretrain --strategy ddp_find_unused_parameters_true train.csv_file=/blue/ewhite/veitchmichaelisj/deeplidar/output/folds/all_annotations_train_fold_0.csv train.root_dir=/blue/ewhite/veitchmichaelisj/deeplidar/output validation.csv_file=/blue/ewhite/veitchmichaelisj/deeplidar/output/folds/all_annotations_val_fold_0.csv validation.root_dir=/blue/ewhite/veitchmichaelisj/deeplidar/output "$@" 