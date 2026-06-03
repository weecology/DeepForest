# Cluster Distributed Runs

This page shows supported patterns for running DeepForest across multiple GPUs and multiple nodes on a Slurm-managed cluster (for example HiPerGator).

## Slurm: `sbatch` and `srun`

`sbatch` requests the allocation (nodes, GPUs, tasks, memory, time). `srun` inside that batch script starts a **job step** within the same allocation. It does **not** submit a second job or double-charge the scheduler.

Match `#SBATCH --ntasks-per-node` to `devices` (one Slurm task per GPU) and `#SBATCH --nodes` to `num_nodes`. For multi-GPU DDP, launch with `srun`. For a single GPU, the cluster train script runs the command directly in the batch step.

Example launchers live under `src/deepforest/scripts/HPC/`.

## Shared Settings

Use the same launch pattern for `train`, `evaluate`, and `predict`:

- `devices=<gpus_per_node>` is the number of GPUs on each node
- `num_nodes=<nnodes>` is the total number of nodes
- `strategy=ddp` enables distributed data parallel execution (use `auto` for single-GPU jobs)
- `workers=0` is required for large-tile prediction with `dataloader_strategy="window"`

## Environment

```bash
ml conda
eval "$(conda shell.bash hook)"
conda activate predict
cd /path/to/DeepForest
mkdir -p slurm_logs
```

## Train

Use `src/deepforest/scripts/HPC/run_cluster_train.sbatch` for production training and smoke tests. The launcher script is `run_cluster_train.sh`.

### Production training (single GPU)

Defaults use `TRAIN_MODE=train` and `CONFIG_NAME=bird`. Submit from the repo root:

```bash
sbatch src/deepforest/scripts/HPC/run_cluster_train.sbatch
```

Hydra overrides and resume:

```bash
export COMET_EXPERIMENT_NAME="exp_lr_0.0005"
sbatch src/deepforest/scripts/HPC/run_cluster_train.sbatch train.lr=0.0005 train.epochs=80

RESUME_CKPT=/path/to/last.ckpt sbatch src/deepforest/scripts/HPC/run_cluster_train.sbatch
```

Multi-GPU or multi-node training: set Slurm resources at submit time and pass matching Hydra settings if needed. The script infers `SCENARIO` from the allocation.

```bash
sbatch --nodes=2 --ntasks-per-node=2 --gpus-per-node=2 --cpus-per-task=8 --mem=128G --time=15:00:00 \
  src/deepforest/scripts/HPC/run_cluster_train.sbatch \
  --strategy ddp devices=2 num_nodes=2
```

### Smoke tests

Smoke tests use bundled OSBS sample data (`TRAIN_MODE=smoke`, `CONFIG_NAME=smoke`, 1 epoch). Set `SCENARIO` and match `#SBATCH` resources:

```bash
# 1 GPU
TRAIN_MODE=smoke SCENARIO=1gpu sbatch --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 \
  --cpus-per-task=8 --mem=32G --time=00:30:00 \
  src/deepforest/scripts/HPC/run_cluster_train.sbatch

# Multi-GPU (one node)
TRAIN_MODE=smoke SCENARIO=multigpu GPUS_PER_NODE=2 sbatch --nodes=1 --ntasks-per-node=2 --gpus-per-node=2 \
  --cpus-per-task=8 --mem=64G --time=00:45:00 \
  src/deepforest/scripts/HPC/run_cluster_train.sbatch

# Multi-node
TRAIN_MODE=smoke SCENARIO=multinode GPUS_PER_NODE=2 NNODES=2 sbatch --nodes=2 --ntasks-per-node=2 --gpus-per-node=2 \
  --cpus-per-task=8 --mem=64G --time=01:00:00 \
  src/deepforest/scripts/HPC/run_cluster_train.sbatch
```

Optional: `export COMET_EXPERIMENT_NAME="my-smoke-run"` before `sbatch`. Disable Comet with `USE_COMET=0`.

### Train directly in a batch script

```bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2

srun uv run deepforest train \
  --strategy ddp \
  accelerator=gpu \
  devices=2 \
  num_nodes=2 \
  train.csv_file=/path/to/train.csv \
  train.root_dir=/path/to/train_images \
  validation.csv_file=/path/to/val.csv \
  validation.root_dir=/path/to/val_images
```

## Evaluate

```bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2

srun uv run deepforest evaluate \
  /path/to/ground_truth.csv \
  --root-dir /path/to/images \
  --save-predictions eval_preds.csv \
  -o eval_metrics.csv \
  --strategy ddp \
  accelerator=gpu \
  devices=2 \
  num_nodes=2
```

## Predict From CSV

For the cluster regression test and example launcher (submit from the repo root):

```bash
sbatch src/deepforest/scripts/HPC/run_cluster_predict_test.sbatch
```

To run your own CSV prediction job directly:

```bash
srun uv run deepforest predict \
  /path/to/images.csv \
  --mode csv \
  --root-dir /path/to/images \
  -o predictions.csv \
  --strategy ddp \
  accelerator=gpu \
  devices=2 \
  num_nodes=2
```

## Predict A Large Tile

For large rasters on a cluster, prefer `predict_tile(..., dataloader_strategy="window")`.

The ready-to-run test launcher is:

```bash
sbatch src/deepforest/scripts/HPC/run_cluster_predict_tile_test.sbatch
```

To run a tiled prediction job directly:

```bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2

srun uv run python tests/cluster_predict_tile_driver.py \
  --input-path /path/to/tile.tif \
  --output-path tile_predictions.csv \
  --model-name weecology/everglades-bird-species-detector \
  --patch-size 1500 \
  --patch-overlap 0 \
  --dataloader-strategy window \
  --devices 2 \
  --num-nodes 2
```

See also the [multi-GPU and multi-node guide](../user_guide/distributed.md).
