# Cluster Distributed Runs

This page shows supported patterns for running DeepForest across multiple GPUs and multiple nodes on a Slurm-managed cluster.

## Shared Settings

Use the same launch pattern for `train`, `evaluate`, and `predict`:

- `devices=<gpus_per_node>` is the number of GPUs on each node
- `num_nodes=<nnodes>` is the total number of nodes
- `strategy=ddp` enables distributed data parallel execution (use `auto` for single-GPU jobs)
- `workers=0` is required for large-tile prediction with `dataloader_strategy="window"`

Launch every job step with **`srun`** so Lightning reads the Slurm environment. Set `#SBATCH --ntasks-per-node` equal to `devices` and `#SBATCH --nodes` equal to `num_nodes`.

## Environment

```bash
ml conda
eval "$(conda shell.bash hook)"
conda activate predict
cd /path/to/DeepForest
```

## Train

For a quick distributed smoke test:

```bash
sbatch --nodes=2 --ntasks-per-node=2 --gpus-per-node=2 --cpus-per-task=4 --mem=80G --time=00:30:00 \
  run_cluster_multinode_smoke.sh
```

For a real training run inside an `sbatch` script:

```bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2

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
#SBATCH --gres=gpu:2

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

For the cluster regression test and example launcher:

```bash
sbatch run_cluster_predict_test.sbatch
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
sbatch run_cluster_predict_tile_test.sbatch
```

To run a tiled prediction job directly:

```bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2

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
