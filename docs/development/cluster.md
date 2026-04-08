# Cluster Distributed Runs

This page shows the shortest supported patterns for running DeepForest across multiple GPUs and multiple nodes on a Slurm-managed cluster.

## Shared Settings

Use the same launch pattern for `train`, `evaluate`, and `predict`:

- `devices=<gpus_per_node>` is the number of GPUs on each node
- `num_nodes=<nnodes>` is the total number of nodes
- `strategy=ddp` enables distributed data parallel execution
- `workers=0` is required for large-tile prediction with `dataloader_strategy="window"`

## Environment

```bash
ml conda
eval "$(conda shell.bash hook)"
conda activate predict
cd /blue/ewhite/everglades/cluster_deepforest/DeepForest
```

## Train

For a quick distributed smoke test, use the helper script:

```bash
salloc --time=05:30:00 --nodes=2 --ntasks-per-node=1 --gpus-per-node=2 --cpus-per-task=4 --mem=80G
GPUS_PER_NODE=2 NNODES=2 srun --nodes=2 --ntasks=2 --ntasks-per-node=1 bash run_cluster_multinode_smoke.sh
```

For a real training run, replace the final command with:

```bash
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export NNODES="${SLURM_NNODES}"
export GPUS_PER_NODE=2

srun --nodes="$NNODES" --ntasks="$NNODES" --ntasks-per-node=1 bash -lc '
ml conda
eval "$(conda shell.bash hook)"
conda activate predict
cd /blue/ewhite/everglades/cluster_deepforest/DeepForest
uv run torchrun \
  --nnodes='"$NNODES"' \
  --nproc_per_node='"$GPUS_PER_NODE"' \
  --node_rank=$SLURM_NODEID \
  --master_addr='"$MASTER_ADDR"' \
  --master_port='"$MASTER_PORT"' \
  -m deepforest.scripts.cli train \
  accelerator=gpu \
  devices='"$GPUS_PER_NODE"' \
  num_nodes='"$NNODES"' \
  strategy=ddp \
  train.csv_file=/path/to/train.csv \
  train.root_dir=/path/to/train_images \
  validation.csv_file=/path/to/val.csv \
  validation.root_dir=/path/to/val_images
'
```

## Evaluate

```bash
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export NNODES="${SLURM_NNODES}"
export GPUS_PER_NODE=2

srun --nodes="$NNODES" --ntasks="$NNODES" --ntasks-per-node=1 bash -lc '
ml conda
eval "$(conda shell.bash hook)"
conda activate predict
cd /blue/ewhite/everglades/cluster_deepforest/DeepForest
uv run torchrun \
  --nnodes='"$NNODES"' \
  --nproc_per_node='"$GPUS_PER_NODE"' \
  --node_rank=$SLURM_NODEID \
  --master_addr='"$MASTER_ADDR"' \
  --master_port='"$MASTER_PORT"' \
  -m deepforest.scripts.cli evaluate \
  /path/to/ground_truth.csv \
  --root-dir /path/to/images \
  --save-predictions eval_preds.csv \
  -o eval_metrics.csv \
  accelerator=gpu \
  devices='"$GPUS_PER_NODE"' \
  num_nodes='"$NNODES"' \
  strategy=ddp
'
```

## Predict From CSV

For the cluster regression test and example launcher, use:

```bash
sbatch run_cluster_predict_test.sbatch
```

To run your own CSV prediction job directly, use the same `torchrun` pattern as evaluation and replace the final module call with:

```bash
-m deepforest.scripts.cli predict /path/to/images.csv \
  --mode csv \
  --root-dir /path/to/images \
  -o predictions.csv \
  accelerator=gpu \
  devices=2 \
  num_nodes=2 \
  strategy=ddp
```

## Predict A Large Tile

For large rasters on a cluster, prefer `predict_tile(..., dataloader_strategy="window")`.

The ready-to-run test launcher is:

```bash
sbatch run_cluster_predict_tile_test.sbatch
```

To run a real tiled prediction job directly, launch the driver with `torchrun`:

```bash
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export NNODES="${SLURM_NNODES}"
export GPUS_PER_NODE=2

srun --nodes="$NNODES" --ntasks="$NNODES" --ntasks-per-node=1 bash -lc '
ml conda
eval "$(conda shell.bash hook)"
conda activate predict
cd /blue/ewhite/everglades/cluster_deepforest/DeepForest
uv run torchrun \
  --nnodes='"$NNODES"' \
  --nproc_per_node='"$GPUS_PER_NODE"' \
  --node_rank=$SLURM_NODEID \
  --master_addr='"$MASTER_ADDR"' \
  --master_port='"$MASTER_PORT"' \
  tests/cluster_predict_tile_driver.py \
  --input-path /path/to/tile.tif \
  --output-path tile_predictions.csv \
  --model-name weecology/everglades-bird-species-detector \
  --patch-size 1500 \
  --patch-overlap 0 \
  --dataloader-strategy window \
  --devices '"$GPUS_PER_NODE"' \
  --num-nodes '"$NNODES"'
'
```
