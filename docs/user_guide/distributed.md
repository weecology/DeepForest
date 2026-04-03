# Multi-GPU and Multi-Node Runs

DeepForest uses PyTorch Lightning distributed execution. For most multi-GPU and multi-node runs, the same four settings matter:

- `accelerator=gpu`
- `devices=<gpus_per_node>`
- `num_nodes=<nnodes>`
- `strategy=ddp`

If you are launching with Slurm on Hipergator, see the [Hipergator developer guide](../development/hipergator.md) for cluster-ready examples.

## Train

```bash
torchrun \
  --nnodes=<nnodes> \
  --nproc_per_node=<gpus_per_node> \
  --node_rank=<node_rank> \
  --master_addr=<master_addr> \
  --master_port=29500 \
  -m deepforest.scripts.cli train \
  accelerator=gpu \
  devices=<gpus_per_node> \
  num_nodes=<nnodes> \
  strategy=ddp \
  train.csv_file=/path/to/train.csv \
  train.root_dir=/path/to/train_images \
  validation.csv_file=/path/to/val.csv \
  validation.root_dir=/path/to/val_images
```

## Evaluate

```bash
torchrun \
  --nnodes=<nnodes> \
  --nproc_per_node=<gpus_per_node> \
  --node_rank=<node_rank> \
  --master_addr=<master_addr> \
  --master_port=29500 \
  -m deepforest.scripts.cli evaluate \
  /path/to/ground_truth.csv \
  --root-dir /path/to/images \
  --save-predictions eval_preds.csv \
  -o eval_metrics.csv \
  accelerator=gpu \
  devices=<gpus_per_node> \
  num_nodes=<nnodes> \
  strategy=ddp
```

## Predict From CSV

```bash
torchrun \
  --nnodes=<nnodes> \
  --nproc_per_node=<gpus_per_node> \
  --node_rank=<node_rank> \
  --master_addr=<master_addr> \
  --master_port=29500 \
  -m deepforest.scripts.cli predict \
  /path/to/images.csv \
  --mode csv \
  --root-dir /path/to/images \
  -o predictions.csv \
  accelerator=gpu \
  devices=<gpus_per_node> \
  num_nodes=<nnodes> \
  strategy=ddp
```

## Predict A Large Tile

For large geospatial rasters, use `predict_tile(..., dataloader_strategy="window")` instead of the simple CLI tile mode.

```python
from deepforest.main import deepforest

m = deepforest()
m.load_model("weecology/everglades-bird-species-detector")
m.config.accelerator = "gpu"
m.config.devices = 2
m.config.num_nodes = 2
m.config.strategy = "ddp"
m.config.workers = 0
m.create_trainer()

results = m.predict_tile(
    path="/path/to/tile.tif",
    patch_size=1500,
    patch_overlap=0,
    dataloader_strategy="window",
)
```

Launch that script with the same `torchrun` pattern shown above. For a complete Slurm example, refer to the [Hipergator developer guide](../development/hipergator.md).
