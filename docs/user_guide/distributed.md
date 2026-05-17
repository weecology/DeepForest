# Multi-GPU and Multi-Node Runs

DeepForest uses PyTorch Lightning distributed execution. For most multi-GPU and multi-node runs, these settings matter:

- `accelerator=gpu`
- `devices=<gpus_per_node>`
- `num_nodes=<nnodes>`
- `strategy=ddp`

On Slurm clusters, launch with **`srun`** inside your job allocation. Match `#SBATCH --ntasks-per-node` to `devices` and `#SBATCH --nodes` to `num_nodes`. See the [cluster developer guide](../development/cluster.md).

Single-GPU jobs can keep the default `strategy=auto`.

## Train

```bash
#SBATCH --nodes=<nnodes>
#SBATCH --ntasks-per-node=<gpus_per_node>
#SBATCH --gres=gpu:<gpus_per_node>

srun uv run deepforest train \
  --strategy ddp \
  accelerator=gpu \
  devices=<gpus_per_node> \
  num_nodes=<nnodes> \
  train.csv_file=/path/to/train.csv \
  train.root_dir=/path/to/train_images \
  validation.csv_file=/path/to/val.csv \
  validation.root_dir=/path/to/val_images
```

## Evaluate

```bash
srun uv run deepforest evaluate \
  /path/to/ground_truth.csv \
  --root-dir /path/to/images \
  --save-predictions eval_preds.csv \
  -o eval_metrics.csv \
  --strategy ddp \
  accelerator=gpu \
  devices=<gpus_per_node> \
  num_nodes=<nnodes>
```

## Predict From CSV

```bash
srun uv run deepforest predict \
  /path/to/images.csv \
  --mode csv \
  --root-dir /path/to/images \
  -o predictions.csv \
  --strategy ddp \
  accelerator=gpu \
  devices=<gpus_per_node> \
  num_nodes=<nnodes>
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

Launch that script with the same `srun` Slurm pattern and trainer settings. For a complete cluster example, see the [cluster developer guide](../development/cluster.md).
