# Scaling DeepForest using PyTorch Lightning

## Increase batch size

It is more efficient to run a larger batch size on a single GPU. This is because the overhead of loading data and moving data between the CPU and GPU is relatively large. By running a larger batch size, we can reduce the overhead of these operations.

```
m.config["batch_size"] = 16
```

## Training

DeepForest's create_trainer argument passes any argument to pytorch lightning. This means we can use pytorch lightnings amazing distributed training specifications. There is a large amount of documentation, but we find the most helpful section is

https://lightning.ai/docs/pytorch/stable/accelerators/gpu_intermediate.html

For example on a SLURM cluster, we use the following line to get 5 gpus on a single node.
```python
m.create_trainer(logger=comet_logger, accelerator="gpu", strategy="ddp", num_nodes=1, devices=devices)
```

### Complete SLURM Example

Here's a complete example for training on 2 GPUs with SLURM that has been tested and works correctly:

**SLURM submission script (`submit_train.sh`):**
```bash
#!/bin/bash
#SBATCH --job-name=train_model
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus=2
#SBATCH --cpus-per-task=10
#SBATCH --mem=200GB
#SBATCH --time=48:00:00
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err

# Use srun without explicit task/gpu flags - SLURM handles process spawning
# The --ntasks-per-node and --gpus directives above control resource allocation
srun uv run python train_script.py \
    --data_dir /path/to/data \
    --batch_size 24 \
    --workers 10 \
    --epochs 30
```

**Python training script:**
```python
import torch
from deepforest import main

# Initialize model
m = main.deepforest()

# Configure training parameters
m.config["train"]["csv_file"] = "train.csv"
m.config["train"]["root_dir"] = "/path/to/data"
m.config["batch_size"] = 24
m.config["workers"] = 10

# Set devices to total GPU count - PyTorch Lightning DDP will handle
# device assignment per process when used with SLURM process spawning
devices = torch.cuda.device_count() if torch.cuda.is_available() else 0

m.create_trainer(
    logger=comet_logger,
    devices=devices,
    strategy="ddp",  # Distributed Data Parallel
    fast_dev_run=False,
)

# Train the model
m.trainer.fit(m)
```

**Key points:**
- `--ntasks-per-node=2` in SBATCH directives spawns 2 processes (one per GPU)
- `--gpus=2` in SBATCH directives allocates 2 GPUs total
- Use plain `srun` without `--ntasks` or `--gpus-per-task` flags - let SLURM handle process spawning
- Set `devices=torch.cuda.device_count()` in Python (not `devices=1`) - PyTorch Lightning DDP coordinates with SLURM's process management
- Batch size is per-GPU: with `batch_size=24` and 2 GPUs, you get 48 images per forward pass total

While we rarely use multi-node GPU's, pytorch lightning has functionality at very large scales. We welcome users to share what configurations worked best.

A few notes that can trip up those less used to multi-gpu training. These are for the default configurations and may vary on a specific system. We use a large University SLURM cluster with 'ddp' distributed data parallel.

1. Batch-sizes are expressed _per_ _gpu_. If you tell DeepForest you want 2 images per batch and request 5 gpus, you are computing 10 images per forward pass across all GPUs. This is crucial for when profiling, make sure to scale any tests by the batch size!

2. Each device gets its own portion of the dataset. This means that they do not interact during forward passes.

3. Make sure to use `srun` when combining with SLURM! This is critical for proper process spawning and will cause training to hang without error if omitted. Use `--ntasks-per-node` in SBATCH directives (not in srun) to control the number of processes. Documented [here](https://lightning.ai/docs/pytorch/latest/clouds/cluster_advanced.html#troubleshooting).


## Prediction

Often we have a large number of tiles we want to predict. DeepForest uses [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) to scale inference. This gives us access to powerful tools for scaling without any changes to user code. DeepForest automatically detects whether you are running on GPU or CPU.

There are three dataset strategies that *balance cpu memory, gpu memory, and gpu utilization* using batch sizes.

```python
prediction_single = m.predict_tile(path=path, patch_size=300, dataloader_strategy="single")
```
The `dataloader_strategy` parameter has three options:

* **single**: Loads the entire image into CPU memory and passes individual windows to GPU.

* **batch**: Loads the entire image into GPU memory and creates views of the image as batches. Requires the entire tile to fit into GPU memory. CPU parallelization is possible for loading images.

* **window**: Loads only the desired window of the image from the raster dataset. Most memory efficient option, but cannot parallelize across windows due to Python's Global Interpreter Lock, workers must be set to 0.

## Data Loading

DeepForest uses PyTorch's DataLoader for efficient data loading. One important parameter for scaling is the number of CPU workers, which controls parallel data loading using multiple CPU processes. This can be set

```
m.config["workers"] = 10
```
0 workers runs without multiprocessing, workers > 1 runs with multiprocessing. Increase this value slowly, as IO constraints can lead to deadlocks among workers.
