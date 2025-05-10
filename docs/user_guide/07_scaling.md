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

While we rarely use multi-node GPU's, pytorch lightning has functionality at very large scales. We welcome users to share what configurations worked best.

A few notes that can trip up those less used to multi-gpu training. These are for the default configurations and may vary on a specific system. We use a large University SLURM cluster with 'ddp' distributed data parallel.

1. Batch-sizes are expressed _per_ _gpu_. If you tell DeepForest you want 2 images per batch and request 5 gpus, you are computing 10 images per forward pass across all GPUs. This is crucial for when profiling, make sure to scale any tests by the batch size!

2. Each device gets its own portion of the dataset. This means that they do not interact during forward passes.

3. Make sure to use srun when combining with SLURM! This is an easy one to miss and will cause training to hang without error. Documented here

https://lightning.ai/docs/pytorch/latest/clouds/cluster_advanced.html#troubleshooting.


## Prediction

Often we have a large number of tiles we want to predict. DeepForest uses [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) to scale inference. This gives us access to powerful tools for scaling without any changes to user code. DeepForest automatically detects whether you are running on GPU or CPU. 

There are three dataset strategies that *balance cpu memory, gpu memory, and gpu utilization* using batch sizes. 

```python
prediction_single = m.predict_tile(path=path, patch_size=300, dataloader_strategy="single")
```
The `dataloader_strategy` parameter has three options:

* **single**: Loads the entire image into CPU memory and passes individual windows to GPU.

* **batch**: Loads the entire image into GPU memory and creates views of the image as batches. Requires the entire tile to fit into GPU memory. CPU parallelization is possible for loading images.

* **window**: Loads only the desired window of the image from the raster dataset. Most memory efficient option, but cannot parallelize across windows due to rasterio's Global Interpreter Lock (GIL), workers must be set to 0. 

## Data Loading

DeepForest uses PyTorch's DataLoader for efficient data loading. One important parameter for scaling is `num_workers`, which controls parallel data loading using multiple CPU processes. This can be set 

```
m.config["workers"] = 10
```
0 workers runs without multiprocessing, workers > 1 runs with multiprocessing. Increase this value slowly, as IO constraints can lead to deadlocks among workers.


