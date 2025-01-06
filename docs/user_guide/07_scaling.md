# Scaling DeepForest using PyTorch Lightning

Often we have a large number of tiles we want to predict. DeepForest uses [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) to scale inference. This gives us access to powerful tools for scaling without any changes to user code. DeepForest automatically detects whether you are running on GPU or CPU. Within a single GPU node, you can scale training without needing to specify any additional arguments, since we use the ['auto' devices](https://lightning.ai/docs/pytorch/stable/common/trainer.html#devices) detection within PyTorch Lightning. For advanced users, DeepForest can [run across multiple SLURM nodes](https://lightning.ai/docs/pytorch/stable/clouds/cluster_advanced.html), each with multiple GPUs. 

## Increase batch size

It is more efficient to run a larger batch size on a single GPU. This is because the overhead of loading data and moving data between the CPU and GPU is relatively large. By running a larger batch size, we can reduce the overhead of these operations. 

```
m.config["batch_size"] = 16
```

## Scaling inference across multiple GPUs

There are a few situations in which it is useful to replicate the DeepForest module across many separate Python processes. This is especially helpful when we have a series of non-interacting tasks, often called 'embarrassingly parallel' processes. In these cases, no DeepForest instance needs to communicate with another instance. Rather than coordinating GPUs with the associated annoyance of overhead and backend errors, we can just launch separate jobs and let them finish on their own. One helpful tool in Python is [Dask](https://www.dask.org/). Dask is a wonderful open-source tool for coordinating large-scale jobs. Dask can be run locally, across multiple machines, and with an arbitrary set of resources.

### Example Dask and DeepForest integration using SLURM

Imagine we have a list of images we want to predict using `deepforest.main.predict_tile()`. DeepForest does not allow multi-GPU inference within each tile, as it is too much of a headache to make sure the threads return the correct overlapping window. Instead, we can parallelize across tiles, such that each GPU takes a tile and performs an action. The general structure is to create a Dask client across multiple GPUs, submit each DeepForest `predict_tile()` instance, and monitor the results. In this example, we are using a SLURMCluster, a common job scheduler for large clusters. There are many similar ways to create a Dask client object that will be specific to a particular organization. The following arguments are specific to the University of Florida cluster, but will be largely similar to other SLURM naming conventions. We use the extra Dask package, `dask-jobqueue`, which helps format the call.


```
from dask_jobqueue import SLURMCluster
from dask.distributed import Client

cluster = SLURMCluster(processes=1,
                        cores=10,
                        memory="40 GB",
                        walltime='24:00:00',
                        job_extra=extra_args,
                        extra=['--resources gpu=1'],
                        nanny=False,
                        scheduler_options={"dashboard_address": ":8787"},
                        local_directory="/orange/idtrees-collab/tmp/",
                        death_timeout=100)
print(cluster.job_script())
cluster.scale(10)

dask_client = Client(cluster)
```

This job script gets a single GPUs with "40GB" of memory with 10 cpus. We then ask for 10 instances of this setup.
Now that we have a dask client, we can send our custom function. 

```
def function_to_parallelize(tile):
    m = main.deepforest()
    m.load_model("weecology/deepforest-tree") # sub in the custom logic to load your own models
    boxes = m.predict_tile(raster_path=tile)
    # save the predictions using the tile pathname
    filename = "{}.csv".format(os.path.splitext(os.path.basename(tile))[0])
    filename = os.path.join(<savedir>,filename)
    boxes.to_csv(filename)

    return filename
```

```
tiles = [<list of tiles to predict>]
futures = []
for tile in tiles:
    future = client.submit(function_to_parallelize, tile)
    futures.append(future)
```

We can wait to see the futures as they complete! Dask also has a beautiful visualization tool using bokeh. 

```
for x in futures:
    completed_filename = x.result()
    print(completed_filename)
```