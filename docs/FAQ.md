# FAQ

## Exception: RuntimeError: The size of tensor a (4) must match the size of tensor b (3) at non-singleton dimension 0

This error is usually caused by the user accidentily adding a 4 channel image to a 3 channel RGB model. Sometimes .png or .jpeg images are saved with a 'alpha' channel controlling their transparency. This needs to be removed.

```
import rasterio as rio
import numpy as np
import PIL as Image

src = rio.open("/orange/ewhite/everglades/Palmyra/palmyra.tif")
numpy_image = src.read()
numpy_image = np.moveaxis(numpy_image,0,2)

#just select first three bands
numpy_image = numpy_image[:,:,:3].astype("uint8")
image = Image.fromarray(numpy_image)
image.save(name)
```

## How do I make training faster?

While it is impossible to anticipate the setup for all users, there are a few guidelines. First, a GPU-enabled processor is key. Training on a CPU can be done, but it will take much longer (100x) and is probably only done if needed. Using Google Colab can be beneficial but prone to errors. Once on the GPU, the configuration includes a "workers" argument. This connects to PyTorch's dataloader. As the number of workers increases, data is fed to the GPU in parallel. Increase the worker argument slowly, we have found that the optimal number of workers varies by system.

```
m.config["workers"] = 5
```

It is not foolproof, and occasionally 0 workers, in which data loading is run on the main thread, is optimal : https://stackoverflow.com/questions/73331758/can-ideal-num-workers-for-a-large-dataset-in-pytorch-be-0.

For large training runs, setting preload_images to True can be helpful. 

```
m.config["preload_images"] = True
```

This will load all data into GPU memory once, at the beginning of the run. This is great, but it requires you to have enough memory space to do so.
Similarly, increasing the batch size can speed up training. Like both of the options above, we have seen examples where performance (and accuracy) improves and decreases depending on batch size. Track experiment results carefully when altering batch size, since it directly [effects the speed of learning](https://www.baeldung.com/cs/learning-rate-batch-size).

```
m.config["batch_size"] = 10
```

```
import torch

#Load checkpoint of previous model
ckpt = torch.load("/Users/benweinstein/Documents/EvergladesWadingBird/Zooniverse/species_model/snapshots/species_model.pl", map_location = torch.device("cpu"))

from deepforest import main
m = main.deepforest(num_classes = 6, label_dict = {"a":0,"b":1,"c":2,"d":3,"e":4,"f":5})
m.load_state_dict(ckpt["state_dict"])
```

## Weakly referenced objects

On some devices and systems we have found an error referencing the model.trainer object that was created in m.create_trainer(). 
We welcome a reproducible issue to address this error as it appears highly variable and relates to upstream issues. It appears more common on google colab and github actions.

In most cases, this error appears when running multiple calls to model.predict or model.train. We believe this occurs because garbage collection has deleted the model.trainer object see:
https://github.com/Lightning-AI/lightning/issues/12233
https://github.com/weecology/DeepForest/issues/338

If you run into this error, users (e.g https://github.com/weecology/DeepForest/issues/443), have found that creating the trainer object within the loop can resolve this issue.

```
for tile in tiles_to_predict:
    m.create_trainer()
    m.predict_tile(tile)
```
Usually creating this object does not cost too much computational time.

### Saving

We have rarely heard that this appears on save:
```
model.save_model("mymodel.pl")
Weakly-reference object no longer exists
```

In this case, just saving the torch model state dict is an easy fix.

```
torch.save(model.model.state_dict(),model_path)
```

and restore

```
model = main.deepforest()
model.model.load_state_dict(torch.load(model_path))
```

### Training

We have heard that this error can appear when trying to deep copy the pytorch lighnting module. The trainer object is not pickleable.
For example, on multi-gpu enviroments when trying to scale the deepforest model the entire module is copied leading to this error.
Setting the trainer object to None and directly using the pytorch object is a reasonable workaround. 

Replace

```
m = main.deepforest()
m.create_trainer()
m.trainer.fit(m)
```

with

```
m.trainer = None
from pytorch_lightning import Trainer

    trainer = Trainer(
        accelerator="gpu",
        strategy="ddp",
        devices=model.config["devices"],
        enable_checkpointing=False,
        max_epochs=model.config["train"]["epochs"],
        logger=comet_logger
    )
trainer.fit(m)
```
The added benefits of this is more control over the trainer object. 
The downside is that it doesn't align with the .config pattern where a user now has to look into the config to create the trainer. 
We are open to changing this to be the default pattern in the future and welcome input from users.

## How do I reduce double counting in overlapping images?

If you have geospatial data for each image this is straightforward. 
Here is a colab link example to project the predictions from image coordinates into geospatial coordinates and then apply non-max suppression.

[https://colab.research.google.com/drive/1T4HC7i5zqe_9AX0pEZSSSzo6BFPFTgFK?usp=sharing](https://colab.research.google.com/drive/1T4HC7i5zqe_9AX0pEZSSSzo6BFPFTgFK?usp=sharing)

## Issues

We welcome feedback on both the python package as well as the algorithm performance. Please submit detailed issues to the github repo.

[https://github.com/weecology/DeepForest/issues](https://github.com/weecology/DeepForest/issues)
