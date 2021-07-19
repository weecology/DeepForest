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

## I cannot reload a saved multi-class model using a checkpoint.

DeepForest is a pytorch lightning module. There seems to be ongoing debate on how to best to initialize a new module with new hyperparameters.

[https://github.com/PyTorchLightning/pytorch-lightning/issues/924](https://github.com/PyTorchLightning/pytorch-lightning/issues/924)

One easy work around is to create an object of the same specifications and load the model weights. Careful to use map_location argument if loading on a different device (gpu v cpu).

```
import torch

#Load checkpoint of previous model
ckpt = torch.load("/Users/benweinstein/Documents/EvergladesWadingBird/Zooniverse/species_model/snapshots/species_model.pl", map_location = torch.device("cpu"))

from deepforest import main
m = main.deepforest(num_classes = 6, label_dict = {"a":0,"b":1,"c":2,"d":3,"e":4,"f":5})
m.load_state_dict(ckpt["state_dict"])
```

## Issues

We welcome feedback on both the python package as well as the algorithm performance. Please submit detailed issues to the github repo.

[https://github.com/weecology/DeepForest/issues](https://github.com/weecology/DeepForest/issues)