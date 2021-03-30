# FAQ

## I cannot reload a saved multi-class model using a checkpoint.

DeepForest-pytorch is a pytorch lightning module. There seems to be ongoing debate on how to best to initialize a new module with new hyperparameters. 

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

[https://github.com/weecology/DeepForest-pytorch/issues](https://github.com/weecology/DeepForest-pytorch/issues)