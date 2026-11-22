# Multi-species models

DeepForest allows training on multiple species annotations.
When creating a deepforest model object, pass the designed number of classes and a label dictionary that maps each numeric class to a character label. The number of classes can be either be specified in the config, or using config_args during creation.

``` python
from deepforest import main
m = main.deepforest(config_args={"num_classes":2, "label_dict": {"Alive":0,"Dead":1}})
```

It is often, but not always, useful to start with a prebuilt model when trying to identify multiple species. This helps the model focus on learning the multiple classes and not waste data and time re-learning bounding boxes. To load the backbone and box prediction portions of the release model, but create a classification model for more than one species.

Here is an example using the alive/dead tree data stored in the package, but the same logic applies to other detectiors.

``` python
import os
from deepforest import main
from deepforest import get_data

# Initialize new Deepforest model ( the model that you will train ) with your classes.
#
# When you override the number of classes and label_dict, deepforest will
# automatically modify the pretrained model to have the correct classification head
# structure, while keeping the backbone the same.
m = main.deepforest(config_args={"model": {"name": "weecology/deepforest-tree"}, # or 'weecology/deepforest-bird'
                                 "num_classes":2,
                                 "label_dict": {"Alive":0, "Dead":1}})
assert m.model.num_classes == 2

# If you'd prefer to start with generic pretrained weights (typically from MS-COCO):
m = main.deepforest(config_args={"model": {"name": None},
                                 "num_classes":2,
                                 "label_dict": {"Alive":0, "Dead":1}})
assert m.model.num_classes == 2

m.config["train"]["csv_file"] = get_data("testfile_multi.csv")
m.config["train"]["root_dir"] = os.path.dirname(get_data("testfile_multi.csv"))
# Run a single batch to check that training works (see the note below)
m.config["train"]["fast_dev_run"] = True
m.config["batch_size"] = 2

m.config["validation"]["csv_file"] = get_data("testfile_multi.csv")
m.config["validation"]["root_dir"] = os.path.dirname(get_data("testfile_multi.csv"))
m.config["validation"]["val_accuracy_interval"] = 1

m.create_trainer()
m.trainer.fit(m)
```

```{note}
This example only checks that multi-class training runs, so don't expect the trained model to predict anything:

- `fast_dev_run = True` runs a single batch of training and validation and ignores `epochs`.
- `testfile_multi.csv` contains only one image (`SOAP_061.png`, with 28 dead and 9 alive trees).

`predict_image` returns `None` when there are no predictions, so calling `m.predict_image(path=get_data("SOAP_061.png"))` after this example returns `None`.

To train a model that makes predictions, remove the `fast_dev_run` line, set `m.config["train"]["epochs"]` before calling `m.create_trainer()`, and use many more annotated images. As a rough guide, when an earlier version of this example (which reused the tree model's backbone and box regression weights) was trained on this one image, the model still predicted nothing after 200 epochs. By around 500 epochs it was making predictions, but only of dead trees.
```

* For more on loading with state_dict: [Pytorch Docs](https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-state-dict-recommended)
