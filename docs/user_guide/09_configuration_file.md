# Config

DeepForest uses a config.yaml file to control hyperparameters related to model training and evaluation. This allows all the relevant parameters to live in one location and be easily changed when exploring new models. This file is packaged with DeepForest is loaded via the OmegaConf library when you create a `deepforest` instance.

DeepForest includes a default sample config file named config.yaml (at `src/conf/config.yaml`). Users have the option to override this file by creating their own custom config file. OmegaConf will look in this folder first, but if you provide a path to a config, that one will be overlaid on top of the default. This means you don't need to specify every parameter, but it is best practice to copy the full configuration.

Please note that if you would like for deepforest to save the config file on reload (using deepforest.save_model), the config.yaml must be updated instead of updating the dictionary of an already loaded model.

```yaml
# Config file for DeepForest pytorch module

# Cpu workers for data loaders
# Dataloaders
workers: 0
devices: auto
accelerator: auto
batch_size: 1

# Model Architecture
architecture: 'retinanet'
num_classes: 1
nms_thresh: 0.05
score_thresh: 0.1

# Set model name to None to initialize from scratch
model:
    name: 'weecology/deepforest-tree'
    revision: 'main'

# If this label dict is specified, and it differs
# from the model downloaded from the hub, the model
# will be updated to reflect the new class list.
label_dict:
    Tree: 0

# Pre-processing parameters
path_to_raster:
patch_size: 400
patch_overlap: 0.05
annotations_xml:
rgb_dir:
path_to_rgb:

train:
    csv_file:
    root_dir:

    # Optimizer initial learning rate
    lr: 0.001
    scheduler:
        type:
        params:
            # Common parameters
            T_max: 10
            eta_min: 0.00001
            lr_lambda: "0.95 ** epoch"  # For lambdaLR and multiplicativeLR
            step_size: 30  # For stepLR
            gamma: 0.1  # For stepLR, multistepLR, and exponentialLR
            milestones: [50, 100]  # For multistepLR

            # ReduceLROnPlateau parameters (used if type is not explicitly mentioned)
            mode: "min"
            factor: 0.1
            patience: 10
            threshold: 0.0001
            threshold_mode: "rel"
            cooldown: 0
            min_lr: 0
            eps: 0.00000001

    # How many epochs to run for
    epochs: 1
    # Useful debugging flag in pytorch lightning, set to True to get a single batch of training to test settings.
    fast_dev_run: False
    # preload images to GPU memory for fast training. This depends on GPU size and number of images.
    preload_images: False

validation:
    csv_file:
    root_dir:
    preload_images: False
    size:

    # For retinanet you may prefer val_classification, but the default val_loss
    # should work with all models
    lr_plateau_target: val_loss

    # Intersection over union evaluation
    iou_threshold: 0.4
    val_accuracy_interval: 20

predict:
    pin_memory: False

```
## Passing config arguments at runtime using a dict

It can often be useful to pass config args directly to a model instead of editing the config file. By using a dict containing the config keys and their values. Values provided in this dict will override values provided in config.yaml.

```python
from deepforest import main

# Default model has 1 class
m = main.deepforest()
print(m.config.num_classes)

# But we can override using config args, make sure to specify a new label dict.
m = main.deepforest(config_args={"num_classes":2}, label_dict={"Alive":0,"Dead":1})
print(m.config.num_classes)

# These can also be nested for train and val arguments
m = main.deepforest(config_args={"train":{"epochs":7}})
print(m.config.train.epochs)
```

## Dataloaders

### workers

Number of workers to perform asynchronous data generation during model training. Corresponds to num_workers in pytorch [base class](https://pytorch.org/docs/stable/data.html). To turn off asynchronous data generation set workers = 0.

### devices

The number of cpus/gpus to use during model training. Deepforest has been tested on up to 8 gpu and follows a [pytorch lightning module](https://pytorch-lightning.readthedocs.io/en/stable/advanced/multi_gpu.html?highlight=multi%20gpu), which means it can inherit any of the scaling functionality from this library, including TPU support.

### accelerator

Most commonly, `cpu`, `gpu` or `tpu` as well as other [options](https://pytorch-lightning.readthedocs.io/en/1.4.0/advanced/multi_gpu.html) listed:

If `gpu`, it can be helpful to specify the data parallelization strategy. This can be done using the `strategy` arg in `main.create_trainer()`

```python
from deepforest import model as m

m.create_trainer(logger=comet_logger, strategy="ddp")
```

This is passed to the pytorch-lightning trainer, documented in the link above for multi-gpu training.

### batch_size

Number of images per batch during training. GPU memory limits this usually between 5-10

### nms_thresh

Non-max suppression threshold. The higher scoring predicted box is kept when predictions overlap by greater than nms_thresh. For details see [torchvision docs](https://pytorch.org/vision/stable/ops.html#torchvision.ops.nms)

### score_thresh

Score threshold of predictions to keep. Predictions with less than this threshold are removed from output.
The score threshold can be updated anytime by modifying the config. For example, if you want predictions with boxes greater than 0.3, update the config

```python
m.config.score_thresh = 0.3
```

This will be updated when you can predict_tile, predict_image, predict_file, or evaluate

## Train

### csv_file

Path to csv_file for training annotations. Annotations are `.csv` files with headers `image_path, xmin, ymin, xmax, ymax, label`. image_path are relative to the root_dir.
For example this file should have entries like `myimage.tif` not `/path/to/myimage.tif`

### root_dir

Directory to search for images in the csv_file image_path column

### lr

Learning rate for the training optimization. By default the optimizer is stochastic gradient descent with momentum. A learning rate scheduler is used based on validation loss

```python
from torch import optim

optim.SGD(self.model.parameters(), lr=self.config.train.lr, momentum=0.9)
```

A learning rate scheduler is used to adjust the learning rate based on validation loss. The default scheduler is ReduceLROnPlateau:

```python
import torch

self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                   factor=0.1, patience=10,
                                                   verbose=True, threshold=0.0001,
                                                   threshold_mode='rel', cooldown=0,
                                                   min_lr=0, eps=1e-08)
```
This default scheduler can be overridden by specifying a different scheduler in the config_args:

```
scheduler_config = {
    "type": "cosine",  # or "lambdaLR", "multiplicativeLR", "stepLR", "multistepLR", "exponentialLR", "reduceLROnPlateau"
    "params": {
        # Scheduler-specific parameters
    }
}

config_args = {
    "train": {
        "lr": 0.01,
        "scheduler": scheduler_config,
        "csv_file": "path/to/annotations.csv",
        "root_dir": "path/to/root_dir",
        "fast_dev_run": False,
        "epochs": 2
    },
    "validation": {
        "csv_file": "path/to/annotations.csv",
        "root_dir": "path/to/root_dir"
    }
}
```

The scheduler types supported are:

- **cosine**: CosineAnnealingLR
- **lambdaLR**: LambdaLR
- **multiplicativeLR**: MultiplicativeLR
- **stepLR**: StepLR
- **multistepLR**: MultiStepLR
- **exponentialLR**: ExponentialLR
- **reduceLROnPlateau**: ReduceLROnPlateau

### Epochs

The number of times to run a full pass of the dataloader during model training.

### fast_dev_run

A useful pytorch lightning flag that will run a debug run to test inputs. See [pytorch lightning docs](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html?highlight=fast_dev_run#fast-dev-run)

### preload_images

For large training runs, the time spent reading each image and passing it to the GPU can be a significant performance bottleneck.

If the training dataset is small enough to fit into GPU memory, pinning the entire dataset to memory before training will increase training speed. Warning, if the pinned memory is too large, the GPU will overflow/core dump and training will crash.

## Validation

Optional validation dataloader to run during training.

### csv_file

Path to csv_file for validation annotations. Annotations are `.csv` files with headers `image_path, xmin, ymin, xmax, ymax, label`. image_path are relative to the root_dir.
For example this file should have entries like `myimage.tif` not `/path/to/myimage.tif`

### root_dir

Directory to search for images in the csv_file image_path column

### val_accuracy_interval

Compute and log the classification accuracy of the predicted results computed every X epochs.
This incurs some reductions in speed of training and is most useful for multi-class models. To deactivate, set to an number larger than epochs.
