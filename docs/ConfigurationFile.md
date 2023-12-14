# Config

Deepforest uses a config.yml to control hyperparameters related to model training and evaluation. This allows all the relevant parameters to live in one location and be easily changed when exploring new models.

Deepforest comes with a sample config file, deepforest_config.yml. Edit this file to change settings while developing models. Please note that if you would like for deepforest to save the config file on reload (using deepforest.save_model), 
the config.yml must be updated instead of updating the dictionary of an already loaded model.

```
# Config file for DeepForest pytorch module

# Cpu workers for data loaders
# Dataloaders
workers: 1
devices: auto
accelerator: auto
batch_size: 1

# Model Architecture
architecture: 'retinanet'
num_classes: 1
nms_thresh: 0.05

# Architecture specific params
retinanet:
    # Non-max supression of overlapping predictions
    score_thresh: 0.1

train:
    csv_file:
    root_dir:
    
    # Optimizer initial learning rate
    lr: 0.001

    # Print loss every n epochs
    epochs: 1
    # Useful debugging flag in pytorch lightning, set to True to get a single batch of training to test settings.
    fast_dev_run: False
    # pin images to GPU memory for fast training. This depends on GPU size and number of images.
    preload_images: False
    
validation:
    # callback args
    csv_file: 
    root_dir:
    # Intersection over union evaluation
    iou_threshold: 0.4
    val_accuracy_interval: 20
```
## Passing config arguments at runtime using a dict

It can often be useful to pass config args directly to a model instead of editing the config file. By using a dict with that matches with the config keys, main.deepforest will update the config after reading from file.

```
m = main.deepforest()
assert not m.config["num_classes"] == 2

m = main.deepforest(config_args={"num_classes":2}, label_dict={"Alive":0,"Dead":1})
assert m.config["num_classes"] == 2

# These call also be nested for train and val arguments
m = main.deepforest()
assert not m.config["train"]["epochs"] == 7

m = main.deepforest(config_args={"train":{"epochs":7}})
assert m.config["train"]["epochs"] == 7
```

## Dataloaders

### workers

Number of workers to perform asynchronous data generation during model training. Corresponds to num_workers in pytorch [base class](https://pytorch.org/docs/stable/data.html). To turn off asynchronous data generation set workers = 0.

### devices

The number of cpus/gpus to use during model training. Deepforest has been tested on up to 8 gpu and follows a [pytorch lightning module](https://pytorch-lightning.readthedocs.io/en/stable/advanced/multi_gpu.html?highlight=multi%20gpu), which means it can inherit any of the scaling functionality from this library, including TPU support.

### accelerator

Most commonly, `cpu`, `gpu` or `tpu` as well as other [options](https://pytorch-lightning.readthedocs.io/en/1.4.0/advanced/multi_gpu.html) listed:

If `gpu`, it can be helpful to specify the data parallelization strategy. This can be done using the `strategy` arg in `main.create_trainer()`

```
model.create_trainer(logger=comet_logger, strategy="ddp")
```

This is passed to the pytorch-lightning trainer, documented in the link above for multi-gpu training.

### batch_size

Number of images per batch during training. GPU memory limits this usually between 5-10

### nms_thresh

Non-max suppression threshold. The higher scoring predicted box is kept when predictions overlap by greater than nms_thresh. For details see [torchvision docs](https://pytorch.org/vision/stable/ops.html#torchvision.ops.nms)

### score_thresh

Score threshold of predictions to keep. Predictions with less than this threshold are removed from output.
The score threshold can be updated anytime by modifying the config. For example, if you want predictions with boxes greater than 0.3, update the config

```
m.config["score_thresh"] = 0.3
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

```
optim.SGD(self.model.parameters(), lr=self.config["train"]["lr"], momentum=0.9)
```

```
self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', 
                                                   factor=0.1, patience=10, 
                                                   verbose=True, threshold=0.0001, 
                                                   threshold_mode='rel', cooldown=0, 
                                                   min_lr=0, eps=1e-08)
```
This scheduler can be overwritten by replacing the model class

```
m = main.deepforest()
m.scheduler = <>
```

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
