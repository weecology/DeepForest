---
name: Model Performance Issue
about: Questions on model performance and accuracy
title: ''
labels: Performance
assignees: ''

---

## Description
Please provide a concise explanation of the problems you are experiencing.

## Target area
Please provide a sample image or screenshot of the type of trees you are trying to predict.

## Have you tried annotating local data and retraining?

The DeepForest prebuilt model is a good starting point for many trees. Adding a small amount of local data will almost always improve performance. 

* How many trees did you annotate?
* Did you remember to label ALL trees in an image?
* Did you confirm the annotations match the image?

Please provide code samples on model training and development that show the deepforest config settings for any model training.

For example
```
reloaded = deepforest.deepforest(weights="example_save_weights.h5")
Reading config file: deepforest_config.yml
reloaded.config
{'batch_size': 1, 'weights': 'None', 'backbone': 'resnet50', 'image-min-side': 800, 'multi-gpu': 1, 'epochs': 1, 'validation_annotations': 'None', 'freeze_layers': 0, 'freeze_resnet': False, 'score_threshold': 0.05, 'multiprocessing': False, 'workers': 1, 'max_queue_size': 10, 'random_transform': False, 'save-snapshot': False, 'save_path': 'snapshots/', 'snapshot_path': 'snapshots/'}
```

## For coarser resolution data, have you tried varying the patch size?

preprocess.split_raster(patch_size=....)

The DeepForest prebuilt model was built using 0.1m data on 400pixel windows. We have found that for coarser resolution data, increasing the patch size is useful.

Any and all code samples, screenshots and sample images will greatly help debugging. We welcome annotations to be submitted to the DeepForest contrib models to allow other users to benefit from your work.
