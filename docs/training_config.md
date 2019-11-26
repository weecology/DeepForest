# Configuration File

For ease of experimentation, DeepForest reads the majority of training parameters from a .yml file. This allows a user to quickly survey and change the training settings without needing to dive into the source code. Deep learning models are complex, and DeepForest tries to set reasonable defaults when possible. To get the best performance, some parameter exploration will be required for most novel applications. To track  Experiments can be tracked using a [comet_ml](comet.ml) dashboard.

## Sample deepforest_config.yml

```
###
# Config file for DeepForest module
###

### Training
### Batch size. If multi-gpu is > 1, this is the total number of images per batch across all GPUs. Must be evenly divisible by multi-gpu.
batch_size: 1
### Model weights to load before training. From keras.model.save()
weights: None
### Retinanet backbone. See the keras-retinanet repo for options. Only resnet50 has been well explored.
backbone: resnet50
### Resize images to min size. Retinanet anchors may need to be remade if signficantly reducing image size.
image-min-side: 800
##Number of GPUs to train
multi-gpu: 1
#Number of full cycles of the input data to train
epochs: 1
#Validation annotations. If training using fit_generator, these will be evaluated as a callback at the end of each epoch.
validation_annotations: None
###Freeze layers. Used for model finetuning, freeze the bottom n layers.
freeze_layers: 0
###Freeze resnet backbone entirely.
freeze_resnet: False

###Evaluation
###Score threshold, above which bounding boxes are included in evaluation predictions
score_threshold: 0.05

#Keras fit_generator methods, these do not apply to tfrecords input_type
multiprocessing: False
workers: 1
max_queue_size: 10
random_transform: True

#save snapshot and images
###Whether to save snapshots at the end of each epoch
save-snapshot: False
#Save directory for images and snapshots
save_path: snapshots/
snapshot_path: snapshots/
```

## Training parameters

These parameters effect training specifications.

### batch_size: 1

Neural networks are often trained in batches of images, since the entire dataset is too large to read into memory at once. The size of these batches effects both the speed of training (larger batches train faster) and the stability of training (larger batches lead to more consistent results). The default batch_size of 1 is chosen because it is not possible to anticipate the available memory. Increase where possible. Typically batch sizes are evenly divisible by the size of the entire dataset.

** Note on batch_size: If using ```multi-gpu > 1```, batch_size refers to the total number of samples across all GPUs. For example, a ```batch_size: 18``` for ```multi-gpu: 3``` would produces 3 batches of 6 images, one set for each GPU. batch_size must therefore be larger than ```multi-gpu``` and evenly divisible.

### weights: None

Neural networks consist of a set of matrix weights that are updated during model training. Starting from scratch with randomly initialized weights can significantly slow down training and decrease model performance. The ```weights:``` parameter allows you to start from previously saved weights, either from prebuilt models or from a custom session.

Saving example:
```


```

### backbone: resnet50

This is the keras retinanet backbone documented:

DeepForest has only been tested on resnet50 backbone.

### image-min-side: 800

Object detection algorithms, such as retinanets, work by specifying anchor boxes to predict objects. This means that the scale and size of boxes should be consistent between training and prediction. To ensure that the bounding boxes fit the image, all images are resized before training and prediction. For example, ```image-min-side: 800``` means that the smallest input image side would be resized to 800 pixels. This is a sensitive parameter and one that may need evaluation when training new data. Images that are too small will miss small trees, such they are outside the expected box size. Images that are too big will not fit in memory.

### multi-gpu: 1

Number of GPUs to run. Ignored if running on CPU, which is automatically detected by tensorflow. As multi-gpu increases, pay close attention to the batch_size, which must be evenly divisible and greater or equal to the number of GPUs. Multi-gpu will yield significant training gains for very large training datasets. See training.md##-Training Hardware for more discussion.

### epochs: 1

Number of times to show each image during training. At the end of an epoch, model checkpoint will save current weights if ```save-snapshot: True```. It is difficult to anticipate a default setting form epochs, which will vary heavily on whether model is being trained from scratch (more epochs), is very similar to target prediction data (more epochs) or is likely to overfit (fewer epochs).

### freeze_layers: 0

Current deprecated. Option to freeze a portion of convolutional layers of the model supplied in the ```weights:``` parameter.

### freeze_resnet: False

Whether to allow training on the resnet background. Advanced feature for fine-tuning. If the target evaluation data is similar to the data in the ```weights: <path_to_file.h5>```, option to turn off classifiction training. Learn more about [finetuning](https://flyyufelix.github.io/2016/10/03/fine-tuning-in-keras-part1.html), and the [retinanet architecture](https://towardsdatascience.com/object-detection-on-aerial-imagery-using-retinanet-626130ba2203)

### score_threshold: 0.05

Each bounding box comes with a probability score. Higher scores equates to more confidence in the bounding box label and extent. The ```score_threshold``` instructs keras-retinanet to ignore boxes below the threshold during model evaluation. For trees, we have found that these scores tend to be quite low compared to more conventional cases.

## Keras fit_generator methods

These methods are not often changed unless experimenting with increasing training speed.

### multiprocessing: False

Turn multiprocessing on (True) or off (False) during keras fit_generator methods. See

[https://keras.io/models/sequential/#fit_generator]

### workers: 1

Number of parallel workers in fit_generator

[https://keras.io/models/sequential/#fit_generator]

### max_queue_size: 10

Number of images to queue in fit_generator

[https://keras.io/models/sequential/#fit_generator]

### random_transform: True

Data augmentations following keras-retinanet. In the fit_generator method, each annotation can be augmented. May be useful for small datasets.

See [the retinanet repo](https://github.com/fizyr/keras-retinanet/blob/5524619f91699732ba24c6f52fb9e4b0b780b019/keras_retinanet/utils/transform.py#L27) for source of transformations.

## Validation

### validation_annotations: None

At the end of each epoch, DeepForest can evaluate a file of annotations and return the mean average precision. ```validation_annotations: <path_to_file.h5>``` should point to a headerless .csv file in the following format

```
image_path, xmin, ymin, xmax, ymax, label
```

For more information on formatting data see training.md##-Gather-annotations

### save-snapshot: False

Whether to save a snapshot of model weights at the end of each epoch. This is useful for restarting training or saving a prediction model.

see [https://keras.io/callbacks/#ModelCheckpoint]

### save_path: snapshots/

Path on disk to save images if validation_annotations is not ```None```.

### snapshot_path: snapshots/

Path on disk to save snapshots if ```save-snapshot: True```
