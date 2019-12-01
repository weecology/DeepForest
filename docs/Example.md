# Illustrated Example

The goal of this document is to provide a walkthrough of using DeepForest to build and test RGB deep learning tree detection models.

## Goal

For this example, we would like to build a RGB tree detection model for a section of Yellowstone National Park. Data for this example can be downloaded from the NEON portal under site code [YELL](), along with hundreds of other tiles from the same site.

### Sample Data

#### Training tile

!["../www/2019_YELL_2_528000_4978000_image.png"]

#### Evaluation tile

!["../www/2019_YELL_2_541000_4977000_image.png"]

## Evaluation Data

For this minimal example, we take two 1km tiles. One will be used to train the model, one will be used to test the model. It is important that training and test data be geographically separate, as the model could easily memorize features from the training data. Before training, it is critical to have a set of evaluation data to understand model performance. Using the program RectLabel, let's create a small evaluation dataset.

### Crop the evaluation tile

Object detection models generally require that all target objects be labeled in an image. This means that we should cut the evaluation/training tiles into sections suitable for hand annotation. For this example, these crops will be small.

After about 25 minutes of annotations, the evaluation tile is complete.

![]("../www/annotated_example.png")

The data are currently stored in an xml format.

```{python}
from deepforest import deepforest
from deepforest import utilities
from deepforest import get_data
from deepforest import preprocess

#convert hand annotations from xml into retinanet format
YELL_xml = get_data("2019_YELL_2_541000_4977000_image_crop.xml")
annotation = utilities.xml_to_annotations(YELL_xml)
annotation.head()
#Write converted dataframe to file. Saved alongside the images
annotation.to_csv("deepforest/data/example.csv", index=False)
```

### Evaluation windows

Split the evaluation tile into windows of 400px with an overlap of 5% among windows. The windows will be named <image_name>_0.jpg, <image_name>_1.jpg, etc.

```{python}
YELL_test = get_data("2019_YELL_2_541000_4977000_image_crop.tiff")
cropped_annotations= preprocess.split_raster(path_to_raster=YELL_test,
                                 annotations_file="deepforest/data/example.csv",
                                 base_dir="tests/data",
                                 patch_size=400,
                                 patch_overlap=0.05)

cropped_annotations.head()

#Write window annotations file without a header row
cropped_annotations.to_csv("cropped_example.csv",index=False, header=None)
```

### Evaluate against Prebuilt model

Before training a new model, it is helpful to know the performance of the current benchmark model.

Evaluate prebuilt model

```{python}
test_model = deepforest.deepforest()
test_model.use_release()

annotations_file="/data/example.csv"
mAP = test_model.evaluate_generator(annotations=annotations_file)
print("Mean Average Precision is: {:.3f}".format(mAP))
```

## Custom model

Starting from the prebuilt model weights

### GPU hardware azure data science VM

## Train

### Config file

Training parameters are saved in a "deepforest_config.yml" file. By default DeepForest will look for this file in the current working directory.

```
###
# Config file for DeepForest module
# The following arguments
###

### Training
### Batch size. If multi-gpu is > 1, this is the total number of images per batch across all GPUs. Must be evenly divisible by multi-gpu.
batch_size: 1
### Model weights to load before training. From keras.model.save_weights()
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
random_transform: False

#save snapshot and images
###Whether to save snapshots at the end of each epoch
save-snapshot: False
#Save directory for images and snapshots
save_path: snapshots/
snapshot_path: snapshots/
```

Using these settings, train a new model starting from the release model. For more visualization of model training, comet_ml is an extremely useful platform for understanding machine learning results. There is a free tier for academic audiences. This is optional, but worth considering if you are going to do significant testing.

```{python}
from comet_ml import Experiment
comet_experiment = Experiment(api_key=<api_key>,
                                  project_name=<project>, workspace=<"username">)
from deepforest import deepforest
from deepforest import get_data

#Load the latest release
test_model = deepforest.deepforest()
test_model.use_release()

# Example run with short training
test_model.config["epochs"] = 1
test_model.config["save-snapshot"] = False
test_model.config["steps"] = 1

comet_experiment.log_parameters(deepforest_model.config)

annotations_file = get_data("testfile_deepforest.csv")

test_model.train(annotations=annotations_file, input_type="fit_generator")
```

## Evaluate

```{python}

```

## Predict

```{python}

```
