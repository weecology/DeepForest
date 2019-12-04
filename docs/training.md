# Training new models

Our work has shown that starting training from the prebuilt model increases performance, regardless of the geographic location of your data. In the majority of cases, it will be useful for the model to have learned general tree representations that can be refined using hand annotated data.

## Design evaluation data

In our experience, defining a clear evaluation dataset and setting a threshold for desired performance is critical before training. It is common to just dive into training new data with only a vague sense of the desired outcome. This is always a mistake. We highly suggest users spend the time to answer 2 questions:

* What kind of data am I trying to predict?

Capturing the variability and the broad range of tree taxonomy and presentation will make development go more smoothly.

* What kind of accuracy do I need to answer my question?

It is natural to want the best model possible, but one can waste a tremendous amount of time trying to eek out another 5% of recall without understanding whether that increase in performance will improve our understanding of a given ecological or natural resource question. Prioritize evaluation data that matches your desired outcomes. Don't obsess over small errors, but rather think about how to propagate and capture this uncertainty in the overall analysis. [All models are wrong, some are useful.](https://en.wikipedia.org/wiki/All_models_are_wrong)

## Gather annotations

DeepForest uses xml files produced by the commonly used annotation program RectLabel. Please note that Rectlabel is an inexpensive program available only for Mac.

![](../www/rectlabel.png)

 For annotations made in RectLabel, DeepForest has a parse function ```preprocess.xml_to_annotations```. For non-mac users, there are many alternative for object detection annotation. DeepForest only requires that the final annotations be in the following format.

```
image_path, xmin, ymin, xmax, ymax, label
```

Please note that for functions which are fed into keras-retinanet, such as ```evaluate_generator```, ```predict_generator``` and ```train``` this annotation file should be saved without column names. For ```preprocess.split_raster``` the column names should be maintained.

As with the [evaluation example](Example.html), collect training labels from a crop of the training tile and split into smaller windows.

![](../www/YELL_train.png)

```{python}
from deepforest import deepforest
from deepforest import utilities
from deepforest import preprocess

#convert hand annotations from xml into retinanet format
YELL_xml = get_data("2019_YELL_2_528000_4978000_image_crop2.xml")
annotation = utilities.xml_to_annotations(YELL_xml)
annotation.head()

#Write converted dataframe to file. Saved alongside the images
annotation.to_csv("deepforest/data/train_example.csv", index=False)

#Find data on path
YELL_train = get_data("2019_YELL_2_528000_4978000_image_crop2.tiff")
crop_dir = "tests/data/"
train_annotations= preprocess.split_raster(path_to_raster=YELL_train,
                                 annotations_file="deepforest/data/train_example.csv",
                                 base_dir=crop_dir,
                                 patch_size=400,
                                 patch_overlap=0.05)
#View output
train_annotations.head()

#Write window annotations file without a header row, same location as the "base_dir" above.
annotations_file= crop_dir + "train_example.csv"
train_annotations.to_csv(annotations_file,index=False, header=None)
```

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

Using these settings, train a new model starting from the release model. We use are very small number of epochs since this is a sample toy dataset.

```{python}
#Load the latest release
test_model = deepforest.deepforest()
test_model.use_release()

# Example run with short training
test_model.config["epochs"] = 1
test_model.config["save-snapshot"] = False
test_model.train(annotations=annotations_file, input_type="fit_generator")
```
* Estimated training time on CPU: ~ 6 min/epoch

* Training time on GPU: 14 sec/epoch

#### Comet visualization

For more visualization of model training, comet_ml is an extremely useful platform for understanding machine learning results. There is a free tier for academic audiences. This is optional, but worth considering if you are going to do significant testing.

```{python}
from comet_ml import Experiment
comet_experiment = Experiment(api_key=<api_key>,
                                  project_name=<project>, workspace=<"username">)

comet_experiment.log_parameters(deepforest_model.config)

test_model.train(annotations=annotations_file, input_type="fit_generator",comet_experiment=comet_experiment)
```

## Training accuracy

Find the training accuracy of the model. We expect this value to be high (mAP > 0.5), if not, consider training for additional epochs.

```{python}
mAP = test_model.evaluate_generator(annotations=annotations_file)
print("Mean Average Precision is: {:.3f}".format(mAP))
```

returns

```
There are 1 unique labels: ['Tree']
Disabling snapshot saving
Running network: 100% (12 of 12) |#######################################################################################################################################################################################################| Elapsed Time: 0:00:02 Time:  0:00:02
Parsing annotations: 100% (12 of 12) |###################################################################################################################################################################################################| Elapsed Time: 0:00:00 Time:  0:00:00
431 instances of class Tree with average precision: 0.5076
mAP using the weighted average of precisions among classes: 0.5076
mAP: 0.5076
```
