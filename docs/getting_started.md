# Getting started

Here is a simple example of how to predict a single image.

```
from deepforest import deepforest
import matplotlib.pyplot as plt
model = deepforest.deepforest()
model.use_release()
img=model.predict_image("/Users/ben/Documents/NeonTreeEvaluation/evaluation/RGB/JERC_019.tif",return_plot=True)

#predict_image returns plot in BlueGreenRed (opencv style), but matplotlib likes RedGreenBlue, switch the channel order.
plt.imshow(img[:,:,::-1])
```

## Prebuilt models

DeepForest has a prebuilt model trained on data from 24 sites from the [National Ecological Observation Network](https://www.neonscience.org/field-sites/field-sites-map). The prebuilt model uses a semi-supervised approach in which millions of moderate quality annotations are generated using a LiDAR unsupervised tree detection algorithm, followed by hand-annotations of RGB imagery from select sites.

![](../www/semi-supervised.png)
For more details on the modeling approach see [citations](landing.html#citation).

Setting the correct window size to match the prebuilt model takes a few tries. The model was trained on 0.1m data with 400m crops. For data of the same resolution, that window size is appropriate. For coarser data, we have experimentally found that larger windows are actually more useful in providing the model context (e.g 1500px windows). At some point windows become too large and the trees are too tiny to classify. Striking a balance is important.

## Sample data

DeepForest comes with a small set of sample data to help run the docs examples. Since users may install in a variety of manners, and it is impossible to know the relative location of the files, the helper function ```get_data``` is used. This function looks to where DeepForest is installed, and finds the deepforest/data/ directory.

```python
import deepforest

YELL_xml = deepforest.get_data("2019_YELL_2_541000_4977000_image_crop.xml")
```

For more complete examples of DeepForest applications, see the DeepForest demo [repo](https://github.com/weecology/DeepForest_demos/).

## Prediction

DeepForest allows convenient prediction of new data based on the prebuilt model or a [custom trained](getting_started.html#Training) model. There are three ways to format data for prediction.

### Predict a single image

For single images, ```predict_image``` can read an image from memory or file and return predicted tree bounding boxes.

```python
from deepforest import deepforest
from deepforest import get_data

test_model = deepforest.deepforest()
test_model.use_release()

# Predict test image and return boxes
# Find path to test image. While it lives in deepforest/data,
# it is best to use the function if installed as a python module.
# For non-tutorial images, you do not need the get_data function,
# just provide the full path to the data anywhere on your computer.
image_path = get_data("OSBS_029.tif")
boxes = test_model.predict_image(image_path=image_path, show=False, return_plot = False)

boxes.head()
```

```
xmin        ymin        xmax        ymax     score label
0  222.136353  211.271133  253.000061  245.222580  0.790797  Tree
1   52.070221   73.605804   82.522354  111.510605  0.781306  Tree
2   96.324028  117.811966  123.224060  145.982407  0.778245  Tree
3  336.983826  347.946747  375.369019  396.250580  0.677282  Tree
4  247.689362   48.813339  279.102570   88.318176  0.675362  Tree
```

### Predict a tile

Large tiles covering wide geographic extents cannot fit into memory during prediction and would yield poor results due to the density of bounding boxes. Often provided as geospatial .tif files, remote sensing data is best suited for the ```predict_tile``` function, which splits the tile into overlapping windows, perform prediction on each of the windows, and then reassembles the resulting annotations.

Let's show an example with a small image. For larger images, patch_size should be increased.

```python
from deepforest import deepforest
from deepforest import get_data

test_model = deepforest.deepforest()
test_model.use_release()

# Find the tutorial data using the get data function.
# For non-tutorial images, you do not need the get_data function,
# provide the full path to the data anywhere on your computer.

raster_path = get_data("OSBS_029.tif")
# Window size of 300px with an overlap of 25% among windows for this small tile.
predicted_raster = test_model.predict_tile(raster_path, return_plot = True, patch_size=300,patch_overlap=0.25)
```

** Please note the predict tile function is sensitive to patch_size, especially when using the prebuilt model on new data**

We encourage users to try out a variety of patch sizes. For 0.1m data, 400-800px per window is appropriate, but it will depend on the density of tree plots. For coarser resolution tiles, >800px patch sizes have been effective, but we welcome feedback from users using a variety of spatial resolutions.

### Predict a set of annotations

During evaluation of ground truth data, it is useful to have a way to predict a set of images and combine them into a single data frame. The ```predict_generator``` method allows a user to point towards a file of annotations and returns the predictions for all images.

Consider a headerless annotations.csv file in the following format

```
image_path, xmin, ymin, xmax, ymax, label
```
with each bounding box on a seperate row. The image path is relative to the local of the annotations file.

We can view predictions by supplying a save dir ("." = current directory). Predictions in green, annotations in black.

```python
from deepforest import deepforest
from deepforest import get_data

test_model = deepforest.deepforest()
test_model.use_release()

# Find the tutorial csv file. For non-tutorial images, you do not need the get_data function, jut provide the full path to the data anywhere on your computer.
annotations_file = get_data("testfile_deepforest.csv")

test_model.config["save_dir"] = "."
boxes = test_model.predict_generator(annotations=annotations_file)
```

For more information on data files, see below.

## Training

The prebuilt models will always be improved by adding data from the target area. In our work, we have found that even one hour's worth of carefully chosen hand-annotation can yield enormous improvements in accuracy and precision. We envision that for the majority of scientific applications atleast some finetuning of the prebuilt model will be worthwhile. When starting from the prebuilt model for training, we have found that 5-10 epochs is sufficient. We have never seen a retraining task that improved after 10 epochs, but it is possible if there are very large datasets with very diverse classes.

Consider an annotations.csv file in the following format

```
image_path, xmin, ymin, xmax, ymax, label
```

testfile_deepforest.csv

```
OSBS_029.jpg,256,99,288,140,Tree
OSBS_029.jpg,166,253,225,304,Tree
OSBS_029.jpg,365,2,400,27,Tree
OSBS_029.jpg,312,13,349,47,Tree
OSBS_029.jpg,365,21,400,70,Tree
OSBS_029.jpg,278,1,312,37,Tree
OSBS_029.jpg,364,204,400,246,Tree
OSBS_029.jpg,90,117,121,145,Tree
OSBS_029.jpg,115,109,150,152,Tree
OSBS_029.jpg,161,155,199,191,Tree
```

and a classes.csv file in the same directory

```
Tree,0
```

```python
from deepforest import deepforest
from deepforest import get_data

test_model = deepforest.deepforest()

# Example run with short training
test_model.config["epochs"] = 1
test_model.config["save-snapshot"] = False
test_model.config["steps"] = 1

annotations_file = get_data("testfile_deepforest.csv")

test_model.train(annotations=annotations_file, input_type="fit_generator")
```

```python
No model initialized, either train or load an existing retinanet model
There are 1 unique labels: ['Tree']
Disabling snapshot saving

Training retinanet with the following args ['--backbone', 'resnet50', '--image-min-side', '800', '--multi-gpu', '1', '--epochs', '1', '--steps', '1', '--batch-size', '1', '--tensorboard-dir', 'None', '--workers', '1', '--max-queue-size', '10', '--freeze-layers', '0', '--score-threshold', '0.05', '--save-path', 'snapshots/', '--snapshot-path', 'snapshots/', '--no-snapshots', 'csv', 'data/testfile_deepforest.csv', 'data/classes.csv']

Creating model, this may take a second...

... [omitting model summary]

Epoch 1/1

1/1 [==============================] - 11s 11s/step - loss: 4.0183 - regression_loss: 2.8889 - classification_loss: 1.1294
```

## Evaluation

Independent analysis of whether a model can generalize from training data to new areas is critical for creating a robust workflow. We stress that evaluation data must be different from training data, as neural networks have millions of parameters and can easily memorize thousands of samples. Therefore, while it would be rather easy to tune the model to get extremely high scores on the training data, it would fail when exposed to new images.

DeepForest uses the keras-retinanet ```evaluate``` method to score images. This consists of an annotations.csv file in the following format

```
image_path, xmin, ymin, xmax, ymax, label
```

```python
from deepforest import deepforest
from deepforest import get_data

test_model = deepforest.deepforest()
test_model.use_release()

annotations_file = get_data("testfile_deepforest.csv")
mAP = test_model.evaluate_generator(annotations=annotations_file)
print("Mean Average Precision is: {:.3f}".format(mAP))
```

```
Running network: 100% (1 of 1) |#########| Elapsed Time: 0:00:02 Time:  0:00:02
Parsing annotations: N/A% (0 of 1) |     | Elapsed Time: 0:00:00 ETA:  --:--:--
Parsing annotations: 100% (1 of 1) |#####| Elapsed Time: 0:00:00 Time:  0:00:00
60 instances of class Tree with average precision: 0.3687
mAP using the weighted average of precisions among classes: 0.3687
mAP: 0.3687
```

The evaluation file can also be run as a callback during training by setting the config file. This will allow the user to see evaluation performance at the end of each epoch.

```python
test_model.config["validation_annotations"] = "testfile_deepforest.csv"
```

## Loading saved models for prediction

DeepForest uses the keras saving workflow, which means users can save the entire model architecture, or just the weights. For more explanation on Keras see [here](https://stackoverflow.com/questions/42621864/difference-between-keras-model-save-and-model-save-weights).

### Saved Model
To access the training model for saving, use

```python
test_model.model.save("example_saved.h5")
```

which can be reloaded

```
reloaded = deepforest.deepforest(saved_model="example_saved.h5")
Reading config file: deepforest_config.yml
Loading saved model
```

The actual prediction model can be accessed

```python
reloaded.prediction_model
<keras.engine.training.Model object at 0x646ca3b70>
```

but in most cases it is better to just use the deepforest workflow functions such as predict_image.

### Model Weights

If you just want to weights of the model layer, you can do.

```python
test_model.model.save_weights("example_save_weights.h5")
reloaded = deepforest.deepforest(weights="example_save_weights.h5")
```

Now you can use

```python
reloaded.predict_image()
```

as described above.
