# Getting Started

## Prebuilt models

DeepForest has a prebuilt model trained on data from 24 sites from the National Ecological Observation Network (https://www.neonscience.org/field-sites/field-sites-map). The prebuilt model uses a semi-supervised approach in which millions of moderate quality annotations are generated using a LiDAR unsupervised tree detection algorithm, followed by hand-annotations of RGB imagery from select sites. For more details on the modeling approach see

Weinstein, B.G.; Marconi, S.; Bohlman, S.; Zare, A.; White, E. Individual Tree-Crown Detection in RGB Imagery Using Semi-Supervised Deep Learning Neural Networks. Remote Sens. 2019, 11, 1309.
https://www.mdpi.com/2072-4292/11/11/1309

Geographic Generalization in Airborne RGB Deep Learning Tree Detection
Ben. G. Weinstein, Sergio Marconi, Stephanie A. Bohlman, Alina Zare, Ethan P. White
bioRxiv 790071; doi: https://doi.org/10.1101/790071

## Prediction

DeepForest allows convenient prediction to new data based on the pre-built model or a trained model described below. There are three ways to format data for prediction.


### Predict a single image

For single images, predict_image can directly read an image from file and return predicted tree bounding boxes.

```{python}
from deepforest import deepforest

test_model = deepforest.deepforest()
test_model.use_release()

#Predict test image and return boxes
boxes = test_model.predict_image(image_path="tests/data/OSBS_029.tif", show=False, return_plot = False)

boxes.head()
         xmin        ymin        xmax        ymax     score label
0  222.136353  211.271133  253.000061  245.222580  0.790797  Tree
1   52.070221   73.605804   82.522354  111.510605  0.781306  Tree
2   96.324028  117.811966  123.224060  145.982407  0.778245  Tree
3  336.983826  347.946747  375.369019  396.250580  0.677282  Tree
4  247.689362   48.813339  279.102570   88.318176  0.675362  Tree
```

### Predict a tile
Large tiles covering wide geographic extents cannot fit into memory during prediction, and would yield poor results. DeepForest has a ```predict_tile``` function to split the image into overlapping windows, perform prediction on each of the windows, and reassemble the resulting annotations.

### Predict a set of images

Consider an annotations.csv file in the following format

```
image_path, xmin, ymin, xmax, ymax, label
```

## Training

The prebuilt models will always be improved by adding data from the target area. In our work, we have found that even one hour's worth of carefully chosen hand-annotation can yield enormous improvements in accuracy and precision. We envision that for the majority of scientific applications atleast some finetuning of the prebuilt model will be worthwhile.

Consider an annotations.csv file in the following format

```
image_path, xmin, ymin, xmax, ymax, label
```

testfile_deepforest.csv

```
head testfile_deepforest.csv
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

```{python}
from deepforest import deepforest
test_model = deepforest.deepforest()

# Example run with short training
test_model.config["epochs"] = 1
test_model.config["save-snapshot"] = False
test_model.config["steps"] = 1

test_model.train(annotations="data/testfile_deepforest.csv", input_type="fit_generator")

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

Independent analysis of whether a model can generalize from training data to new areas is critical for creating a robust model. We stress that evaluation data must be different from training data, as neural networks have millions of parameters and can easily memorize thousands of samples. Therefore, while it would be rather easy to tune the model to get extremely high scores on the training data, it would fail when exposed to new images.

DeepForest uses the keras-retinanet ```evaluate``` method to score images. This consists of an annotations.csv file in the following format

```
image_path, xmin, ymin, xmax, ymax, label
```

```{python}


```

For more on evaluation, see the [Evaluation Overview]()

## NEON Benchmark

To standardize model evaluation, we have collected and published a benchmark dataset of nearly 20,000 crowns from sites in the National Ecological Observation Network.

https://github.com/weecology/NeonTreeEvaluation
