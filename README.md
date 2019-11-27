# DeepForest

[![Build Status](https://travis-ci.org/Weecology/DeepForest.svg?branch=master)](https://travis-ci.org/Weecology/DeepForest)
[![Documentation Status](https://readthedocs.org/projects/deepforest/badge/?version=latest)](http://deepforest.readthedocs.io/en/latest/?badge=latest)

Python package for training and predicting individual tree crowns in airborne imagery.

## Installation

```
git clone https://github.com/weecology/DeepForest.git
```

This package depends on keras-retinainet for object detection.

```
git clone https://github.com/fizyr/keras-retinanet.git
cd keras-retinanet
pip install .
python setup.py build_ext --inplace
```

### Python dependencies

DeepForest uses conda as a packgae manager.

```
conda env create --file=environment.yml
```

## Documentation

https://deepforest.readthedocs.io.

## Usage

### Prediction

Using DeepForest, users can predict individual tree crowns by loading pre-built models and applying them to RGB images.

Currently there is 1 prebuilt model, "NEON", which was trained using a semi-supervised process from imagery from the National Ecological Observation Network.
For more information on the pre-built models see [citations](https://github.com/weecology/DeepForest#citation).

```{python}
import matplotlib.pyplot as plt
from deepforest import deepforest
from deepforest import utilities

#Download latest model release from github
utilities.use_release()    

#Load model class with release weights
test_model = deepforest.deepforest(weights="data/NEON.h5")

#predict image
image = test_model.predict_image(image_path = "tests/data/OSBS_029.tif")

#Show image, matplotlib expects RGB channel order, but keras-retinanet predicts in BGR
plt.imshow(image[...,::-1])
```

![test image](www/image.png)

## Training

DeepForest allows training through a keras-retinanet CSV generator. Input files must be formatted, without a header, in the following format:

```
image_path, xmin, ymin, xmax, ymax, label
```

Training config parameters are stored in deepforest_config.yml. They also can be changed at runtime.

```{python}
#Load model class
test_model = deepforest.deepforest()

#Change config
test_model.config["epochs"] = 1
test_model.config["save-snapshot"] = False
test_model.config["steps"] = 1

#Train
test_model.train(annotations="tests/data/testfile_deepforest.csv")

#save
test_model.model.save("snapshots/final_model.h5")
```

DeepForest is developed using comet_ml dashboards for training visualization. Simply pass a comet_experiment object to train to log metrics and performance. See more at www.comet.ml  

## Web Demo

Thanks to Microsoft AI4Earth grant for hosting a azure web demo of the trained model.

http://tree.westus.cloudapp.azure.com/shiny/

## License
* Free software: [MIT license](https://github.com/weecology/DeepForest/blob/master/LICENSE)

## Citation

[Weinstein, B.G.; Marconi, S.; Bohlman, S.; Zare, A.; White, E. Individual Tree-Crown Detection in RGB Imagery Using Semi-Supervised Deep Learning Neural Networks.
Remote Sens. 2019, 11, 1309](https://www.mdpi.com/2072-4292/11/11/1309)

[Geographic Generalization in Airborne RGB Deep Learning Tree Detection Ben Weinstein, Sergio Marconi, Stephanie Bohlman, Alina Zare, Ethan P White
bioRxiv 790071; doi: https://doi.org/10.1101/790071](https://www.biorxiv.org/content/10.1101/790071v1.abstract)

### Where can I get sample data?

We are organizing a benchmark dataset for individual tree crown prediction in RGB imagery from the National Ecological Observation

https://github.com/weecology/NeonTreeEvaluation
