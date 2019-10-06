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

## Usage

```{python}
from deepforest import deepforest
from deepforest import utilities

#Download latest model release from github
utilities.use_release()    

#Load model class with release weights
test_model = deepforest.deepforest(weights="data/universal_model_july30.h5")

#predict image
image = test_model.predict_image(image_path = "tests/data/OSBS_029.tif")
```

![www/test_image.png]

## Web Demo

Thanks to Microsoft AI4Earth grant for hosting a azure web demo of the trained model.

http://tree.westus.cloudapp.azure.com/shiny/

## License
* Free software: MIT license
* Documentation: https://deepforest.readthedocs.io.

## Citation

Geographic Generalization in Airborne RGB Deep Learning Tree Detection
Ben Weinstein, Sergio Marconi, Stephanie Bohlman, Alina Zare, Ethan P White
bioRxiv 790071; doi: https://doi.org/10.1101/790071

### Where can I get sample data?

We are organizing a benchmark dataset for individual tree crown prediction in RGB imagery from the National Ecological Observation

https://github.com/weecology/NeonTreeEvaluation


