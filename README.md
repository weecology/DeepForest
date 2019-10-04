# DeepForest

[![Build Status](https://travis-ci.org/Weecology/DeepForest.svg?branch=master)](https://travis-ci.org/Weecology/DeepForest) 
[![Documentation Status](https://readthedocs.org/projects/deepforest/badge/?version=latest)](http://deepforest.readthedocs.io/en/latest/?badge=master)

Python package for training and predicting individual tree crowns in airborne imagery.

## Installation

## Usage

```
git clone https://github.com/weecology/DeepForest.git
```

Depends on keras-retinainet for object detection.

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

## Web Demo

* Free software: MIT license
* Documentation: https://deepforest.readthedocs.io.

## Citation

### Where can I get sample data?

We are organizing a benchmark dataset for individual tree crown prediction in RGB imagery from the National Ecological Observation 
