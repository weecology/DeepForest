# Individual tree-crown detection in RGB imagery using semi-supervised deep learning neural networks 

Ben. G. Weinstein, Sergio Marconi, Stephanie Bohlman, Alina Zare, Ethan White

## Abstract
Remote sensing can transform the speed, scale, and cost of biodiversity and forestry surveys. Data acquisition currently outpaces the ability to identify individual organisms in high resolution imagery. We outline an approach for identifying tree-crowns in RGB imagery using a semi-supervised deep learning detection network. Individual crown delineation has been a long-standing challenge in remote sensing and available algorithms produce mixed results. We show that deep learning models can leverage existing lidar-based unsupervised delineation to generate trees used for training an initial RGB crown detection model. Despite limitations in the original unsupervised detection approach, this noisy training data may contain information from which the neural network can learn initial tree features. We then refine the initial model using a small number of higher-quality hand-annotated RGB images. We validate our proposed approach using an open-canopy site in the National Ecological Observation Network. Our results show that a model using 434,551 self-generated trees with the addition of 2,848 hand-annotated trees yields accurate predictions in natural landscapes. Using an intersection-over-union threshold of 0.5, the full model had an average tree crown recall of 0.69, with a precision of 0.61 for visually-annotated data. The model had an average tree detection rate of 0.82 for field collected stems. The addition of a small number of hand-annotated trees improved performance over the initial self-supervised model. This semi-supervised deep learning approach demonstrates that remote sensing can overcome a lack of labeled training data by generating noisy data for initial training using unsupervised methods and retraining the resulting models with high quality labeled data.

## Dependencies

DeepForest uses conda environments to manage python dependencies

Some key dependencies that have some build specific installs include

* OpenCV
* Keras + Tensorflow
* GDAL

I've forked my own version of the wonderful keras retinanet implementation.

https://github.com/bw4sz/keras-retinanet

```
conda env create --name DeepForest -f=environment.yml
```

All configurations are in the _config.yml 

To install the retinanet, activate the conda env and pip install

For example:
```
MacBook-Pro:keras-retinanet ben$ conda activate DeepLidar_dask
(DeepLidar_dask) MacBook-Pro:keras-retinanet ben$ pip install .
Processing /Users/ben/Documents/keras-retinanet
```

Pretraining data comes from the liDR R package. I've created a wrapper here: https://github.com/weecology/TreeSegmentation/blob/master/analysis/detection_training.R

## Model Training

Generate data in h5 format for efficient training

```
python dask_generate.py
```

## Training 

Two modes, see config. --train will run the pretraining Silva annotations
```
python train.py --train
```

--retrain will run the hand annotations

```
python train.py --retrain
```

## Evalution

```
python eval.py
```

## Published articles

Our first article is in review at *Remote Sensing*. The prepint can be found [here](https://www.biorxiv.org/content/10.1101/532952v1). 

The results of the full model can be found on our [comet page](https://www.comet.ml/bw4sz/deeplidar/2645e41bf83b47e68a313f3c933aff8a). To recreate this analysis, make sure to turn lidar post-processing off by setting min_density to a very high value (e.g 100) in the _config.yml file.

## Data

See the data repo: https://github.com/weecology/NeonTreeEvaluation

* With the exception of the hand annotated tiles that are too big to fit on github. 

Data from [SJER](https://www.neonscience.org/field-sites/field-sites-map/SJER) were used in the first publication. They can be downloaded directly from NEON (e.g 2018_SJER_3_259000_4110000_image.tif from https://data.neonscience.org/browse-data?showAllDates=true&siteCode=SJER&dpCode=DP3.30010.001)
