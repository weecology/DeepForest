# Individual tree-crown detection in RGB imagery using self-supervised deep learning neural networks

Ben. G. Weinstein, Sergio Marconi, Stephanie Bohlman, Alina Zare, Ethan White

# Abstract
Remote sensing has the potential to transform the speed, scale, and cost of biodiversity surveys. Data acquisition currently outpaces the ability to identify individual organisms in high resolution imagery. We outline an approach to predict individual tree-crowns in RGB imagery using a deep learning detection network. Individual crown delineation is a persistent challenge in forestry and has been largely addressed using three-dimensional LIDAR. We show that deep learning models can leverage existing lidar-based unsupervised delineation approaches to train an RGB crown detection model, which is then refined using a small number of hand-annotated RGB images. We validate our proposed approach using an open-canopy site in the National Ecological Observation Network. Our results show that combining LIDAR and RGB methods in a self-supervised model improves predictions of trees in natural landscapes. The addition of a small number of hand-annotated images improved performance over the initial self-supervised model. While undercounting of individual trees in complex canopy conditions remains an area of continued development, deep learning can greatly increase the available data pool for remotely sensed tree surveys.

## Dependencies

DeepForest uses conda environments to manage python dependencies

Some key dependencies that have some build specific installs include

* OpenCV
* Tensorflow
* GDAL

```
conda env create --name DeepForest -f=environment.yml
```

All configurations are in the _config.yml 

## Training

```
python train.py
```

## Evalution

```
python eval.py
```
