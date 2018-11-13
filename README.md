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

# Results

<div align="center">
<img src="/Figures/Conceptual.png" width = "400" height="600"/>
</div>

Figure 1. A conceptual figure of the proposed pipeline. A LIDAR-based unsupervised classification generates initial training data for a self-supervised RGB deep learning model. The model is then retrained based on a small number of hand-annotated trees to create the full model.

![Alt text](/Figures/Panel.png?raw=true height="400" "Conceptual Figure")

Figure 2. Predicted individual tree crowns for the unsupervised lidar, self-supervised RGB and full model for two NEON tower plots (SJER_015, SJER_053) at the San Joaquin, CA site. For the full model, the field-collected tree centroids are shown in red points. NEON field teams only label a subset of trees (and only those with DBH > 10cm), leading to a smaller number of ground truth points than observed trees in a plot image.  
