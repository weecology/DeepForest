# DeepForest

[![Build status](https://travis-ci.org/weecology/DeepForest.svg?master)](https://travis-ci.org/weecology)

Tree Crown Segmentation and Species Classification for the National Ecological Observation Network Sites

## Contributors

Ben Weinstein

Sergio Marconi

Sarah Graves

Ethan White

# Summary

The following is tested on the University of Florida Hipergator High Performance Cluster
The goal of this repo is to investigate and implement as many tree crown delination approaches as possible. Nearly 30 methods have been published in the past 10 years. The vast majority use unsupervised classification approaches to deliniate tree crown polygons. By treating these unsupervised classification methods as a feature ensemble for a deep learning neural network, we can leverage the benefits of each approach. Using field collected crown polygons as ground truth, the ensemble network will be run across NEON sites.

# Pipeline


## Non-pypi Dependencies
It is highly suggested that a user runs in a clean virtualenv. 

* GDAL

## Dependencies

```
pip install -r requirements.txt
```

The main executition script is RunJob.sh

* RunJob.sh

## Tree Crown Implementations (R and Python)
[Watershed](http://neondataskills.org/lidar/calc-biomass-py/) in python

[Watershed and Standard Window, silva?](http://adam-erickson.github.io/gapfraction/) in R

[Li 2012](https://pypi.python.org/pypi/forestutils) in python

[Hamraz 2016](http://cs.uky.edu/~hhamraz/lidar/manual.htm) in python

[Li 2012](https://github.com/Jean-Romain/lidR/wiki/Tree-segmentation-from-A-to-Z) in R

[Silva 2016](https://rdrr.io/rforge/rLiDAR/man/FindTreesCHM.html) in R

[Dalponte 2016](https://www.rdocumentation.org/packages/lidR/versions/1.4.0/topics/lastrees) in R

[Ayrey, E. et al. 2017. Layer Stacking: A Novel Algorithm for Individual Forest Tree Segmentation from LiDAR Point Clouds. - Can. J. Remote Sens. 43: 16–27.](https://github.com/bw4sz/Layer-Stacking)

### Authors still to be contacted (Bibliography)

* Xu, S. et al. 2018. A supervoxel approach to the segmentation of individual trees from LiDAR point clouds. - Remote Sens. Lett. 9: 515–523.

* Ferraz, A. et al. 2016. Lidar detection of individual tree size in tropical forests. - Remote Sens. Environ. 183: 318–333.

* Vega, C. et al. 2014. PTrees: A point-based approach to forest tree extractionfrom lidar data. - Int. J. Appl. Earth Obs. Geoinf. 33: 98–108.

* Lee, J. et al. 2017. A graph cut approach to 3D tree delineation, using integrated airborne LiDAR and hyperspectral imagery.: 1–24.

* Zhen, Z. et al. 2016. Trends in automatic individual tree crown detection and delineation-evolution of LiDAR data. - Remote Sens. 8: 1–26.  (review)

* Wallace, L. et al. 2014. Evaluating tree detection and segmentation routines on very high resolution UAV LiDAR ata. - IEEE Trans. Geosci. Remote Sens. 52: 7619–7628. (review)

* Wallace, L. et al. 2016. Assessment of forest structure using two UAV techniques: A comparison of airborne laser scanning and structure from motion (SfM) point clouds. - Forests 7: 1–16.

* Jaskierniak, D. et al. 2015. Using tree detection algorithms to predict stand sapwood area, basal area and stocking density in Eucalyptus regnans forest. - Remote Sens. 7: 7298–7323.
