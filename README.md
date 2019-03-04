# Combining LIDAR and RGB airborne imagery for Individual tree-crown detection

Ben. G. Weinstein

## Dependencies

DeepForest uses conda environments to manage python dependencies

Some key dependencies that have some build specific installs include

* OpenCV
* Tensorflow
* GDAL

```
conda env create --name DeepForest -f=environment.yml
```

Depending on conda env, you may need to download laszip directly see: https://stackoverflow.com/questions/49500149/laspy-cannot-find-laszip-when-is-installed-from-source-laszip-is-in-path

All configurations are in the _config.yml 

Generate data

```
python dask_generate.py
```

## Training

```
python train.py
```

## Evalution

```
python eval.py
```

## Published articles

Our first article is in review *Remote Sensing*. The prepint can be found [here](https://www.biorxiv.org/content/10.1101/532952v1). 

The results of the full model can be found on our [comet page](https://www.comet.ml/bw4sz/deeplidar/2645e41bf83b47e68a313f3c933aff8a). To recreate this analysis, make sure to turn lidar post-processing off by setting min_density to a very high value (e.g 100) in the _config.yml file.
