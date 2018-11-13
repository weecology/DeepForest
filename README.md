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

All configurations are in the _config.yml 

## Training

```
python train.py
```

## Evalution

```
python eval.py
```
