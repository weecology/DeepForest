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
