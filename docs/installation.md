# Installation

DeepForest can be installed from source using the github repository.

```
git clone https://github.com/weecology/DeepForest.git
cd DeepForest
```

## Conda environment

The python package dependencies are managed by conda. For help installing conda see: [conda quickstart](https://docs.conda.io/projects/conda/en/latest/user-guide/install/). The conda installation of DeepForest installs a [fork](https://github.com/bw4sz/keras-retinanet.git) of the keras-retinanet to perform object detection.

```
conda env create --file=environment.yml
conda activate DeepForest
```

```
python
(test) MacBook-Pro:DeepForest ben$ python
Python 3.6.9 |Anaconda, Inc.| (default, Jul 30 2019, 13:42:17)
[GCC 4.2.1 Compatible Clang 4.0.1 (tags/RELEASE_401/final)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import deepforest
>>> deepforest.__version__
'0.2.2'
```
