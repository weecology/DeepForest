# Installation


DeepForest has a two step install process from pypi.

```
pip install DeepForest
pip install git+git://github.com/bw4sz/keras-retinanet.git
```

DeepForest can alternatively be installed from source using the github repository.

```
git clone https://github.com/weecology/DeepForest.git --depth 1
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
'0.2.3'
```

### Windows installation

Windows users have reported needing to first install the major dependencies for DeepForest before pip install. The standard windows approach to installing site packages wheels will work and reduce clutter in installation. The following packages are the major functions in DeepForest

* tensorflow==1.14.0
1.14.0 has been tested, any version < 2.0 should work as well. Deepforest relies on keras retinanet which does not yet have a tensorflow 2.0 build.
* keras
* OpenCV
