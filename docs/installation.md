# Installation


DeepForest has Windows, Linux and OSX prebuilt wheels on pypi. We *strongly* recommend using a conda or virtualenv to create a clean installation container.

```
pip install DeepForest
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

### Tensorflow dependency

Some users have choosen to install tensorflow seperately. We recommend to just pip install DeepForest to ensure the latest dependencies are downloaded. If you do want to go down this route for some reason, please be sure to use tensorflow < 2.0. The keras retinanet is not 2.0 ready at this time.
