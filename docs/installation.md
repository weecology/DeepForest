# Installation


DeepForest can be installed from source using the github repository.

```
git clone https://github.com/weecology/DeepForest.git
```

## Conda environment

The python package dependencies are managed by conda. For help installing conda see: [conda quickstart]()

```
conda env create --file=environment.yml
conda activate DeepForest
```

## Dependencies

DeepForest depends on a fork of the keras-retinanet to perform object detection

```
git clone https://github.com/bw4sz/keras-retinanet.git
cd keras-retinanet
pip install .
python setup.py build_ext --inplace
```

Note: The installation of tensorflow varies widely among systems and may need to be installed seperately.
