# Installation


DeepForest can be installed from source using the github repository.

```
git clone https://github.com/weecology/DeepForest.git
```

## Dependencies

DeepForest depends on keras-retinanet to perform object detection

```
git clone https://github.com/fizyr/keras-retinanet.git
cd keras-retinanet
pip install .
```

## Conda environment

The python package dependencies are managed by conda.

```
conda install -f=environment.yml
```

Note: The installation of tensorflow varies widely among systems and may need to be installed seperately. 
