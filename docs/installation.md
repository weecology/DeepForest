# Installation

DeepForest has Windows, Linux and OSX prebuilt wheels on pypi. We *strongly* recommend using a conda or virtualenv to create a clean installation container.

```
pip install DeepForest-pytorch
```

For questions on conda-forge installation, please submit issues to the feedstock repo: https://github.com/conda-forge/deepforest-feedstock

## Source Installation

DeepForest can alternatively be installed from source using the github repository. The python package dependencies are managed by conda.

```
git clone https://github.com/weecology/DeepForest-pytorch.git
cd DeepForest-pytorch
conda env create --file=environment.yml
conda activate deepforest_pytorch
```
