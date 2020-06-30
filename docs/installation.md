# Installation

DeepForest has Windows, Linux and OSX prebuilt wheels on pypi. We *strongly* recommend using a conda or virtualenv to create a clean installation container.

```
pip install DeepForest
```

DeepForest is also available on conda-forge to help users compile code and manage dependencies. Conda builds are currently available for osx and linux, python 3.6 or 3.7. For help installing conda see: [conda quickstart](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).

For example, to create a env test with python 3.7
```
conda create --name test python=3.7
conda activate test
conda install -c conda-forge deepforest
```

For questions on conda-forge installation, please submit issues to the feedstock repo: https://github.com/conda-forge/deepforest-feedstock

## Source Installation

DeepForest can alternatively be installed from source using the github repository. The python package dependencies are managed by conda.

```
git clone https://github.com/weecology/DeepForest.git
cd DeepForest
conda env create --file=environment.yml
conda activate DeepForest
#build c extentions for retinanet
python setup.py build_ext --inplace
```
