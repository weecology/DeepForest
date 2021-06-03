# Installation

DeepForest has Windows, Linux and OSX prebuilt wheels on pypi. We *strongly* recommend using a conda or virtualenv to create a clean installation container.

```
pip install DeepForest
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

## GPU support

Pytorch can be run on GPUs to allow faster model training and prediction. Deepforest-pytorch is a pytorch lightning module, as automatically distributes data to available GPUs.
If using a release model with training, the module can be moved from CPU to GPU for prediction is the pytorch.to() method.

```
from deepforest import main
m = main.deepforest()
m.use_release()
print("Current device is {}".format(m.device))
m.to("cuda")
print("Current device is {}".format(m.device))
```

```
Current device is cuda:0
```

Distributed multi-gpu prediction outside of the training module is not yet implemented. We welcome pull requests for additional support.