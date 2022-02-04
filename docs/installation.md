# Installation

DeepForest has Windows, Linux and OSX prebuilt wheels on pypi. We *strongly* recommend using a conda or virtualenv to create a clean installation container.

```
pip install DeepForest
```

DeepForest itself is pure Python and will work on all major operating systems, but has spatial and deep learning dependencies that can be harder to install, particularly on Windows. To make this easier DeepForest can also be installed using conda and mamba.

## conda/mamba CPU

Simple installs from conda-forge have been fragile due to issues with pytorch and torch vision in that repository.
Therefore we recommend first installing those dependencies from the official pytorch repo and then install DeepForest. 

```
conda create -n deepforest python=3 pytorch torchvision -c pytorch
conda activate deepforest
conda install deepforest -c conda-forge
```

Due to the complex dependency tree conda-based installs can be slow.
We recommend using [mamba](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html) to speed them up.

## conda/mamba GPU

Depending on the GPU you will need with `cudatoolkit=10.2` or `cudatoolkit=11.3`:

```
conda create -n deepforest python=3 pytorch torchvision cudatoolkit=10.2 -c pytorch
conda activate deepforest
conda install deepforest -c conda-forge
```

## Source Installation

DeepForest can alternatively be installed from source using the github repository. The python package dependencies are managed by conda.

```
git clone https://github.com/weecology/DeepForest.git
cd DeepForest
conda env create --file=environment.yml
conda activate deepforest
```

## GPU support

Pytorch can be run on GPUs to allow faster model training and prediction. Deepforest is a pytorch lightning module, as automatically distributes data to available GPUs.
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