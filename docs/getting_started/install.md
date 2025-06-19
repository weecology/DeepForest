1.5
(install)=
# Installation

## TL;DR – one-liner install

```bash
# CPU-only
pip install deepforest

# OR – GPU (PyTorch/cuDNN installed separately)
# pip install deepforest  # then install the appropriate torch wheel
```

DeepForest supports **Python 3.9 – 3.12** on Linux, macOS and Windows. We strongly recommend installing inside a fresh virtual environment – either `venv` or `conda` – to avoid dependency clashes.

```bash
python -m venv .venv          # or: conda create -n deepforest python=3.11
source .venv/bin/activate     # conda activate deepforest
pip install --upgrade pip
pip install deepforest
```

Verify the installation works (downloads the ~200 MB pretrained model on first run):

```bash
python - <<'PY'
from deepforest import main
m = main.deepforest(); m.use_release()
print("DeepForest ready – num classes:", m.config.num_classes)
PY
```

---

DeepForest has Windows, Linux, and OSX prebuilt wheels on PyPI. We *strongly* recommend using a conda or virtualenv to create a clean installation container.

For example:

```bash
conda create -n DeepForest python=3.11
conda activate DeepForest
```

```bash
pip install DeepForest
```

DeepForest itself is pure Python and will work on all major operating systems, but it has spatial and deep learning dependencies that can be harder to install, particularly on Windows. To make this easier, DeepForest can also be installed using conda and mamba.

## conda/mamba CPU

Simple installs from conda-forge have been fragile due to issues with PyTorch and TorchVision in that repository. Therefore, we recommend first installing those dependencies from the official PyTorch repo and then installing DeepForest.

```bash
conda create -n deepforest python=3 pytorch torchvision -c pytorch
conda activate deepforest
conda install deepforest -c conda-forge
```

Due to the complex dependency tree, conda-based installs can be slow. We recommend using [mamba](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html) to speed them up.

## conda/mamba GPU

Depending on the GPU, you will need either `cudatoolkit=10.2` or `cudatoolkit=11.3`:

```bash
conda create -n deepforest python=3 pytorch torchvision cudatoolkit=10.2 -c pytorch
conda activate deepforest
conda install deepforest -c conda-forge
```

## Geopandas Errors

Occasionally, it is easier to conda install geopandas, followed by pip install DeepForest. Geopandas has compiled libraries that need conda to work best (unless you already have GDAL installed). Once geopandas is installed, DeepForest can find it, and pip can be much faster. See the [geopandas documentation](https://geopandas.org/en/latest/getting_started/install.html) for more solutions.

## Source Installation

DeepForest can alternatively be installed from source using the GitHub repository. The Python package dependencies are managed by conda. In this case, we use a virtual environment.

```bash
git clone https://github.com/weecology/DeepForest.git
cd DeepForest
conda env create --file=environment.yml
conda activate DeepForest  # the name of the env in environment.yml is DeepForest
```

## GPU support

PyTorch can be run on GPUs to allow faster model training and prediction. DeepForest is a PyTorch Lightning module, which automatically distributes data to available GPUs. If using a release model with training, the module can be moved from CPU to GPU for prediction using the `pytorch.to()` method.

```python
from deepforest import main
m = main.deepforest()
m.load_model("weecology/deepforest-tree")
print("Current device is {}".format(m.device))
m.to("cuda")
print("Current device is {}".format(m.device))
```

```text
Current device is cpu
Current device is cuda:0
```

Distributed multi-GPU prediction outside of the training module is not yet implemented. We welcome pull requests for additional support.
