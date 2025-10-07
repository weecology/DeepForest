(install)=
# Installation

DeepForest has Windows, Linux, and macOS prebuilt wheels on PyPI. We *strongly* recommend using uv or a virtualenv to create a clean installation container.

```bash
pip install deepforest
```

```bash
uv add deepforest
```


## Source Installation

DeepForest can alternatively be installed from source using the GitHub repository.
This can be done directly using pip or uv.

```bash
pip install git+https://github.com/weecology/DeepForest.git
```

```bash
uv add "deepforest @ git+https://github.com/weecology/deepforest"
```

Or you can clone and install locally.

```bash
git clone https://github.com/weecology/DeepForest.git
cd DeepForest
pip install .
```

```bash
git clone https://github.com/weecology/DeepForest.git
cd DeepForest
uv sync --all-extras -dev
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
