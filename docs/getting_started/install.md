(install)=
# Installation

DeepForest has Windows, Linux, and macOS prebuilt wheels on PyPI. We *strongly* recommend using uv or a virtualenv to create a clean installation container.
### Python Version Compatibility

DeepForest relies on several dependencies that may not yet support the latest Python releases.

| Python Version | Status | Notes |
| :--- | :--- | :--- |
| **3.10 / 3.11** | **Stable** | **Highly Recommended** |
| 3.12 | Compatible | Most features work stable. |
| 3.13 / 3.14+ | Experimental | Known `AST` issues via `docformatter` (see [PyCQA/docformatter#327](https://github.com/PyCQA/docformatter/issues/327)). |

**Recommendation:** If you encounter installation errors on newer Python versions, please use a virtual environment with Python 3.11.

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
uv sync --all-extras --dev
```

## Development Installation

For developers who want to contribute to DeepForest, install the package in development mode with all dependencies:

```bash
git clone https://github.com/weecology/DeepForest.git
cd DeepForest
pip install .'[dev,docs]'
```

Or using uv:

```bash
git clone https://github.com/weecology/DeepForest.git
cd DeepForest
uv sync --all-extras --dev
```

This installs DeepForest in editable mode with development and documentation dependencies.

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
