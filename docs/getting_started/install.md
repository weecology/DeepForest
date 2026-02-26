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

## HuggingFace Authentication (Optional)

Models are downloaded from [Hugging Face Hub](https://huggingface.co). Authentication is optional for public models but recommended for higher rate limits and faster downloads.

To authenticate:
- Run `huggingface-cli login`, or
- Set the `HF_TOKEN` environment variable

For details, see the [Hugging Face token documentation](https://huggingface.co/docs/hub/security-tokens).

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
