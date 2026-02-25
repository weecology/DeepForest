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


## HuggingFace Authentication (Optional)

When downloading models from HuggingFace Hub, you may see a warning about unauthenticated requests. Setting a HuggingFace token provides higher rate limits and faster downloads.

### Get Your Token

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Go to [Settings → Access Tokens](https://huggingface.co/settings/tokens)
3. Create a new token (read access is sufficient)

### Set Your Token

**Linux/Mac:**

```bash
export HF_TOKEN=your_token_here
```

**Windows (Command Prompt):**

```bash
set HF_TOKEN=your_token_here
```

**Windows (PowerShell):**

```PowerShell
$env:HF_TOKEN="your_token_here"
```

**Python:**

```Python
import os
os.environ["HF_TOKEN"] = "your_token_here"
```

This is optional but recommended for frequent model downloads.
