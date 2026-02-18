# Developer's Guide

Depends on Python 3.10+

## Getting started

1. Quickstart by forking the [main repository](https://github.com/weecology/DeepForest)

2. Clone your copy of the repository.

   - **Using ssh**:

     ```bash
     git clone git@github.com:[your user name]/DeepForest.git
     ```

   - **Using https**:

     ```bash
     git clone https://github.com/[your user name]/DeepForest.git
     ```

3. Link or point your cloned copy to the main repository. (I always name it upstream)

   ```bash
   git remote add upstream https://github.com/weecology/DeepForest.git
   ```

4. Check or confirm your settings using `git remote -v`

```bash
   origin git@github.com:[your user name]/DeepForest.git (fetch)
   origin git@github.com:[your user name]/DeepForest.git (push)
   upstream https://github.com/weecology/DeepForest.git (fetch)
   upstream https://github.com/weecology/DeepForest.git (push)
```

5. Install the package from the main directory.

Deepforest can be installed using any system that uses PyPI as the source including pip and uv.

**Install using Pip**

Installing with Pip uses [dev_requirements.txt](https://github.com/weecology/DeepForest/blob/main/dev_requirements.txt).

```bash
pip install .'[dev,docs]'
```

**Install using uv**

```bash
uv sync --all-extras --dev
```

6. Verify the installation by running this simple test:

```python
from deepforest import main
model = main.deepforest()
print("DeepForest successfully installed!")
```

## Testing

### Running tests locally

```bash
$ pip install . --upgrade  # or python setup.py install
$ pytest -v
```

### Code Quality and Style

We use [pre-commit](https://pre-commit.com/) to ensure consistent code quality and style across the project. Pre-commit runs automated checks and fixes before each commit to catch issues early.

#### Setting up pre-commit

1. **Install pre-commit** (if not already installed):
   ```bash
   pip install pre-commit
   ```

2. **Install the pre-commit hooks** in your local repository:
   ```bash
   pre-commit install
   ```

3. **Run pre-commit on all files** (optional, for initial setup):
   ```bash
   pre-commit run --all-files
   ```

#### What pre-commit does

Our pre-commit configuration (`.pre-commit-config.yaml`) includes:

- **Code formatting**: Uses [Ruff](https://docs.astral.sh/ruff/) for fast Python linting and formatting
- **Import sorting**: Automatically sorts and organizes imports
- **Docstring formatting**: Uses [docformatter](https://github.com/PyCQA/docformatter) to format docstrings
- **Notebook formatting**: Formats Jupyter notebooks in the documentation
- **File checks**: Ensures files end with newlines, removes trailing whitespace, checks YAML/JSON syntax
- **Large file detection**: Prevents accidentally committing large files

#### Running pre-commit manually

You can run pre-commit checks manually at any time:

```bash
# Run on staged files only
pre-commit run

# Run on all files
pre-commit run --all-files

# Run a specific hook
pre-commit run ruff
```

#### Fixing pre-commit issues

Most pre-commit hooks will automatically fix issues for you. If a hook fails:

1. **Check the output** - pre-commit will show you what needs to be fixed
2. **Re-run the hook** - many hooks can auto-fix issues:
   ```bash
   pre-commit run --all-files
   ```
3. **Stage the fixed files** and commit again:
   ```bash
   git add .
   git commit -m "Your commit message"
   ```

#### Bypassing pre-commit (not recommended)

If you need to bypass pre-commit for a specific commit (not recommended for regular development):

```bash
git commit --no-verify -m "Your commit message"
```

#### Editor integration

For the best development experience, consider integrating these tools directly into your editor:

- **VS Code**: Install the Ruff extension for real-time linting and formatting
- **PyCharm**: Configure Ruff as an external tool
- **Vim/Neovim**: Use plugins like `nvim-lspconfig` with the Ruff language server

For more information, see the [pre-commit documentation](https://pre-commit.com/).

## Contributing Code

### Pull Request Quality Standards

We welcome contributions! To ensure smooth reviews and maintain code quality, please follow these guidelines:

#### Before Submitting a Pull Request

- **Issue discussion**: For non-trivial changes, discuss your approach in an issue first. This can be done by either responding to an open issue indicating an interest in addressing the issue and an outline of how you plan to do so, or by creating a new issue describing the changes you think should be made (e.g., what the bug or feature is), that you would be interested in helping implement them, and what your general plan is for doing so.
- **Code style**: Follow project conventions (Ruff formatting, type hints, docstrings)
- **Tests**: Add or update tests for your changes. All tests must pass locally
- **Documentation**: Update relevant documentation for new features or API changes
- **Pre-commit checks**: Run `uv run pre-commit run --all-files` and ensure all checks pass
- **Breaking changes**: Discuss breaking changes with maintainers before implementing

#### Pull Request Description Requirements

Your PR description should include:

- **Problem statement**: What issue does this PR address? Link to related issues
- **Solution approach**: How does this PR solve the problem?
- **Testing**: How was this tested? Include test results or examples
- **Breaking changes**: Are there any breaking changes? If so, how should users migrate?
- **Screenshots/Examples**: For visualization if applicable

#### AI-Assisted Contributions Policy

We recognize that AI tools can be helpful in development. However, we have specific expectations:

- **Transparency**: If you used AI tools (e.g., GitHub Copilot, ChatGPT, etc.), please mention this in your PR description
- **Understanding required**: You must understand the code you're submitting. Don't submit code you can't explain or debug
- **Review and validation**: AI-generated code must be thoroughly reviewed, tested, and validated by you before submission
- **Context matters**: Ensure AI suggestions fit our project's architecture, patterns, and coding standards

#### Review Process Expectations

- **Each PR**: Has an issue and has passed CI tests
- **Be responsive**: Respond to review feedback promptly and constructively
- **Be open to suggestions**: Maintainers may suggest alternative approaches - be open to discussion
- **Iterate**: It's normal for PRs to go through multiple rounds of review
- **Inactive PRs**: PRs that are inactive for 30+ days may be closed. Please communicate if you need more time

## Documentation

We are using [Sphinx](http://www.sphinx-doc.org/en/stable/) and [Read the Docs](https://readthedocs.org/) for the documentation.

We use [Docformatter](https://pypi.org/project/docformatter/) for formatting and style checking.

```bash
$ docformatter --in-place --recursive src/deepforest/
```

### Update Documentation

The documentation is automatically updated for changes in functions.
However, the documentation should be updated after the addition of new functions or modules.

Change to the docs directory and use `sphinx-apidoc` to update the doc's `source`. Exclude the tests and setup.py documentation.

Run:

```bash
sphinx-apidoc -f -o ./source ../ ../tests/* ../setup.py
```

The `source` is the destination folder for the source rst files. `../` is the path to where the deepforest source code is located relative to the doc directory.

### Test documentation locally

```bash
cd docs  # Go to the docs directory and install the current changes.
pip install ../ -U
make clean  # Run
make html  # Run
```

## Create Release

### Start

1. **Run Pytest tests** – seriously, run them now. And Test build artifacts
   - Run `Pytest -v`
   - Run `pip install build && python -m build && twine check dist/*`
2. Ensure `HISTORY.rst` is up to date with all changes since the last release.
3. Use `bump-my-version show-bump` to determine the appropriate version bump.
4. Update the version for release: `bump-my-version bump [minor | patch | pre_l | pre_n]`. If show-bump does not have the right option, we can manually set it `bump-my-version bump --new-version 1.4.0`
5. Publish the release to PyPi
   - All releases are done on GitHub Actions when a new tag is pushed.
   - `git tag v1.0.0`
   - `git push origin v1.0.0`
6. Post-release, update the version to the next development iteration:
   - Run `bump-my-version show-bump` to check the target version.
   - Then, execute `bump-my-version bump [minor | patch | pre_l | pre_n]`.

Note: Do not commit the build directory after making html.

## Upload to Hugging Face Hub

To upload a trained model to the weecology organization space on Hugging Face Hub:

1. Train or load your model checkpoint
2. Set the label dictionary to match your classes
3. Use `push_to_hub` with the weecology organization name

For example:

```python
from deepforest import main

# Load model from checkpoint
model = main.deepforest.load_from_checkpoint("path/to/checkpoint.ckpt")

# Set label dictionary mapping class names to indices
model.label_dict = {"Livestock": 0}

# Push to weecology organization space
model.model.push_to_hub("weecology/deepforest-livestock")

# reload later
model.from_pretrained("weecology/deepforest-livestock")
```

The model will be uploaded to [https://huggingface.co/weecology/[model-name]](https://huggingface.co/weecology/[model-name])

### CropModel

```python
from deepforest.model import CropModel

crop_model = CropModel()
crop_model.push_to_hub("weecology/cropmodel-deadtrees")

# Reload it later
crop_model.from_pretrained("Weecology/cropmodel-deadtrees")

```

Please name the cropmodel based on what is being classified.

Note: You must have appropriate permissions in the weecology organization to upload models to weecology. If you are not already an active collaborator we recommend initially uploading new models to your own huggingface account and then letting us know and the model and whether or not you are interested in having them hosted on weecology's account.

#### Contributing a user-trained CropModel

If you have trained a classification model and want others to be able to load it through DeepForest, follow these steps.

##### 1. Model format

Your model must be a `torchvision` classification backbone (e.g. `resnet18`, `resnet50`) with the final FC layer sized to your number of classes.

The uploaded repo needs two files:

- `model.safetensors` with state dict keys prefixed by `model.` (e.g. `model.layer1.0.conv1.weight`)
- `config.json` with a `cropmodel` section containing at minimum `label_dict`, `architecture`, and `balance_classes`

Example `config.json`:

```json
{
  "cropmodel": {
    "architecture": "resnet18",
    "label_dict": {"ClassA": 0, "ClassB": 1},
    "balance_classes": false,
    "batch_size": 4,
    "num_workers": 0,
    "lr": 0.0001,
    "resize": [224, 224],
    "expand": 0,
    "scheduler": {
      "type": "ReduceLROnPlateau",
      "params": {
        "mode": "min",
        "factor": 0.5,
        "patience": 5,
        "threshold": 0.0001,
        "threshold_mode": "rel",
        "cooldown": 0,
        "min_lr": 0,
        "eps": 1e-08
      }
    }
  }
}
```

##### 2. Label dictionary

The model must have a `label_dict` mapping class names to integer indices: `{"ClassA": 0, "ClassB": 1, ...}`. This is stored in `config.json` and loaded automatically by `CropModel.load_model()`.

##### 3. Normalization

CropModel applies ImageNet normalization by default through `self.normalize()`. If your model was trained with standard ImageNet preprocessing, no changes are needed.

##### 4. Upload

You can either use `CropModel.push_to_hub()` directly, or build the two files manually and upload via `huggingface_hub.HfApi.upload_folder()`.

Using `push_to_hub()`:

```python
from deepforest.model import CropModel

crop_model = CropModel(config_args={"architecture": "resnet18"})
crop_model.create_model(num_classes=10)
# ... load your trained weights ...
crop_model.label_dict = {"SpeciesA": 0, "SpeciesB": 1}
crop_model.push_to_hub("your-username/cropmodel-yourmodel")
```

##### 5. Verify the model loads correctly

```python
from deepforest.model import CropModel
import torch

m = CropModel.load_model("your-username/cropmodel-yourmodel")
x = torch.rand(1, 3, 224, 224)
out = m(x)
assert out.shape[1] == len(m.label_dict)
```

##### 6. Document the model

Add a brief description of your model to `docs/user_guide/02_prebuilt.md` under the Crop Classifiers section. Include the species or categories it classifies, the training data source, and a code snippet showing how to load it.
