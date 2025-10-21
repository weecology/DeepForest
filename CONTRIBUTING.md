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
