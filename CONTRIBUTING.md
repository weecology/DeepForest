# Developer's Guide

Depends on Python 3.9+

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

   ```text
   origin git@github.com:[your user name]/DeepForest.git (fetch)
   origin git@github.com:[your user name]/DeepForest.git (push)
   upstream https://github.com/weecology/DeepForest.git (fetch)
   upstream https://github.com/weecology/DeepForest.git (push)
   ```

5. Install the package from the main directory.

Deepforest can be installed using either pip or conda.

**Install using Pip**

Installing with Pip uses [dev_requirements.txt](https://github.com/weecology/DeepForest/blob/main/dev_requirements.txt).

```bash
$ pip install -r dev_requirements.txt
$ pip install . -U
```

**Install using Conda**

Installing with Conda uses [environment.yaml](https://github.com/weecology/DeepForest/blob/main/environment.yml).

Conda-based installs can be slow. We recommend using [mamba](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html#quickstart) to speed them up.

```bash
$ conda create -n deepforest python=3
$ conda activate deepforest
$ pip install . -U
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

### Checking and fixing code style

#### Using Yapf

We use [yapf](https://github.com/google/yapf) for code formatting and style checking.

The easiest way to make sure your code is formatted correctly is to integrate it into your editor.
See [EDITOR SUPPORT](https://github.com/google/yapf/blob/main/EDITOR%20SUPPORT.md).

You can also run yapf from the command line to cleanup the style in your changes:

```bash
yapf -i --recursive src/deepforest/ --style=.style.yapf
```

If the style tests fail on a pull request, running the above command is the easiest way to fix this.

#### Using pre-commit

We configure all our checks using the `.pre-commit-config.yaml` file. To verify your code styling before committing, you should run `pre-commit install` to set up the hooks, followed by `pre-commit run` to execute them. This will apply the formatting rules specified in the `.style.yapf` file. For additional information, please refer to the [pre-commit documentation](https://pre-commit.com/index.html).

### Testing the Conda Deepforest Build

We use the `conda_recipe/meta.yaml` to make sure that the conda build can build the package.

```bash
$ cd conda_recipe
$ conda build conda_recipe/meta.yaml -c conda-forge -c defaults
```

### Conda staged recipe update

Update the Conda recipe after every release.

Clone the [Weecology staged recipes](https://github.com/weecology/staged-recipes).
Checkout the deepforest branch, update the `deepforest/meta.yaml` with the new version and the sha256 values. Sha256 values are obtained from the source on [PYPI download files](https://pypi.org/project/deepforest/#files) using the deepforest-{version-number}.tar.gz.

```jinja
{% set version = "fill new" %}
{% set sha256 = "fill new" %}
```

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
5. Publish the release to PyPi and update the Conda package.
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
model.push_to_hub("weecology/deepforest-livestock")
```

The model will be uploaded to [https://huggingface.co/weecology/[model-name]](https://huggingface.co/weecology/[model-name])

Note: You must have appropriate permissions in the weecology organization to upload models.
