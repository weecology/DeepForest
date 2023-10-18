# Developer's Guide

Deepends on Python 3.5+

## Getting started

1. Quickstart by forking the [main repository](https://github.com/weecology/DeepForest)

2. Clone your copy of the repository.

> - **Using ssh**
> `git clone git@github.com:[your user name]/DeepForest.git`

> - **Using https**
> `git clone https://github.com/[your user name]/DeepForest.git`

3. Link or point your cloned copy to the main repository. (I always
name it upstream)

> - `git remote add upstream https://github.com/weecology/DeepForest.git`

4. Check or confirm your settings using `git remote -v`

```J
origin git@github.com:[your user name]/DeepForest.git (fetch)
origin git@github.com:[your user name]/DeepForest.git (push)
upstream https://github.com/weecology/DeepForest.git (fetch)
upstream https://github.com/weecology/DeepForest.git (push)
```

6. Install the package from the main directory.

Deepforest can be installed using either pip or conda.

**Install using Pip**

Installing with Pip uses [dev_requirements.txt](https://github.com/weecology/DeepForest/blob/main/dev_requirements.txt).

``` bash
$ pip install -r dev_requirements.txt
$ pip install . -U
```

**Install using Conda**

Installing with Conda uses [environment yaml](https://github.com/weecology/DeepForest/blob/main/environment.yml).

Conda-based installs can be slow. We recommend using
[mamba](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html#quickstart)
to speed them up.

``` bash
$ conda create -n deepforest python=3
$ conda activate deepforest
$ pip install . -U
```

7. Check if the package was installed; please test using the [sample
code](https://deepforest.readthedocs.io/en/latest/getting_started.html).

## Testing

### Running tests locally

``` bash
$ pip install . --upgrade # or python setup.py install
$ pytest -v
```

### Testing the Conda Deepforest Build

We use the conda_recipe/meta.yaml to make sure that the conda build can
build the package

``` bash
$ cd conda_recipe
$ conda build conda_recipe/meta.yaml -c conda-forge -c defaults
```

### Conda staged recipe update

Update the Conda recipe after every release.

Clone the [Weecology staged recipes](https://github.com/weecology/staged-recipes).
Checkout deepforest branch, update the `deepforeset/meta.yaml` with
the new version and the sha256 values. Sha256 values are obtained from
the source on [PYPI download files](https://pypi.org/project/deepforest/#files)
using the deepforest-{version-number}.tar.gz

``` 
{% set version = "fill new" %}
{% set sha256 = "fill new" %}
```

## Documentation

We are using [Sphinx](http://www.sphinx-doc.org/en/stable/) and [Read
the Docs](https://readthedocs.org//) for the documentation.

**Update Documentation**

The documentation is automatically updated for changes in functions.
However, the documentation should be updated after the addition of new
functions or modules.

Change to the docs directory and use sphinx-apidoc to update the doc's
`source`. Exclude the tests and setup.py documentation.

Run

``` bash
sphinx-apidoc -f -o ./source ../ ../tests/* ../setup.py
```

The `source` is the destination folder for the source rst files. `../`
is the path to where the deepforest source code is located relative to
the doc directory.

**Test documentation locally**

``` bash
cd docs # Go to the docs directory and install the current changes.

pip install ../ -U
make clean # Run
make html # Run

Note:
Do not commit the build directory after making html.
```