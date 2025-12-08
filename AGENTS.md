# Project Overview

DeepForest is a Python library for object detection in aerial images for biodiversity applications such as tree detection and wildlife observation. It uses pytorch and Pytorch Lightning framework for model training, prediction and evaluation.

Imagery is of a relatively high resolution and is generally not satellite data.

## Setup and usage

- We use `uv` for package management
- NEVER edit dependencies in pyproject.toml directly, ALWAYS use `uv add`
- To set up and run tests, use `uv sync --all-extras --dev`. You can reference the CI workflow in .github
- Use the `deepforest` command line interface with Hydra overrides for simple tests
- When you've finished, run `uv run pre-commit` with appropriate arguments to run formatting and linters on your work.

## Folder Structure

- `/src`: Contains the source code for the DeepForest library
- `/src/conf`: Contains configuration files in OmegaConf/Hydra format, and the schema.py
- `/src/scripts`: Contains CLI interface(s) to DeepForest
- `/tests`: Contains the unit tests; we use pytest
- `/docs`: Contains documentation for the project, build using Sphinx and hosted on readthedocs.

## Libraries

- torch, torchvision, torchmetrics, transformers, pandas, numpy, scipy
- pytorch lightning
- visualization using matplotlib and supervision

## Data formats

- Annotations are usually provided in CSV with Pascal VOC convention for bounding box coordinates. For example:

```
image_path,xmin,ymin,xmax,ymax,label
OSBS_029.tif,203,67,227,90,Tree
OSBS_029.tif,256,99,288,140,Tree
OSBS_029.tif,166,253,225,304,Tree
```

- Other input types are possible using utilities.read_file
- Example CSV and shapefile inputs are in `/src/data`

## Configuration

- Use the configuration system to specify user-facing parameters instead of hardcoding/magic numbers in code
- Make sure the schema is up to date with the config files

## Coding standards

- Simple and maintainable
- Write docstrings and use type annotation
- Limited comments, highlighting important details
- Keep documentation in sync with code changes
- We use ruff for linting with options set in pyproject.toml
- Enforced by .pre-commit.yaml
- Take advantage of implicit vectorization using pandas, numpy, etc. Avoid explicit loops if you can.

## Code Style

-  According to PEP 8, "avoid trailing whitespace anywhere. Because it’s usually invisible, it can be confusing"
