# Project Overview

DeepForest is a Python library for object detection in aerial images for biodiversity applications such as tree detection and wildlife observation. It uses pytorch and Pytorch Lightning framework for model training, prediction and evaluation.

Imagery is of a relatively high resolution and is generally not satellite data.

## Setup and usage

- We use `uv` for package management
- NEVER edit dependencies in pyproject.toml directly, ALWAYS use `uv add`
- To set up and run tests, use `uv sync --all-extras --dev`. You can reference the CI workflow in .github
- When you've finished, run `uv run pre-commit` with appropriate arguments to run formatting and linters on your work.

## Training, prediction, evaluation and config checking

```bash
usage: deepforest [-h] [--config-dir CONFIG_DIR] [--config-name CONFIG_NAME]
                  {train,predict,evaluate,config} ...

DeepForest CLI

positional arguments:
  {train,predict,evaluate,config}
    train               Train a model.
    predict             Run prediction on input image or CSV file
    evaluate            Run evaluation on ground truth annotations. Use
                        --predictions-csv to provide existing predictions, or
                        omit to generate them.
    config              Show the current config

options:
  -h, --help            show this help message and exit
  --config-dir CONFIG_DIR
                        Path to custom configuration directory
  --config-name CONFIG_NAME
                        Name of configuration file to use
```
- For example, train with a specific config: `uv run deepforest --config-name treeformer train train.csv_file=/new/path`

## Folder Structure

- `/src`: Contains the source code for the DeepForest library
- `/src/conf`: Contains configuration files in OmegaConf/Hydra format, and the schema.py
- `/src/scripts`: Contains CLI interface(s) to DeepForest
- `/tests`: Contains the unit tests; we use pytest
- `/docs`: Contains documentation for the project
- Use `/scratch`: for throw-away test scripts and other debug files.
- `/lightning_logs`: Is the default log directory for training runs.

## Libraries

- torch, torchvision, torchmetrics, transformers, pandas, numpy, scipy
- pytorch lightning
- visualization using matplotlib and supervision

## Test data
You can use data in the following location. These files contain bounding box annotations in standard DeepForest format (e.g. load with a standard dataset without any concerns)

-  /Users/josh/data/Neon_benchmark/images
- /Users/josh/data/Neon_benchmark/test.csv for validation (standard test dataset with known eval performance)
- /Users/josh/data/Neon_benchmark/train.csv train data, human annotated

- This data can be loaded directly as a Bounding Box dataset or a Keypoint dataset, as the library will convert it on-the-fly. You do not need to test whether they can be loaded.

## Data formats

- Annotations are usually provided in CSV with Pascal VOC convention for bounding box coordinates. For example:

```
image_path,xmin,ymin,xmax,ymax,label
OSBS_029.tif,203,67,227,90,Tree
OSBS_029.tif,256,99,288,140,Tree
OSBS_029.tif,166,253,225,304,Tree
```

- Other input types are possible using utilities.read_file (keypoints)
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
- ASCII characters only, no unicode/special characters
- Do not remove existing comments unless necessary, these often contain important context
