# Project Overview

DeepForest is a Python library for object detection in aerial images, primarily trees, but also supporting models for livestock and other objects of interest. It uses pytorch throughout and the Lightning framework for model training, prediction and evaluation.

Imagery is of a relatively high resolution and is generally not satellite data.

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

- Annotations are usually provided in CSV with Pascal VOC convention for bounding box coordinates
- Other input types are possible using utilities.read_file
- Example CSV inputs are in `/src/data`

## Configuration

- Use the configuration system to specify user-facing parameters instead of hardcoding/magic numbers in code
- Make sure the schema is up to date with the config files

## Coding standards

- Simple and maintanable
- Write docstrings and use type annotation
- Limited comments, highlighting important details
- Keep documentation in sync with code changes
- We use ruff for linting with options set in pyproject.toml
- Enforced by .pre-commit.yaml
