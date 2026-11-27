(getting_started)=
# Getting Started

## Installation


::::{grid} 1 2 2 2
:gutter: 4
:margin: 0 5 0 0

:::{grid-item-card} Install with pip
:class-card: install-card
:shadow: md

To install with pip from [`PyPI`](https://pypi.org/project/deepforest):

```bash
pip install deepforest
```
:::

:::{grid-item-card} Install with uv
:class-card: install-card
:shadow: md

To install with uv:

```bash
uv add deepforest
```
:::
::::

::::{grid} 1 1 1 1
:gutter: 4

:::{grid-item-card} In-depth instructions?
:class-card: install-card
:shadow: md
:link: install
:link-type: doc
:link-alt: View Installation Guide

Installing a specific version? Installing from source? Check the advanced installation page.
:::
::::

## Intro to DeepForest

DeepForest is a python package for airborne object detection and classification.

::::{grid} 1 2 2 2
:gutter: 4
:margin: 2 2 0 0

:::{grid-item-card}
:footer: *Tree crown prediction using DeepForest*
```{image} ../../www/OSBS_sample.png
:width: 100%
:alt: Tree crown detection
:class: shadow
```
:::

:::{grid-item-card}
:footer: *Bird detection using DeepForest*
```{image} ../../www/bird_panel.jpg
:width: 100%
:alt: Bird detection
:class: shadow
```
:::
::::

**DeepForest** is a python package for training and predicting ecological objects in airborne imagery. DeepForest comes with prebuilt models for immediate use and fine-tuning by annotating and training custom models on your own data.

```{toctree}
:maxdepth: 2
:hidden:

install
overview
intro_tutorials/index
comparison
spotlight
```
