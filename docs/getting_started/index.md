(getting_started)=
# Getting Started

## Installation


::::{grid} 1 2 2 2
:gutter: 4
:margin: 0 5 0 0

:::{grid-item-card} Working with conda?
:class-card: install-card
:shadow: md

DeepForest is part of the [`Anaconda`](https://docs.continuum.io/anaconda/) distribution and can be installed with Anaconda or Miniconda:

```bash
conda install -c conda-forge deepforest
```
:::

:::{grid-item-card} Prefer pip?
:class-card: install-card
:shadow: md

DeepForest can be installed via pip from [`PyPI`](https://pypi.org/project/deepforest):

```bash
pip install deepforest
```
:::
::::


::::{grid} 1 1 1 1
:gutter: 4

:::{grid-item-card} In-depth instructions?
:class-card: install-card
:shadow: md

Installing a specific version? Installing from source? Check the advanced installation page.

+++
```{button-ref} install
:ref-type: doc
:color: secondary
:click-parent: true
:align: center
View Installation Guide
```
:::
::::

## Intro to DeepForest

DeepForest is a python package for airborne object detection and classification.

::::{grid} 1 2 2 2
:gutter: 4
:margin: 2 2 0 0

:::{grid-item}
```{image} ../../www/OSBS_sample.png
:width: 100%
:alt: Tree crown detection
:class: shadow
```
+++
*Tree crown prediction using DeepForest*
:::

:::{grid-item}
```{image} ../../www/bird_panel.jpg
:width: 100%
:alt: Bird detection
:class: shadow
```
+++
*Bird detection using DeepForest*
:::
::::

**DeepForest** is a python package for training and predicting ecological objects in airborne imagery. DeepForest comes with prebuilt models for immediate use and fine-tuning by annotating and training custom models on your own data.

```{toctree}
:maxdepth: 2
:hidden:

quickstart
install
overview
intro_tutorials/index
comparison
```

