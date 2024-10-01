.. _getting_started:

===============
Getting started
===============

Installation
------------

.. grid:: 1 2 2 2
    :gutter: 4

    .. grid-item-card:: Working with conda?
        :class-card: install-card
        :columns: 12 12 6 6
        :padding: 3

        DeepForest is part of the `Anaconda <https://docs.continuum.io/anaconda/>`__
        distribution and can be installed with Anaconda or Miniconda:

        ++++++++++++++++++++++

        .. code-block:: bash

            conda install -c conda-forge deepforest

    .. grid-item-card:: Prefer pip?
        :class-card: install-card
        :columns: 12 12 6 6
        :padding: 3

        DeepForest can be installed via pip from `PyPI <https://pypi.org/project/deepforest>`__.

        ++++

        .. code-block:: bash

            pip install deepforest

    .. grid-item-card:: In-depth instructions?
        :class-card: install-card
        :columns: 12
        :padding: 3

        Installing a specific version? Installing from source? Check the advanced
        installation page.

        +++

        .. button-ref:: install
            :ref-type: ref
            :click-parent:
            :color: secondary
            :expand:

            Learn more


.. _gentle_intro:

Intro to DeepForest
-------------------
DeepForest is a python package for airborne object detection and classification.


|Tree crown prediction using DeepForest| |Bird detection using DeepForest|

.. |Tree crown prediction using DeepForest| image:: ../../www/OSBS_sample.png
   :width: 45%

.. |Bird detection using DeepForest| image:: ../../www/bird_panel.jpg
   :width: 45%


**DeepForest** is a python package for training and predicting ecological objects in airborne imagery. DeepForest comes with prebuilt models for immediate use and fine-tuning by annotating and training custom models on your own data.



.. toctree::
    :maxdepth: 2
    :hidden:

    install
    overview
    intro_tutorials/index
    comparison

