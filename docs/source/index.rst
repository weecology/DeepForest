Welcome to DeepForest!
**********************

DeepForest is a python package for airborne object detection and classification.

**Tree crown prediction using DeepForest**

.. image:: ../../www/OSBS_sample.png

**Bird detection using DeepForest**

.. image:: ../../www/bird_panel.jpg


Why DeepForest?
===============

Observing the abundance and distribution of individual organisms is one of the foundations of ecology. Connecting broad-scale changes in organismal ecology, such as those associated with climate change, shifts in land use, and invasive species require fine-grained data on individuals at wide spatial and temporal extents.

To capture these data, ecologists are turning to airborne data collection from uncrewed aerial vehicles, piloted aircraft, and earth-facing satellites. Computer vision, a type of image-based artificial intelligence, has become a reliable tool for converting images into ecological information by detecting and classifying ecological objects within airborne imagery.

There have been many studies demonstrating that, with sufficient labeling, computer vision can yield near-human-level performance for ecological analysis. However, almost all of these studies rely on isolated computer vision models that require extensive technical expertise and human data labeling.
In addition, the speed of innovation in computer vision makes it difficult for even experts to keep up with new innovations.

To address these challenges, the next phase of ecological computer vision needs to reduce the technical barriers and move towards general models that can be applied across space, time, and taxa.

DeepForest aims to be **simple**, **customizable**, and **modular**. DeepForest makes an effort to keep unnecessary complexity hidden from the ordinary user by developing straightforward functions like "predict_tile." The majority of machine learning projects actually fail due to poor data and project management, not clever models. DeepForest makes an effort to generate straightforward defaults, utilize already-existing labeling tools and interfaces, and minimize the effect of learning new APIs and code.

Where can I get help, learn from others, and report bugs?
=========================================================

Given the enormous array of forest types and image acquisition environments, it is unlikely that your image will be perfectly predicted by a prebuilt model. Below are some tips and some general guidelines to improve predictions.

Get suggestions on how to improve a model by using the [discussion board](https://github.com/weecology/DeepForest/discussions). Please be aware that only feature requests or bug reports should be posted on the issues page. The most helpful thing you can do is leave feedback on DeepForest `issue page`_. No feature or issue, or positive affirmation is too small. Please do it now!


`Source code`_ is available on GitHub.

.. toctree::
   :maxdepth: 2

   ../introduction/index
   ../installation_and_setup/index
   ../getting_started/index
   ../training_and_evaluation/index
   ../data_annotation/index
   ../advanced_features/index
   ../developer_resources/index
   ../deepforestr
   modules.rst
   ../related_work/index


Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _issue page: https://github.com/weecology/DeepForest/issues
.. _Source code: https://github.com/weecology/DeepForest.git
