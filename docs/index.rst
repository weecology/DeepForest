Welcome to DeepForest!
==============================================

DeepForest is a python package for airborne object detection and classification.

In DeepForest you can use pretrained models for tree crown prediction

.. image:: ../www/OSBS_sample.png

For bird detection

.. image:: ../www/bird_panel.jpg

or train your own object detection and classification model. 

*Why DeepForest?*

Observing the abundance and distribution of individual organisms is one of the foundations of ecology. Connecting broad-scale changes in organismal ecology, such as those associated with climate change, shifts in land use,
and invasive species requires fine-grained data on individuals at wide spatial and temporal extents. To capture these data, ecologists are turning to airborne data collection from uncrewed aerial vehicles, piloted aircraft,
and earth-facing satellites. Computer vision, a type of image-based artificial intelligence, has become a reliable tool for converting images into ecological information by detecting and classifying ecological objects within airborne imagery.
There have been many studies demonstrating that with sufficient labeling, computer vision can yield near-human level performance for ecological analysis. 
However, almost all of these studies rely on isolated computer vision models that require extensive technical expertise and human data labeling. 
In addition, the speed of innovation in computer vision makes it difficult for even experts to keep up with new innovations.
To address these challenges, the next phase for ecological computer vision needs to reduce the technical barriers and move towards general models that can be applied across space, time and taxa.

The goal of DeepForest is to be *simple*, *customizable*, and *modular*. DeepForest tries to create simple functions, like 'predict_tile', and keep unneeded details away from average user. The truth is that most machine learning projects fail not because of fancy models, but because of data and project organization. DeepForest tries to create simple defaults, use existing labeling tools and interfaces, and tries to be minimally impactful in learning new code and API.

*Feedback? How are you using DeepForest?*

The most helpful thing you can do is leave feedback on DeepForest git `repo https://github.com/weecology/DeepForest/issues`_. No feature or issue, or positive affirmation is too small. Please do it now!

Source code is available here: (https://github.com/weecology/DeepForest.git).

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   landing
   installation
   getting_started
   ConfigurationFile
   Evaluation
   deepforestr
   FAQ
   better
   ExtendingModule
   code_of_conduct.rst
   bird_detector
   history.rst
   source/deepforest
   authors.rst
   source/modules.rst
   use
   developer

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
