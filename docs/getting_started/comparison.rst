.. _comparison:

===========================
Comparison with other tools
===========================

Similar tools
-------------

There are many open-source projects for training machine learning models. We see DeepForest as a complement to many existing and excellent packages.

* Roboflow

  The `supervision <https://supervision.roboflow.com/latest/>`_, `inference <https://inference.roboflow.com/>`_ and related packages within Roboflow's ecosystem are well executed and used throughout DeepForest. The inference machine underlying Roboflow requires connection to Roboflow, a computer vision software company which requires an API key, and has a range of commercial and license structures. We think of DeepForest as a small set of curated models that are targeted towards the ecological and environmental monitoring community. Finding robust models is challenging amongst the thousands of Roboflow projects. Roboflow is designed to be an all-encompassing ecosystem, whereas DeepForest is intentionally small and aimed at existing pipelines.

* Torchgeo

  `Torchgeo <https://github.com/microsoft/torchgeo>`_ is a Python library written by developers at Microsoft to help automate remote sensing machine learning. Torchgeo has general structures, but the documents and general structure are focused on raster-based remote sensing, especially using earth-facing satellite data. Torchgeo has a number of useful datasets and curates pretrained models for remote sensing applications. The Torchgeo audience is generally more experienced with machine learning than the average DeepForest user.

We hope to continue to connect with both Roboflow and Torchgeo to improve interoperability among all model types and training. The future of open-source depends on collaboration, and we welcome users from all packages to submit ideas on how best to serve the community and reduce any duplication and wasted effort. There are many packages that hold useful individual models (e.g., `DetectTree2 <https://github.com/PatBall1/detectree2>`_) related to individual scientific publications. Our hope with DeepForest is to wrap general routines beyond individual research projects to make machine learning applications to environmental monitoring easier.