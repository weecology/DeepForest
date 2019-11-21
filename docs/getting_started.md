# What is DeepForest?

DeepForest is a python package for training and predicting individual tree crowns from RGB imagery.

# How does it work?
DeepForest uses deep learning object detection networks to predict bounding boxes corresponding to individual trees in RGB imagery. DeepForest is built on top of the keras-retinanet () package and designed to make training models for tree detection simpler.

# Prebuilt models

DeepForest has a prebuilt model trained on data from 21 sites from the National Ecological Observation Network (). The prebuilt model uses a semi-supervised approach in which annotations are generated using unsupervised tree detection algorithms, followed by hand annotations of RGB imagery. For more details on the modeling approach see

Paper 1
Paper 2.

# Training

# Evaluation

## NEON Benchmark

# Prediction
