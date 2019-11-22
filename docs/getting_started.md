# What is DeepForest?

DeepForest is a python package for training and predicting individual tree crowns from RGB imagery.

# How does it work?
DeepForest uses deep learning object detection networks to predict bounding boxes corresponding to individual trees in RGB imagery. DeepForest is built on top of the keras-retinanet () package and designed to make training models for tree detection simpler.

# Prebuilt models

DeepForest has a prebuilt model trained on data from 21 sites from the National Ecological Observation Network (). The prebuilt model uses a semi-supervised approach in which annotations are generated using unsupervised tree detection algorithms, followed by hand annotations of RGB imagery. For more details on the modeling approach see

Weinstein, B.G.; Marconi, S.; Bohlman, S.; Zare, A.; White, E. Individual Tree-Crown Detection in RGB Imagery Using Semi-Supervised Deep Learning Neural Networks. Remote Sens. 2019, 11, 1309.
https://www.mdpi.com/2072-4292/11/11/1309

Geographic Generalization in Airborne RGB Deep Learning Tree Detection
Ben. G. Weinstein, Sergio Marconi, Stephanie A. Bohlman, Alina Zare, Ethan P. White
bioRxiv 790071; doi: https://doi.org/10.1101/790071

# Training

While the prebuilt models outperform available tree crown detection tools, they can always be improved by adding data from the target area. In our work, we have found that even one hour worth of carefully chosen hand annotation can yield enormous improvements in accuracy and precision. We envision that for the majority of scientific applications, atleast some finetuning of the prebuilt model will be worthwhile.

```{python}



```

# Evaluation


## NEON Benchmark

To standardize model evaluation, we have collected and published a benchmark dataset of nearly 20,000 crowns from sites in the National Ecological Observation Network.

https://github.com/weecology/NeonTreeEvaluation

# Prediction

Once a satisfactory model has been trained, DeepForest allows convenient prediction to new data. There are three ways to format data for prediction. For single images, predict_image can directly read an image from file and return predicted tree bounding boxes.

```{python}


```

For large tiles that cannot fit into memory, DeepForest has a ```predict_tile``` function to split the image into overlapping windows, perform prediction on each of the windows, and reassemble the resulting annotations.
