# What is DeepForest?

DeepForest is a python package for training and predicting individual tree crowns from RGB imagery.

# How does it work?
DeepForest uses deep learning object detection networks to predict bounding boxes corresponding to individual trees in RGB imagery. DeepForest is built on a fork of the [keras-retinanet](https://github.com/fizyr/keras-retinanet) package and designed to make training models for tree detection simpler.

# Prebuilt models

DeepForest has a prebuilt model trained on data from 24 sites from the National Ecological Observation Network (https://www.neonscience.org/field-sites/field-sites-map). The prebuilt model uses a semi-supervised approach in which millions of moderate quality annotations are generated using a LiDAR unsupervised tree detection algorithm, followed by hand-annotations of RGB imagery from select sites. For more details on the modeling approach see

Weinstein, B.G.; Marconi, S.; Bohlman, S.; Zare, A.; White, E. Individual Tree-Crown Detection in RGB Imagery Using Semi-Supervised Deep Learning Neural Networks. Remote Sens. 2019, 11, 1309.
https://www.mdpi.com/2072-4292/11/11/1309

Geographic Generalization in Airborne RGB Deep Learning Tree Detection
Ben. G. Weinstein, Sergio Marconi, Stephanie A. Bohlman, Alina Zare, Ethan P. White
bioRxiv 790071; doi: https://doi.org/10.1101/790071

# Training

The prebuilt models will always be improved by adding data from the target area. In our work, we have found that even one hour's worth of carefully chosen hand-annotation can yield enormous improvements in accuracy and precision. We envision that for the majority of scientific applications atleast some finetuning of the prebuilt model will be worthwhile.

```{python}



```

# Evaluation

Independent analysis of whether a model can generalize from training data to new areas is critical for creating a robust model. We stress that evaluation data must be different from training data, as neural networks have millions of parameters and can easily memorize thousands of samples. Therefore, while it would be rather easy to tune the model to get extremely high scores on the training data, it would fail when exposed to new images.

DeepForest uses the keras-retinanet ```evaluate``` method to score images. This consists of an annotations.csv file in the following format

```
image_path, xmin, ymin, xmax, ymax, label
```

```{python}


```

For more on evaluation, see the [Evaluation Overview]()

## NEON Benchmark

To standardize model evaluation, we have collected and published a benchmark dataset of nearly 20,000 crowns from sites in the National Ecological Observation Network.

https://github.com/weecology/NeonTreeEvaluation

# Prediction

Once a satisfactory model has been trained, DeepForest allows convenient prediction to new data. There are three ways to format data for prediction. For single images, predict_image can directly read an image from file and return predicted tree bounding boxes.

```{python}


```

For large tiles that cannot fit into memory, DeepForest has a ```predict_tile``` function to split the image into overlapping windows, perform prediction on each of the windows, and reassemble the resulting annotations.
