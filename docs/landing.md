# What is DeepForest?

DeepForest is a python package for training and predicting individual tree crowns from airborne RGB imagery. DeepForest comes with a prebuilt model trained on data from the National Ecological Observation Network. Users can extend this model by annotating and training custom models.

![](../www/image.png)

## How does it work?
DeepForest uses deep learning object detection networks to predict bounding boxes corresponding to individual trees in RGB imagery. DeepForest is built on a fork of the [keras-retinanet](https://github.com/fizyr/keras-retinanet) package and designed to make training models for tree detection simpler.

## Demo
[http://tree.westus.cloudapp.azure.com/shiny/](http://tree.westus.cloudapp.azure.com/shiny/)

## License
Free software: [MIT license](https://github.com/weecology/DeepForest/blob/master/LICENSE)

## Why DeepForest?

Remote sensing can transform the speed, scale, and cost of biodiversity and forestry surveys. Data acquisition currently outpaces the ability to identify individual organisms in high resolution imagery. Individual crown delineation has been a long-standing challenge in remote sensing and available algorithms produce mixed results. DeepForest is the first open source implementation of a deep learning model for crown detection. Deep learning has made enormous strides in a range of computer vision tasks but requires significant amounts of training data. By including a trained model, we hope to simplify the process of retraining deep learning models for a range of forests, sensors, and spatial resolutions.  

## Citation

[Weinstein, B.G.; Marconi, S.; Bohlman, S.; Zare, A.; White, E. Individual Tree-Crown Detection in RGB Imagery Using Semi-Supervised Deep Learning Neural Networks.
Remote Sens. 2019, 11, 1309](https://www.mdpi.com/2072-4292/11/11/1309)

[Geographic Generalization in Airborne RGB Deep Learning Tree Detection Ben Weinstein, Sergio Marconi, Stephanie Bohlman, Alina Zare, Ethan P White
bioRxiv 790071; doi: https://doi.org/10.1101/790071](https://www.biorxiv.org/content/10.1101/790071v1.abstract)
