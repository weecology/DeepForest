# What is DeepForest?

DeepForest is a python package for training and predicting ecological objects in airborne imagery. DeepForest currently comes with a tree crown object detection model and a bird detection model. Both are single class modules that can be extended to species classification based on new data. Users can extend these models by annotating and training custom models.

![](../www/image.png)

## How does deepforest work?

DeepForest uses deep learning object detection networks to predict bounding boxes corresponding to individual trees in RGB imagery. 
DeepForest is built on the object detection module from the [torchvision package](http://pytorch.org/vision/stable/index.html) and designed to make training models for tree detection simpler.

For more about the motivation behind DeepForest, see some recent talks we have given on computer vision for ecology and practical applications to machine learning in environmental monitoring.


**Towards a general AI for computer vision for ecology**

<iframe width="560" height="315" src="https://www.youtube.com/embed/ifsl6zwvWw0?si=wLGtZtRa1jZZMa3V" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>


**Computer Vision for Ecology**

<iframe width="560" height="315" src="https://www.youtube.com/embed/r7zqn4AZmb0?start=1080" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Where can I get help, learn from others, and report bugs?

Given the enormous array of forest types and image acquisition environments, it is unlikely that your image will be perfectly predicted by the prebuilt model. Below are some tips and some general guidelines to improve predictions.

Get suggestions on how to improve a model by using the [discussion board](https://github.com/weecology/DeepForest/discussions). Please be aware that only feature requests or bug reports should be posted on the [issues page](https://github.com/weecology/DeepForest/issues).

## License

Free software: [MIT license](https://github.com/weecology/DeepForest/blob/master/LICENSE)

## Why DeepForest?

Remote sensing can transform the speed, scale, and cost of biodiversity and forestry surveys. Data acquisition currently outpaces the ability to identify individual organisms in high resolution imagery. Individual crown delineation has been a long-standing challenge in remote sensing and available algorithms produce mixed results. DeepForest is the first open source implementation of a deep learning model for crown detection. Deep learning has made enormous strides in a range of computer vision tasks but requires significant amounts of training data. By including a trained model, we hope to simplify the process of retraining deep learning models for a range of forests, sensors, and spatial resolutions.

## Citation

[Weinstein, B.G.; Marconi, S.; Bohlman, S.; Zare, A.; White, E. Individual Tree-Crown Detection in RGB Imagery Using Semi-Supervised Deep Learning Neural Networks.
Remote Sens. 2019, 11, 1309](https://www.mdpi.com/2072-4292/11/11/1309)

[Geographic Generalization in Airborne RGB Deep Learning Tree Detection Ben Weinstein, Sergio Marconi, Stephanie Bohlman, Alina Zare, Ethan P White
bioRxiv 790071; doi: https://doi.org/10.1101/790071](https://www.biorxiv.org/content/10.1101/790071v1.abstract)
