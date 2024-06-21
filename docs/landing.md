# What is DeepForest?

DeepForest is a python package for training and predicting ecological objects in airborne imagery. DeepForest currently comes with a tree crown object detection model and a bird detection model. Both are single class modules that can be extended to species classification based on new data. Users can extend these models by annotating and training custom models. 

![](../www/image.png)

## How does deepforest work?

DeepForest uses deep learning object detection networks to predict bounding boxes corresponding to individual trees in RGB imagery. 
DeepForest is built on the object detection module from the [torchvision package](http://pytorch.org/vision/stable/index.html) and designed to make training models for ecological object detection simpler.

For more about the motivation behind DeepForest, see some recent talks we have given on computer vision for ecology and practical applications to machine learning in environmental monitoring.

**Airborne Ecology**

<iframe width="560" height="315" src="https://www.youtube.com/embed/O4K95-0W5FE?si=Vw8-yFLgRWaVIdbu" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

**Practical Intro to Computer Vision in Ecology Research**

[<iframe width="560" height="315" src="https://www.youtube.com/embed/r7zqn4AZmb0?start=1080" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>](https://youtu.be/wRBG74STulc?si=SRMWh6n9VlRU8kff)

## Where can I get help, learn from others, and report bugs?

Given the enormous array of forest types and image acquisition environments, it is unlikely that your image will be perfectly predicted by a prebuilt model. Below are some tips and some general guidelines to improve predictions.

Get suggestions on how to improve a model by using the [discussion board](https://github.com/weecology/DeepForest/discussions). Please be aware that only feature requests or bug reports should be posted on the [issues page](https://github.com/weecology/DeepForest/issues).

## License

Free software: [MIT license](https://github.com/weecology/DeepForest/blob/master/LICENSE)

## Why DeepForest?

Remote sensing can transform the speed, scale, and cost of biodiversity and forestry surveys. Data acquisition currently outpaces the ability to identify individual organisms in high resolution imagery. Individual crown delineation has been a long-standing challenge in remote sensing and available algorithms produce mixed results. DeepForest is the first open source implementation of a deep learning model for crown detection. Deep learning has made enormous strides in a range of computer vision tasks but requires significant amounts of training data. By including a trained model, we hope to simplify the process of retraining deep learning models for a range of forests, sensors, and spatial resolutions.

## How can I contribute?

DeepForest is an open-source python project that depends on user contributions. Users can help by

* Making recommendations to the API and workflow. Please open an issue for anything that could help reduce friction and improve user experience.

* Leading implementations of new features. Check out the 'good first issue' tag on the repo and get in touch with mantainers and tell us about your skills. 

* Data contributions! The DeepForest backbone tree and bird models are not perfect. Please consider posting any annotations you make on zenodo, or sharing them with DeepForest mantainers. Open an [issue](https://github.com/weecology/DeepForest/issues) and tell us about the RGB data and annotations. For example, we are collecting tree annotations to create an [open-source benchmark](https://milliontrees.idtrees.org/). Please consider sharing data to make the models stronger and benefit you and other users. 

## Citation

Most usage of DeepForest should cite two papers.

The first is the DeepForest paper, which describes the package:

[Weinstein, B.G., Marconi, S., Aubry‐Kientz, M., Vincent, G., Senyondo, H. and White, E.P., 2020. DeepForest: A Python package for RGB deep learning tree crown delineation. Methods in Ecology and Evolution, 11(12), pp.1743-1751. https://doi.org/10.1111/2041-210X.13472](https://doi.org/10.1111/2041-210X.13472)

The second is the paper describing the model.

For the tree detection model cite:

[Weinstein, B.G.; Marconi, S.; Bohlman, S.; Zare, A.; White, E.P., 2019. Individual Tree-Crown Detection in RGB Imagery Using Semi-Supervised Deep Learning Neural Networks. Remote Sensing 11, 1309 https://doi.org/10.3390/rs11111309](https://doi.org/10.3390/rs11111309)

For the bird detection model cite:

[Weinstein, B.G., L. Garner, V.R. Saccomanno, A. Steinkraus, A. Ortega, K. Brush, G.M. Yenni, A.E. McKellar, R. Converse, C.D. Lippitt, A. Wegmann, N.D. Holmes, A.J. Edney, T. Hart, M.J. Jessopp, R.H. Clarke, D. Marchowski, H. Senyondo, R. Dotson, E.P. White, P. Frederick, S.K.M. Ernest. 2022. A general deep learning model for bird detection in high‐resolution airborne imagery. Ecological Applications: e2694 https://doi.org/10.1002/eap.2694](https://doi.org/10.1002/eap.2694)