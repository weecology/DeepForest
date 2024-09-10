# What is DeepForest?

DeepForest is a python package for training and predicting ecological objects in airborne imagery. DeepForest comes with prebuilt models for immediate use and fine-tuning by annotating and training custom models on their own data. Deepforest models can also be extended to species classification based on new data. DeepForest is designed for 1) applied researchers with limited machine learning experience, 2) applications with limited data that can be supported by prebuilt models, 3) scientists looking for a easy to use baseline to compare methods against. DeepForest uses deep learning object detection networks to predict the location of ecological objects in airborne imagery. The design of DeepForest is intended to be simple, modular and reproducible. 

![](../../www/image.png)

For more about the motivation behind DeepForest, see some recent talks we have given on computer vision for ecology and practical applications to machine learning in environmental monitoring.

**Airborne Ecology**

<iframe width="560" height="315" src="https://www.youtube.com/embed/O4K95-0W5FE?si=Vw8-yFLgRWaVIdbu" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

**Practical Intro to Computer Vision in Ecology Research**

[<iframe width="560" height="315" src="https://www.youtube.com/embed/r7zqn4AZmb0?start=1080" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>](https://youtu.be/wRBG74STulc?si=SRMWh6n9VlRU8kff)

## Where can I get help, learn from others, and report bugs?

Get suggestions on how to improve a model by using the [discussion board](https://github.com/weecology/DeepForest/discussions). Please be aware that only feature requests or bug reports should be posted on the [issues page](https://github.com/weecology/DeepForest/issues).


## Why DeepForest?

Airborne imagery can transform the speed, scale, and cost of biodiversity and forestry surveys. Data acquisition currently outpaces the ability to identify individual organisms in high resolution images. Global models for key ecological classes, such as 'Bird' and 'Tree' will reduce the need to collect large training datasets for each project. Deep learning has made enormous strides in a range of computer vision tasks but requires significant amounts of training data. By including trained models, we hope to simplify the process of retraining deep learning models for a range of backgrounds, sensors, and spatial resolutions.

## How can I contribute?

DeepForest is an open-source python project that depends on user contributions. Users can help by

* Making recommendations to the API and workflow. Please open an issue for anything that could help reduce friction and improve user experience.

* Leading implementations of new features. Check out the 'good first issue' tag on the repo and get in touch with mantainers and tell us about your skills. 

* Data contributions! The DeepForest backbone tree and bird models are not perfect. Please consider posting any annotations you make on zenodo, or sharing them with DeepForest mantainers. Open an [issue](https://github.com/weecology/DeepForest/issues) and tell us about the RGB data and annotations. For example, we are collecting tree annotations to create an [open-source benchmark](https://milliontrees.idtrees.org/). Please consider sharing data to make the models stronger and benefit you and other users. 

## License

Free software: [MIT license](https://github.com/weecology/DeepForest/blob/master/LICENSE)

## Citation

Most usage of DeepForest should cite two papers:

The first is the DeepForest paper, which describes the python package:

[Weinstein, B.G., Marconi, S., Aubry‐Kientz, M., Vincent, G., Senyondo, H. and White, E.P., 2020. DeepForest: A Python package for RGB deep learning tree crown delineation. Methods in Ecology and Evolution, 11(12), pp.1743-1751. https://doi.org/10.1111/2041-210X.13472](https://doi.org/10.1111/2041-210X.13472)

The second is the paper describing the particular model. See (prebuilt models)[prebuilt.md] for citations for each model.

