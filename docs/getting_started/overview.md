# Package overview

## What is DeepForest?

DeepForest is a Python package for training and predicting ecological objects in airborne imagery. DeepForest comes with prebuilt models for immediate use and fine-tuning by annotating and training custom models on your own data. DeepForest models can also be extended to classification (e.g., species) based on new data. DeepForest is designed for:

1. Applied researchers with limited machine learning experience
2. Applications with limited data that can be supported by prebuilt models
3. Scientists looking for an easy-to-use baseline to compare methods against

DeepForest uses deep learning object detection networks to predict the location of ecological objects in airborne imagery. The design of DeepForest is intended to be simple, modular, and reproducible.

![Tree crown prediction using DeepForest](../../www/image.png)

For more about the motivation behind DeepForest, see some recent talks we have given on computer vision for ecology and practical applications to machine learning in environmental monitoring.

### DeepForest Walkthroughs and Presentations

#### Ecological Society of America Statistical Seminar

This video provides an overview of fine-tuning a prebuilt model. The accompanying notebook is also available ([Bird FineTuning.ipynb](https://github.com/weecology/DeepForest/blob/main/docs/user_guide/examples/Bird_FineTuning.ipynb)).

<iframe width="560" height="315" src="https://www.youtube.com/embed/fhlC0W2kDMQ?si=fwrc9Lpoi1L-sGyl" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

<iframe width="560" height="315" src="https://www.youtube.com/embed/O4K95-0W5FE?si=Vw8-yFLgRWaVIdbu" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

#### Practical Intro to Computer Vision in Ecology Research

<a href="https://youtu.be/wRBG74STulc?si=SRMWh6n9VlRU8kff">
    <iframe width="560" height="315" src="https://www.youtube.com/embed/r7zqn4AZmb0?start=1080" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</a>

### Where can I get help, learn from others, and report bugs?

Given the enormous array of taxa, background, and image acquisition environments, it is unlikely that your image will be perfectly predicted by a prebuilt model. Check out the 'training', 'annotation', and 'predicting' sections of the documentation for more information on how to improve predictions using your own data.

Get suggestions on how to improve a model by using the [discussion board](https://github.com/weecology/DeepForest/discussions). Please be aware that only feature requests or bug reports should be posted on the issues page. The most helpful thing you can do is leave feedback on the DeepForest [issue page](https://github.com/weecology/DeepForest/issues). No feature, issue, or positive affirmation is too small. Please do it now!

## Why DeepForest?

Observing the abundance and distribution of individual organisms is one of the foundations of ecology. Connecting broad-scale changes in organismal ecology, such as those associated with climate change, shifts in land use, and invasive species, requires fine-grained data on individuals at wide spatial and temporal extents.

To capture these data, ecologists are turning to airborne data collection from uncrewed aerial vehicles, piloted aircraft, and earth-facing satellites. Computer vision, a type of image-based artificial intelligence, has become a reliable tool for converting images into ecological information by detecting and classifying ecological objects within airborne imagery.

There have been many studies demonstrating that, with sufficient labeling, computer vision can yield near-human-level performance for ecological analysis. However, almost all of these studies rely on isolated computer vision models that require extensive technical expertise and human data labeling. In addition, the speed of innovation in computer vision makes it difficult for even experts to keep up with new innovations.

To address these challenges, the next phase of ecological computer vision needs to reduce the technical barriers and move towards general models that can be applied across space, time, and taxa.

DeepForest aims to be **simple**, **customizable**, and **modular**. DeepForest makes an effort to keep unnecessary complexity hidden from the ordinary user by developing straightforward functions like `predict_tile`. The majority of machine learning projects actually fail due to poor data and project management, not clever models. DeepForest makes an effort to generate straightforward defaults, utilize already-existing labeling tools and interfaces, and minimize the effect of learning new APIs and code.

## How can I contribute?

DeepForest is an open-source Python project that depends on user contributions. Users can help by:

- Making recommendations to the API and workflow. Please open an issue for anything that could help reduce friction and improve the user experience.
- Leading implementations of new features. Check out the 'good first issue' tag on the repo and get in touch with the maintainers to tell us about your skills.
- Data contributions! The DeepForest backbone models are not perfect. Please consider posting any annotations you make on Zenodo, or sharing them with DeepForest maintainers. Open an [issue](https://github.com/weecology/DeepForest/issues) and tell us about the RGB data and annotations. For example, we are collecting tree annotations to create an [open-source benchmark](https://milliontrees.idtrees.org/). Please consider sharing data to make the models stronger and benefit you and other users.

## Citation

Most usage of DeepForest should cite two papers:

The first is the DeepForest paper, which describes the Python package:

```{note}
Weinstein, B.G., Marconi, S., Aubry‐Kientz, M., Vincent, G., Senyondo, H. and White, E.P., 2020. DeepForest: A Python package for RGB deep learning tree crown delineation. Methods in Ecology and Evolution, 11(12), pp.1743-1751. [https://doi.org/10.1111/2041-210X.13472](https://doi.org/10.1111/2041-210X.13472)
```

The second is the paper describing the particular model. See [Prebuilt Setup](../user_guide/02_prebuilt.md) for citations for each model.

## License

:::{card}
```{include} ../../LICENSE
:relative-docs: docs/
:relative-images:
```
:::

8.5

## Try it now – 30-second demo

The example below loads a pretrained tree-crown detector, runs it on a bundled sample image, and prints the first few predicted boxes. Everything runs on CPU and therefore works on any laptop.

```python
from deepforest import main
from importlib import resources

# Load the pretrained tree model (≈200 MB download on first use)
model = main.deepforest()
model.use_release()

# Locate the small demo RGB image that ships with DeepForest
sample_img = resources.files("deepforest.data") / "OSBS_029.png"

# Predict tree crowns
crowns = model.predict_image(path=str(sample_img))
print(crowns.head())
```

Output (columns: bounding-box coordinates, predicted label, confidence score):

```text
   xmin   ymin   xmax   ymax label     score
0   159    122    247    224  Tree  0.89
1   334    333    421    417  Tree  0.86
...
```

For a visual overview of the detections, pass the result to `deepforest.visualize.plot_predictions` (see Quick-start on the home page).