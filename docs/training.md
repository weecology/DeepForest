# Training New Models

Our work has shown that starting training from the prebuilt model increases performance, regardless of the geographic location of your data. In the majority of cases, it will be useful for the model to have learned general tree representations that can be refined using hand annotated data.

## Design evaluation data

In our experience, defining a clear evaluation dataset and setting a threshold for desired performance is critical before training. It is common to just dive into training new data with only a vague sense of the desired outcome. This is always a mistake. We highly suggest users spend the time to answer 2 questions:

* What kind of data am I trying to predict?

Capturing the variability and the broad range of tree taxonomy and presentation will make development go more smoothly.

* What kind of accuracy do I need to answer my question?

It is natural to want the best model possible, but one can waste a tremendous amount of time trying to eek out another 5% of recall without understanding whether that increase in performance will improve our understanding of a given ecological or natural resource question. Prioritize evaluation data that matches your desired outcomes. Don't obsess over small errors, but rather think about how to propagate and capture this uncertainty in the overall analysis. [All models are wrong, some are useful.](https://en.wikipedia.org/wiki/All_models_are_wrong).

## Gather annotations

DeepForest uses xml files produced by the commonly used annotation program RectLabel. Please note that Rectlabel is an inexpensive program available only for Mac.

![](../www/rectlabel.png)

 For annotations made in RectLabel, DeepForest has a parse function ```preprocess.xml_to_annotations```. For non-mac users, there are many alternative for object detection annotation. DeepForest only requires that the final annotations be in the following format.

```
image_path, xmin, ymin, xmax, ymax, label
```

Please note that for functions which are fed into keras-retinanet, such as ```evaluate_generator```, ```predict_generator``` and ```train``` this annotation file should be saved without column names. For ```preprocess.split_raster``` the column names should be maintained.

## Training Hardware

Training neural networks is computationally intensive. While small amounts of data, on the order of several hundred trees, can be trained on a laptop in a few hours, large amounts of data are best trained on dedicated graphical processing units (GPUs). Many university clusters have GPUs available, and they can be rented for short periods of time on cloud servers (AWS, Google Cloud, Azure).

## Fit_generator versus tfrecords

There are currently two ways to train a deepforest model, directly using the annotations file described above, or wrapping those data into a tfrecords files. The benefits of annotations file, which uses a keras ```fit_generator``` method, is its simplicity and transparency. The downside is training speed. For the vast majority of projects, using a single GPU will be sufficient for training data. However, if you using any pretraining or semi-supervised approach, and have millions or tens of millions of samples, the ```fit_generator``` does not scale well across multiple GPUs. To create a tfrecords file:

TODO which of these have headers?

0. Optional -> generate crops from training tiles

```{python}
annotations_file = preprocess.split_raster(<path_to_raster>, config["annotations_file"], "tests/data/",config["patch_size"], config["patch_overlap"])
```

1. Generate the anchors for training from the annotations file

```{python}
created_records = tfrecords.create_tfrecords(annotations_file="tests/data/testfile_tfrecords.csv",
                           class_file="tests/data/classes.csv",
                           image_min_side=config["image-min-side"],
                           backbone_model=config["backbone"],
                           size=100,
                           savedir="tests/data/")
```

2. Train the model by supplying a list of tfrecords and the original file

```{python}
test_model.train(annotations="tests/data/testfile_tfrecords.csv",input_type="tfrecord", list_of_tfrecords=created_records)
```

This approach is >10X faster when used to scale across 8 GPUs on a single machine. Please note that tfrecords are very large on disk.
