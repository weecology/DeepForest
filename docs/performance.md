# Performance

## Training Hardware

Training neural networks is computationally intensive. While small amounts of data, on the order of several hundred trees, can be trained on a laptop in a few hours, large amounts of data are best trained on dedicated graphical processing units (GPUs). Many university clusters have GPUs available, and they can be rented for short periods of time on cloud servers (AWS, Google Cloud, Azure).

## Fit_generator versus tfrecords

There are currently two ways to train a deepforest model, directly using the annotations file described above, or wrapping those data into a tfrecords files. The benefits of annotations file, which uses a keras ```fit_generator``` method, is its simplicity and transparency. The downside is training speed. For the vast majority of projects, using a single GPU will be sufficient for training data. However, if you using any pretraining or semi-supervised approach, and have millions or tens of millions of samples, the ```fit_generator``` does not scale well across multiple GPUs. To create a tfrecords file:

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
