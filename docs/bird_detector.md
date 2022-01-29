# Bird Detector

Utilizing the same workflow as the tree detection model, we have trained a bird detection model for airborne imagery.

```
m = main.deepforest()
m.use_bird_release()
```

![](../www/bird_panel.jpg)

We have created a [GPU colab tutorial](https://colab.research.google.com/drive/1e9_pZM0n_v3MkZpSjVRjm55-LuCE2IYE?usp=sharing
) to demonstrate the workflow for using the bird model.

For more information, or specific questions about the bird detection, please create issues on the [BirdDetector repo](https://github.com/weecology/BirdDetector)

## Annotating new images

If you would like to train a model, here is a quick video on a simple way to annotate images.

<div style="position: relative; padding-bottom: 62.5%; height: 0;"><iframe src="https://www.loom.com/embed/e1639d36b6ef4118a31b7b892344ba83" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe></div>

Using a shapefile we could turn it into a dataframe of bounding box annotations by converting the points into boxes


```
df = shapefile_to_annotations(
    shapefile="annotations.shp", 
    rgb="image_path", box_points=True, buffer_size=0.15
)
```

Optionally we can split these annotations into crops if the image is large and will not fit into memory. This is often the case.

```
df.to_csv("full_annotations.csv",index=False)
annotations = preprocess.split_raster(
    path_to_raster=image_path,
    annotations_file="full_annotations.csv",
    patch_size=450,
    patch_overlap=0,
    base_dir=directory_to_save_crops,
    allow_empty=False
)
```

## Multi-species models

DeepForest allows training on multiple species annotations. It is often, but not always, useful to start from the general bird detector when trying to identify multiple species. This helps the model focus on learning the multiple classes and not wasting data and time re-learning bird bounding boxes.
To load the backboard and box prediction portions of the release model, but create a classification model for more than one species.

Here is an example using the alive/dead tree data stored in the package, but the same logic applies to the bird detector.

```
m = main.deepforest(num_classes=2, label_dict={"Alive":0,"Dead":0})
deepforest_release_model = main.deepforest()
deepforest_release_model.use_bird_release()
m.model.backbone.load_state_dict(deepforest_release_model.model.backbone.state_dict())
m.model.head.regression_head.load_state_dict(deepforest_release_model.model.head.regression_head.state_dict())

m.config["train"]["csv_file"] = get_data("testfile_multi.csv") 
m.config["train"]["root_dir"] = os.path.dirname(get_data("testfile_multi.csv"))
m.config["train"]["fast_dev_run"] = True
m.config["batch_size"] = 2
    
m.config["validation"]["csv_file"] = get_data("testfile_multi.csv") 
m.config["validation"]["root_dir"] = os.path.dirname(get_data("testfile_multi.csv"))
m.config["validation"]["val_accuracy_interval"] = 1

m.create_trainer()
m.trainer.fit(m)
assert m.num_classes == 2
```

## Citation

A general deep learning model for bird detection in high resolution airborne imagery
Ben G. Weinstein, Lindsey Garner, Vienna R. Saccomanno, Ashley Steinkraus, Andrew Ortega, Kristen Brush, Glenda Yenni, Ann E. McKellar, Rowan Converse, Christopher D. Lippitt, Alex Wegmann, Nick D. Holmes, Alice J. Edney, Tom Hart, Mark J. Jessopp, Rohan H Clarke, Dominik Marchowski, Henry Senyondo, Ryan Dotson, Ethan P. White, Peter Frederick, S.K. Morgan Ernest
bioRxiv 2021.08.05.455311; doi: https://doi.org/10.1101/2021.08.05.455311

https://www.biorxiv.org/content/10.1101/2021.08.05.455311v1.abstract
