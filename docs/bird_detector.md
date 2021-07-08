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

## Citation

The detector is currently in prep, please check back for a published citation. Cite the github release until a preprint is available.
