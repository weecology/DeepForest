# How do I predict on large geospatial tiles?

## Predict a tile

Large tiles covering wide geographic extents cannot fit into memory during prediction and would yield poor results due to the density of bounding boxes. Often provided as geospatial .tif files, remote sensing data is best suited for the `predict_tile` function, which splits the tile into overlapping windows, performs prediction on each of the windows, and then reassembles the resulting annotations. Overlapping detections are removed based on the `iou_threshold` parameter.

Let’s show an example with a small image. For larger images, patch_size should be increased.

1.5
# Predict very large rasters (.tif) without running out of memory

DeepForest handles huge aerial/satellite rasters by sliding a fixed-size window across the image, running the model on each crop and stitching the results back together. Here is the minimal working example.

```python
from deepforest import main, get_data

model = main.deepforest()
model.use_release()            # load pretrained tree model

# A small demo tile ships with the package; replace with your own .tif
raster = get_data("OSBS_029.tif")

preds = model.predict_tile(
    path=raster,
    patch_size=400,       # pixels per window
    patch_overlap=0.15    # fraction overlap between windows
)
print(preds.head())        # xmin, ymin, xmax, ymax, label, score, image_path
```

Tip: visualise results with `deepforest.visualize.plot_results(preds)` or export to GeoJSON using `preds.to_file()` after converting to geographic coordinates (see User Guide: 15_Writing_data).

```{note}
The `predict_tile` function is sensitive to `patch_size`, especially when using the prebuilt model on new data. We encourage users to experiment with various patch sizes. For 0.1m data, 400-800px per window is appropriate, but it will depend on the density of tree plots. For coarser resolution tiles, >800px patch sizes have been effective.
```