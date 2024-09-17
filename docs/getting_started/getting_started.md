# Getting started

# Demo

[Try out the DeepForest models online!](https://huggingface.co/spaces/weecology/deepforest-demo)

## How do I use a pretrained model to predict an image?

```python
from deepforest import main, get_data
from deepforest.visualize import plot_results
import os

model = main.deepforest()
model.use_release()

sample_image_path = get_data("OSBS_029.png")
results = model.predict_image(path=sample_image_path)
plot_results(results, root_dir=os.path.dirname(sample_image_path))
```

<img src="../www/getting_started1.png" height="300px">

### Predict a tile

Large tiles covering wide geographic extents cannot fit into memory during prediction and would yield poor results due to the density of bounding boxes. Often provided as geospatial .tif files, remote sensing data is best suited for the `predict_tile` function, which splits the tile into overlapping windows, performs prediction on each of the windows, and then reassembles the resulting annotations.

![](../../www/getting_started1.png)

Let's show an example with a small image. For larger images, patch_size should be increased.

```python
raster_path = get_data("OSBS_029.tif")
# Window size of 300px with an overlap of 25% among windows for this small tile.
boxes = model.predict_tile(raster_path, patch_size=300, patch_overlap=0.25)
plot_results(boxes)
```

** Please note the predict tile function is sensitive to patch_size, especially when using the prebuilt model on new data**

We encourage users to try out a variety of patch sizes. For 0.1m data, 400-800px per window is appropriate, but it will depend on the density of tree plots. For coarser resolution tiles, >800px patch sizes have been effective, but we welcome feedback from users using a variety of spatial resolutions.

### Sample package data

DeepForest comes with a small set of sample data that can be used to test out the provided examples. The data resides in the DeepForest data directory. Use the `get_data` helper function to locate the path to this directory, if needed.

```python
sample_image = get_data("OSBS_029.png")
sample_image
'/Users/benweinstein/Documents/DeepForest/deepforest/data/OSBS_029.png'
```

To use images other than those in the sample data directory, provide the full path for the images.

```python
image_path = get_data("OSBS_029.png")
boxes = model.predict_image(path=image_path)
```

```
>>> boxes
     xmin   ymin   xmax   ymax label     score    image_path
0   330.0  342.0  373.0  391.0  Tree  0.802979  OSBS_029.png
1   216.0  206.0  248.0  242.0  Tree  0.778803  OSBS_029.png
2   325.0   44.0  363.0   82.0  Tree  0.751573  OSBS_029.png
3   261.0  238.0  296.0  276.0  Tree  0.748605  OSBS_029.png
4   173.0    0.0  229.0   33.0  Tree  0.738210  OSBS_029.png
5   258.0  198.0  291.0  230.0  Tree  0.716250  OSBS_029.png
6    97.0  305.0  152.0  363.0  Tree  0.711664  OSBS_029.png
7    52.0   72.0   85.0  108.0  Tree  0.698782  OSBS_029.png