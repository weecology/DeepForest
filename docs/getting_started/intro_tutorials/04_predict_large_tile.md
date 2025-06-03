# How do I predict on large geospatial tiles?

## Predict a tile

Large tiles covering wide geographic extents cannot fit into memory during prediction and would yield poor results due to the density of bounding boxes. Often provided as geospatial .tif files, remote sensing data is best suited for the `predict_tile` function, which splits the tile into overlapping windows, performs prediction on each of the windows, and then reassembles the resulting annotations. Overlapping detections are removed based on the `iou_threshold` parameter.

Let’s show an example with a small image. For larger images, patch_size should be increased.

```python
from deepforest import main
from deepforest import get_data
from deepforest.visualize import plot_results
import matplotlib.pyplot as plt

# Initialize the model class
model = main.deepforest()

# Load a pretrained tree detection model from Hugging Face
model.load_model(model_name="weecology/deepforest-tree", revision="main")

# Predict on large geospatial tiles using overlapping windows
path = get_data("OSBS_029.tif")
predicted_image = model.predict_tile(paths=path, patch_size=300, patch_overlap=0.25)
plot_results(predicted_image)
```

```