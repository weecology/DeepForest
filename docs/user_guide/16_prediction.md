# Prediction

There are atleast four ways to make predictions with DeepForest.
1. Predict an image using [model.predict_image](https://deepforest.readthedocs.io/en/latest/source/deepforest.html#deepforest.main.deepforest.predict_image)
2. Predict a tile using [model.predict_tile](https://deepforest.readthedocs.io/en/latest/source/deepforest.html#deepforest.main.deepforest.predict_tile) 
3. Predict a directory of using a csv file using [model.predict_file](https://deepforest.readthedocs.io/en/latest/source/deepforest.html#deepforest.main.deepforest.predict_file)
4. Predict a batch of images using [model.predict_batch](https://deepforest.readthedocs.io/en/latest/source/deepforest.html#deepforest.main.deepforest.predict_batch)

## Predict an image using model.predict_image

```python
from deepforest import main
from deepforest import get_data
from deepforest.visualize import plot_results

# Initialize the model class
model = main.deepforest()

# Load a pretrained tree detection model from Hugging Face 
model.load_model(model_name="weecology/deepforest-tree", revision="main")

sample_image_path = get_data("OSBS_029.png")
img = model.predict_image(path=sample_image_path)
plot_results(img)
```

## Predict a tile using model.predict_tile

Large tiles covering wide geographic extents cannot fit into memory during prediction and would yield poor results due to the density of bounding boxes. Often provided as geospatial .tif files, remote sensing data is best suited for the ``predict_tile`` function, which splits the tile into overlapping windows, performs prediction on each of the windows, and then reassembles the resulting annotations.

Let's show an example with a small image. For larger images, patch_size should be increased.

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
raster_path = get_data("OSBS_029.tif")
predicted_raster = model.predict_tile(raster_path, patch_size=300, patch_overlap=0.25)
plot_results(predicted_raster)
```

### Patch Size

   The *predict_tile* function is sensitive to *patch_size*, especially when using the prebuilt model on new data.
   We encourage users to experiment with various patch sizes. For 0.1m data, 400-800px per window is appropriate, but it will depend on the density of tree plots. For coarser resolution tiles, >800px patch sizes have been effective.

## Predict a directory of using a csv file using model.predict_file

For a list of images with annotations in a csv file, the `predict_file` function will return a dataframe with the predicted bounding boxes for each image as a single dataframe. This is useful for making predictions on a large number of images that have ground truth annotations.

```python

csv_file = get_data("OSBS_029.csv")
original_file = pd.read_csv(csv_file)
df = m.predict_file(csv_file, root_dir=os.path.dirname(csv_file))
```

```
>>> print(df.head())
         xmin        ymin        xmax        ymax  label     score    image_path                                           geometry
0  330.080566  342.662140  373.715454  391.686005      0  0.802979  OSBS_029.tif  POLYGON ((373.715 342.662, 373.715 391.686, 33...
1  216.171234  206.591583  248.594879  242.545593      0  0.778803  OSBS_029.tif  POLYGON ((248.595 206.592, 248.595 242.546, 21...
2  325.359222   44.049034  363.431244   82.248329      0  0.751573  OSBS_029.tif  POLYGON ((363.431 44.049, 363.431 82.248, 325....
3  261.008606  238.633163  296.410034  276.705475      0  0.748605  OSBS_029.tif  POLYGON ((296.410 238.633, 296.410 276.705, 26...
4  173.029999    0.000000  229.023438   33.749977      0  0.738210  OSBS_029.tif  POLYGON ((229.023 0.000, 229.023 33.750, 173.0...
```

## Predict a batch of images using model.predict_batch

For existing dataloaders, the `predict_batch` function will return a list of dataframes, one for each batch. This is more efficient than using predict_image since multiple images can be processed in a single forward pass.

```python
from deepforest import dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

raster_path = get_data("OSBS_029.tif")
tile = np.array(Image.open(raster_path))
ds = dataset.TileDataset(tile=tile, patch_overlap=0.1, patch_size=100)
dl = DataLoader(ds, batch_size=3)
    
# Perform prediction
predictions = []
for batch in dl:
    prediction = m.predict_batch(batch)
    predictions.append(prediction)
```