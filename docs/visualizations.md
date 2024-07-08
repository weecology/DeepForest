
# Visualization

# `convert_to_sv_format` Function

## Description

The `convert_to_sv_format` function converts DeepForest prediction results into a `supervision.Detections` object. This object contains bounding boxes, class IDs, confidence scores, and class names. It is designed to facilitate the visualization and further processing of object detection results using [supervision](https://supervision.roboflow.com/latest/) library.

## Arguments

- `df (pd.DataFrame)`: The DataFrame containing the results from `predict_image` or `predict_tile`. The DataFrame is expected to have the following columns:
  - `xmin`: The minimum x-coordinate of the bounding box.
  - `ymin`: The minimum y-coordinate of the bounding box.
  - `xmax`: The maximum x-coordinate of the bounding box.
  - `ymax`: The maximum y-coordinate of the bounding box.
  - `label`: The class label of the detected object.
  - `score`: The confidence score of the detection.
  - `image_path`: The path to the image (optional, not used in this function).

## Returns

- `sv.Detections`: A `supervision.Detections` object containing:
  - `xyxy (ndarray)`: A 2D numpy array of shape (_, 4) with bounding box coordinates.
  - `class_id (ndarray)`: A numpy array of integer class IDs.
  - `confidence (ndarray)`: A numpy array of confidence scores.
  - `data (Dict[str, List[str]])`: A dictionary containing additional data, including class names.

## Usage Example

### Example 1: Converting Prediction Results and Annotating an Image

```python
import supervision as sv
from deepforest import main
from deepforest import get_data
from deepforest.visualize import convert_to_sv_format
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd

# Initialize the DeepForest model
m = main.deepforest()
m.use_release()

# Load image
img_path = get_data("OSBS_029.tif")
image = cv2.imread(img_path)

# Predict using DeepForest
result = m.predict_image(img_path)


# Convert custom prediction results to supervision format
sv_detections = convert_to_sv_format(result)

# You can now use `sv_detections` for further processing or visualization
```
To show bounding boxes:
```python
# You can visualize predicted bounding boxes

bounding_box_annotator = sv.BoundingBoxAnnotator()
annotated_frame = bounding_box_annotator.annotate(
    scene=image.copy(),
    detections=sv_detections
)


# Display the image using Matplotlib
plt.imshow(annotated_frame)
plt.axis('off')  # Hide axes for a cleaner look
plt.show()

```

![Bounding Boxes](figures/tree_predicted_bounding_boxes.jpeg)

To show labels:

``` python

label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
annotated_frame = label_annotator.annotate(
    scene=image.copy(),
    detections=sv_detections,
    labels=sv_detections['class_name']
)

# Display the image using Matplotlib
plt.imshow(annotated_frame)
plt.axis('off')  # Hide axes for a cleaner look
plt.show()
```
![Labels](figures/tree_predicted_labels.jpeg)
---