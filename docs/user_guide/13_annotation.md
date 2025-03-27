# Annotation

Annotations play a crucial role in machine learning projects. If you're unhappy with your model's performance, annotating new samples is the best first step to improving it.

> **Note:** DeepForest >1.4.0 supports annotations in COCO and Pascal VOC format.

The machine learning annotation space is moving very quickly, and there are dozens of annotation tools and formats. DeepForest supports the following formats using the `read_file` function:

- CSV (`.csv`)
- Shapefile (`.shp`)
- GeoPackage (`.gpkg`)
- COCO (`.json`)
- Pascal VOC (`.xml`)

An incomplete list of annotation tools DeepForest users have reported success with:

- QGIS
- Label Studio
- CVAT
- Labelme
- Agentic
- AWS Ground Truth
- LabelBox
- Roboflow
- and many more  

We intentionally do not create our own annotation tools, but rather focus on supporting community-created tools. Look for exports in `.xml`, `.json`, or `.csv` formats, which are all common in the above tools.

## How do we annotate images?

For quick annotations of a few images, we use QGIS either as projected or unprojected data. You can create a shapefile for each image.

```{figure} ../../www/QGIS_annotation.png
:alt: QGIS annotation
:align: center
:width: 80%
```
*QGIS annotation example.*

### Label Studio

For long-term projects, we recommend using [Label Studio](https://labelstud.io/) as an annotation platform. It offers many useful features and is easy to set up.

```{figure} ../../www/label_studio.png
:alt: Label Studio annotation
:align: center
:width: 80%
```
*Label Studio annotation platform.*

## Exporting Annotations for DeepForest

To ensure compatibility with DeepForest, export your annotations from Label Studio in **Pascal VOC XML format**. This format is widely used for object detection tasks and can be directly read by DeepForest's `read_pascal_voc` function.

### Steps to Export in Pascal VOC Format

1. Navigate to your project in Label Studio.
2. Click on the **Export** button.
3. Select **Pascal VOC XML** as the export format.
4. Save the exported XML file.

For more details on reading Pascal VOC annotations in DeepForest, see: [reading-xml-annotations](#reading-xml-annotations-in-pascal-voc-format).

## Do I Need to Annotate All Objects in My Image?

Yes! Object detection models use non-annotated areas of an image as negative data. While annotating all objects in an image can be challenging, missing annotations will cause the model to *ignore* objects that should be treated as positive samples, leading to poor performance.

## How Can I Speed Up Annotation?

1. **Select Important Images**: Duplicate backgrounds or objects contribute little to model generalization. Focus on gathering a wide variety of object appearances.
2. **Avoid Over-splitting Labels**: Often, using a superclass for detection followed by a separate model for classification is more effective. See the [`CropModel`](03_cropmodels.md) for an example.
3. **Balance Accuracy and Practicality**: Depending on the goal (e.g., object counting or detection), keypoints can sometimes be used instead of precise boxes to simplify the process.

## Quick Video on Annotating Images

Here is a video demonstrating a simple way to annotate images:

```{raw} html
<div style="position:relative;padding-bottom:56.25%;height:0;overflow:hidden;">
<iframe src="https://www.loom.com/embed/e1639d36b6ef4118a31b7b892344ba83" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position:absolute;top:0;left:0;width:100%;height:100%;"></iframe>
</div>
```

## Converting Shapefile Annotations to DataFrame

You can convert shapefile points into bounding box annotations using the following code:

```python
df = shapefile_to_annotations(
    shapefile="annotations.shp",
    rgb="image_path",
    convert_to_boxes=True,
    buffer_size=0.15
)
```

## Cutting Large Tiles into Pieces

Annotating large airborne imagery can be challenging. DeepForest has a utility to crop images into smaller, more manageable chunks.

```python
raster = get_data("2019_YELL_2_528000_4978000_image_crop2.png")
output_crops = preprocess.split_raster(
    path_to_raster=raster,
    annotations_file=None,
    save_dir=tmpdir,
    patch_size=500,
    patch_overlap=0
)
```

## Starting Annotations from Pre-labeled Imagery

You can speed up new annotations by starting with model predictions. Below is an example of predicting detections and saving them as shapefiles, which can then be edited in a tool like QGIS.

```python
from deepforest import main
from deepforest.visualize import plot_predictions
from deepforest.utilities import boxes_to_shapefile
import rasterio as rio
import geopandas as gpd
from glob import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from shapely import geometry

PATH_TO_DIR = "/path/to/directory"
files = glob(f"{PATH_TO_DIR}/*.JPG")
m = main.deepforest(label_dict={"Bird": 0})
m.load_model(model_name="weecology/deepforest-bird", revision="main")

for path in files:
    boxes = m.predict_image(path=path)
    rio_src = rio.open(path)
    image = rio_src.read()

    if boxes is None:
        continue

    image = np.rollaxis(image, 0, 3)
    fig = plot_predictions(df=boxes, image=image)
    plt.imshow(fig)

    basename = os.path.splitext(os.path.basename(path))[0]
    shp = boxes_to_shapefile(boxes, root_dir=PATH_TO_DIR, projected=False)
    shp.to_file(f"{PATH_TO_DIR}/{basename}.shp")
```

## Reading XML Annotations in Pascal VOC Format

DeepForest can read annotations in Pascal VOC format, a widely-used dataset format for visual object detection. The `read_pascal_voc` function reads XML annotations and converts them into a format suitable for use with models like RetinaNet.

### Example:

```python
from deepforest import get_data
from deepforest.utilities import read_pascal_voc

xml_path = get_data("OSBS_029.xml")
df = read_pascal_voc(xml_path)
print(df)
```

This prints:

```text
    image_path  xmin  ymin  xmax  ymax  label
0   OSBS_029.tif   203    67   227    90   Tree
1   OSBS_029.tif   256    99   288   140   Tree
2   OSBS_029.tif   166   253   225   304   Tree
3   OSBS_029.tif   365     2   400    27   Tree
...
```

## Fast Iterations for Annotation Success

Avoid collecting all annotations before model testing. Start with a small number of annotations and let the model highlight which images are most needed. Fast iterations lead to quicker model improvement. For an example in wildlife sensing, see [Kellenberger et al., 2019](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8807383).

# Please Make Your Annotations Open-Source!

DeepForest's models are not perfect. Please consider sharing your annotations with the community to make the models stronger. You can post your annotations on **Zenodo** or open an [issue](https://github.com/weecology/DeepForest/issues) to share your data with the maintainers.