# Annotation
Annotation is likely the most important part of machine learning projects. Fancy models are nice, but data are always paramount. If you aren't happy with model performance, annotating new samples is the best idea.

## How should I annotate images?
For quick annotations of a few images, we recommend using QGIS or ArcGIS. Either as project or unprojected data. Create a shapefile for each image.

![QGISannotation](../www/QGIS_annotation.png)

### Label-studio

For longer term projects, we recommend [label-studio](https://labelstud.io/) as an annotation platform. It has many useful features and is easy to set up.

![QGISannotation](../www/label_studio.png)

## Do I need annotate all objects in my image?
Yes! Object detection models use the non-annotated areas of an image as negative data. We know that it can be exceptionally hard to annotate all trees in an image, or determine the classes of all birds in an image. However, if you have objects in the image that are not annotated, the model is learning *to ignore* those portion of the image. This can severely affect model performance.

## Cutting large tiles into pieces

It is often difficult to annotate very large airborne imagery. DeepForest has a small utility to crop images into smaller chunks that can be annotated more easily.

```python
raster = get_data("2019_YELL_2_528000_4978000_image_crop2.png")

output_crops = preprocess.split_raster(path_to_raster=raster,
                                        annotations_file=None,
                                        save_dir=tmpdir,
                                        patch_size=500,
                                        patch_overlap=0)

# Returns a list of crop filenames.
assert len(output_crops) == 25

# Assert that all output_crops exist
for crop in output_crops:
    assert os.path.exists(crop)
```

## How can I view current predictions as shapefiles?

It is often useful to train new training annotations starting from current predictions. This allows users to more quickly find and correct errors. The following example shows how to create a list of files, predict detections in each, and save as shapefiles. A user can then edit these shapefiles in a program like QGIS.

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

PATH_TO_DIR = "/Users/benweinstein/Dropbox/Weecology/everglades_species/easyidp/HiddenLittle_03_24_2022"
files = glob("{}/*.JPG".format(PATH_TO_DIR))
m = main.deepforest(label_dict={"Bird":0})
m.use_bird_release()
for path in files:
    #use predict_tile if each object is a orthomosaic
    boxes = m.predict_image(path=path)
    #Open each file and get the geospatial information to convert output into a shapefile
    rio_src = rio.open(path)
    image = rio_src.read()
    
    #Skip empty images
    if boxes is None:
        continue
    
    #View result
    image = np.rollaxis(image, 0, 3)
    fig = plot_predictions(df=boxes, image=image)   
    plt.imshow(fig)
    
    #Create a shapefile, in this case img data was unprojected
    shp = boxes_to_shapefile(boxes, root_dir=PATH_TO_DIR, projected=False)
    
    #Get name of image and save a .shp in the same folder
    basename = os.path.splitext(os.path.basename(path))[0]
    shp.to_file("{}/{}.shp".format(PATH_TO_DIR,basename))
```
## Understanding Pascal VOC and the read_pascal_voc Function
### What is Pascal VOC?
The Pascal Visual Object Classes (VOC) dataset is a benchmark in visual object category recognition and detection, providing a standard dataset of images and annotation, and standard evaluation procedures. 

Pascal VOC annotations are typically stored in XML format, which includes information about the image and the bounding boxes of the objects present within it. Each bounding box is defined by its coordinates (xmin, ymin, xmax, ymax) and is associated with a class label. The annotations also include other metadata about the image, such as its filename. You can know more about Pascal VOC from [Here](https://roboflow.com/formats/pascal-voc-xml)



### The `read_pascal_voc` Function
The `read_pascal_voc` function is designed to read these XML annotations and convert them into a format suitable for use with object detection models, such as RetinaNet. This function parses the XML file, extracts the relevant information, and constructs a pandas DataFrame containing the image path and the bounding box coordinates along with the class labels.

Example:
```python
from deepforest import get_data
from deepforest.utilities import read_pascal_voc

xml_path = get_data("OSBS_029.xml")
df = read_pascal_voc(xml_path)
print(df)
```
This prints:
```
      image_path  xmin  ymin  xmax  ymax label
0   OSBS_029.tif   203    67   227    90  Tree
1   OSBS_029.tif   256    99   288   140  Tree
2   OSBS_029.tif   166   253   225   304  Tree
3   OSBS_029.tif   365     2   400    27  Tree
4   OSBS_029.tif   312    13   349    47  Tree
..           ...   ...   ...   ...   ...   ...
56  OSBS_029.tif    60   292    96   332  Tree
57  OSBS_029.tif    89   362   114   390  Tree
58  OSBS_029.tif   236   132   253   152  Tree
59  OSBS_029.tif   316   174   346   214  Tree
60  OSBS_029.tif   220   208   251   244  Tree
```
---

## Fast iterations are the key to annotation success

Many projects have a linear concept of annotations with all the annotations collected before model testing. This is often a mistake. Especially in multi-class scenerios, start with a small number of annotations and allow the model to decide which images are most needed. This can be done in an automated way, or simply by looking at confusion matrices and predicted images. Imagine model developement as a pipeline, the more times you can iterate, the more rapidly your model will improve. For an example in airborne wildlife remote sensing, see the excellent paper by [B. Kellenberger et al. 2019](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8807383&casa_token=ZCCfJk3Fy-IAAAAA:UyZyftM25on1ZUijB1o4gBUWw8JBD5lmVkAvzZqL7PlQTvQMzOIK2n9A73swGUpYZYhARUbw&tag=1).

# Please consider making your annotations open-source!

The DeepForest backbone tree and bird models are not perfect. Please consider posting any annotations you make on zenodo, or sharing them with DeepForest mantainers. Open an [issue](https://github.com/weecology/DeepForest/issues) and tell us about the RGB data and annotations. For example, we are collecting tree annotations to create an [open-source benchmark](https://milliontrees.idtrees.org/). Please consider sharing data to make the models stronger and benefit you and other users. 

# How can I get new airborne data?

Many remote sensing assets are stored as an ImageServer within ArcGIS REST protocol. As part of automating airborne image workflows, we have tools that help work with these assets. For example [California NAIP data](https://map.dfg.ca.gov/arcgis/rest/services/Base_Remote_Sensing/NAIP_2020_CIR/ImageServer). 

More work is needed to encompass the *many* different param settings and specifications. We welcome pull requests from those with experience with  [WebMapTileServices](https://enterprise.arcgis.com/en/server/latest/publish-services/windows/wmts-services.htm).

## Specify a lat-long box and crop an ImageServer asset 

```python
from deepforest import utilities
import matplotlib.pyplot as plt
import rasterio as rio
import os
import asyncio
from aiolimiter import AsyncLimiter

async def main():
    url = "https://map.dfg.ca.gov/arcgis/rest/services/Base_Remote_Sensing/NAIP_2020_CIR/ImageServer/"
    xmin, ymin, xmax, ymax = -124.112622, 40.493891, -124.111536, 40.49457
    tmpdir = "<download_location>"
    image_name = "example_crop.tif"
    
    # Create semaphore and rate limiter
    semaphore = asyncio.Semaphore(1)
    limiter = AsyncLimiter(1, 0.05)

    # Ensure the directory exists
    os.makedirs(tmpdir, exist_ok=True)

    # Download the image
    filename = await utilities.download_ArcGIS_REST(semaphore, limiter, url, xmin, ymin, xmax, ymax, "EPSG:4326", savedir=tmpdir, image_name=image_name)

    # Check the saved file exists
    assert os.path.exists(os.path.join(tmpdir, image_name))

    # Confirm file has crs and show
    with rio.open(os.path.join(tmpdir, image_name)) as src:
        assert src.crs is not None
        # Show
        plt.imshow(src.read().transpose(1, 2, 0))
        plt.show()

# Run the async function
asyncio.run(main())
```

## Downloading a batch of images

```python
import asyncio
import pandas as pd
from aiolimiter import AsyncLimiter
from deepforest import utilities

async def download_crops(result_df, tmp_dir):
    url = 'https://map.dfg.ca.gov/arcgis/rest/services/Base_Remote_Sensing/NAIP_2022/ImageServer'
    
    # Semaphore to limit the number of concurrent downloads to 20
    semaphore = asyncio.Semaphore(20)

    # Rate limiter to control the download rate (1 request per 0.05 seconds)
    limiter = AsyncLimiter(1, 0.05)
    tasks = []
    for idx, row in result_df.iterrows():
        xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
        os.makedirs(tmp_dir, exist_ok=True)
        image_name = f"image_{idx}.tif"
        # Create an async task for each image download
        task = utilities.download_ArcGIS_REST(semaphore, limiter, url, xmin, ymin, xmax, ymax, "EPSG:4326", savedir=tmp_dir, image_name=image_name)
        tasks.append(task)
    
    # Wait for all tasks to complete
    await asyncio.gather(*tasks)

# Create a sample DataFrame
data = {
    'xmin': [-121.5, -122.0],
    'ymin': [37.0, 37.5],
    'xmax': [-121.0, -121.5],
    'ymax': [37.5, 38.0]
}
df = pd.DataFrame(data)
asyncio.run(download_crops(df, "/path/to/tmp_dir"))
```
