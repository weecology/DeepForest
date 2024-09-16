# Reading in data

The most time confusing part of many open source projects in getting the data in and out. This is because there are so many formats and ways a user might arrive at the package. DeepForest has collated many use cases into a single read_file function that will attempt to read many common data formats, projected/unprojected data and create a dataframe ready for DeepForest functions.

## Annotation geometries and coordinate systems

DeepForest was originally designed for bounding box annotations. As of DeepForest 1.4.0, Point and Polygon annotation are also supported. There are two ways to format annotations, depending on what kind of annotation platform you were using. 'read_file' can read points, polygons, boxes, in both coordinates with reference to image origin (top left 0,0), as well as projected coordinates on the earth's surface.

**Note:** For csv files, coordinates are expected to be in the image coordinate system, not the projected coordinate (such as lat long or UTM).

### Boxes

#### CSV

Here the annotations are in plain csv files, with coordinates relative to image origin.

```
image_path,xmin,ymin,xmax,ymax,label
OSBS_029.tif,203,67,227,90,Tree
OSBS_029.tif,256,99,288,140,Tree
OSBS_029.tif,166,253,225,304,Tree
OSBS_029.tif,365,2,400,27,Tree
OSBS_029.tif,312,13,349,47,Tree
OSBS_029.tif,365,21,400,70,Tree
OSBS_029.tif,278,1,312,37,Tree
OSBS_029.tif,364,204,400,246,Tree
```

```
filename = get_data("OSBS_029.csv")
utilities.read_file(filename)
      image_path  xmin  ymin  xmax  ymax label                                           geometry
0   OSBS_029.tif   203    67   227    90  Tree  POLYGON ((227.000 67.000, 227.000 90.000, 203....
1   OSBS_029.tif   256    99   288   140  Tree  POLYGON ((288.000 99.000, 288.000 140.000, 256...
2   OSBS_029.tif   166   253   225   304  Tree  POLYGON ((225.000 253.000, 225.000 304.000, 16...
3   OSBS_029.tif   365     2   400    27  Tree  POLYGON ((400.000 2.000, 400.000 27.000, 365.0...
4   OSBS_029.tif   312    13   349    47  Tree  POLYGON ((349.000 13.000, 349.000 47.000, 312....
```
> **Note:**  To mantain continuity with <1.4.0 versions, the function for boxes continues to output xmin, ymin, xmax, ymax columns as individual columns as well. *

### Shapefiles

Geographic data can also be saved as shapefiles with projected coordinates

```
gdf.iloc[0]
geometry      POLYGON ((404222.4 3285121.5, 404222.4 3285122...
label                                                      Tree
image_path    /Users/benweinstein/Documents/DeepForest/deepf...
Name: 0, dtype: object
```

These coordinates are made relative to the image origin when the file is read.

```
shp = utilities.read_file(input="/path/to/boxes_shapefile.shp")
shp.head()
  label    image_path                                           geometry
0  Tree  OSBS_029.tif  POLYGON ((105.000 214.000, 95.000 214.000, 95....
1  Tree  OSBS_029.tif  POLYGON ((205.000 214.000, 195.000 214.000, 19...
```

## Points

### CSV

```
x,y,label
10,20,Tree
15,30,Tree
```

### Shapefile
```
shp = utilities.read_file(input="/path/to/points_shapefile.shp")
annotations.head()
  label    image_path                 geometry
0  Tree  OSBS_029.tif  POINT (100.000 209.000)
1  Tree  OSBS_029.tif  POINT (200.000 209.000)
```

## Polygons

## CSV

Polygons are expressed well-known-text format. Learn more about (wkt)[https://en.wikipedia.org/wiki/Well-known_text_representation_of_geometry]. 

```
"POLYGON ((0 0, 0 2, 1 1, 1 0, 0 0))",Tree,OSBS_029.png
"POLYGON ((2 2, 2 4, 3 3, 3 2, 2 2))",Tree,OSBS_029.png
```

## Shapefile

```
shp = utilities.read_file(input="/path/to/polygons_shapefile.shp")
annotations.head()
  label    image_path                                           geometry
0  Tree  OSBS_029.png  POLYGON ((0.00000 0.00000, 0.00000 2.00000, 1....
1  Tree  OSBS_029.png  POLYGON ((2.00000 2.00000, 2.00000 4.00000, 3....
```

# Writing Data
 
Objects read in by deepforest.utilities.read_file are [geopandas geodataframes](https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.html). They can be exported as csv files or shapefiles. 

```
annotations.to_csv("annotations.csv")
annotations.to_file("annotations.shp")
```

## Converting from image to geographic coordinates

Standard computer visions models have no concept of geographic information, which is why we use the image coordinate system to represent coordinates within DeepForest. If you want to convert predictions back into geographic coordinates, we provide a utility to go from image to geo coordinates based on coordinate reference system of the file in the image_path column.

```
from deepforest import get_data
import rasterio as rio
import geopandas as gpd
from matplotlib import pyplot as plt

annotations = get_data("2018_SJER_3_252000_4107000_image_477.csv")
path_to_raster = get_data("2018_SJER_3_252000_4107000_image_477.tif")
src = rio.open(path_to_raster)
original = utilities.read_file(annotations)

geo_coords = utilities.image_to_geo_coordinates(original, root_dir=os.path.dirname(path_to_raster))
src_window = geometry.box(*src.bounds)

fig, ax = plt.subplots(figsize=(10, 10))
gpd.GeoSeries(src_window).plot(ax=ax, color="blue", alpha=0.5)
geo_coords.plot(ax=ax, color="red")
plt.show()
```