# Reading in Data

The most time-consuming part of many open-source projects is getting the data in and out. This is because there are so many formats and ways a user might interact with the package. DeepForest has collated many use cases into a single `read_file` function that will attempt to read many common data formats, both projected and unprojected, and create a dataframe ready for DeepForest functions.

## Annotation Geometries and Coordinate Systems

DeepForest was originally designed for bounding box annotations. As of DeepForest 1.4.0, point and polygon annotations are also supported. There are two ways to format annotations, depending on the annotation platform you are using. `read_file` can read points, polygons, and boxes, in both image coordinate systems (relative to image origin at top-left 0,0) as well as projected coordinates on the Earth's surface. The `read_file` method also appends the location of the current image directory as an attribute. To access this attribute use

```
filename = get_data("OSBS_029.csv")

```

**Note:** For CSV files, coordinates are expected to be in the image coordinate system, not projected coordinates (such as latitude/longitude or UTM).

### Boxes

#### CSV

Here, the annotations are in plain CSV files, with coordinates relative to the image origin.

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

```python
filename = get_data("OSBS_029.csv")
df = utilities.read_file(filename)
```

Example output:

```
      image_path  xmin  ymin  xmax  ymax label                                           geometry
0   OSBS_029.tif   203    67   227    90  Tree  POLYGON ((227.000 67.000, 227.000 90.000, 203....
1   OSBS_029.tif   256    99   288   140  Tree  POLYGON ((288.000 99.000, 288.000 140.000, 256...
2   OSBS_029.tif   166   253   225   304  Tree  POLYGON ((225.000 253.000, 225.000 304.000, 16...
3   OSBS_029.tif   365     2   400    27  Tree  POLYGON ((400.000 2.000, 400.000 27.000, 365.0...
4   OSBS_029.tif   312    13   349    47  Tree  POLYGON ((349.000 13.000, 349.000 47.000, 312....
```

**Note:** To maintain continuity with versions < 1.4.0, the function for boxes continues to output `xmin`, `ymin`, `xmax`, and `ymax` columns as individual columns as well.

The location of these image files is saved in the root_dir attribute

```
df.root_dir
'/Users/benweinstein/Documents/DeepForest/deepforest/data'
```


#### Shapefiles

Geographic data can also be saved as shapefiles with projected coordinates.

Example:

```
gdf.iloc[0]
geometry      POLYGON ((404222.4 3285121.5, 404222.4 3285122...
label                                                      Tree
image_path    /Users/benweinstein/Documents/DeepForest/deepf...
Name: 0, dtype: object
```

These coordinates are made relative to the image origin when the file is read.

```python
shp = utilities.read_file(input="/path/to/boxes_shapefile.shp")
shp.head()
```

Example output:

```
  label    image_path                                           geometry
0  Tree  OSBS_029.tif  POLYGON ((105.000 214.000, 95.000 214.000, 95....
1  Tree  OSBS_029.tif  POLYGON ((205.000 214.000, 195.000 214.000, 19...
```

### Points

#### CSV

Example:

```
x,y,label
10,20,Tree
15,30,Tree
```

#### Shapefile

```python
shp = utilities.read_file(input="/path/to/points_shapefile.shp")
annotations.head()
```

Example output:

```
  label    image_path                 geometry
0  Tree  OSBS_029.tif  POINT (100.000 209.000)
1  Tree  OSBS_029.tif  POINT (200.000 209.000)
```

### Polygons

#### CSV

Polygons are expressed in well-known-text (WKT) format. Learn more about [WKT](https://en.wikipedia.org/wiki/Well-known_text_representation_of_geometry).

```
"POLYGON ((0 0, 0 2, 1 1, 1 0, 0 0))",Tree,OSBS_029.png
"POLYGON ((2 2, 2 4, 3 3, 3 2, 2 2))",Tree,OSBS_029.png
```

#### Shapefile

```python
shp = utilities.read_file(input="/path/to/polygons_shapefile.shp")
annotations.head()
```

Example output:

```
  label    image_path                                           geometry
0  Tree  OSBS_029.png  POLYGON ((0.00000 0.00000, 0.00000 2.00000, 1....
1  Tree  OSBS_029.png  POLYGON ((2.00000 2.00000, 2.00000 4.00000, 3....
```
