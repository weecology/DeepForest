# Writing Data

Objects read in by `deepforest.utilities.read_file` are [geopandas GeoDataFrames](https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.html). They can be exported as CSV files or shapefiles.

```python
annotations.to_csv("annotations.csv")
annotations.to_file("annotations.shp")
```

## Converting from Image to Geographic Coordinates

Standard computer vision models have no concept of geographic information, which is why we use the image coordinate system to represent coordinates within DeepForest. If you want to convert predictions back into geographic coordinates, we provide a utility to go from image to geo coordinates based on the coordinate reference system of the file in the `image_path` column.

```python
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