import os
import warnings
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
import xmltodict
import yaml
from tqdm import tqdm
from typing import Union

from PIL import Image
from deepforest import _ROOT
import json
import urllib.request
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import RevisionNotFoundError, HfHubHTTPError
from omegaconf import DictConfig, OmegaConf


def load_config(config_name: str = "config.yaml",
                overrides: Union[DictConfig, dict] = {}) -> DictConfig:
    """Load yaml configuration file via Hydra."""

    if not config_name.endswith('yaml'):
        config_name += '.yaml'

    if overrides is None:
        overrides = {}

    config_root = os.path.abspath(os.path.join(_ROOT, "conf"))
    config = OmegaConf.load(os.path.join(config_root, config_name))
    config.merge_with(overrides)

    return config


class DownloadProgressBar(tqdm):
    """Download progress bar class."""

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        Update class attributes
        Args:
            b:
            bsize:
            tsize:

        Returns:

        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def read_pascal_voc(xml_path):
    """Load annotations from xml format (e.g. RectLabel editor) and convert
    them into retinanet annotations format.

    Args:
        xml_path (str): Path to the annotations xml, formatted by RectLabel
    Returns:
        Annotations (pandas dataframe): in the
            format -> path-to-image.png,x1,y1,x2,y2,class_name
    """
    # parse
    with open(xml_path) as fd:
        doc = xmltodict.parse(fd.read())

    # grab xml objects
    try:
        tile_xml = doc["annotation"]["object"]
    except Exception as e:
        raise Exception("error {} for path {} with doc annotation{}".format(
            e, xml_path, doc["annotation"]))

    xmin = []
    xmax = []
    ymin = []
    ymax = []
    label = []

    if isinstance(tile_xml, list):
        # Construct frame if multiple trees
        for tree in tile_xml:
            xmin.append(tree["bndbox"]["xmin"])
            xmax.append(tree["bndbox"]["xmax"])
            ymin.append(tree["bndbox"]["ymin"])
            ymax.append(tree["bndbox"]["ymax"])
            label.append(tree['name'])
    else:
        xmin.append(tile_xml["bndbox"]["xmin"])
        xmax.append(tile_xml["bndbox"]["xmax"])
        ymin.append(tile_xml["bndbox"]["ymin"])
        ymax.append(tile_xml["bndbox"]["ymax"])
        label.append(tile_xml['name'])

    rgb_name = os.path.basename(doc["annotation"]["filename"])

    # set dtypes, check for floats and round
    xmin = [round_with_floats(x) for x in xmin]
    xmax = [round_with_floats(x) for x in xmax]
    ymin = [round_with_floats(x) for x in ymin]
    ymax = [round_with_floats(x) for x in ymax]

    annotations = pd.DataFrame({
        "image_path": rgb_name,
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax,
        "label": label
    })

    return annotations


def convert_point_to_bbox(gdf, buffer_size):
    """Convert an input point type annotation to a bounding box by buffering
    the point with a fixed size.

    Args:
        gdf (GeoDataFrame): The input point type annotation.
        buffer_size (float): The size of the buffer to be applied to the point.

    Returns:
        gdf (GeoDataFrame): The output bounding box type annotation.
    """
    # define in image coordinates and buffer to create a box
    gdf["geometry"] = [
        shapely.geometry.Point(x, y)
        for x, y in zip(gdf.geometry.x.astype(float), gdf.geometry.y.astype(float))
    ]
    gdf["geometry"] = [
        shapely.geometry.box(left, bottom, right, top)
        for left, bottom, right, top in gdf.geometry.buffer(buffer_size).bounds.values
    ]

    return gdf


def xml_to_annotations(xml_path):

    warnings.warn(
        "xml_to_annotations will be deprecated in 2.0. Please use read_pascal_voc instead.",
        DeprecationWarning)

    return read_pascal_voc(xml_path)


def shapefile_to_annotations(shapefile,
                             rgb=None,
                             root_dir=None,
                             buffer_size=None,
                             convert_point=False,
                             geometry_type=None,
                             save_dir=None):
    """Convert a shapefile of annotations into annotations csv file for
    DeepForest training and evaluation.

    Args:
        shapefile: Path to a shapefile on disk. If a label column is present, it will be used, else all labels are assumed to be "Tree"
        rgb: Path to the RGB image on disk
        root_dir: Optional directory to prepend to the image_path column
    Returns:
        results: a pandas dataframe
    """
    # Deprecation of previous arguments
    if geometry_type:
        warnings.warn(
            "geometry_type argument is deprecated and will be removed in DeepForest 2.0. The function will infer geometry from the shapefile directly.",
            DeprecationWarning)
    if save_dir:
        warnings.warn(
            "save_dir argument is deprecated and will be removed in DeepForest 2.0. The function will return a pandas dataframe instead of saving to disk.",
            DeprecationWarning)

    # Read shapefile
    if isinstance(shapefile, str):
        gdf = gpd.read_file(shapefile)
    else:
        gdf = shapefile.copy(deep=True)

    if rgb is None:
        if "image_path" not in gdf.columns:
            raise ValueError(
                "No image_path column found in shapefile, please specify rgb path")
        else:
            rgb = gdf.image_path.unique()[0]
            print("Found image_path column in shapefile, using {}".format(rgb))

    # Determine geometry type and report to user
    if gdf.geometry.type.unique().shape[0] > 1:
        raise ValueError(
            "Multiple geometry types found in shapefile. Please ensure all geometries are of the same type."
        )
    else:
        geometry_type = gdf.geometry.type.unique()[0]
        print("Geometry type of shapefile is {}".format(geometry_type))

    # Convert point to bounding box if desired
    if convert_point:
        if geometry_type == "Point":
            if buffer_size is None:
                raise ValueError(
                    "buffer_size must be specified to convert point to bounding box")
            gdf = convert_point_to_bbox(gdf, buffer_size)
        else:
            raise ValueError("convert_point is True, but geometry type is not Point")

    # raster bounds
    if root_dir:
        rgb = os.path.join(root_dir, rgb)
    with rasterio.open(rgb) as src:
        raster_crs = src.crs

    if gdf.crs:
        # If epsg is 4326, then the buffer size is in degrees, not meters, see https://github.com/weecology/DeepForest/issues/694
        if gdf.crs.to_string() == "EPSG:4326":
            raise ValueError(
                "The shapefile crs is in degrees. This function works for UTM and meter based crs only. see https://github.com/weecology/DeepForest/issues/694"
            )

        # Check matching the crs
        if not gdf.crs.to_string() == raster_crs.to_string():
            warnings.warn(
                "The shapefile crs {} does not match the image crs {}".format(
                    gdf.crs.to_string(), src.crs.to_string()), UserWarning)

        if src.crs is not None:
            print("CRS of shapefile is {}".format(gdf.crs))
            print("CRS of image is {}".format(raster_crs))
            gdf = geo_to_image_coordinates(gdf, src.bounds, src.res[0])

    # check for label column
    if "label" not in gdf.columns:
        raise ValueError(
            "No label column found in shapefile. Please add a column named 'label' to your shapefile."
        )
    else:
        gdf["label"] = gdf["label"]

    # add filename
    gdf["image_path"] = os.path.basename(rgb)

    return gdf


def determine_geometry_type(df):
    """Determine the geometry type of a prediction or annotation
    Args:
        df: a pandas dataframe
    Returns:
        geometry_type: a string of the geometry type
    """
    if type(df) in [pd.DataFrame, gpd.GeoDataFrame]:
        columns = df.columns
        if "geometry" in columns:
            df = gpd.GeoDataFrame(geometry=df['geometry'])
            geometry_type = df.geometry.type.unique()[0]
            if geometry_type == "Polygon":
                if (df.geometry.area == df.envelope.area).all():
                    return 'box'
                else:
                    return 'polygon'
            else:
                return 'point'
        elif "xmin" in columns and "ymin" in columns and "xmax" in columns and "ymax" in columns:
            geometry_type = "box"
        elif "polygon" in columns:
            geometry_type = "polygon"
        elif "x" in columns and "y" in columns:
            geometry_type = 'point'
        else:
            raise ValueError(
                "Could not determine geometry type from columns {}".format(columns))

    elif type(df) == dict:
        if 'boxes' in df.keys():
            geometry_type = "box"
        elif 'polygon' in df.keys():
            geometry_type = "polygon"
        elif 'points' in df.keys():
            geometry_type = "point"

    return geometry_type


def read_coco(json_file):
    """Read a COCO format JSON file and return a pandas dataframe.

    Args:
        json_file: Path to the COCO segmentation JSON file
    Returns:
        df: A pandas dataframe with image_path and geometry columns
    """
    with open(json_file, "r") as f:
        coco_data = json.load(f)

        polygons = []
        filenames = []
        image_ids = {image["id"]: image["file_name"] for image in coco_data["images"]}

        for annotation in coco_data["annotations"]:
            segmentation_mask = annotation["segmentation"][0]
            # Convert flat list to coordinate pairs
            pairs = [(segmentation_mask[i], segmentation_mask[i + 1])
                     for i in range(0, len(segmentation_mask), 2)]
            polygon = shapely.geometry.Polygon(pairs)
            filenames.append(image_ids[annotation["image_id"]])
            polygons.append(polygon.wkt)

        return pd.DataFrame({"image_path": filenames, "geometry": polygons})


def read_file(input, root_dir=None, image_path=None, label=None):
    """Read a file and return a geopandas dataframe.

    Args:
        input: A path to a file, a pandas DataFrame, or a geopandas GeoDataFrame.
        root_dir (str): Location of the image files, if not in the same directory as the annotations file.
        image_path (str, optional): If provided, this value will be assigned to a new 'image_path' column
            for every row in the dataframe. Only use this when the file contains annotations from a single image.
        label (str, optional): If provided, this value will be assigned to a new 'label' column
            for every row in the dataframe. Only use this when all annotations share the same label.

    Returns:
        df: A geopandas dataframe with the properly formatted geometry column.
    """
    if image_path is not None:
        warnings.warn(
            "You have passed an image_path. This value will be assigned to every row in the dataframe. "
            "Only use this if the file contains annotations for a single image.",
            UserWarning)

    if label is not None:
        warnings.warn(
            "You have passed a label. This value will be assigned to every row in the dataframe. "
            "Only use this if all annotations share the same label.", UserWarning)

    # read file
    if isinstance(input, str):
        if input.endswith(".csv"):
            df = pd.read_csv(input)
        elif input.endswith(".json"):
            df = read_coco(input)
        elif input.endswith((".shp", ".gpkg")):
            df = shapefile_to_annotations(input, root_dir=root_dir)
        elif input.endswith(".xml"):
            df = read_pascal_voc(input)
        else:
            raise ValueError(
                "File type {} not supported. DeepForest currently supports .csv, .shp, .gpkg, .xml, and .json files. See https://deepforest.readthedocs.io/en/latest/annotation.html "
                .format(df))
    else:
        # Explicitly check for GeoDataFrame first
        if isinstance(input, gpd.GeoDataFrame):
            return shapefile_to_annotations(input, root_dir=root_dir)
        elif isinstance(input, pd.DataFrame):
            df = input.copy(deep=True)
        else:
            raise ValueError(
                "Input must be a path to a file, geopandas or a pandas dataframe")

    if isinstance(df, pd.DataFrame):
        if df.empty:
            raise ValueError("No annotations in dataframe")
        # If the geometry column is present, convert to geodataframe directly
        if "geometry" in df.columns:
            if pd.api.types.infer_dtype(df['geometry']) == 'string':
                df['geometry'] = gpd.GeoSeries.from_wkt(df['geometry'])
        else:
            # Detect geometry type
            geom_type = determine_geometry_type(df)

            # Check for uppercase names and set to lowercase
            df.columns = [x.lower() for x in df.columns]

            # convert to geodataframe
            if geom_type == "box":
                df['geometry'] = df.apply(
                    lambda x: shapely.geometry.box(x.xmin, x.ymin, x.xmax, x.ymax),
                    axis=1)
            elif geom_type == "polygon":
                df['geometry'] = gpd.GeoSeries.from_wkt(df["polygon"])
            elif geom_type == "point":
                df["geometry"] = [
                    shapely.geometry.Point(x, y)
                    for x, y in zip(df.x.astype(float), df.y.astype(float))
                ]
            else:
                raise ValueError("Geometry type {} not supported".format(geom_type))

    # convert to geodataframe
    df = gpd.GeoDataFrame(df, geometry='geometry')

    # Handle missing 'image_path' and 'label' columns if not provided in the shapefile
    if "image_path" not in df.columns and image_path is not None:
        df["image_path"] = image_path
    elif "image_path" not in df.columns:
        warnings.warn(
            "'image_path' column is missing from shapefile, please specify the image path",
            UserWarning)

    if "label" not in df.columns and label is not None:
        df["label"] = label
    elif "label" not in df.columns:
        warnings.warn("'label' column is missing from shapefile, using default label",
                      UserWarning)
        df["label"] = "Unknown"  # Set default label if not provided

    # If root_dir is specified, add as attribute
    if root_dir is not None:
        df.root_dir = root_dir
    else:
        try:
            df.root_dir = os.path.dirname(input)
        except TypeError:
            warnings.warn(
                "root_dir argument for the location of images should be specified if input is not a path, returning without results.root_dir attribute",
                UserWarning)

    return df


def crop_raster(bounds, rgb_path=None, savedir=None, filename=None, driver="GTiff"):
    """
    Crop a raster to a bounding box, save as projected or unprojected crop
    Args:
        bounds: a tuple of (left, bottom, right, top) bounds
        rgb_path: path to the rgb image
        savedir: directory to save the crop
        filename: filename to save the crop "{}.tif".format(filename)"
        driver: rasterio driver to use, default to GTiff, can be 'GTiff' for projected data or 'PNG' unprojected data
    Returns:
        filename: path to the saved crop, if savedir specified
        img: a numpy array of the crop, if savedir not specified
    """
    left, bottom, right, top = bounds
    src = rasterio.open(rgb_path)
    if src.crs is None:
        # Read unprojected data using PIL and crop numpy array
        img = np.array(Image.open(rgb_path))
        img = img[bottom:top, left:right, :]
        img = np.rollaxis(img, 2, 0)
        cropped_transform = None
        if driver == "GTiff":
            warnings.warn(
                "Driver {} not supported for unprojected data, setting to 'PNG',".format(
                    driver), UserWarning)
            driver = "PNG"
    else:
        # Read projected data using rasterio and crop
        img = src.read(window=rasterio.windows.from_bounds(
            left, bottom, right, top, transform=src.transform))
        cropped_transform = rasterio.windows.transform(
            rasterio.windows.from_bounds(left,
                                         bottom,
                                         right,
                                         top,
                                         transform=src.transform), src.transform)
    if img.size == 0:
        raise ValueError("Bounds {} does not create a valid crop for source {}".format(
            bounds, src.transform))
    if savedir:
        res = src.res[0]
        height = (top - bottom) / res
        width = (right - left) / res

        # Write the cropped image to disk with transform
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        if driver == "GTiff":
            filename = "{}/{}.tif".format(savedir, filename)
            with rasterio.open(filename,
                               "w",
                               driver="GTiff",
                               height=height,
                               width=width,
                               count=img.shape[0],
                               dtype=img.dtype,
                               transform=cropped_transform) as dst:
                dst.write(img)
        elif driver == "PNG":
            # PNG driver does not support transform
            filename = "{}/{}.png".format(savedir, filename)
            with rasterio.open(filename,
                               "w",
                               driver="PNG",
                               height=height,
                               width=width,
                               count=img.shape[0],
                               dtype=img.dtype) as dst:
                dst.write(img)
        else:
            raise ValueError("Driver {} not supported".format(driver))

    if savedir:
        return filename
    else:
        return img


def crop_annotations_to_bounds(gdf, bounds):
    """
    Crop a geodataframe of annotations to a bounding box
    Args:
        gdf: a geodataframe of annotations
        bounds: a tuple of (left, bottom, right, top) bounds
    Returns:
        gdf: a geodataframe of annotations cropped to the bounds
    """
    # unpack image bounds
    left, bottom, right, top = bounds

    # Crop the annotations
    gdf.geometry = gdf.geometry.translate(xoff=-left, yoff=-bottom)

    return gdf


def geo_to_image_coordinates(gdf, image_bounds, image_resolution):
    """
    Convert from projected coordinates to image coordinates
    Args:
        gdf: a pandas type dataframe with columns: name, xmin, ymin, xmax, ymax. Name is the relative path to the root_dir arg.
        image_bounds: bounds of the image
        image_resolution: resolution of the image
    Returns:
        gdf: a geopandas dataframe with the transformed to image origin. CRS is removed
        """

    if len(image_bounds) != 4:
        raise ValueError("image_bounds must be a tuple of (left, bottom, right, top)")

    transformed_gdf = gdf.copy(deep=True)
    # unpack image bounds
    left, bottom, right, top = image_bounds

    transformed_gdf.geometry = transformed_gdf.geometry.translate(xoff=-left, yoff=-top)
    transformed_gdf.geometry = transformed_gdf.geometry.scale(xfact=1 / image_resolution,
                                                              yfact=-1 / image_resolution,
                                                              origin=(0, 0))
    transformed_gdf.crs = None

    return transformed_gdf


def round_with_floats(x):
    """Check if string x is float or int, return int, rounded if needed."""

    try:
        result = int(x)
    except BaseException:
        warnings.warn(
            "Annotations file contained non-integer coordinates. "
            "These coordinates were rounded to nearest int. "
            "All coordinates must correspond to pixels in the image coordinate system. "
            "If you are attempting to use projected data, "
            "first convert it into image coordinates see FAQ for suggestions.")
        result = int(np.round(float(x)))

    return result


def check_image(image):
    """Check an image is three channel, channel last format
        Args:
           image: numpy array
        Returns: None, throws error on assert
    """
    if not image.shape[2] == 3:
        raise ValueError("image is expected have three channels, channel last format, "
                         "found image with shape {}".format(image.shape))


def image_to_geo_coordinates(gdf, root_dir=None, flip_y_axis=False):
    """Convert from image coordinates to geographic coordinates.

    Args:
        gdf: A geodataframe.
        root_dir: Directory of images to lookup image_path column. If None, it will attempt to use gdf.root_dir.
        flip_y_axis: If True, reflect predictions over y axis to align with raster data in QGIS, which uses a negative y origin compared to numpy.

    Returns:
        transformed_gdf: A geospatial dataframe with the boxes optionally transformed to the target crs.
    """
    transformed_gdf = gdf.copy(deep=True)

    # Attempt to use root_dir from gdf if not provided
    if root_dir is None:
        if hasattr(gdf, "root_dir") and gdf.root_dir:
            root_dir = gdf.root_dir
        else:
            raise ValueError(
                "root_dir is not provided and could not be inferred from the predictions."
                "Ensure that the predictions have a root_dir attribute or pass root_dir explicitly."
            )

    plot_names = transformed_gdf.image_path.unique()
    if len(plot_names) > 1:
        raise ValueError(
            "This function projects a single plot's worth of data. Multiple plot names found: {}"
            .format(plot_names))
    else:
        plot_name = plot_names[0]

    rgb_path = "{}/{}".format(root_dir, plot_name)
    with rasterio.open(rgb_path) as dataset:
        bounds = dataset.bounds
        left, bottom, right, top = bounds
        pixelSizeX, pixelSizeY = dataset.res
        crs = dataset.crs
        transform = dataset.transform

    geom_type = determine_geometry_type(transformed_gdf)
    projected_geometry = []
    if geom_type == "box":
        # Convert image pixel locations to geographic coordinates
        coordinates = transformed_gdf.geometry.bounds
        xmin_coords, ymin_coords = rasterio.transform.xy(transform=transform,
                                                         rows=coordinates.miny,
                                                         cols=coordinates.minx,
                                                         offset='center')
        xmax_coords, ymax_coords = rasterio.transform.xy(transform=transform,
                                                         rows=coordinates.maxy,
                                                         cols=coordinates.maxx,
                                                         offset='center')

        for left, bottom, right, top in zip(xmin_coords, ymin_coords, xmax_coords,
                                            ymax_coords):
            geom = shapely.geometry.box(left, bottom, right, top)
            projected_geometry.append(geom)

    elif geom_type == "polygon":
        for geom in transformed_gdf.geometry:
            polygon_vertices = []
            for x, y in geom.exterior.coords:
                projected_vertices = rasterio.transform.xy(transform=transform,
                                                           rows=y,
                                                           cols=x,
                                                           offset='center')
                polygon_vertices.append(projected_vertices)
            geom = shapely.geometry.Polygon(polygon_vertices)
            projected_geometry.append(geom)

    elif geom_type == "point":
        x_coords, y_coords = rasterio.transform.xy(transform=transform,
                                                   rows=transformed_gdf.geometry.y,
                                                   cols=transformed_gdf.geometry.x,
                                                   offset='center')
        for x, y in zip(x_coords, y_coords):
            geom = shapely.geometry.Point(x, y)
            projected_geometry.append(geom)

    transformed_gdf.geometry = projected_geometry
    if flip_y_axis:
        # Numpy uses top left 0,0 origin, flip along y axis.
        # See https://gis.stackexchange.com/questions/306684/why-does-qgis-use-negative-y-spacing-in-the-default-raster-geotransform
        transformed_gdf.geometry = transformed_gdf.geometry.scale(xfact=1,
                                                                  yfact=-1,
                                                                  origin=(0, 0))

    # Assign crs
    transformed_gdf.crs = crs

    return transformed_gdf


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return tuple(zip(*batch))


def boxes_to_shapefile(df, root_dir, projected=True, flip_y_axis=False):
    """
    Convert from image coordinates to geographic coordinates
    Note that this assumes df is just a single plot being passed to this function
    Args:
       df: a pandas type dataframe with columns: name, xmin, ymin, xmax, ymax. Name is the relative path to the root_dir arg.
       root_dir: directory of images to lookup image_path column
       projected: If True, convert from image to geographic coordinates, if False, keep in image coordinate system
       flip_y_axis: If True, reflect predictions over y axis to align with raster data in QGIS, which uses a negative y origin compared to numpy. See https://gis.stackexchange.com/questions/306684/why-does-qgis-use-negative-y-spacing-in-the-default-raster-geotransform
    Returns:
       df: a geospatial dataframe with the boxes optionally transformed to the target crs
    """

    warnings.warn(
        "This function will be deprecated in DeepForest 2.0, as it only can process boxes and the API now includes point and polygon annotations. Please use image_to_geo_coordinates instead.",
        DeprecationWarning)

    # Raise a warning and confirm if a user sets projected to True when flip_y_axis is True.
    if flip_y_axis and projected:
        warnings.warn(
            "flip_y_axis is {}, and projected is {}. In most cases, projected should be False when inverting y axis. Setting projected=False"
            .format(flip_y_axis, projected), UserWarning)
        projected = False

    plot_names = df.image_path.unique()
    if len(plot_names) > 1:
        raise ValueError("This function projects a single plots worth of data. "
                         "Multiple plot names found {}".format(plot_names))
    else:
        plot_name = plot_names[0]

    rgb_path = "{}/{}".format(root_dir, plot_name)
    with rasterio.open(rgb_path) as dataset:
        bounds = dataset.bounds
        pixelSizeX, pixelSizeY = dataset.res
        crs = dataset.crs
        transform = dataset.transform

    if projected:
        # Convert image pixel locations to geographic coordinates
        xmin_coords, ymin_coords = rasterio.transform.xy(transform=transform,
                                                         rows=df.ymin,
                                                         cols=df.xmin,
                                                         offset='center')

        xmax_coords, ymax_coords = rasterio.transform.xy(transform=transform,
                                                         rows=df.ymax,
                                                         cols=df.xmax,
                                                         offset='center')

        # One box polygon for each tree bounding box
        # Careful of single row edge case where
        # xmin_coords comes out not as a list, but as a float
        if type(xmin_coords) == float:
            xmin_coords = [xmin_coords]
            ymin_coords = [ymin_coords]
            xmax_coords = [xmax_coords]
            ymax_coords = [ymax_coords]

        box_coords = zip(xmin_coords, ymin_coords, xmax_coords, ymax_coords)
        box_geoms = [
            shapely.geometry.box(xmin, ymin, xmax, ymax)
            for xmin, ymin, xmax, ymax in box_coords
        ]

        geodf = gpd.GeoDataFrame(df, geometry=box_geoms)
        geodf.crs = crs

        return geodf

    else:
        if flip_y_axis:
            # See https://gis.stackexchange.com/questions/306684/why-does-qgis-use-negative-y-spacing-in-the-default-raster-geotransform
            # Numpy uses top left 0,0 origin, flip along y axis.
            df['geometry'] = df.apply(
                lambda x: shapely.geometry.box(x.xmin, -x.ymin, x.xmax, -x.ymax), axis=1)
        else:
            df['geometry'] = df.apply(
                lambda x: shapely.geometry.box(x.xmin, x.ymin, x.xmax, x.ymax), axis=1)
        df = gpd.GeoDataFrame(df, geometry="geometry")

        return df


def annotations_to_shapefile(df, transform, crs):
    """Convert output from predict_image and  predict_tile to a geopandas
    data.frame.

    Args:
        df: prediction data.frame with columns  ['xmin','ymin','xmax','ymax','label','score']
        transform: A rasterio affine transform object
        crs: A rasterio crs object
    Returns:
        results: a geopandas dataframe where every entry is the bounding box for a detected tree.
    """

    raise NotImplementedError(
        "This function is deprecated. Please use image_to_geo_coordinates instead.")


def project_boxes(df, root_dir, transform=True):
    """
    Convert from image coordinates to geographic coordinates
    Note that this assumes df is just a single plot being passed to this function
    df: a pandas type dataframe with columns: name, xmin, ymin, xmax, ymax.
    Name is the relative path to the root_dir arg.
    root_dir: directory of images to lookup image_path column
    transform: If true, convert from image to geographic coordinates
    """
    raise NotImplementedError(
        "This function is deprecated. Please use image_to_geo_coordinates instead.")

    return df
