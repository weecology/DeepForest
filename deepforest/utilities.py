import os
import warnings
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
import shapely
import xmltodict
import yaml
from tqdm import tqdm
import aiohttp

from PIL import Image
from deepforest import _ROOT
from shapely.geometry import Point, box
from pyproj import CRS
import json
import warnings
import geopandas as gpd
import rasterio
import shapely
from tqdm import tqdm
from deepforest import _ROOT
from shapely.geometry import box
import json
import requests
import urllib.request


def read_config(config_path):
    """Read config yaml file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

    except Exception as e:
        raise FileNotFoundError("There is no config at {}, yields {}".format(
            config_path, e))

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


def use_bird_release(
        save_dir=os.path.join(_ROOT, "data/"), prebuilt_model="bird", check_release=True):
    """
    Check the existence of, or download the latest model release from github
    Args:
        save_dir: Directory to save filepath, default to "data" in deepforest repo
        prebuilt_model: Currently only accepts "NEON", but could be expanded to include other prebuilt models. The local model will be called prebuilt_model.h5 on disk.
        check_release (logical): whether to check github for a model recent release. In cases where you are hitting the github API rate limit, set to False and any local model will be downloaded. If no model has been downloaded an error will raise.
    Returns: release_tag, output_path (str): path to downloaded model

    """

    # Naming based on pre-built model
    output_path = os.path.join(save_dir, prebuilt_model + ".pt")

    if check_release:
        # Find latest github tag release from the DeepLidar repo
        _json = json.loads(
            urllib.request.urlopen(
                urllib.request.Request(
                    'https://api.github.com/repos/Weecology/BirdDetector/releases/latest',
                    headers={'Accept': 'application/vnd.github.v3+json'},
                )).read())
        asset = _json['assets'][0]
        url = asset['browser_download_url']

        # Check the release tagged locally
        try:
            release_txt = pd.read_csv(save_dir + "current_bird_release.csv")
        except BaseException:
            release_txt = pd.DataFrame({"current_bird_release": [None]})

        # Download the current release it doesn't exist
        if not release_txt.current_bird_release[0] == _json["html_url"]:

            print("Downloading model from BirdDetector release {}, see {} for details".
                  format(_json["tag_name"], _json["html_url"]))

            with DownloadProgressBar(unit='B',
                                     unit_scale=True,
                                     miniters=1,
                                     desc=url.split('/')[-1]) as t:
                urllib.request.urlretrieve(url,
                                           filename=output_path,
                                           reporthook=t.update_to)

            print("Model was downloaded and saved to {}".format(output_path))

            # record the release tag locally
            release_txt = pd.DataFrame({"current_bird_release": [_json["html_url"]]})
            release_txt.to_csv(save_dir + "current_bird_release.csv")
        else:
            print("Model from BirdDetector Repo release {} was already downloaded. "
                  "Loading model from file.".format(_json["html_url"]))

        return _json["html_url"], output_path
    else:
        try:
            release_txt = pd.read_csv(save_dir + "current_release.csv")
        except BaseException:
            raise ValueError("Check release argument is {}, but no release has been "
                             "previously downloaded".format(check_release))

        return release_txt.current_release[0], output_path


def use_release(
        save_dir=os.path.join(_ROOT, "data/"), prebuilt_model="NEON", check_release=True):
    """
    Check the existence of, or download the latest model release from github
    Args:
        save_dir: Directory to save filepath, default to "data" in deepforest repo
        prebuilt_model: Currently only accepts "NEON", but could be expanded to include other prebuilt models. The local model will be called prebuilt_model.h5 on disk.
        check_release (logical): whether to check github for a model recent release. In cases where you are hitting the github API rate limit, set to False and any local model will be downloaded. If no model has been downloaded an error will raise.
        
    Returns: release_tag, output_path (str): path to downloaded model

    """
    # Naming based on pre-built model
    output_path = os.path.join(save_dir, prebuilt_model + ".pt")

    if check_release:
        # Find latest github tag release from the DeepLidar repo
        _json = json.loads(
            urllib.request.urlopen(
                urllib.request.Request(
                    'https://api.github.com/repos/Weecology/DeepForest/releases/latest',
                    headers={'Accept': 'application/vnd.github.v3+json'},
                )).read())
        asset = _json['assets'][0]
        url = asset['browser_download_url']

        # Check the release tagged locally
        try:
            release_txt = pd.read_csv(save_dir + "current_release.csv")
        except BaseException:
            release_txt = pd.DataFrame({"current_release": [None]})

        # Download the current release it doesn't exist
        if not release_txt.current_release[0] == _json["html_url"]:

            print("Downloading model from DeepForest release {}, see {} "
                  "for details".format(_json["tag_name"], _json["html_url"]))

            with DownloadProgressBar(unit='B',
                                     unit_scale=True,
                                     miniters=1,
                                     desc=url.split('/')[-1]) as t:
                urllib.request.urlretrieve(url,
                                           filename=output_path,
                                           reporthook=t.update_to)

            print("Model was downloaded and saved to {}".format(output_path))

            # record the release tag locally
            release_txt = pd.DataFrame({"current_release": [_json["html_url"]]})
            release_txt.to_csv(save_dir + "current_release.csv")
        else:
            print("Model from DeepForest release {} was already downloaded. "
                  "Loading model from file.".format(_json["html_url"]))

        return _json["html_url"], output_path
    else:
        try:
            release_txt = pd.read_csv(save_dir + "current_release.csv")
        except BaseException:
            raise ValueError("Check release argument is {}, but no release "
                             "has been previously downloaded".format(check_release))

        return release_txt.current_release[0], output_path


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


# TO DO -> Should this whole function hae a deprecation warning? Shouldn't users just use the read_file function?
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
        print("CRS of shapefile is {}".format(src.crs))
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


def read_file(input, root_dir=None):
    """Read a file and return a geopandas dataframe.

    This is the main entry point for reading annotations into deepforest.
    Args:
        input: a path to a file or a pandas dataframe
        root_dir: Optional directory to prepend to the image_path column
    Returns:
        df: a geopandas dataframe with the properly formatted geometry column
    """
    # read file
    if isinstance(input, str):
        if input.endswith(".csv"):
            df = pd.read_csv(input)
        elif input.endswith((".shp", ".gpkg")):
            df = shapefile_to_annotations(input, root_dir=root_dir)
        elif input.endswith(".xml"):
            df = xml_to_annotations(input)
        else:
            raise ValueError(
                "File type {} not supported. DeepForest currently supports .csv, .shp or .xml files. See https://deepforest.readthedocs.io/en/latest/annotation.html "
                .format(df))
    else:
        if type(input) == pd.DataFrame:
            df = input.copy(deep=True)
        elif type(input) == gpd.GeoDataFrame:
            return shapefile_to_annotations(input, root_dir=root_dir)
        else:
            raise ValueError(
                "Input must be a path to a file, geopandas or a pandas dataframe")

    if type(df) == pd.DataFrame:
        if df.empty:
            raise ValueError("No annotations in dataframe")
        # If the geometry column is present, convert to geodataframe directly
        if "geometry" in df.columns:
            df['geometry'] = gpd.GeoSeries.from_wkt(df['geometry'])
            df.crs = None
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


def image_to_geo_coordinates(gdf, root_dir, flip_y_axis=False):
    """Convert from image coordinates to geographic coordinates.

    Args:
        gdf: A geodataframe.
        root_dir: Directory of images to lookup image_path column.
        flip_y_axis: If True, reflect predictions over y axis to align with raster data in QGIS, which uses a negative y origin compared to numpy.

    Returns:
        transformed_gdf: A geospatial dataframe with the boxes optionally transformed to the target crs.
    """
    transformed_gdf = gdf.copy(deep=True)
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


async def download_ArcGIS_REST(semaphore,
                               limiter,
                               url,
                               xmin,
                               ymin,
                               xmax,
                               ymax,
                               bbox_crs,
                               savedir,
                               additional_params=None,
                               image_name="image.tiff",
                               download_service="exportImage"):
    """
    Fetch data from a web server using geographic boundaries and save it as a GeoTIFF file.
    This function is used to download data from an ArcGIS REST service, not WMTS or WMS services.
    Example url: https://gis.calgary.ca/arcgis/rest/services/pub_Orthophotos/CurrentOrthophoto/ImageServer/
    
    Parameters:
    - semaphore: An asyncio.Semaphore instance to limit concurrent downloads.
    - limiter: An asyncio-based rate limiter to control the download rate.
    - url: The base URL of the ArcGIS REST service 
    - xmin: The minimum x-coordinate (longitude).
    - ymin: The minimum y-coordinate (latitude).
    - xmax: The maximum x-coordinate (longitude).
    - ymax: The maximum y-coordinate (latitude).
    - bbox_crs: The coordinate reference system (CRS) of the bounding box.
    - savedir: The directory to save the downloaded image.
    - additional_params: Additional query parameters to include in the request (default is None).
    - image_name: The name of the image file to be saved (default is "image.tiff").
    - download_service: The specific service to use for downloading the image (default is "exportImage").

    Returns:
    - The file path of the saved image if the download is successful.
    - None if the download fails.
    
    Function usage:
        import asyncio
        from asyncio import Semaphore
        from aiolimiter import AsyncLimiter
        from deepforest import utilities

        async def main():
            semaphore = Semaphore(10)
            limiter = AsyncLimiter(1, 0.5)
            url = "https://map.dfg.ca.gov/arcgis/rest/services/Base_Remote_Sensing/NAIP_2020_CIR/ImageServer/
            xmin, ymin, xmax, ymax = -114.0719, 51.0447, -114.001, 51.075
            bbox_crs = "EPSG:4326"
            savedir = "/path/to/save"
            image_name = "example_image.tiff"
            await utilities.download_ArcGIS_REST(semaphore, limiter, url, xmin, ymin, xmax, ymax, bbox_crs, savedir, image_name=image_name)

        asyncio.run(main())
        
    For more details on the function usage, refer to https://deepforest.readthedocs.io/en/latest/annotation.html.
 
    """

    params = {"f": "json"}

    async with aiohttp.ClientSession() as session:
        await semaphore.acquire()
        async with limiter:
            try:
                async with session.get(url, params=params) as resp:
                    response = await resp.read()
                    response_dict = json.loads(response)
                    spatialReference = response_dict["spatialReference"]
                    if "latestWkid" in spatialReference:
                        wkid = spatialReference["latestWkid"]
                        crs = CRS.from_epsg(wkid)
                    elif 'wkt' in spatialReference:
                        crs = CRS.from_wkt(spatialReference['wkt'])

                bbox = f"{xmin},{ymin},{xmax},{ymax}"
                bounds = gpd.GeoDataFrame(geometry=[
                    shapely.geometry.box(xmin, ymin, xmax, ymax)
                ],
                                          crs=bbox_crs).to_crs(crs).bounds

                params.update({
                    "bbox":
                        f"{bounds.minx[0]},{bounds.miny[0]},{bounds.maxx[0]},{bounds.maxy[0]}",
                    "f":
                        "image",
                    'format':
                        'tiff',
                })

                if additional_params:
                    params.update(additional_params)

                download_url_service = f"{url}/{download_service}"
                async with session.get(download_url_service, params=params) as resp:
                    response = await resp.read()
                    if resp.status == 200:
                        filename = f"{savedir}/{image_name}"
                        with open(filename, "wb") as f:
                            f.write(response)
                        return filename
                    else:
                        raise Exception(f"Failed to fetch data: {resp.status}")

            except Exception as e:
                print(f"Error downloading image {image_name}: {e}")
            finally:
                semaphore.release()
