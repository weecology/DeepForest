"""Utilities model"""
import json
import os
import urllib
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
import xmltodict
import yaml
from tqdm import tqdm

from deepforest import _ROOT


def read_config(config_path):
    """Read config yaml file"""
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


def use_release(save_dir=os.path.join(_ROOT, "data/"), prebuilt_model="NEON"):
    """
    Check the existence of, or download the latest model release from github
    Args:
        save_dir: Directory to save filepath, default to "data" in deepforest repo
        prebuilt_model: Currently only accepts "NEON", but could be expanded to include other prebuilt models. The local model will be called prebuilt_model.h5 on disk.

    Returns: release_tag, output_path (str): path to downloaded model

    """
    # Find latest github tag release from the DeepLidar repo
    _json = json.loads(
        urllib.request.urlopen(
            urllib.request.Request(
                'https://api.github.com/repos/Weecology/DeepForest-pytorch/releases/latest',
                headers={'Accept': 'application/vnd.github.v3+json'},
            )).read())
    asset = _json['assets'][0]
    url = asset['browser_download_url']

    # Naming based on pre-built model
    output_path = os.path.join(save_dir, prebuilt_model + ".pt")

    # Check the release tagged locally
    try:
        release_txt = pd.read_csv(save_dir + "current_release.csv")
    except BaseException:
        release_txt = pd.DataFrame({"current_release": [None]})

    # Download the current release it doesn't exist
    if not release_txt.current_release[0] == _json["html_url"]:

        print("Downloading model from DeepForest release {}, see {} for details".format(
            _json["tag_name"], _json["html_url"]))

        with DownloadProgressBar(unit='B',
                                 unit_scale=True,
                                 miniters=1,
                                 desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

        print("Model was downloaded and saved to {}".format(output_path))

        # record the release tag locally
        release_txt = pd.DataFrame({"current_release": [_json["html_url"]]})
        release_txt.to_csv(save_dir + "current_release.csv")
    else:
        print("Model from DeepForest release {} was already downloaded. "
              "Loading model from file.".format(_json["html_url"]))

    return _json["html_url"], output_path


def xml_to_annotations(xml_path):
    """
    Load annotations from xml format (e.g. RectLabel editor) and convert
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
    return (annotations)


def shapefile_to_annotations(shapefile, rgb, savedir="."):
    """
    Convert a shapefile of annotations into annotations csv file for DeepForest training and evaluation
    Args:
        shapefile: Path to a shapefile on disk. If a label column is present, it will be used, else all labels are assumed to be "Tree"
        rgb: Path to the RGB image on disk
        savedir: Directory to save csv files
    Returns:
        results: a pandas dataframe
    """
    # Read shapefile
    gdf = gpd.read_file(shapefile)

    # get coordinates
    df = gdf.geometry.bounds

    # raster bounds
    with rasterio.open(rgb) as src:
        left, bottom, right, top = src.bounds
        resolution = src.res[0]

    # Transform project coordinates to image coordinates
    df["tile_xmin"] = (df.minx - left) / resolution
    df["tile_xmin"] = df["tile_xmin"].astype(int)

    df["tile_xmax"] = (df.maxx - left) / resolution
    df["tile_xmax"] = df["tile_xmax"].astype(int)

    # UTM is given from the top, but origin of an image is top left

    df["tile_ymax"] = (top - df.miny) / resolution
    df["tile_ymax"] = df["tile_ymax"].astype(int)

    df["tile_ymin"] = (top - df.maxy) / resolution
    df["tile_ymin"] = df["tile_ymin"].astype(int)

    # Add labels is they exist
    if "label" in gdf.columns:
        df["label"] = gdf["label"]
    else:
        df["label"] = "Tree"

    # add filename
    df["image_path"] = os.path.basename(rgb)

    # select columns
    result = df[[
        "image_path", "tile_xmin", "tile_ymin", "tile_xmax", "tile_ymax", "label"
    ]]
    result = result.rename(columns={
        "tile_xmin": "xmin",
        "tile_ymin": "ymin",
        "tile_xmax": "xmax",
        "tile_ymax": "ymax"
    })

    # ensure no zero area polygons due to rounding to pixel size
    result = result[~(result.xmin == result.xmax)]
    result = result[~(result.ymin == result.ymax)]

    return result


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


def check_file(df):
    """Check a file format for correct column names and structure"""

    if not all(x in df.columns
               for x in ["image_path", "xmin", "xmax", "ymin", "ymax", "label"]):
        raise IOError(
            "Input file has incorrect column names, the following columns must exist 'image_path','xmin','ymin','xmax','ymax','label'."
        )

    return df


def check_image(image):
    """Check an image is three channel, channel last format
        Args:
           image: numpy array
        Returns: None, throws error on assert
    """
    if not image.shape[2] == 3:
        raise ValueError("image is expected have three channels, channel last format, found image with shape {}".format(image.shape))
    
def project_boxes(df, root_dir, transform=True):
    """
    Convert from image coordinates to geopgraphic cooridinates
    Note that this assumes df is just a single plot being passed to this function
    df: a pandas type dataframe with columns name, xmin, ymin, xmax, ymax, name. Name is the relative path to the root_dir arg.
    root_dir: directory of images
    transform: If true, convert from image to geographic coordinates
    """
    plot_names = df.image_path.unique()
    if len(plot_names) > 1:
        raise ValueError(
            "This function projects a single plots worth of data. Multiple plot names found {}"
            .format(plot_names))
    else:
        plot_name = plot_names[0]

    rgb_path = "{}/{}".format(root_dir, plot_name)
    with rasterio.open(rgb_path) as dataset:
        bounds = dataset.bounds
        pixelSizeX, pixelSizeY = dataset.res
        crs = dataset.crs

    if transform:
        # subtract origin. Recall that numpy origin is top left! Not bottom
        # left.
        df["xmin"] = (df["xmin"].astype(float) * pixelSizeX) + bounds.left
        df["xmax"] = (df["xmax"].astype(float) * pixelSizeX) + bounds.left
        df["ymin"] = bounds.top - (df["ymin"].astype(float) * pixelSizeY)
        df["ymax"] = bounds.top - (df["ymax"].astype(float) * pixelSizeY)

    # combine column to a shapely Box() object, save shapefile
    df['geometry'] = df.apply(
        lambda x: shapely.geometry.box(x.xmin, x.ymin, x.xmax, x.ymax), axis=1)
    df = gpd.GeoDataFrame(df, geometry='geometry')

    df.crs = crs

    return df


def collate_fn(batch):
    batch = list(filter(lambda x : x is not None, batch))
        
    return tuple(zip(*batch))
