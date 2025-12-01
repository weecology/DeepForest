import json
import os
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
import xmltodict
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

from deepforest import _ROOT
from deepforest.conf.schema import Config as StructuredConfig


def load_config(
    config_name: str = "config.yaml",
    overrides: DictConfig | dict | None = None,
    strict: bool = False,
) -> DictConfig:
    """Loads the DeepForest structured config, merges with YAML and overrides.

    If config_name is found to be a valid path, it will be loaded, otherwise
    it's assumed to be a named config in the deepforest package. The loaded
    config will be validated against the schema.

    You can load a config in strict mode, which will not allow any additional keys.
    This may be useful for debugging, but it may cause issues due to the way OmegaConf
    handles dictionary config items, like label_dict.

    Args:
        config_name (str): Path to config file
        overrides (DictConfig or dict): Overrides to config
        strict (bool): If True, disallows unexpected keys.

    Returns:
        config (DictConfig): composed configuration
    """

    if not config_name.endswith(".yaml"):
        config_name += ".yaml"

    if overrides is None:
        overrides = {}

    if os.path.exists(config_name):
        yaml_path = config_name
    else:
        config_root = os.path.abspath(os.path.join(_ROOT, "conf"))
        yaml_path = os.path.join(config_root, config_name)

    # Load base configuration from struct
    base = OmegaConf.structured(StructuredConfig)
    OmegaConf.set_struct(base, strict)

    # Label dict has model-specific keys, so needs to be mutable.
    # Validated elsewhere
    OmegaConf.set_struct(base.label_dict, False)

    # Load target (potentially derived) config
    yaml_cfg = OmegaConf.load(yaml_path)

    # Drop Hydra-specific overrides
    yaml_cfg.pop("defaults", None)

    # Merge in sequence (base, derived config, overrides)
    config = OmegaConf.merge(base, yaml_cfg, overrides)

    # This hack is necessary because OmegaConf will merge rather than
    # replace label_dict by default.
    yaml_cfg_label_dict = yaml_cfg.get("label_dict", None)  # type: ignore
    if yaml_cfg_label_dict:
        config.label_dict = yaml_cfg_label_dict

    override_label_dict = overrides.get("label_dict", None)
    if override_label_dict:
        config.label_dict = override_label_dict

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


class DeepForest_DataFrame(gpd.GeoDataFrame):
    """Custom GeoDataFrame that preserves a root_dir attribute if present."""

    _metadata = ["root_dir"]

    def __init__(self, *args, **kwargs):
        root_dir = getattr(args[0], "root_dir", None) if args else None
        super().__init__(*args, **kwargs)
        if root_dir is not None:
            self.root_dir = root_dir

    @property
    def _constructor(self):
        return DeepForest_DataFrame


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
        raise Exception(
            "error {} for path {} with doc annotation{}".format(
                e, xml_path, doc["annotation"]
            )
        ) from e

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
            label.append(tree["name"])
    else:
        xmin.append(tile_xml["bndbox"]["xmin"])
        xmax.append(tile_xml["bndbox"]["xmax"])
        ymin.append(tile_xml["bndbox"]["ymin"])
        ymax.append(tile_xml["bndbox"]["ymax"])
        label.append(tile_xml["name"])

    rgb_name = os.path.basename(doc["annotation"]["filename"])

    # set dtypes, check for floats and round
    xmin = [round_with_floats(x) for x in xmin]
    xmax = [round_with_floats(x) for x in xmax]
    ymin = [round_with_floats(x) for x in ymin]
    ymax = [round_with_floats(x) for x in ymax]

    annotations = pd.DataFrame(
        {
            "image_path": rgb_name,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "label": label,
        }
    )

    return annotations


def convert_point_to_bbox(gdf: gpd.GeoDataFrame, buffer_size: float) -> gpd.GeoDataFrame:
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
        for x, y in zip(
            gdf.geometry.x.astype(float), gdf.geometry.y.astype(float), strict=False
        )
    ]
    gdf["geometry"] = [
        shapely.geometry.box(left, bottom, right, top)
        for left, bottom, right, top in gdf.geometry.buffer(buffer_size).bounds.values
    ]

    return gdf


def shapefile_to_annotations(
    shapefile: str | gpd.GeoDataFrame,
    rgb: str | None = None,
    root_dir: str | None = None,
    buffer_size: float | None = None,
    convert_point: bool = False,
    label: str | None = None,
) -> gpd.GeoDataFrame:
    raise DeprecationWarning(
        "shapefile_to_annotations is deprecated, use read_file instead"
    )

    if buffer_size is not None and convert_point:
        raise DeprecationWarning(
            "buffer_size argument is deprecated, use convert_point_to_bbox instead"
        )

    return __shapefile_to_annotations__(shapefile)


def __assign_image_path__(gdf, image_path: str) -> str:
    if image_path is None:
        if "image_path" not in gdf.columns:
            raise ValueError(
                "No image_path column found in GeoDataframe and image_path argument not specified, please specify the root_dir and image_path arguements: read_file(input=df, root_dir='path/to/images/', image_path='image.tif', ...)"
            )
        else:
            # Image Path columns exists, leave it unchanged.
            pass
    else:
        if "image_path" in gdf.columns:
            existing_image_path = gdf.image_path.unique()[0]
            if len(existing_image_path) > 1:
                warnings.warn(
                    f"Multiple image_paths found in dataframe: {existing_image_path}, overriding and assigning {image_path} to all rows!",
                    stacklevel=2,
                )
            if existing_image_path != image_path:
                warnings.warn(
                    f"Image path {existing_image_path} found in dataframe, overriding and assigning {image_path} to all rows!",
                    stacklevel=2,
                )
            gdf["image_path"] = image_path
        else:
            gdf["image_path"] = image_path

    return gdf


def __shapefile_to_annotations__(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Convert geospatial annotations to DeepForest format.

    Args:
        gdf: A GeoDataFrame with a geometry column and an image_path column.

    Returns:
        GeoDataFrame with annotations in DeepForest format.
    """
    # Determine geometry type and report to user
    if gdf.geometry.type.unique().shape[0] > 1:
        raise ValueError(
            "Multiple geometry types found in shapefile. "
            "Please ensure all geometries are of the same type."
        )
    else:
        geometry_type = gdf.geometry.type.unique()[0]
        print(f"Geometry type of shapefile is {geometry_type}")

    # raster bounds
    full_image_path = os.path.join(gdf.root_dir, gdf.image_path.unique()[0])
    with rasterio.open(full_image_path) as src:
        raster_crs = src.crs

    if gdf.crs:
        # If epsg is 4326, then the buffer size is in degrees, not meters,
        # see https://github.com/weecology/DeepForest/issues/694
        if gdf.crs.to_string() == "EPSG:4326":
            raise ValueError(
                "The shapefile crs is in degrees. "
                "This function works for UTM and meter based crs only. "
                "see https://github.com/weecology/DeepForest/issues/694"
            )

        # Check matching the crs
        if not gdf.crs.to_string() == raster_crs.to_string():
            warnings.warn(
                f"The shapefile crs {gdf.crs.to_string()} does not match the image crs {src.crs.to_string()}",
                UserWarning,
                stacklevel=2,
            )

        if src.crs is not None:
            print(f"CRS of shapefile is {gdf.crs}")
            print(f"CRS of image is {raster_crs}")
            gdf = geo_to_image_coordinates(gdf, src.bounds, src.res[0])

    return gdf


def determine_geometry_type(df):
    """Determine the geometry type of a prediction or annotation
    Args:
        df: a pandas dataframe
    Returns:
        geometry_type: a string of the geometry type
    """
    if type(df) in [pd.DataFrame, gpd.GeoDataFrame, DeepForest_DataFrame]:
        columns = df.columns
        if "geometry" in columns:
            df = gpd.GeoDataFrame(geometry=df["geometry"])
            geometry_type = df.geometry.type.unique()[0]
            if geometry_type == "Polygon":
                if (df.geometry.area == df.envelope.area).all():
                    return "box"
                else:
                    return "polygon"
            else:
                return "point"
        elif (
            "xmin" in columns
            and "ymin" in columns
            and "xmax" in columns
            and "ymax" in columns
        ):
            geometry_type = "box"
        elif "polygon" in columns:
            geometry_type = "polygon"
        elif "x" in columns and "y" in columns:
            geometry_type = "point"
        else:
            raise ValueError(f"Could not determine geometry type from columns {columns}")

    elif isinstance(df, dict):
        if "boxes" in df.keys():
            geometry_type = "box"
        elif "polygon" in df.keys():
            geometry_type = "polygon"
        elif "points" in df.keys():
            geometry_type = "point"

    return geometry_type


def format_geometry(predictions, scores=True, geom_type=None):
    """Format a retinanet prediction into a pandas dataframe for a batch of images
    Args:
        predictions: a list of dictionaries with keys 'boxes' and 'labels' coming from a retinanet
        scores: Whether boxes come with scores, during prediction, or without scores, as in during training.
    Returns:
        df: a pandas dataframe
        None if the dataframe is empty
    """

    # Detect geometry type
    if geom_type is None:
        geom_type = determine_geometry_type(predictions)

    if geom_type == "box":
        df = format_boxes(predictions, scores=scores)
        if df is None:
            return None

    elif geom_type == "polygon":
        raise ValueError("Polygon predictions are not yet supported for formatting")
    elif geom_type == "point":
        raise ValueError("Point predictions are not yet supported for formatting")

    return df


def format_boxes(prediction, scores=True):
    """Format a retinanet prediction into a pandas dataframe for a single
    image.

    Args:
        prediction: a dictionary with keys 'boxes' and 'labels' coming from a retinanet
        scores: Whether boxes come with scores, during prediction, or without scores, as in during training.
    Returns:
        df: a pandas dataframe
    """
    if len(prediction["boxes"]) == 0:
        return None

    df = pd.DataFrame(
        prediction["boxes"].cpu().detach().numpy(),
        columns=["xmin", "ymin", "xmax", "ymax"],
    )
    df["label"] = prediction["labels"].cpu().detach().numpy()

    if scores:
        df["score"] = prediction["scores"].cpu().detach().numpy()

    df["geometry"] = df.apply(
        lambda x: shapely.geometry.box(x.xmin, x.ymin, x.xmax, x.ymax), axis=1
    )
    return df


def read_coco(json_file):
    """Read a COCO format JSON file and return a pandas dataframe.

    Args:
        json_file: Path to the COCO segmentation JSON file
    Returns:
        df: A pandas dataframe with image_path, geometry, and label columns
    """
    with open(json_file) as f:
        coco_data = json.load(f)

    # Create mapping from image IDs to filenames
    image_ids = {image["id"]: image["file_name"] for image in coco_data["images"]}

    # Create mapping from category IDs to category names
    category_id_to_name = {
        category["id"]: category["name"] for category in coco_data["categories"]
    }

    polygons = []
    filenames = []
    labels = []

    for annotation in coco_data["annotations"]:
        segmentation = annotation.get("segmentation")
        if not segmentation:
            continue
        # COCO polygons are usually a list of lists; take the first (assume "single part")
        segmentation_mask = segmentation[0]
        # Convert flat list to coordinate pairs
        pairs = [
            (segmentation_mask[i], segmentation_mask[i + 1])
            for i in range(0, len(segmentation_mask), 2)
        ]
        polygon = shapely.geometry.Polygon(pairs)
        filenames.append(image_ids[annotation["image_id"]])
        polygons.append(polygon.wkt)
        cat_id = annotation.get("category_id")
        label = category_id_to_name.get(cat_id, cat_id)
        labels.append(label)

    return pd.DataFrame({"image_path": filenames, "geometry": polygons, "label": labels})


def __pandas_to_geodataframe__(df: pd.DataFrame):
    """Create a geometry column from a pandas dataframe with coordinates".

    Args:
        df: a pandas dataframe with columns: xmin, ymin, xmax, ymax, or x, y, or polygon
    Returns:
        gdf: a geodataframe with a geometry column
    """
    # If the geometry column is present, convert to geodataframe directly
    if "geometry" in df.columns:
        if pd.api.types.infer_dtype(df["geometry"]) == "string":
            df["geometry"] = gpd.GeoSeries.from_wkt(df["geometry"])
    else:
        geom_type = determine_geometry_type(df)
        if geom_type == "box":
            df["geometry"] = df.apply(
                lambda x: shapely.geometry.box(x.xmin, x.ymin, x.xmax, x.ymax), axis=1
            )
        elif geom_type == "polygon":
            df["geometry"] = gpd.GeoSeries.from_wkt(df["polygon"])
        elif geom_type == "point":
            df["geometry"] = gpd.GeoSeries(
                [
                    shapely.geometry.Point(x, y)
                    for x, y in zip(df.x.astype(float), df.y.astype(float), strict=False)
                ]
            )
    gdf = gpd.GeoDataFrame(df, geometry="geometry")
    gdf = DeepForest_DataFrame(gdf)

    return gdf


def __check_and_assign_label__(
    df: pd.DataFrame | gpd.GeoDataFrame, label: str | None = None
):
    if label is None:
        if "label" not in df.columns:
            raise ValueError(
                "No label specified and no label column found in dataframe, please specify label in label argument: read_file(input=df, label='YourLabel', ...)"
            )
    else:
        if "label" in df.columns:
            existing_labels = df.label.unique()
            if len(existing_labels) > 1:
                warnings.warn(
                    f"Multiple labels found in dataframe: {existing_labels}, the label argument in read_file will override these labels!",
                    stacklevel=2,
                )
            if existing_labels[0] != label:
                warnings.warn(
                    f"Label {existing_labels[0]} found in dataframe, overriding and assigning {label} to all rows!",
                    stacklevel=2,
                )
        else:
            df["label"] = label

    return df


def __assign_root_dir__(
    input,
    gdf: gpd.GeoDataFrame,
    root_dir: str | None = None,
):
    if root_dir is not None:
        gdf.root_dir = root_dir
    else:
        # If the user specified a path to file, use that root_dir as default.
        if isinstance(input, str):
            gdf.root_dir = os.path.dirname(input)
        else:
            raise ValueError(
                "root_dir argument not specified and input is a dataframe, where are the images stored?"
            )

    return gdf


def _pandas_to_deepforest_format__(input, df, image_path, root_dir, label):
    df = __check_and_assign_label__(df, label=label)
    gdf = __pandas_to_geodataframe__(df)
    gdf = __assign_image_path__(gdf, image_path=image_path)
    gdf = __assign_root_dir__(input, gdf, root_dir=root_dir)
    gdf = DeepForest_DataFrame(gdf)

    return gdf


def read_file(
    input: str | pd.DataFrame,
    root_dir: str | None = None,
    image_path: str | None = None,
    label: str | None = None,
) -> gpd.GeoDataFrame:
    """Read file and return GeoDataFrame in DeepForest format.

    Args:
        input: Path to file, DataFrame, or GeoDataFrame
        root_dir: Root directory for image files
        image_path: Path relative to root_dir to a single image that will be assigned as the image_path column for all annotations. The full path will be constructed by joining the root_dir and the image_path. Overrides any image_path column in input.
        label: Single label to be assigned as the label for all annotations. Overrides any label column in input.

    Returns:
        GeoDataFrame with geometry, image_path, and label columns
    """
    # Check arguments
    if image_path is not None and root_dir is None:
        raise ValueError(
            "root_dir argument must be specified if image_path argument is used"
        )

    # read file
    if isinstance(input, str):
        if input.endswith(".csv"):
            df = pd.read_csv(input)
            gdf = _pandas_to_deepforest_format__(input, df, image_path, root_dir, label)
        elif input.endswith(".json"):
            df = read_coco(input)
            gdf = _pandas_to_deepforest_format__(input, df, image_path, root_dir, label)
        elif input.endswith(".xml"):
            df = read_pascal_voc(input)
            gdf = _pandas_to_deepforest_format__(input, df, image_path, root_dir, label)
        elif input.endswith((".shp", ".gpkg")):
            gdf = gpd.read_file(input)
            gdf = DeepForest_DataFrame(gdf)
            gdf = __assign_image_path__(gdf, image_path=image_path)
            gdf = __check_and_assign_label__(gdf, label=label)
            gdf = __assign_root_dir__(input=input, gdf=gdf, root_dir=root_dir)
            gdf = __shapefile_to_annotations__(gdf)
        else:
            raise ValueError(
                f"File type {input} not supported. "
                "DeepForest currently supports .csv, .shp, .gpkg, .xml, and .json files. "
                "See https://deepforest.readthedocs.io/en/latest/annotation.html "
            )
    elif isinstance(input, gpd.GeoDataFrame):
        gdf = input.copy(deep=True)
        gdf = __assign_image_path__(gdf, image_path=image_path)
        gdf = __assign_root_dir__(input, gdf, root_dir=root_dir)
        gdf = DeepForest_DataFrame(gdf)
        gdf_list = []
        for image_path in gdf.image_path.unique():
            image_annotations = gdf[gdf.image_path == image_path]
            gdf = __shapefile_to_annotations__(image_annotations)
            gdf_list.append(gdf)

        # When concat, need to reform GeoPandas GeoDataFrame
        gdf = pd.concat(gdf_list)
        gdf = gpd.GeoDataFrame(gdf)
        gdf = DeepForest_DataFrame(gdf)
        gdf = __check_and_assign_label__(gdf, label=label)

    elif isinstance(input, pd.DataFrame):
        input = input.copy(deep=True)
        if input.empty:
            raise ValueError("No annotations in dataframe")
        gdf = __pandas_to_geodataframe__(input)
        gdf = __assign_image_path__(gdf, image_path=image_path)
        gdf = __assign_root_dir__(input, gdf, root_dir=root_dir)
        gdf = __check_and_assign_label__(gdf, label=label)
        gdf = DeepForest_DataFrame(gdf)
    else:
        raise ValueError(
            "Input must be a path to a file, geopandas or a pandas dataframe"
        )

    return gdf


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
                f"Driver {driver} not supported for unprojected data, setting to 'PNG',",
                UserWarning,
                stacklevel=2,
            )
            driver = "PNG"
    else:
        # Read projected data using rasterio and crop
        img = src.read(
            window=rasterio.windows.from_bounds(
                left, bottom, right, top, transform=src.transform
            )
        )
        cropped_transform = rasterio.windows.transform(
            rasterio.windows.from_bounds(
                left, bottom, right, top, transform=src.transform
            ),
            src.transform,
        )
    if img.size == 0:
        raise ValueError(
            f"Bounds {bounds} does not create a valid crop for source {src.transform}"
        )
    if savedir:
        res = src.res[0]
        height = (top - bottom) / res
        width = (right - left) / res

        # Write the cropped image to disk with transform
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        if driver == "GTiff":
            filename = f"{savedir}/{filename}.tif"
            with rasterio.open(
                filename,
                "w",
                driver="GTiff",
                height=height,
                width=width,
                count=img.shape[0],
                dtype=img.dtype,
                transform=cropped_transform,
            ) as dst:
                dst.write(img)
        elif driver == "PNG":
            # PNG driver does not support transform
            filename = f"{savedir}/{filename}.png"
            with rasterio.open(
                filename,
                "w",
                driver="PNG",
                height=height,
                width=width,
                count=img.shape[0],
                dtype=img.dtype,
            ) as dst:
                dst.write(img)
        else:
            raise ValueError(f"Driver {driver} not supported")

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
    transformed_gdf.geometry = transformed_gdf.geometry.scale(
        xfact=1 / image_resolution, yfact=-1 / image_resolution, origin=(0, 0)
    )
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
            "This is expected for some annotation formats.",
            UserWarning,
            stacklevel=2,
        )
        result = int(np.round(float(x)))

    return result


def check_image(image):
    """Check an image is three channel, channel last format
    Args:
       image: numpy array
    Returns: None, throws error on assert
    """
    if not image.shape[2] == 3:
        raise ValueError(
            "image is expected have three channels, channel last format, "
            f"found image with shape {image.shape}"
        )


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
            f"This function projects a single plot's worth of data. Multiple plot names found: {plot_names}"
        )
    else:
        plot_name = plot_names[0]

    rgb_path = f"{root_dir}/{plot_name}"
    with rasterio.open(rgb_path) as dataset:
        bounds = dataset.bounds
        left, bottom, right, top = bounds
        crs = dataset.crs
        transform = dataset.transform

    geom_type = determine_geometry_type(transformed_gdf)
    projected_geometry = []
    if geom_type == "box":
        # Convert image pixel locations to geographic coordinates
        coordinates = transformed_gdf.geometry.bounds
        xmin_coords, ymin_coords = rasterio.transform.xy(
            transform=transform,
            rows=coordinates.miny,
            cols=coordinates.minx,
            offset="center",
        )
        xmax_coords, ymax_coords = rasterio.transform.xy(
            transform=transform,
            rows=coordinates.maxy,
            cols=coordinates.maxx,
            offset="center",
        )

        for left, bottom, right, top in zip(
            xmin_coords, ymin_coords, xmax_coords, ymax_coords, strict=False
        ):
            geom = shapely.geometry.box(left, bottom, right, top)
            projected_geometry.append(geom)

    elif geom_type == "polygon":
        for geom in transformed_gdf.geometry:
            polygon_vertices = []
            for x, y in geom.exterior.coords:
                projected_vertices = rasterio.transform.xy(
                    transform=transform, rows=y, cols=x, offset="center"
                )
                polygon_vertices.append(projected_vertices)
            geom = shapely.geometry.Polygon(polygon_vertices)
            projected_geometry.append(geom)

    elif geom_type == "point":
        x_coords, y_coords = rasterio.transform.xy(
            transform=transform,
            rows=transformed_gdf.geometry.y,
            cols=transformed_gdf.geometry.x,
            offset="center",
        )
        for x, y in zip(x_coords, y_coords, strict=False):
            geom = shapely.geometry.Point(x, y)
            projected_geometry.append(geom)

    transformed_gdf.geometry = projected_geometry

    # Update xmin, xmax, ymin, ymax columns to match transformed geometry
    if geom_type == "box":
        bounds = transformed_gdf.geometry.bounds
        transformed_gdf["xmin"] = bounds.minx
        transformed_gdf["xmax"] = bounds.maxx
        transformed_gdf["ymin"] = bounds.miny
        transformed_gdf["ymax"] = bounds.maxy

    if flip_y_axis:
        # Numpy uses top-left origin, flip y-axis for QGIS compatibility
        # See GIS StackExchange for details on negative y-spacing
        transformed_gdf.geometry = transformed_gdf.geometry.scale(
            xfact=1, yfact=-1, origin=(0, 0)
        )

    # Assign crs
    transformed_gdf.crs = crs

    return transformed_gdf


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return tuple(zip(*batch, strict=False))
