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
) -> gpd.GeoDataFrame:
    """Convert shapefile annotations to DeepForest format.

    Args:
        shapefile: Path to shapefile or GeoDataFrame
        rgb: Path to RGB image
        root_dir: Directory to prepend to image paths
        buffer_size: Buffer size for point-to-polygon conversion
        convert_point: Convert points to bounding boxes

    Returns:
        GeoDataFrame with annotations
    """

    # Read shapefile
    if isinstance(shapefile, str):
        gdf = gpd.read_file(shapefile)
    else:
        gdf = shapefile.copy(deep=True)

    if rgb is None:
        if "image_path" not in gdf.columns:
            raise ValueError(
                "No image_path column found in shapefile, please specify rgb path"
            )
        else:
            rgb = gdf.image_path.unique()[0]
            print(f"Found image_path column in shapefile, using {rgb}")

    # Determine geometry type and report to user
    if gdf.geometry.type.unique().shape[0] > 1:
        raise ValueError(
            "Multiple geometry types found in shapefile. "
            "Please ensure all geometries are of the same type."
        )
    else:
        geometry_type = gdf.geometry.type.unique()[0]
        print(f"Geometry type of shapefile is {geometry_type}")

    # Convert point to bounding box if desired
    if convert_point:
        if geometry_type == "Point":
            if buffer_size is None:
                raise ValueError(
                    "buffer_size must be specified to convert point to bounding box"
                )
            gdf = convert_point_to_bbox(gdf, buffer_size)
        else:
            raise ValueError("convert_point is True, but geometry type is not Point")

    # raster bounds
    if root_dir:
        rgb = os.path.join(root_dir, rgb)
    with rasterio.open(rgb) as src:
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
        df: A pandas dataframe with image_path and geometry columns
    """
    with open(json_file) as f:
        coco_data = json.load(f)

        polygons = []
        filenames = []
        image_ids = {image["id"]: image["file_name"] for image in coco_data["images"]}

        for annotation in coco_data["annotations"]:
            segmentation_mask = annotation["segmentation"][0]
            # Convert flat list to coordinate pairs
            pairs = [
                (segmentation_mask[i], segmentation_mask[i + 1])
                for i in range(0, len(segmentation_mask), 2)
            ]
            polygon = shapely.geometry.Polygon(pairs)
            filenames.append(image_ids[annotation["image_id"]])
            polygons.append(polygon.wkt)

        return pd.DataFrame({"image_path": filenames, "geometry": polygons})


def read_file(
    input: str | pd.DataFrame,
    root_dir: str | None = None,
    image_path: str | None = None,
    label: str | None = None,
) -> gpd.GeoDataFrame:
    """Read file and return GeoDataFrame.

    Args:
        input: Path to file, DataFrame, or GeoDataFrame
        root_dir: Root directory for image files
        image_path: Assign to all rows (use only for single image)
        label: Assign to all rows (use only for single label)

    Returns:
        GeoDataFrame with geometry column
    """
    if image_path is not None:
        warnings.warn(
            "You have passed an image_path. "
            "This value will be assigned to every row in the dataframe. "
            "Only use this if the file contains annotations for a single image.",
            UserWarning,
            stacklevel=2,
        )

    if label is not None:
        warnings.warn(
            "You have passed a label. This value will be assigned to every row in the dataframe. "
            "Only use this if all annotations share the same label.",
            UserWarning,
            stacklevel=2,
        )

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
                f"File type {df} not supported. "
                "DeepForest currently supports .csv, .shp, .gpkg, .xml, and .json files. "
                "See https://deepforest.readthedocs.io/en/latest/annotation.html "
            )
    else:
        # Explicitly check for GeoDataFrame first
        if isinstance(input, gpd.GeoDataFrame):
            return shapefile_to_annotations(input, root_dir=root_dir)
        elif isinstance(input, pd.DataFrame):
            df = input.copy(deep=True)
        else:
            raise ValueError(
                "Input must be a path to a file, geopandas or a pandas dataframe"
            )

    if isinstance(df, pd.DataFrame):
        if df.empty:
            raise ValueError("No annotations in dataframe")
        # If the geometry column is present, convert to geodataframe directly
        if "geometry" in df.columns:
            if pd.api.types.infer_dtype(df["geometry"]) == "string":
                df["geometry"] = gpd.GeoSeries.from_wkt(df["geometry"])
        else:
            # Detect geometry type
            geom_type = determine_geometry_type(df)

            # convert to geodataframe
            if geom_type == "box":
                df["geometry"] = df.apply(
                    lambda x: shapely.geometry.box(x.xmin, x.ymin, x.xmax, x.ymax), axis=1
                )
                df = gpd.GeoDataFrame(df, geometry="geometry")
            elif geom_type == "polygon":
                df["geometry"] = gpd.GeoSeries.from_wkt(df["polygon"])
                df = gpd.GeoDataFrame(df, geometry="geometry")
            elif geom_type == "point":
                df["geometry"] = gpd.GeoSeries(
                    [
                        shapely.geometry.Point(x, y)
                        for x, y in zip(
                            df.x.astype(float), df.y.astype(float), strict=False
                        )
                    ]
                )
                df = gpd.GeoDataFrame(df, geometry="geometry")
            else:
                raise ValueError(f"Geometry type {geom_type} not supported")

    # Add missing columns if not provided
    if "image_path" not in df.columns and image_path is not None:
        df["image_path"] = image_path
    elif "image_path" not in df.columns:
        warnings.warn(
            "'image_path' column is missing from shapefile, please specify the image path",
            UserWarning,
            stacklevel=2,
        )

    if "label" not in df.columns and label is not None:
        df["label"] = label
    elif "label" not in df.columns:
        warnings.warn(
            "'label' column is missing from shapefile, using default label",
            UserWarning,
            stacklevel=2,
        )
        df["label"] = "Unknown"  # Set default label if not provided

    # If root_dir is specified, add as attribute
    if root_dir is not None:
        df.root_dir = root_dir
    else:
        try:
            df.root_dir = os.path.dirname(input)
        except TypeError:
            warnings.warn(
                "root_dir argument for the location of images should be specified "
                "if input is not a path, returning without results.root_dir attribute",
                UserWarning,
                stacklevel=2,
            )

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
