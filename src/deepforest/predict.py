# Prediction utilities
import os

import numpy as np
import pandas as pd
import shapely
import torch
from scipy.spatial import cKDTree
from shapely import affinity
from torchvision.ops import nms

from deepforest import distributed
from deepforest.datasets import cropmodel


def translate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Shift window-relative predictions into image coordinates using geometry.

    Args:
        predictions: DataFrame with geometry and window_xmin/window_ymin offset columns.

    Returns:
        DataFrame with geometry (and coordinate columns) shifted by the window origin.
    """
    predictions = predictions.copy()
    is_box = {"xmin", "ymin", "xmax", "ymax"}.issubset(predictions.columns)
    is_point = {"x", "y"}.issubset(predictions.columns)

    predictions["geometry"] = [
        affinity.translate(geom, xoff=dx, yoff=dy)
        for geom, dx, dy in zip(
            predictions.geometry,
            predictions.window_xmin,
            predictions.window_ymin,
            strict=True,
        )
    ]

    if is_box:
        bounds = shapely.bounds(np.array(predictions["geometry"]))
        predictions[["xmin", "ymin", "xmax", "ymax"]] = bounds.astype(int)
    elif is_point:
        coords = shapely.get_coordinates(np.array(predictions["geometry"]))
        predictions["x"] = coords[:, 0]
        predictions["y"] = coords[:, 1]
    # Polygons carry all positional information in the translated geometry,
    # so no coordinate columns need updating.

    return predictions.drop(columns=["window_xmin", "window_ymin"]).reset_index(drop=True)


def _run_nms(
    predictions: pd.DataFrame,
    boxes: "torch.Tensor",
    output_columns: list[str],
    iou_threshold: float,
) -> pd.DataFrame:
    """Run torchvision NMS and return the surviving rows with selected
    columns."""
    if predictions.shape[0] <= 1:
        return predictions[output_columns].reset_index(drop=True).copy()

    print(
        f"{predictions.shape[0]} predictions in overlapping windows, applying non-max suppression"
    )
    scores = torch.tensor(predictions.score.values, dtype=torch.float32)
    keep_idx = nms(boxes=boxes, scores=scores, iou_threshold=iou_threshold).numpy()
    filtered = predictions.iloc[keep_idx].reset_index(drop=True)
    print(f"{filtered.shape[0]} predictions kept after non-max suppression")
    return filtered[output_columns].reset_index(drop=True).copy()


def reduce_boxes(predictions: pd.DataFrame, iou_threshold: float) -> pd.DataFrame:
    """Reduce overlapping box predictions with torchvision NMS.

    Args:
        predictions: DataFrame of image-space box predictions.
        iou_threshold: IoU threshold for NMS.

    Returns:
        DataFrame containing the filtered box predictions in the public box schema.
    """
    output_columns = ["xmin", "ymin", "xmax", "ymax", "label", "score"]
    boxes = torch.tensor(
        predictions[["xmin", "ymin", "xmax", "ymax"]].values, dtype=torch.float32
    )
    return _run_nms(predictions, boxes, output_columns, iou_threshold)


def reduce_points(predictions: pd.DataFrame, nms_thresh: float) -> pd.DataFrame:
    """Reduce nearby point predictions with distance-based suppression.

    Args:
        predictions: DataFrame of image-space point predictions.
        nms_thresh: Distance threshold in pixels used to suppress duplicates.

    Returns:
        Filtered point predictions with all non-coordinate columns preserved.
    """
    predictions = predictions.reset_index(drop=True)
    if nms_thresh <= 0 or len(predictions) <= 1:
        return predictions

    coords = predictions[["x", "y"]].values
    scores = predictions["score"].values
    tree = cKDTree(coords)
    order = np.argsort(scores)[::-1]
    kept = np.ones(len(coords), dtype=bool)

    for idx in order:
        if not kept[idx]:
            continue

        for neighbor_idx in tree.query_ball_point(coords[idx], r=nms_thresh):
            if neighbor_idx != idx:
                kept[neighbor_idx] = False

    return predictions.iloc[np.flatnonzero(kept)].reset_index(drop=True)


def reduce_polygons(predictions: pd.DataFrame, iou_threshold: float) -> pd.DataFrame:
    """Reduce overlapping polygon predictions with NMS on polygon bounds.

    Non-max suppression is computed on the axis-aligned bounding box of each
    polygon (its envelope), which parallels :func:`reduce_boxes` and avoids the
    cost of all-pairs polygon IoU.

    Args:
        predictions: DataFrame of image-space polygon predictions with a
            ``geometry`` column.
        iou_threshold: IoU threshold for NMS.

    Returns:
        DataFrame containing the filtered polygon predictions.
    """
    output_columns = ["geometry", "label", "score"]
    boxes = torch.tensor(
        shapely.bounds(np.array(predictions.geometry)), dtype=torch.float32
    )
    return _run_nms(predictions, boxes, output_columns, iou_threshold)


def mosaic(
    predictions: pd.DataFrame,
    iou_threshold: float = 0.1,
    nms_distance_thresh: float = 5.0,
) -> pd.DataFrame:
    """Mosaic predictions from overlapping windows.

    Args:
        predictions: A pandas dataframe containing predictions from overlapping windows from a single image.
        iou_threshold: The IoU threshold for non-max suppression (box predictions).
        nms_distance_thresh: Distance in pixels below which two points are duplicates (point predictions).

    Returns:
        A pandas dataframe of predictions.
    """
    if predictions.empty:
        return predictions.copy()

    is_box_predictions = {"xmin", "ymin", "xmax", "ymax"}.issubset(predictions.columns)
    is_point_predictions = {"x", "y"}.issubset(predictions.columns)
    is_polygon_predictions = "geometry" in predictions.columns

    # Validate before translating, which needs a geometry column to work with.
    if not (is_box_predictions or is_point_predictions or is_polygon_predictions):
        raise ValueError(
            "Predictions must include box, point or polygon (geometry) coordinates."
        )

    translated_predictions = translate_predictions(predictions)

    if is_box_predictions:
        return reduce_boxes(translated_predictions, iou_threshold=iou_threshold)

    if is_point_predictions:
        return reduce_points(translated_predictions, nms_thresh=nms_distance_thresh)

    return reduce_polygons(translated_predictions, iou_threshold=iou_threshold)


def _flatten_prediction_batches_(batched_results):
    """Flatten prediction batches returned by Lightning predict()."""
    flattened = []
    for batch in batched_results:
        if isinstance(batch, pd.DataFrame):
            if not batch.empty:
                flattened.append(batch)
            continue

        for item in batch:
            if isinstance(item, pd.DataFrame) and not item.empty:
                flattened.append(item)

    if not flattened:
        return pd.DataFrame()

    return pd.concat(flattened, ignore_index=True)


def reduce_overlapping(image_results, config, task="box"):
    """Applies non-max suppression: across-class NMS for multi-class boxes,
    distance-based suppression for points, envelope NMS for polygons.

    Args:
        image_results: predictions for one image.
        config: model configuration providing the NMS thresholds.
        task: model task, "box", "point" or "polygon".

    Returns:
        Reduced predictions for the image.
    """
    if task == "box":
        if image_results.label.nunique() > 1:
            image_results = reduce_boxes(image_results, iou_threshold=config.nms_thresh)
    elif task == "point":
        image_results = reduce_points(
            image_results, nms_thresh=config.point.nms_distance_thresh
        )
    elif task == "polygon":
        image_results = reduce_polygons(image_results, iou_threshold=config.nms_thresh)

    return image_results


def _dataloader_wrapper_(model, trainer, dataloader, crop_model=None, root_dir=None):
    """Run inference over a dataloader and reduce predictions per image.

    Returns a plain dataframe of image-space predictions with numeric labels.
    Callers are responsible for label mapping and read_file/root_dir formatting.

    Args:
        model: deepforest.main object
        trainer: a pytorch lightning trainer object
        dataloader: pytorch dataloader object
        crop_model: Optional CropModel (or list) to classify detected crops.
            Requires root_dir.
        root_dir: directory of images on disk
    Returns:
        results: pandas dataframe with predictions for each image in the dataloader
    """
    batched_results = trainer.predict(model, dataloader)
    results = distributed.gather_dataframe(_flatten_prediction_batches_(batched_results))

    if results.empty:
        return pd.DataFrame()

    # dropna=False keeps a null image_path when image_path is not available.
    processed_results = []
    for _, group in results.groupby("image_path", dropna=False):
        processed_results.append(
            reduce_overlapping(group, model.config, task=model.model.task)
        )

    results = pd.concat(processed_results, ignore_index=True)

    if crop_model is not None:
        if root_dir is None:
            raise ValueError("crop_model requires a path/root_dir ")
        results.root_dir = root_dir
        results = _crop_models_wrapper_(
            crop_models=crop_model, trainer=trainer, results=results
        )

    return results


def _predict_crop_model_(
    crop_model,
    trainer,
    results,
    path,
    transform=None,
    augmentations=None,
    model_index=0,
    is_single_model=False,
):
    """Predicts crop model on a raster file.

    Args:
        crop_model: The crop model to be used for prediction.
        trainer: The PyTorch Lightning trainer object for prediction.
        results: The results dataframe to store the predicted labels and scores.
        path: The path to the raster file.
        is_single_model: Boolean flag to determine column naming.

    Returns:
        The updated results dataframe with predicted labels and scores.
    """
    if results.empty:
        print("No predictions to run crop model on, returning empty dataframe")
        return results

    # Remove invalid boxes
    results = results[results.xmin != results.xmax]
    results = results[results.ymin != results.ymax]

    # Get config from crop_model if not using custom transform
    resize = None
    resize_interpolation = "bilinear"
    normalize = None
    expand = 0
    if transform is None and hasattr(crop_model, "config"):
        cropmodel_cfg = crop_model.config.get("cropmodel", {})
        resize = cropmodel_cfg.get("resize", [224, 224])
        resize_interpolation = cropmodel_cfg.get("resize_interpolation", "bilinear")
        norm_transform = crop_model.normalize()
        if norm_transform is None:
            normalize = False
        else:
            normalize = norm_transform
        expand = cropmodel_cfg.get("expand", 0)

    # Create dataset
    bounding_box_dataset = cropmodel.BoundingBoxDataset(
        results,
        root_dir=os.path.dirname(path),
        transform=transform,
        augmentations=augmentations,
        resize=resize,
        resize_interpolation=resize_interpolation,
        normalize=normalize,
        expand=expand,
    )

    # Create dataloader
    crop_dataloader = crop_model.predict_dataloader(bounding_box_dataset)

    # Run prediction
    crop_results = trainer.predict(crop_model, crop_dataloader)

    # Process results
    label, score = crop_model.postprocess_predictions(crop_results)

    # Determine column names
    if is_single_model:
        label_column = "cropmodel_label"
        score_column = "cropmodel_score"
    else:
        label_column = f"cropmodel_label_{model_index}"
        score_column = f"cropmodel_score_{model_index}"

    if crop_model.numeric_to_label_dict is None:
        raise ValueError(
            f"The numeric_to_label_dict is not set, and the label_dict is "
            f"{crop_model.label_dict}, set either when loading CropModel(label_dict=), "
            f"which creates the numeric_to_label_dict, or load annotations from CropModel."
            f"load_from_disk(), which creates the dictionaries based on file contents."
        )

    results[label_column] = [crop_model.numeric_to_label_dict[x] for x in label]

    results[score_column] = score

    return results


def _crop_models_wrapper_(
    crop_models, trainer, results, transform=None, augmentations=None
):
    if crop_models is not None and not isinstance(crop_models, list):
        crop_models = [crop_models]

    # Run predictions
    crop_results = []
    if crop_models:
        is_single_model = (
            len(crop_models) == 1
        )  # Flag to check if only one model is passed
        for i, crop_model in enumerate(crop_models):
            for path in results.image_path.unique():
                path = os.path.join(results.root_dir, path)
                crop_result = _predict_crop_model_(
                    crop_model=crop_model,
                    results=results,
                    path=path,
                    trainer=trainer,
                    model_index=i,
                    transform=transform,
                    augmentations=augmentations,
                    is_single_model=is_single_model,
                )
                crop_results.append(crop_result)

    # Concatenate results
    crop_results = pd.concat(crop_results)

    return crop_results
