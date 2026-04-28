# Prediction utilities
import os

import numpy as np
import pandas as pd
import shapely
import torch
from scipy.spatial import cKDTree
from shapely import affinity
from torchvision.ops import nms

from deepforest import utilities
from deepforest.datasets import cropmodel
from deepforest.utilities import read_file


def _predict_image_(
    model,
    image: np.ndarray | None = None,
    path: str | None = None,
    iou_threshold: float = 0.15,
    nms_distance_thresh: float = 5.0,
):
    """Predict a single image with a deepforest model.

    Args:
        model: a deepforest.main.model object
        image: a tensor of shape (channels, height, width)
        path: optional path to read image from disk instead of passing image arg
        iou_threshold: IoU threshold for box non-max suppression
        nms_distance_thresh: Distance threshold in pixels for point NMS, see config.point.nms_distance_thresh
    Returns:
        df: A pandas dataframe of predictions (Default)
        img: The input with predictions overlaid (Optional)
    """
    image = torch.tensor(image).permute(2, 0, 1)
    image = image / 255

    with torch.no_grad():
        prediction = model(image.unsqueeze(0))

    prediction = prediction[0]
    geom_type = utilities.determine_geometry_type(prediction)
    df = utilities.format_geometry(prediction, geom_type=geom_type)

    # return None for no predictions
    if df is None:
        return None

    if geom_type == "box" and df.label.nunique() > 1:
        df = across_class_nms(df, iou_threshold=iou_threshold)
    elif geom_type == "point":
        df = reduce_points(df, nms_thresh=nms_distance_thresh)

    # Add image path if provided
    if path is not None:
        df["image_path"] = os.path.basename(path)

    return df


def translate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Shift window-relative predictions into image coordinates using geometry.

    Args:
        predictions: DataFrame with geometry and window_xmin/window_ymin offset columns.

    Returns:
        DataFrame with geometry (and coordinate columns) shifted by the window origin.
    """
    predictions = predictions.copy()
    is_box = {"xmin", "ymin", "xmax", "ymax"}.issubset(predictions.columns)

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
    else:
        coords = shapely.get_coordinates(np.array(predictions["geometry"]))
        predictions["x"] = coords[:, 0]
        predictions["y"] = coords[:, 1]

    return predictions.drop(columns=["window_xmin", "window_ymin"]).reset_index(drop=True)


def reduce_boxes(predictions: pd.DataFrame, iou_threshold: float) -> pd.DataFrame:
    """Reduce overlapping box predictions with torchvision NMS.

    Args:
        predictions: DataFrame of image-space box predictions.
        iou_threshold: IoU threshold for NMS.

    Returns:
        DataFrame containing the filtered box predictions in the public box schema.
    """
    box_output_columns = ["xmin", "ymin", "xmax", "ymax", "label", "score"]
    if predictions.shape[0] <= 1:
        return predictions[box_output_columns].reset_index(drop=True).copy()

    print(
        f"{predictions.shape[0]} predictions in overlapping windows, applying non-max suppression"
    )

    boxes = torch.tensor(
        predictions[["xmin", "ymin", "xmax", "ymax"]].values, dtype=torch.float32
    )
    scores = torch.tensor(predictions.score.values, dtype=torch.float32)
    keep_idx = nms(boxes=boxes, scores=scores, iou_threshold=iou_threshold).numpy()

    filtered_predictions = predictions.iloc[keep_idx].reset_index(drop=True)
    print(f"{filtered_predictions.shape[0]} predictions kept after non-max suppression")
    return filtered_predictions[box_output_columns].reset_index(drop=True).copy()


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
    translated_predictions = translate_predictions(predictions)

    if is_box_predictions:
        return reduce_boxes(translated_predictions, iou_threshold=iou_threshold)

    if is_point_predictions:
        return reduce_points(translated_predictions, nms_thresh=nms_distance_thresh)

    raise ValueError("Predictions must include either box or point coordinates.")


def across_class_nms(predicted_boxes, iou_threshold=0.15):
    """Perform non-max suppression for a dataframe of results (see
    visualize.format_boxes) to remove boxes that overlap by iou_thresholdold of
    IoU."""
    # Skip NMS if there's is one or less prediction
    if predicted_boxes.shape[0] <= 1:
        return predicted_boxes

    # move prediciton to tensor
    boxes = torch.tensor(
        predicted_boxes[["xmin", "ymin", "xmax", "ymax"]].values, dtype=torch.float32
    )
    scores = torch.tensor(predicted_boxes.score.values, dtype=torch.float32)
    labels = predicted_boxes.label.values

    bbox_left_idx = nms(boxes=boxes, scores=scores, iou_threshold=iou_threshold)
    bbox_left_idx = bbox_left_idx.numpy()
    new_boxes, new_labels, new_scores = (
        boxes[bbox_left_idx].type(torch.int),
        labels[bbox_left_idx],
        scores[bbox_left_idx],
    )

    # Recreate box dataframe
    image_detections = np.concatenate(
        [
            new_boxes,
            np.expand_dims(new_labels, axis=1),
            np.expand_dims(new_scores, axis=1),
        ],
        axis=1,
    )

    new_df = pd.DataFrame(
        image_detections, columns=["xmin", "ymin", "xmax", "ymax", "label", "score"]
    )

    return new_df


def _dataloader_wrapper_(model, trainer, dataloader, root_dir, crop_model):
    """

    Args:
        model: deepforest.main object
        trainer: a pytorch lightning trainer object
        dataloader: pytorch dataloader object
        root_dir: directory of images. If none, uses "image_dir" in config
        nms_thresh: Non-max suppression threshold, see config.nms_thresh
        crop_model: Optional. A list of crop models to be used for prediction.
    Returns:
        results: pandas dataframe with bounding boxes, label and scores for each image in the csv file
    """
    batched_results = trainer.predict(model, dataloader)

    # Flatten list from batched prediction
    prediction_list = []
    global_image_idx = 0
    for _idx, batch in enumerate(batched_results):
        for _image_idx, image_result in enumerate(batch):
            formatted_result = dataloader.dataset.postprocess(
                image_result, global_image_idx
            )
            global_image_idx += 1
            prediction_list.append(formatted_result)

    # Postprocess predictions, return empty dataframe if no predictions
    if not prediction_list:
        return pd.DataFrame()

    results = pd.concat(prediction_list)

    if results.empty:
        return results

    # Apply across class NMS for each image
    processed_results = []
    for image_path in results.image_path.unique():
        image_results = results[results.image_path == image_path].copy()

        if crop_model:
            # Flag to check if only one model is passed
            is_single_model = len(crop_model) == 1

            for i, crop_model_item in enumerate(crop_model):
                crop_model_results = _predict_crop_model_(
                    crop_model=crop_model_item,
                    results=image_results,
                    path=image_path,
                    trainer=trainer,
                    model_index=i,
                    is_single_model=is_single_model,
                )

            processed_results.append(crop_model_results)

    results = read_file(results, root_dir)

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
