# Prediction utilities
import os
from warnings import warn

import numpy as np
import pandas as pd
import torch
from torchvision.ops import nms

from deepforest import utilities
from deepforest.datasets import cropmodel
from deepforest.utilities import read_file


def _predict_image_(
    model,
    image: np.ndarray | None = None,
    path: str | None = None,
    nms_thresh: float = 0.15,
):
    """Predict a single image with a deepforest model.

    Args:
        model: a deepforest.main.model object
        image: a tensor of shape (channels, height, width)
        path: optional path to read image from disk instead of passing image arg
        nms_thresh: Non-max suppression threshold, see config.nms_thresh (only for box detection)
    Returns:
        df: A pandas dataframe of predictions (Default)
        img: The input with predictions overlaid (Optional)
    """

    image = torch.tensor(image).permute(2, 0, 1)
    image = image / 255

    with torch.no_grad():
        prediction = model(image.unsqueeze(0))

    df = utilities.format_geometry(prediction[0])

    # return None for no predictions
    if df is None:
        return None

    # NMS for boxes only
    if "xmin" in df.columns and df.label.nunique() > 1:
        df = across_class_nms(df, iou_threshold=nms_thresh)

    # Add image path if provided
    if path is not None:
        df["image_path"] = os.path.basename(path)

    return df


def transform_coordinates(predictions):
    """Transform coordinates from window space to original image space.

    Args:
        predictions: DataFrame of predictions with coordinate columns and window_xmin, window_ymin

    Returns:
        DataFrame with transformed coordinates
    """
    predictions = predictions.copy()

    # Handle box coordinates
    if "xmin" in predictions.columns:
        predictions["xmin"] += predictions["window_xmin"]
        predictions["xmax"] += predictions["window_xmin"]
        predictions["ymin"] += predictions["window_ymin"]
        predictions["ymax"] += predictions["window_ymin"]

        # Cast to int
        predictions["xmin"] = predictions["xmin"].astype(int)
        predictions["ymin"] = predictions["ymin"].astype(int)
        predictions["xmax"] = predictions["xmax"].astype(int)
        predictions["ymax"] = predictions["ymax"].astype(int)

    # Handle keypoint coordinates
    elif "x" in predictions.columns and "y" in predictions.columns:
        predictions["x"] += predictions["window_xmin"]
        predictions["y"] += predictions["window_ymin"]

        # Cast to int
        predictions["x"] = predictions["x"].astype(int)
        predictions["y"] = predictions["y"].astype(int)

    return predictions


def apply_nms(boxes, scores, labels, iou_threshold):
    """Apply non-maximum suppression to boxes.

    Args:
        boxes: tensor of shape (N, 4) containing box coordinates
        scores: tensor of shape (N,) containing confidence scores
        labels: array of shape (N,) containing labels
        iou_threshold: IoU threshold for NMS

    Returns:
        DataFrame with filtered boxes
    """
    bbox_left_idx = nms(boxes=boxes, scores=scores, iou_threshold=iou_threshold)
    bbox_left_idx = bbox_left_idx.numpy()

    new_boxes = boxes[bbox_left_idx].type(torch.int)
    new_labels = labels[bbox_left_idx]
    new_scores = scores[bbox_left_idx]

    # Recreate box dataframe
    image_detections = np.concatenate(
        [
            new_boxes,
            np.expand_dims(new_labels, axis=1),
            np.expand_dims(new_scores, axis=1),
        ],
        axis=1,
    )

    return pd.DataFrame(
        image_detections, columns=["xmin", "ymin", "xmax", "ymax", "label", "score"]
    )


def mosaic(predictions, iou_threshold=0.1):
    """Mosaic predictions from overlapping windows.

    Args:
        predictions: A pandas dataframe containing predictions from overlapping windows from a single image.
        iou_threshold: The IoU threshold for non-max suppression.

    Returns:
        A pandas dataframe of predictions.
    """
    predicted_results = transform_coordinates(predictions)

    # Skip NMS if there's is one or less prediction
    if predicted_results.shape[0] <= 1:
        return predicted_results

    # TODO: Should probably have an aggregation function here.
    # For keypoints, eturn transformed coordinates
    if "x" in predicted_results.columns and "y" in predicted_results.columns:
        warn(
            "Keypoint merging for overlapping windows is not yet supported, returning all points.",
            stacklevel=2,
        )
        return predicted_results

    print(
        f"{predicted_results.shape[0]} box predictions in overlapping windows, applying non-max suppression"
    )

    # Convert to tensors
    boxes = torch.tensor(
        predicted_results[["xmin", "ymin", "xmax", "ymax"]].values, dtype=torch.float32
    )
    scores = torch.tensor(predicted_results.score.values, dtype=torch.float32)
    labels = predicted_results.label.values

    # Apply NMS
    filtered_boxes = apply_nms(boxes, scores, labels, iou_threshold)
    print(f"{filtered_boxes.shape[0]} predictions kept after non-max suppression")

    return filtered_boxes


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
    for batch in batched_results:
        for images in batch:
            prediction_list.append(images)

    # Postprocess predictions
    results = dataloader.dataset.postprocess(prediction_list)

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

    # Get resize dimensions from crop_model config if not using custom transform
    resize = None
    if transform is None and hasattr(crop_model, "config"):
        resize = crop_model.config.get("cropmodel", {}).get("resize", [224, 224])

    # Create dataset
    bounding_box_dataset = cropmodel.BoundingBoxDataset(
        results,
        root_dir=os.path.dirname(path),
        transform=transform,
        augmentations=augmentations,
        resize=resize,
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
