# Prediction utilities
import pandas as pd
import numpy as np
import os

import torch
from torchvision.ops import nms
import typing

from deepforest import visualize, dataset
from deepforest.utilities import read_file


def _predict_image_(model,
                    image: typing.Optional[np.ndarray] = None,
                    path: typing.Optional[str] = None,
                    nms_thresh: float = 0.15,
                    return_plot: bool = False,
                    thickness: int = 1,
                    color: typing.Optional[tuple] = (0, 165, 255)):
    """Predict a single image with a deepforest model.

    Args:
        model: a deepforest.main.model object
        image: a tensor of shape (channels, height, width)
        path: optional path to read image from disk instead of passing image arg
        nms_thresh: Non-max suppression threshold, see config.nms_thresh
        return_plot: Return image with plotted detections
        thickness: thickness of the rectangle border line in px
        color: color of the bounding box as a tuple of BGR color, e.g. orange annotations is (0, 165, 255)
    Returns:
        df: A pandas dataframe of predictions (Default)
        img: The input with predictions overlaid (Optional)
    """
    with torch.no_grad():
        prediction = model(image.unsqueeze(0))

    # return None for no predictions
    if len(prediction[0]["boxes"]) == 0:
        return None

    df = visualize.format_boxes(prediction[0])
    df = across_class_nms(df, iou_threshold=nms_thresh)

    if return_plot:
        # Bring to gpu
        image = image.cpu()

        # Cv2 likes no batch dim, BGR image and channels last, 0-255
        image = np.array(image.squeeze(0))
        image = np.rollaxis(image, 0, 3)
        image = image[:, :, ::-1] * 255
        image = image.astype("uint8")
        image = visualize.plot_predictions(image, df, color=color, thickness=thickness)

        return image
    else:
        if path:
            df["image_path"] = os.path.basename(path)

    return df


def transform_coordinates(boxes):
    """Transform box coordinates from window space to original image space.
    
    Args:
        boxes: DataFrame of predictions with xmin, ymin, xmax, ymax, window_xmin, window_ymin columns
        
    Returns:
        DataFrame with transformed coordinates
    """
    boxes = boxes.copy()
    boxes["xmin"] += boxes["window_xmin"]
    boxes["xmax"] += boxes["window_xmin"]
    boxes["ymin"] += boxes["window_ymin"]
    boxes["ymax"] += boxes["window_ymin"]
    
    return boxes


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
    image_detections = np.concatenate([
        new_boxes,
        np.expand_dims(new_labels, axis=1),
        np.expand_dims(new_scores, axis=1)
    ], axis=1)

    return pd.DataFrame(image_detections,
                       columns=["xmin", "ymin", "xmax", "ymax", "label", "score"])


def mosiac(boxes, iou_threshold=0.1):
    """Mosaic predictions from overlapping windows.

    Args:
        boxes: A list of pandas dataframes containing predictions from overlapping windows.
        iou_threshold: The IoU threshold for non-max suppression.

    Returns:
        A pandas dataframe of predictions.
    """
    # Transform coordinates for each image separately
    transformed_boxes = []
    for box_df in boxes:
        if box_df is not None and not box_df.empty:
            transformed_box = transform_coordinates(box_df)
            transformed_boxes.append(transformed_box)
    
    if not transformed_boxes:
        return pd.DataFrame()
        
    predicted_boxes = pd.concat(transformed_boxes)
    
    final_boxes = []
    if predicted_boxes["image_path"].isnull().all():
        image_boxes = predicted_boxes
        
        # Convert to tensors
        boxes = torch.tensor(image_boxes[["xmin", "ymin", "xmax", "ymax"]].values,
                           dtype=torch.float32)
        scores = torch.tensor(image_boxes.score.values, dtype=torch.float32)
        labels = image_boxes.label.values
        
        # Apply NMS
        filtered_boxes = apply_nms(boxes, scores, labels, iou_threshold)
        filtered_boxes["image_path"] = None
        print(f"{filtered_boxes.shape[0]} predictions kept after non-max suppression")
        final_boxes.append(filtered_boxes)
    else:
        # Apply NMS per image
        for image_path in predicted_boxes["image_path"].unique():
            print(f"{predicted_boxes.shape[0]} predictions in overlapping windows, applying non-max suppression")
            image_boxes = predicted_boxes[predicted_boxes["image_path"] == image_path]
            
            # Convert to tensors
            boxes = torch.tensor(image_boxes[["xmin", "ymin", "xmax", "ymax"]].values,
                            dtype=torch.float32)
            scores = torch.tensor(image_boxes.score.values, dtype=torch.float32)
            labels = image_boxes.label.values
            
            # Apply NMS
            filtered_boxes = apply_nms(boxes, scores, labels, iou_threshold)
            filtered_boxes["image_path"] = image_path
            print(f"{filtered_boxes.shape[0]} predictions kept after non-max suppression")
            final_boxes.append(filtered_boxes)
        
    if not final_boxes:
        return pd.DataFrame()
        
    mosaic_df = pd.concat(final_boxes, ignore_index=True)

    return mosaic_df


def across_class_nms(predicted_boxes, iou_threshold=0.15):
    """Perform non-max suppression for a dataframe of results (see
    visualize.format_boxes) to remove boxes that overlap by iou_thresholdold of
    IoU."""

    # move prediciton to tensor
    boxes = torch.tensor(predicted_boxes[["xmin", "ymin", "xmax", "ymax"]].values,
                         dtype=torch.float32)
    scores = torch.tensor(predicted_boxes.score.values, dtype=torch.float32)
    labels = predicted_boxes.label.values

    bbox_left_idx = nms(boxes=boxes, scores=scores, iou_threshold=iou_threshold)
    bbox_left_idx = bbox_left_idx.numpy()
    new_boxes, new_labels, new_scores = boxes[bbox_left_idx].type(
        torch.int), labels[bbox_left_idx], scores[bbox_left_idx]

    # Recreate box dataframe
    image_detections = np.concatenate([
        new_boxes,
        np.expand_dims(new_labels, axis=1),
        np.expand_dims(new_scores, axis=1)
    ],
                                      axis=1)

    new_df = pd.DataFrame(image_detections,
                          columns=["xmin", "ymin", "xmax", "ymax", "label", "score"])

    return new_df


def _dataloader_wrapper_(model,
                         trainer,
                         dataloader,
                         root_dir,
                         annotations,
                         nms_thresh):
    """Create a dataset and predict entire annotation file.

    Csv file format is .csv file with the columns "image_path", "xmin","ymin","xmax","ymax" for the image name and bounding box position.
    Image_path is the relative filename, not absolute path, which is in the root_dir directory. One bounding box per line.

    Args:
        model: deepforest.main object
        trainer: a pytorch lightning trainer object
        dataloader: pytorch dataloader object
        root_dir: directory of images. If none, uses "image_dir" in config
        annotations: a pandas dataframe of annotations
        nms_thresh: Non-max suppression threshold, see config.nms_thresh
        savedir: Optional. Directory to save image plots.
    Returns:
        results: pandas dataframe with bounding boxes, label and scores for each image in the csv file
    """
    paths = annotations.image_path.unique()
    batched_results = trainer.predict(model, dataloader)

    # Flatten list from batched prediction
    prediction_list = []
    for batch in batched_results:
        for images in batch:
            prediction_list.append(images)

    results = []
    for index, prediction in enumerate(prediction_list):
        # If there is more than one class, apply NMS Loop through images and apply cross
        if len(prediction.label.unique()) > 1:
            prediction = across_class_nms(prediction, iou_threshold=nms_thresh)

        prediction["image_path"] = paths[index]
        results.append(prediction)

    results = pd.concat(results, ignore_index=True)
    if results.empty:
        results["geometry"] = None
        return results

    results = read_file(results, root_dir)

    return results


def _predict_crop_model_(crop_model,
                         trainer,
                         results,
                         raster_path,
                         transform=None,
                         augment=False,
                         model_index=0,
                         is_single_model=False):
    """Predicts crop model on a raster file.

    Args:
        crop_model: The crop model to be used for prediction.
        trainer: The PyTorch Lightning trainer object for prediction.
        results: The results dataframe to store the predicted labels and scores.
        raster_path: The path to the raster file.
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

    # Create dataset
    bounding_box_dataset = dataset.BoundingBoxDataset(
        results,
        root_dir=os.path.dirname(raster_path),
        transform=transform,
        augment=augment)

    # Create dataloader
    crop_dataloader = crop_model.predict_dataloader(bounding_box_dataset)

    # Run prediction
    crop_results = trainer.predict(crop_model, crop_dataloader)

    # Process results
    stacked_outputs = np.vstack(np.concatenate(crop_results))
    label = np.argmax(stacked_outputs, axis=1)  # Get class with highest probability
    score = np.max(stacked_outputs, axis=1)  # Get confidence score

    # Determine column names
    if is_single_model:
        label_column = "cropmodel_label"
        score_column = "cropmodel_score"
    else:
        label_column = f"cropmodel_label_{model_index}"
        score_column = f"cropmodel_score_{model_index}"

    if crop_model.numeric_to_label_dict is None:
        raise ValueError(
            f"The numeric_to_label_dict is not set, and the label_dict is {crop_model.label_dict}, set either when loading CropModel(label_dict=), which creates the numeric_to_label_dict, or load annotations from CropModel.load_from_disk(), which creates the dictionaries based on file contents."
        )

    results[label_column] = [crop_model.numeric_to_label_dict[x] for x in label]

    results[score_column] = score

    return results
