# Prediction utilities
import cv2
import pandas as pd
import numpy as np
import os
from PIL import Image
import warnings

import torch
from torchvision.ops import nms
import typing

from deepforest import visualize


def _predict_image_(model,
                    image: typing.Optional[np.ndarray] = None,
                    path: typing.Optional[str] = None,
                    nms_thresh: float = 0.15,
                    return_plot: bool = False,
                    thickness: int = 1,
                    color: typing.Optional[tuple] = (0, 165, 255)):
    """Predict a single image with a deepforest model
            
    Args:
        model: a deepforest.main.model object
        image: a tensor of shape (channels, height, width)
        path: optional path to read image from disk instead of passing image arg
        nms_thresh: Non-max supression threshold, see config["nms_thresh"]
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


def mosiac(boxes, windows, sigma=0.5, thresh=0.001, iou_threshold=0.1):
    # transform the coordinates to original system
    for index, _ in enumerate(boxes):
        xmin, ymin, xmax, ymax = windows[index].getRect()
        boxes[index].xmin += xmin
        boxes[index].xmax += xmin
        boxes[index].ymin += ymin
        boxes[index].ymax += ymin

    predicted_boxes = pd.concat(boxes)
    print(
        f"{predicted_boxes.shape[0]} predictions in overlapping windows, applying non-max supression"
    )
    # move prediciton to tensor
    boxes = torch.tensor(predicted_boxes[["xmin", "ymin", "xmax", "ymax"]].values,
                         dtype=torch.float32)
    scores = torch.tensor(predicted_boxes.score.values, dtype=torch.float32)
    labels = predicted_boxes.label.values
    # Performs non-maximum suppression (NMS) on the boxes according to
    # their intersection-over-union (IoU).
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

    mosaic_df = pd.DataFrame(image_detections,
                             columns=["xmin", "ymin", "xmax", "ymax", "label", "score"])

    print(f"{mosaic_df.shape[0]} predictions kept after non-max suppression")

    return mosaic_df


def across_class_nms(predicted_boxes, iou_threshold=0.15):
    """perform non-max suppression for a dataframe of results (see visualize.format_boxes) to remove boxes that overlap by iou_thresholdold of IoU"""

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
                         nms_thresh,
                         savedir=None,
                         color=None,
                         thickness=1):
    """Create a dataset and predict entire annotation file

    Csv file format is .csv file with the columns "image_path", "xmin","ymin","xmax","ymax" for the image name and bounding box position.
    Image_path is the relative filename, not absolute path, which is in the root_dir directory. One bounding box per line.

    Args:
        model: deepforest.main object
        trainer: a pytorch lightning trainer object
        dataloader: pytorch dataloader object
        root_dir: directory of images. If none, uses "image_dir" in config
        annotations: a pandas dataframe of annotations
        nms_thresh: Non-max supression threshold, see config["nms_thresh"]
        savedir: Optional. Directory to save image plots.
        color: color of the bounding box as a tuple of BGR color, e.g. orange annotations is (0, 165, 255)
        thickness: thickness of the rectangle border line in px
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

    if savedir:
        visualize.plot_prediction_dataframe(results,
                                            root_dir=root_dir,
                                            savedir=savedir,
                                            color=color,
                                            thickness=thickness)

    return results
