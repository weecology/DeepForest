# Prediction utilities
import cv2
import pandas as pd
import numpy as np
import warnings

import torch
from torchvision.ops import nms

from deepforest import preprocess
from deepforest import visualize

def predict_image(model,
                  image,
                  return_plot,
                  device,
                  iou_threshold=0.1,
                  color=None,
                  thickness=1):
    """Predict an image with a deepforest model

    Args:
        image: a numpy array of a RGB image ranged from 0-255
        path: optional path to read image from disk instead of passing image arg
        return_plot: Return image with plotted detections
        device: pytorch device of 'cuda' or 'cpu' for gpu prediction. Set internally.
        color: color of the bounding box as a tuple of BGR color, e.g. orange annotations is (0, 165, 255)
        thickness: thickness of the rectangle border line in px
    Returns:
        boxes: A pandas dataframe of predictions (Default)
        img: The input with predictions overlaid (Optional)
    """

    if image.dtype != "float32":
        warnings.warn(f"Image type is {image.dtype}, transforming to float32. "
                      f"This assumes that the range of pixel values is 0-255, as "
                      f"opposed to 0-1.To suppress this warning, transform image "
                      f"(image.astype('float32')")
        image = image.astype("float32")
    image = preprocess.preprocess_image(image)

    with torch.no_grad():
        prediction = model(image.unsqueeze(0))

    # return None for no predictions
    if len(prediction[0]["boxes"]) == 0:
        return None

    # This function on takes in a single image.
    df = visualize.format_boxes(prediction[0])
    df = across_class_nms(df, iou_threshold=iou_threshold)

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
        return df


def mosiac(boxes, windows, use_soft_nms=False, sigma=0.5, thresh=0.001, iou_threshold=0.1):
    # transform the coordinates to original system
    for index, _ in enumerate(boxes):
        xmin, ymin, xmax, ymax = windows[index].getRect()
        boxes[index].xmin += xmin
        boxes[index].xmax +=  xmin
        boxes[index].ymin +=  ymin
        boxes[index].ymax +=  ymin
    
    predicted_boxes = pd.concat(boxes)
    print(
        f"{predicted_boxes.shape[0]} predictions in overlapping windows, applying non-max supression"
    )
    # move prediciton to tensor
    boxes = torch.tensor(predicted_boxes[["xmin", "ymin", "xmax", "ymax"]].values,
                         dtype=torch.float32)
    scores = torch.tensor(predicted_boxes.score.values, dtype=torch.float32)
    labels = predicted_boxes.label.values

    if use_soft_nms:
        # Performs soft non-maximum suppression (soft-NMS) on the boxes.
        bbox_left_idx = soft_nms(boxes=boxes,
                                 scores=scores,
                                 sigma=sigma,
                                 thresh=thresh)
    else:
        # Performs non-maximum suppression (NMS) on the boxes according to
        # their intersection-over-union (IoU).
        bbox_left_idx = nms(boxes=boxes,
                            scores=scores,
                            iou_threshold=iou_threshold)

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

    mosaic_df = pd.DataFrame(
        image_detections,
        columns=["xmin", "ymin", "xmax", "ymax", "label", "score"])

    print(f"{mosaic_df.shape[0]} predictions kept after non-max suppression")

    return mosaic_df

def soft_nms(boxes, scores, sigma=0.5, thresh=0.001):
    """
    Perform python soft_nms to reduce the confidances of the proposals proportional  to IoU value
    Paper: Improving Object Detection With One Line of Code
    Code : https://github.com/DocF/Soft-NMS/blob/master/softnms_pytorch.py
    Args:
        boxes: predicitons bounding boxes tensor format [x1,y1,x2,y2]
        scores: the score corresponding to each box tensors
        sigma: variance of Gaussian function
        thresh: score thresh
    Return:
        idxs_keep: the index list of the selected boxes

    """
    # indexes concatenate boxes with the last column
    N = boxes.shape[0]
    indexes = torch.arange(0, N, dtype=torch.float).view(N, 1)

    boxes = torch.cat((boxes, indexes), dim=1)

    # The order of boxes coordinate is [x1,y1,y2,x2]
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tscore = scores[i].clone()
        pos = i + 1

        if i != N - 1:
            maxscore, maxpos = torch.max(scores[pos:], dim=0)
            if tscore < maxscore:
                boxes[i], boxes[maxpos.item() + i +
                                1] = boxes[maxpos.item() + i +
                                           1].clone(), boxes[i].clone()
                scores[i], scores[maxpos.item() + i +
                                  1] = scores[maxpos.item() + i +
                                              1].clone(), scores[i].clone()
                areas[i], areas[maxpos + i + 1] = areas[maxpos + \
                                                        i + 1].clone(), areas[i].clone()

        # IoU calculate
        xx1 = np.maximum(boxes[i, 0].numpy(), boxes[pos:, 0].numpy())
        yy1 = np.maximum(boxes[i, 1].numpy(), boxes[pos:, 1].numpy())
        xx2 = np.minimum(boxes[i, 2].numpy(), boxes[pos:, 2].numpy())
        yy2 = np.minimum(boxes[i, 3].numpy(), boxes[pos:, 3].numpy())

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = torch.tensor(w * h)
        ovr = torch.div(inter, (areas[i] + areas[pos:] - inter))

        # Gaussian decay
        weight = torch.exp(-(ovr * ovr) / sigma)
        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    idxs_keep = boxes[:, 4][scores > thresh].int()

    return idxs_keep


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
