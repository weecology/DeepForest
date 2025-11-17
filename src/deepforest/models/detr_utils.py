"""Shared utility functions for DETR-based models."""

import torch
from torchvision.ops import nms


def prepare_targets(targets):
    """Translate BoxDataset targets into MS-COCO format for transformers models.

    Args:
        targets: List of target dictionaries with 'labels' and 'boxes' keys

    Returns:
        List of COCO-formatted target dictionaries
    """
    if not isinstance(targets, list):
        targets = [targets]

    coco_targets = []

    for target in targets:
        annotations_for_target = []
        for i, (label, box) in enumerate(
            zip(target["labels"], target["boxes"], strict=False)
        ):
            if isinstance(box, torch.Tensor):
                box = box.tolist()

            if isinstance(label, torch.Tensor):
                label = label.item()

            # Convert from [xmin, ymin, xmax, ymax] to COCO format [x, y, width, height]
            xmin, ymin, xmax, ymax = box
            coco_bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
            area = (xmax - xmin) * (ymax - ymin)

            annotations_for_target.append(
                {
                    "id": i,
                    "image_id": i,
                    "category_id": label,
                    "bbox": coco_bbox,
                    "area": area,
                    "iscrowd": 0,
                }
            )

        coco_targets.append({"image_id": 0, "annotations": annotations_for_target})

    return coco_targets


def apply_nms(predictions, iou_thresh):
    """Apply class-wise NMS to a list of predictions.

    Args:
        predictions: List of prediction dictionaries with 'boxes', 'scores', 'labels'
        iou_thresh: IoU threshold for NMS

    Returns:
        List of filtered prediction dictionaries
    """
    filtered = []
    for pred in predictions:
        boxes = pred["boxes"]
        scores = pred["scores"]
        labels = pred["labels"]

        keep = []
        for cls in labels.unique():
            cls_mask = labels == cls
            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]
            cls_keep = nms(cls_boxes, cls_scores, iou_thresh)
            cls_indices = torch.nonzero(cls_mask).squeeze(1)[cls_keep]
            keep.append(cls_indices)

        if keep:
            keep = torch.cat(keep)
            filtered.append(
                {
                    "boxes": boxes[keep],
                    "scores": scores[keep],
                    "labels": labels[keep],
                }
            )
        else:
            filtered.append(
                {
                    "boxes": boxes,
                    "scores": scores,
                    "labels": labels,
                }
            )

    return filtered


def handle_padding_and_postprocess(processor, preds, encoded_inputs, images, config, num_queries=None):
    """Handle padded batches and post-process predictions with proper coordinate scaling.

    When images of different sizes are batched, the processor pads them to a uniform size.
    Predictions are in normalized [0, 1] coordinates relative to the padded image space,
    so they must be scaled by padded dimensions and then clipped to original bounds.

    Args:
        processor: HuggingFace image processor with post_process_object_detection method
        preds: Raw model predictions
        encoded_inputs: Dictionary containing 'pixel_values' tensor
        images: List of original image tensors
        config: Model configuration with score_thresh
        num_queries: Optional top_k parameter for DeformableDetr/ConditionalDetr

    Returns:
        List of post-processed prediction dictionaries
    """
    original_sizes = [i.shape[-2:] for i in images] if isinstance(images, list) else [images.shape[-2:]]
    batch_size = encoded_inputs['pixel_values'].shape[0]
    encoded_h, encoded_w = encoded_inputs['pixel_values'].shape[-2:]
    target_sizes_padded = [(encoded_h, encoded_w)] * batch_size

    kwargs = {
        "threshold": config.score_thresh,
        "target_sizes": target_sizes_padded,
    }

    if num_queries is not None:
        kwargs["top_k"] = num_queries

    results = processor.post_process_object_detection(preds, **kwargs)

    if isinstance(images, list):
        for i, (result, orig_size) in enumerate(zip(results, original_sizes)):
            orig_h, orig_w = orig_size
            if result['boxes'].shape[0] > 0:
                result['boxes'][:, [0, 2]] = torch.clamp(result['boxes'][:, [0, 2]], min=0, max=orig_w)
                result['boxes'][:, [1, 3]] = torch.clamp(result['boxes'][:, [1, 3]], min=0, max=orig_h)

    return results
