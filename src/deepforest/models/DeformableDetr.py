import warnings
from pathlib import Path

import torch
from torch import nn
from torchvision.ops import nms
from transformers import (
    DeformableDetrForObjectDetection,
    DeformableDetrImageProcessor,
    logging,
)

from deepforest.model import BaseModel

# Suppress huge amounts of unnecessary warnings from transformers.
logging.set_verbosity_error()


class DeformableDetrWrapper(nn.Module):
    """This class wraps a transformers DeformableDetrForObjectDetection model
    so that input pre- and post-processing happens transparently."""

    task: str = "box"

    def __init__(self, config, name, revision, use_nms=False, **hf_args):
        """Initialize a DeformableDetrForObjectDetection model.

        We assume that the provided name applies to both model and
        processor. By default this function creates a model with MS-COCO
        initialized weights, but can be overridden if needed.
        """
        super().__init__()
        self.config = config
        self.use_nms = use_nms

        # This suppresses a bunch of messages which are specific to DETR,
        # but do not impact model function.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)

            # If the user passed in a different number of classes to the model,
            # then the model will be modified on load. So we ignore
            # mismatched sizes here.
            model_kwargs = {
                "ignore_mismatched_sizes": True,
                **hf_args,
            }

            # Only override label mapping if config provides it
            if self.config.label_dict:
                model_kwargs["label2id"] = dict(self.config.label_dict)
                model_kwargs["id2label"] = {
                    v: k for k, v in self.config.label_dict.items()
                }
                model_kwargs["num_labels"] = len(self.config.label_dict)

            self.net = DeformableDetrForObjectDetection.from_pretrained(
                name,
                revision=revision,
                **model_kwargs,
            )
            self.processor = DeformableDetrImageProcessor.from_pretrained(
                name, revision=revision, **hf_args
            )

            # For consistency with other DeepForest components
            self.label_dict = self.net.config.label2id
            self.num_classes = self.net.config.num_labels

    def _prepare_targets(self, targets):
        """This is an internal function which translates BoxDataset targets
        into MS-COCO format, for use with transformers-like models."""
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

    def _apply_nms(self, predictions, iou_thresh):
        """Apply class-wise NMS to a list of predictions."""
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

    def forward(self, images, targets=None, prepare_targets=True):
        """DeformableDetrForObjectDetection forward pass. If targets are
        provided the function returns a loss dictionary, otherwise it returns
        processed predictions. For details, see the transformers documentation
        for "post_process_object_detection".

        Returns:
            predictions: list of dictionaries with "score", "boxes" and "labels", or
                          a loss dict for training.
        """
        if targets and prepare_targets:
            targets = self._prepare_targets(targets)

        encoded_inputs = self.processor.preprocess(
            images=images, annotations=targets, return_tensors="pt", do_rescale=False
        )

        # Tensor "movement" is not automatic here, this
        # could be refactored to the dataloader (collate_fn)
        # later.
        for k, v in encoded_inputs.items():
            if isinstance(v, torch.Tensor):
                encoded_inputs[k] = v.to(self.net.device)

        if isinstance(encoded_inputs.get("labels"), list):
            [target.to(self.net.device) for target in encoded_inputs["labels"]]

        preds = self.net(**encoded_inputs)

        if targets is None or not self.training:
            results = self.processor.post_process_object_detection(
                preds,
                threshold=self.config.score_thresh,
                target_sizes=[i.shape[-2:] for i in images]
                if isinstance(images, list)
                else [images.shape[-2:]],
            )

            # DETR is specifically designed to be NMS-free, however we've seen cases
            # where it still predicts duplicate boxes
            if self.use_nms:
                results = self._apply_nms(results, iou_thresh=self.config.nms_thresh)

            return results
        else:
            return preds.loss_dict

    def save_pretrained(self, save_directory, push_to_hub=False, **kwargs):
        """Save the model and processor to a directory or push to HF Hub."""
        self.net.save_pretrained(save_directory, push_to_hub=push_to_hub, **kwargs)
        self.processor.save_pretrained(save_directory, push_to_hub=push_to_hub, **kwargs)


class Model(BaseModel):
    def __init__(self, config, **kwargs):
        """
        Args:
        """
        super().__init__(config)

    def create_model(
        self,
        pretrained: str | Path | None = "SenseTime/deformable-detr",
        *,
        revision: str | None = "main",
        map_location: str | torch.device | None = None,
        **hf_args,
    ) -> DeformableDetrWrapper:
        """Create a Deformable DETR model from pretrained weights.

        The number of classes set via config and will override the
        downloaded checkpoint. The default weights will load a model
        trained on MS-COCO that should fine-tune well on other tasks.
        """

        # Take class mapping from config if the user plans to pretrain,
        # otherwise it should be defined by the hub model.
        if pretrained is None:
            hf_args.setdefault("id2label", self.config.numeric_to_label_dict)

        return DeformableDetrWrapper(
            self.config, name=pretrained, revision=revision, **hf_args
        ).to(map_location)
