import warnings
from transformers import DeformableDetrForObjectDetection, DeformableDetrImageProcessor, logging
from deepforest.model import BaseModel
import torch
from pathlib import Path
from torch import nn
from torchvision.ops import nms
# Suppress huge amounts of unnecessary warnings from transformers.
logging.set_verbosity_error()


class DeformableDetrWrapper(nn.Module):
    """This class wraps a transformers DeformableDetrForObjectDetection model
    so that input pre- and post-processing happens transparently."""

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

            self.net = DeformableDetrForObjectDetection.from_pretrained(
                name,
                revision=revision,
                num_labels=self.config.num_classes,
                ignore_mismatched_sizes=True,
                **hf_args)
            self.processor = DeformableDetrImageProcessor.from_pretrained(
                name, revision=revision, **hf_args)

            self.label_dict = self.net.config.label2id

    def _prepare_targets(self, targets):

        if not isinstance(targets, list):
            targets = [targets]

        coco_targets = []

        for target in targets:
            coco_targets.append({
                "image_id":
                    0,
                "annotations": [{
                    "id": i,
                    "image_id": i,
                    "category_id": label,
                    "bbox": box.tolist(),
                    "area": (box[3] - box[1]) * (box[2] - box[0]),
                    "iscrowd": 0,
                } for i, (label, box) in enumerate(zip(target["labels"], target["boxes"]))
                               ]
            })

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
                filtered.append({
                    "boxes": boxes[keep],
                    "scores": scores[keep],
                    "labels": labels[keep],
                })
            else:
                filtered.append({
                    "boxes": boxes,
                    "scores": scores,
                    "labels": labels,
                })

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

        encoded_inputs = self.processor.preprocess(images=images,
                                                   annotations=targets,
                                                   return_tensors="pt",
                                                   do_rescale=False)

        preds = self.net(**encoded_inputs)

        if targets is None or not self.training:
            results = self.processor.post_process_object_detection(
                preds,
                threshold=self.config.score_thresh,
                target_sizes=[i.shape[-2:] for i in images]
                if isinstance(images, list) else [images.shape[-2:]])

            # DETR is specifically designed to be NMS-free, however we've seen cases
            # where it still predicts duplicate boxes
            if self.use_nms:
                results = self._apply_nms(results, iou_thresh=self.config.nms_thresh)

            return results
        else:
            return preds.loss_dict


class Model(BaseModel):

    def __init__(self, config, **kwargs):
        """
        Args:
        """
        super().__init__(config)

    def create_model(self,
                     pretrained: str | Path | None = "SenseTime/deformable-detr",
                     *,
                     revision: str | None = "main",
                     map_location: str | torch.device | None = None,
                     **hf_args) -> DeformableDetrWrapper:
        """Create a Deformable DETR model from pretrained weights.

        The number of classes set via config and will override the
        downloaded checkpoint. The default weights will load a model
        trained on MS-COCO that should fine-tune well on other tasks.
        """

        # Take class mapping from config if the user plans to pretrain,
        # otherwise it should be defined by the hub model.
        if pretrained is None:
            hf_args.setdefault('id2label', self.config.numeric_to_label_dict)

        return DeformableDetrWrapper(self.config,
                                     name=pretrained,
                                     revision=revision,
                                     **hf_args).to(map_location)
