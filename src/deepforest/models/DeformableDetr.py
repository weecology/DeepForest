import warnings
from pathlib import Path

import torch
from torch import nn
from transformers import (
    DeformableDetrForObjectDetection,
    DeformableDetrImageProcessor,
    logging,
)

from deepforest.model import BaseModel
from deepforest.models import detr_utils

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

            # If the user passed in a different number of classes to the model,
            # then the model will be modified on load. So we ignore
            # mismatched sizes here.
            self.net = DeformableDetrForObjectDetection.from_pretrained(
                name,
                revision=revision,
                num_labels=self.config.num_classes,
                ignore_mismatched_sizes=True,
                auxiliary_loss=self.config.train.auxiliary_loss,
                **hf_args,
            )
            self.processor = DeformableDetrImageProcessor.from_pretrained(
                name,
                do_resize=False,
                do_rescale=False,
                do_normalize=True,
                revision=revision,
                **hf_args,
            )

            # If user-provided label_dict doesn't match the model's id2label:
            if self.net.config.label2id != self.config.label_dict:
                warnings.warn(
                    "Your supplied label dict differs from the model."
                    "This is expected if you plan to fine-tune this model on your own data.",
                    stacklevel=2,
                )
                self.net.config.label2id = self.config.label_dict
                self.net.config.id2label = {
                    v: k for k, v in self.config.label_dict.items()
                }

            # For consistency with other DeepForest components
            self.label_dict = self.net.config.label2id
            self.num_classes = self.net.config.num_labels


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
            targets = detr_utils.prepare_targets(targets)

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

        # Prediction + Validation
        if targets is None or not self.training:
            # Handle padding for mixed-size batches
            original_sizes = [i.shape[-2:] for i in images] if isinstance(images, list) else [images.shape[-2:]]
            batch_size = encoded_inputs['pixel_values'].shape[0]
            encoded_h, encoded_w = encoded_inputs['pixel_values'].shape[-2:]
            target_sizes_padded = [(encoded_h, encoded_w)] * batch_size

            results = self.processor.post_process_object_detection(
                preds,
                threshold=self.config.score_thresh,
                target_sizes=target_sizes_padded,
                top_k=self.net.config.num_queries,
            )

            # Clip boxes in padding area
            if isinstance(images, list):
                for i, (result, orig_size) in enumerate(zip(results, original_sizes)):
                    orig_h, orig_w = orig_size
                    if result['boxes'].shape[0] > 0:
                        result['boxes'][:, [0, 2]] = torch.clamp(result['boxes'][:, [0, 2]], min=0, max=orig_w)
                        result['boxes'][:, [1, 3]] = torch.clamp(result['boxes'][:, [1, 3]], min=0, max=orig_h)

            # DETR is specifically designed to be NMS-free, however we've seen cases
            # where it still predicts duplicate boxes
            if self.use_nms:
                results = self._apply_nms(results, iou_thresh=self.config.nms_thresh)

            return results
        else:
            # Drop cardinality error as it's incorrect for DeformableDETR
            preds.loss_dict.pop('cardinality_error', None)
            return preds.loss_dict


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
