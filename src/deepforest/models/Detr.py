import warnings
from pathlib import Path

import torch
from torch import nn
from transformers import (
    DetrForObjectDetection,
    DetrImageProcessor,
    logging,
)

from deepforest.model import BaseModel
from deepforest.models import detr_utils

# Suppress huge amounts of unnecessary warnings from transformers.
logging.set_verbosity_error()


class DetrWrapper(nn.Module):
    """This class wraps a transformers DetrForObjectDetection model so that
    input pre- and post-processing happens transparently."""

    def __init__(self, config, name, revision, use_nms=False, **hf_args):
        """Initialize a DetrForObjectDetection model.

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
            self.net = DetrForObjectDetection.from_pretrained(
                name,
                revision=revision,
                num_labels=self.config.num_classes,
                num_queries=300,
                ignore_mismatched_sizes=True,
                auxiliary_loss=self.config.train.auxiliary_loss,
                **hf_args,
            )
            self.processor = DetrImageProcessor.from_pretrained(
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
        """DetrForObjectDetection forward pass. If targets are provided the
        function returns a loss dictionary, otherwise it returns processed
        predictions. For details, see the transformers documentation for
        "post_process_object_detection".

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

        if targets is None or not self.training:
            results = detr_utils.handle_padding_and_postprocess(
                self.processor, preds, encoded_inputs, images, self.config
            )

            if self.use_nms:
                results = detr_utils.apply_nms(results, iou_thresh=self.config.nms_thresh)

            return results
        else:
            return preds.loss_dict


class Model(BaseModel):
    def __init__(self, config, **kwargs):
        """
        Args:
        """
        super().__init__(config)

    def create_model(
        self,
        pretrained: str | Path | None = "facebook/detr-resnet-50",
        *,
        revision: str | None = "main",
        map_location: str | torch.device | None = None,
        **hf_args,
    ) -> DetrWrapper:
        """Create a DETR model from pretrained weights.

        The number of classes set via config and will override the
        downloaded checkpoint. The default weights will load a model
        trained on MS-COCO that should fine-tune well on other tasks.
        """

        # Take class mapping from config if the user plans to pretrain,
        # otherwise it should be defined by the hub model.
        if pretrained is None:
            hf_args.setdefault("id2label", self.config.numeric_to_label_dict)

        return DetrWrapper(self.config, name=pretrained, revision=revision, **hf_args).to(
            map_location
        )
