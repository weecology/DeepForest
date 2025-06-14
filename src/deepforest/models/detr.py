import warnings
from transformers import AutoModelForObjectDetection, AutoImageProcessor
from deepforest.model import Model
from torch import nn


class TransformersWrapper(nn.Module):
    """This class wraps a transformers AutoModelForObjectDetection model so
    that input pre- and post-processing happens transparently."""

    def __init__(self, config):
        """Initialize an AutoModelForObjectDetection model.

        We assume that the provided model.name specified via config
        applies to both the model and the processor.
        """
        super().__init__()
        self.config = config

        # This suppresses a bunch of messages which are specific to DETR,
        # but do not impact model function.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)

            self.net = AutoModelForObjectDetection.from_pretrained(
                self.config.model.name,
                revision=self.config.model.revision,
                num_labels=self.config.num_classes,
                ignore_mismatched_sizes=True,
            )

        self.processor = AutoImageProcessor.from_pretrained(
            self.config.model.name,
            revision=self.config.model.revision,
        )

    def forward(self, images, targets=None):
        """AutoModelForObjectDetection forward pass. If targets are provided
        the function returns a loss dictionary, otherwise it returns processed
        predictions. For details, see the transformers documentation for
        "post_process_object_detection".

        Returns:
            predictions: list of dictionaries with "score", "boxes" and "labels", or
                          a loss dict for training.
        """
        encoded_inputs = self.processor.preprocess(images=images,
                                                   annotations=targets,
                                                   return_tensors="pt",
                                                   do_rescale=False)

        preds = self.net(**encoded_inputs)

        if targets is None:
            return self.processor.post_process_object_detection(
                preds,
                threshold=self.config.score_thresh,
                target_sizes=[i.shape[-2:] for i in images]
                if isinstance(images, list) else [images.shape[-2:]])
        else:
            return preds.loss_dict


class Model(Model):

    def __init__(self, config, **kwargs):
        """
        Args:
        """
        super().__init__(config)

    def create_model(self):
        """Create a Deformable DETR model from pretrained weights.

        The number of classes set via config and will override the
        downloaded checkpoint, which is expected if training from a
        model derived from MS-COCO.
        """
        return TransformersWrapper(self.config)
