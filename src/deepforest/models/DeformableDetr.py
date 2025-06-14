import warnings
from transformers import DeformableDetrForObjectDetection, DeformableDetrImageProcessor, logging
from deepforest.model import Model
from torch import nn

# Suppress huge amounts of unnecessary warnings from transformers.
logging.set_verbosity_error()


class TransformersWrapper(nn.Module):
    """This class wraps a transformers AutoModelForObjectDetection model so
    that input pre- and post-processing happens transparently."""

    def __init__(self, config, name, revision):
        """Initialize an AutoModelForObjectDetection model.

        We assume that the provided name applies to both model and
        processor. By default this function creates a model with MS-COCO
        initialized weights, but can be overridden if needed.
        """
        super().__init__()
        self.config = config

        # This suppresses a bunch of messages which are specific to DETR,
        # but do not impact model function.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)

            self.net = DeformableDetrForObjectDetection.from_pretrained(
                name,
                revision=revision,
                num_labels=self.config.num_classes,
                ignore_mismatched_sizes=True)
            self.processor = DeformableDetrImageProcessor.from_pretrained(
                name, revision=revision)

    def prepare_targets(self, targets):

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

    def forward(self, images, targets=None, prepare_targets=True):
        """AutoModelForObjectDetection forward pass. If targets are provided
        the function returns a loss dictionary, otherwise it returns processed
        predictions. For details, see the transformers documentation for
        "post_process_object_detection".

        Returns:
            predictions: list of dictionaries with "score", "boxes" and "labels", or
                          a loss dict for training.
        """

        if targets and prepare_targets:
            targets = self.prepare_targets(targets)

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

    def create_model(self, name="SenseTime/deformable-detr", revision="main"):
        """Create a Deformable DETR model from pretrained weights.

        The number of classes set via config and will override the
        downloaded checkpoint, which is expected if training from a
        model derived from MS-COCO.
        """
        return TransformersWrapper(self.config, name, revision)
