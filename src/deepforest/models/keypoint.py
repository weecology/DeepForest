"""This code is largely derived from the transformers
DeformableDetrForObjectDetection class, with additional support for processing
and loss calculation. Several functions have sections that are copied mostly
verbatim due to only a few lines changing.

Under the Apache 2.0 license, transformers code is copyright 2018- The
Hugging Face team. All rights reserved.
https://github.com/huggingface/transformers?tab=Apache-2.0-1-ov-file
"""

from dataclasses import dataclass

import numpy as np
import PIL
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from transformers import (
    DeformableDetrConfig,
    DeformableDetrModel,
    DeformableDetrPreTrainedModel,
)
from transformers.image_utils import ChannelDimension, get_image_size
from transformers.loss.loss_deformable_detr import DeformableDetrImageLoss
from transformers.loss.loss_for_object_detection import HungarianMatcher
from transformers.models.deformable_detr.image_processing_deformable_detr import (
    DeformableDetrImageProcessor,
)
from transformers.models.deformable_detr.modeling_deformable_detr import (
    DeformableDetrMLPPredictionHead,
    inverse_sigmoid,
)
from transformers.utils import ModelOutput


class DeformableDetrKeypointConfig(DeformableDetrConfig):
    """Configuration for Deformable DETR keypoint detection.

    Extends DeformableDetrConfig with keypoint-specific parameters.
    """

    def __init__(
        self,
        point_cost: float = 5.0,
        point_loss_coefficient: float = 5.0,
        point_loss_type: str = "l1",
        **kwargs,
    ):
        """
        Args:
            point_cost: The relative weight of the point distance in the matching cost.
            point_loss_coefficient: The coefficient for the point loss in the total loss.
            point_loss_type: Type of loss for point coordinates. Options: "l1" (default, standard for DETR) or "mse" (L2).
            **kwargs: Additional arguments passed to DeformableDetrConfig.
        """
        super().__init__(**kwargs)
        self.point_cost = point_cost
        self.point_loss_coefficient = point_loss_coefficient
        if point_loss_type not in ["l1", "mse"]:
            raise ValueError(
                f"point_loss_type must be 'l1' or 'mse', got '{point_loss_type}'"
            )
        self.point_loss_type = point_loss_type


@dataclass
class DeformableDetrKeypointDetectionOutput(ModelOutput):
    r"""init_reference_points (`torch.FloatTensor` of shape  `(batch_size,
    num_queries, 2)`):

    Initial reference points sent through the Transformer decoder.
    last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
        Sequence of hidden-states at the output of the last layer of the decoder of the model.
    intermediate_hidden_states (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`):
        Stacked intermediate hidden states (output of each layer of the decoder).
    intermediate_reference_points (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, 2)`):
        Stacked intermediate reference points (reference points of each layer of the decoder).
    enc_outputs_class (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`, *optional*):
        NOT CURRENTLY SUPPORTED. Would be used for two-stage detection where encoder predicts
        initial keypoint proposals. Currently always None.
    enc_outputs_coord_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, 2)`, *optional*):
        NOT CURRENTLY SUPPORTED. Would be used for two-stage detection where encoder predicts
        initial keypoint coordinates. Currently always None.
    """

    loss: torch.FloatTensor | None = None
    loss_dict: dict | None = None
    logits: torch.FloatTensor | None = None
    pred_points: torch.FloatTensor | None = None
    auxiliary_outputs: list[dict] | None = None
    init_reference_points: torch.FloatTensor | None = None
    last_hidden_state: torch.FloatTensor | None = None
    intermediate_hidden_states: torch.FloatTensor | None = None
    intermediate_reference_points: torch.FloatTensor | None = None
    decoder_hidden_states: tuple[torch.FloatTensor] | None = None
    decoder_attentions: tuple[torch.FloatTensor] | None = None
    cross_attentions: tuple[torch.FloatTensor] | None = None
    encoder_last_hidden_state: torch.FloatTensor | None = None
    encoder_hidden_states: tuple[torch.FloatTensor] | None = None
    encoder_attentions: tuple[torch.FloatTensor] | None = None
    enc_outputs_class: torch.FloatTensor | None = None
    enc_outputs_coord_logits: torch.FloatTensor | None = None


class DeformableDetrKeypointMatcher(HungarianMatcher):
    """Hungarian matcher for keypoint detection using L2 distance for matching
    cost.

    Note: The matcher always uses L2 (Euclidean) distance for computing matching cost.
    The actual training loss (L1 vs MSE) is configured separately in DeformableDetrKeypointLoss.

    Args:
        class_cost: Relative weight of the classification error in the matching cost.
        point_cost: Relative weight of the L2 point distance in the matching cost.
    """

    def __init__(self, class_cost: float = 1, point_cost: float = 1):
        # Map point_cost to parent's bbox_cost parameter (semantically it's for points here)
        super().__init__(class_cost=class_cost, bbox_cost=point_cost, giou_cost=0)

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Matches predicted keypoints to ground truth using:
        - Classification cost (focal loss)
        - L2 point distance cost
        """
        batch_size, num_queries = outputs["logits"].shape[:2]

        # Flatten to compute cost matrices in a batch
        out_prob = (
            outputs["logits"].flatten(0, 1).sigmoid()
        )  # [batch_size * num_queries, num_classes]
        out_points = outputs["pred_points"].flatten(0, 1)  # [batch_size * num_queries, 2]

        # Concatenate target labels and points
        target_ids = torch.cat([v["class_labels"] for v in targets])
        target_points = torch.cat([v["points"] for v in targets])

        # Compute approximate classification cost
        class_cost = -out_prob[:, target_ids]

        # Compute L1 point distance cost
        point_cost = torch.cdist(out_points, target_points, p=1)

        # Final cost matrix
        # Note: self.bbox_cost was set to point_cost value in __init__
        cost_matrix = self.class_cost * class_cost + self.bbox_cost * point_cost
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

        sizes = [len(v["points"]) for v in targets]

        indices = [
            linear_sum_assignment(c[i])
            for i, c in enumerate(cost_matrix.split(sizes, -1))
        ]
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]


class DeformableDetrKeypointLoss(DeformableDetrImageLoss):
    """Loss for keypoint detection using focal loss and configurable point
    distance loss.

    Inherits loss_labels, loss_cardinality, and forward from DeformableDetrImageLoss.
    Only adds loss computation for points via get_loss override.

    Args:
        loss_type: Type of loss for coordinates - "l1" (default, standard for DETR) or "mse" (L2).
    """

    def __init__(self, matcher, num_classes, focal_alpha, losses, loss_type="l1"):
        super().__init__(matcher, num_classes, focal_alpha, losses)
        if loss_type not in ["l1", "mse"]:
            raise ValueError(f"loss_type must be 'l1' or 'mse', got '{loss_type}'")
        self.loss_type = loss_type

    def loss_points(self, outputs, targets, indices, num_objects):
        """Distance loss for keypoint coordinates (L1 or MSE based on
        config)."""
        idx = self._get_source_permutation_idx(indices)
        src_points = outputs["pred_points"][idx]
        target_points = torch.cat(
            [t["points"][i] for t, (_, i) in zip(targets, indices, strict=False)], dim=0
        )

        # Apply configured loss type
        if self.loss_type == "l1":
            loss_point = nn.functional.l1_loss(
                src_points, target_points, reduction="none"
            )
        else:  # mse
            loss_point = nn.functional.mse_loss(
                src_points, target_points, reduction="none"
            )

        losses = {f"loss_point_{self.loss_type}": loss_point.sum() / num_objects}

        return losses

    def get_loss(self, loss, outputs, targets, indices, num_objects):
        """Extend parent loss map to support 'points'."""
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "points": self.loss_points,
        }
        if loss not in loss_map:
            raise ValueError(f"Loss {loss} not supported")
        return loss_map[loss](outputs, targets, indices, num_objects)


def DeformableDetrForKeypointDetectionLoss(
    logits,
    labels,
    device,
    pred_points,
    config,
    outputs_class=None,
    outputs_coord=None,
    **kwargs,
):
    """Loss function for keypoint detection."""
    # Point matching can use Hungarian, just like boxes
    point_cost = getattr(
        config, "point_cost", config.bbox_cost
    )  # Fallback for backwards compatibility
    matcher = DeformableDetrKeypointMatcher(
        class_cost=config.class_cost, point_cost=point_cost
    )

    # Setup the criterion, default L1 but L2/MSE is also allowed
    losses = ["labels", "points", "cardinality"]
    loss_type = getattr(config, "point_loss_type", "l1")
    criterion = DeformableDetrKeypointLoss(
        matcher=matcher,
        num_classes=config.num_labels,
        focal_alpha=config.focal_alpha,
        losses=losses,
        loss_type=loss_type,
    )
    criterion.to(device)

    # Compute individual losses
    outputs_loss = {}
    auxiliary_outputs = None
    outputs_loss["logits"] = logits
    outputs_loss["pred_points"] = pred_points
    if config.auxiliary_loss:
        # Adapt _set_aux_loss for points instead of boxes
        auxiliary_outputs = [
            {"logits": a, "pred_points": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1], strict=False)
        ]
        outputs_loss["auxiliary_outputs"] = auxiliary_outputs

    loss_dict = criterion(outputs_loss, labels)
    # Compute total loss
    point_loss_coefficient = getattr(
        config, "point_loss_coefficient", config.bbox_loss_coefficient
    )
    weight_dict = {"loss_ce": 1, f"loss_point_{loss_type}": point_loss_coefficient}
    if config.auxiliary_loss:
        aux_weight_dict = {}
        for i in range(config.decoder_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)

    return loss, loss_dict, auxiliary_outputs


def prepare_keypoint_annotation(
    image,
    target,
    input_data_format: ChannelDimension | str | None = None,
):
    """Convert keypoint annotations into the format expected by DeformableDetr.

    Expected input format:
        {
            "image_id": int,
            "annotations": [
                {
                    "category_id": int,
                    "keypoints": [x1,y1,...],
                    "keypoints": [x, y] or [[x1, y1], [x2, y2], ...],  # Single or multiple keypoints
                },
                ...
            ]
        }

    Output format:
        {
            "image_id": array,
            "labels": array of shape (num_keypoints,),
            "points": array of shape (num_keypoints, 2),  # Internal representation
            "orig_size": array of shape (2,)  # [height, width]
        }
    """
    image_height, image_width = get_image_size(image, channel_dim=input_data_format)

    image_id = target["image_id"]
    image_id = np.asarray([image_id], dtype=np.int64)

    # Get all annotations for the given image
    annotations = target["annotations"]

    classes = [obj["category_id"] for obj in annotations]
    classes = np.asarray(classes, dtype=np.int64)

    # Extract keypoints
    keypoints_list = []
    for obj in annotations:
        if "keypoints" in obj:
            kpts = obj["keypoints"]
            kpts = np.asarray(kpts, dtype=np.float32)

            # Handle different keypoint formats
            if kpts.ndim == 1:
                # Single keypoint: [x, y]
                if len(kpts) != 2:
                    raise ValueError(
                        f"Expected keypoint to have 2 coordinates (x, y), got {len(kpts)}"
                    )
                kpts = kpts.reshape(1, 2)
            elif kpts.ndim == 2:
                # Multiple keypoints: [[x1, y1], [x2, y2], ...]
                if kpts.shape[1] != 2:
                    raise ValueError(
                        f"Expected keypoints to have 2 coordinates per point (x, y), got {kpts.shape[1]}"
                    )
            else:
                raise ValueError(f"Invalid keypoint format with {kpts.ndim} dimensions")
        # If objects, they're separate
        elif "bbox" in obj:
            x, y, w, h = obj["bbox"]
            kpt = np.asarray([[x + w / 2, y + h / 2]], dtype=np.float32)

        keypoints_list.append(kpts)

    # Flatten all keypoints and their corresponding class labels
    all_points = []
    all_classes = []

    for i, kpts in enumerate(keypoints_list):
        for kpt in kpts:
            all_points.append(kpt)
            all_classes.append(classes[i])

    points = np.array(all_points, dtype=np.float32).reshape(-1, 2)

    # Clip points to image boundaries
    points[:, 0] = points[:, 0].clip(min=0, max=image_width)
    points[:, 1] = points[:, 1].clip(min=0, max=image_height)

    new_target = {}
    new_target["image_id"] = image_id
    new_target["labels"] = np.array(all_classes, dtype=np.int64)
    # Transformers library expects "class_labels" key for loss computation
    new_target["class_labels"] = new_target["labels"]
    new_target["points"] = points
    new_target["orig_size"] = np.asarray(
        [int(image_height), int(image_width)], dtype=np.int64
    )

    return new_target


def normalize_keypoint_annotation(annotation: dict, image_size: tuple[int, int]) -> dict:
    """Normalize keypoint annotations to [0, 1] coordinate space.

    Args:
        annotation: Dictionary containing "points" key with shape (num_points, 2)
        image_size: Tuple of (height, width)

    Returns:
        Normalized annotation dictionary with points scaled to [0, 1]
    """
    image_height, image_width = image_size
    norm_annotation = annotation.copy()

    # Normalize points: divide x by width, y by height
    if "points" in annotation:
        norm_annotation["points"] = annotation["points"] / np.array(
            [image_width, image_height], dtype=np.float32
        )

    # Ensure both "labels" and "class_labels" are preserved for transformers compatibility
    if "labels" in norm_annotation and "class_labels" not in norm_annotation:
        norm_annotation["class_labels"] = norm_annotation["labels"]

    return norm_annotation


class DeformableDetrKeypointImageProcessor(DeformableDetrImageProcessor):
    """Image processor for keypoint detection with Deformable DETR.

    Extends DeformableDetrImageProcessor to handle keypoint annotations
    instead of bounding boxes. Uses "keypoints" in external API and
    "points" for internal model representation.
    """

    def prepare_annotation(
        self,
        image: np.ndarray,
        target: dict,
        format: str | None = None,
        return_segmentation_masks=None,
        masks_path=None,
        input_data_format: str | ChannelDimension | None = None,
    ) -> dict:
        """Prepare a keypoint annotation for feeding into DeformableDetr model.

        Overrides parent to handle keypoint annotations instead of
        bounding boxes.
        """
        return prepare_keypoint_annotation(
            image, target, input_data_format=input_data_format
        )

    def normalize_annotation(self, annotation: dict, image_size: tuple[int, int]) -> dict:
        """Normalize keypoint annotations to [0, 1] coordinate space.

        Called by parent's preprocess() method when
        do_convert_annotations=True.
        """
        return normalize_keypoint_annotation(annotation, image_size)

    def resize_annotation(
        self,
        annotation,
        orig_size,
        size,
        resample: PIL.Image.Resampling = PIL.Image.Resampling.NEAREST,
    ) -> dict:
        """Resize the annotation to match the resized image.

        This is an override of the existing function in transformers to
        handle keypoints only (since we don't care about other
        annotations here). Since the processor may resize samples, we
        also need to scale keypoints to match (even though they are then
        scaled to [0,1] anyway).
        """
        ratios = tuple(
            float(s) / float(s_orig) for s, s_orig in zip(size, orig_size, strict=False)
        )
        ratio_height, ratio_width = ratios

        new_annotation = dict(annotation)
        new_annotation["points"] *= np.array([ratio_height, ratio_width])

        return new_annotation

    def post_process_keypoint_detection(
        self,
        outputs,
        threshold: float = 0.5,
        target_sizes: torch.Tensor | list[tuple] = None,
        top_k: int = 100,
    ):
        """Converts the raw output of DeformableDetrForKeypointDetection into
        final keypoints.

        Args:
            outputs: Raw outputs of the model with 'logits' and 'pred_points'
            threshold: Score threshold to keep keypoint predictions
            target_sizes: Tensor of shape (batch_size, 2) or list of tuples (height, width)
            top_k: Keep only top k keypoints before filtering by threshold

        Returns:
            List of dictionaries with 'scores', 'labels', and 'keypoints' for each image
        """
        out_logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
        out_points = (
            outputs.pred_points
            if hasattr(outputs, "pred_points")
            else outputs["pred_points"]
        )

        if target_sizes is not None:
            if len(out_logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

        # Get class probabilities
        prob = out_logits.sigmoid()
        prob = prob.view(out_logits.shape[0], -1)
        k_value = min(top_k, prob.size(1))
        topk_values, topk_indexes = torch.topk(prob, k_value, dim=1)
        scores = topk_values

        # Get corresponding point indices and labels
        topk_points_idx = torch.div(
            topk_indexes, out_logits.shape[2], rounding_mode="floor"
        )
        labels = topk_indexes % out_logits.shape[2]

        # Gather the corresponding points
        points = torch.gather(
            out_points, 1, topk_points_idx.unsqueeze(-1).repeat(1, 1, 2)
        )

        # Convert from relative [0, 1] to absolute [0, height/width] coordinates
        if target_sizes is not None:
            if isinstance(target_sizes, list):
                img_h = torch.Tensor([i[0] for i in target_sizes])
                img_w = torch.Tensor([i[1] for i in target_sizes])
            else:
                img_h, img_w = target_sizes.unbind(1)
            scale_fct = torch.stack([img_w, img_h], dim=1).to(points.device)
            points = points * scale_fct[:, None, :]

        # Filter by threshold and return results for each item in batch
        results = []
        for result in zip(scores, labels, points, strict=False):
            score, label, point = result

            # Filter all outputs by score threshold
            mask = score > threshold
            score = score[mask]
            label = label[mask]
            keypoint = point[mask]

            results.append(
                {
                    "scores": score,
                    "labels": label,
                    "keypoints": keypoint,
                }
            )

        return results


class DeformableDetrForKeypointDetection(DeformableDetrPreTrainedModel):
    # When using clones, all layers > 0 will be clones, but layer 0 *is* required
    _tied_weights_keys = [r"point_embed\.[1-9]\d*", r"class_embed\.[1-9]\d*"]
    # We can't initialize the model on meta device as some weights are modified during the initialization
    _no_split_modules = None

    def __init__(self, config: DeformableDetrConfig):
        super().__init__(config)

        # Deformable DETR encoder-decoder model
        self.model = DeformableDetrModel(config)

        # Detection heads on top
        self.class_embed = nn.Linear(config.d_model, config.num_labels)

        # 2D output for x/y
        self.point_embed = DeformableDetrMLPPredictionHead(
            input_dim=config.d_model,
            hidden_dim=config.d_model,
            output_dim=2,
            num_layers=3,
        )

        # Currently no support for with_box_refine (iterative refinement) or two_stage
        # with_box_refine: Would create independent prediction heads per decoder layer
        #                  for iterative point coordinate refinement
        # two_stage: Would add encoder-based proposal generation before decoder refinement
        #           (requires with_box_refine=True)
        num_pred = config.decoder_layers

        # Weight-tied prediction heads across decoder layers
        self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
        self.point_embed = nn.ModuleList([self.point_embed for _ in range(num_pred)])
        self.model.decoder.point_embed = None

        if config.two_stage:
            raise NotImplementedError(
                "Two-stage keypoint detection is not currently supported. "
                "This would require implementing encoder-side proposal generation. "
                "Set config.two_stage=False to use standard single-stage detection."
            )

        # Initialize weights and apply final processing
        self.post_init()

    def loss_function(self, *args, **kwargs):
        """Wrapper for the keypoint detection loss function."""
        return DeformableDetrForKeypointDetectionLoss(*args, **kwargs)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: torch.LongTensor | None = None,
        decoder_attention_mask: torch.FloatTensor | None = None,
        encoder_outputs: torch.FloatTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        decoder_inputs_embeds: torch.FloatTensor | None = None,
        labels: list[dict] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple[torch.FloatTensor] | DeformableDetrKeypointDetectionOutput:
        r"""For full documentation, look at DeformableDetrForObjectDetection.

        labels (`list[Dict]` of len `(batch_size,)`, *optional*):
            Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
            following 2 keys: 'labels' and 'points' (the class labels and object centers (points) of an image in the batch
            respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of objects
            in the image,)` and the points a `torch.FloatTensor` of shape `(number of points in the image, 2)`.
        ```
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # First, send images through DETR base model to obtain encoder + decoder outputs
        outputs = self.model(
            pixel_values,
            pixel_mask=pixel_mask,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs.intermediate_hidden_states if return_dict else outputs[2]
        init_reference = outputs.init_reference_points if return_dict else outputs[0]
        inter_references = (
            outputs.intermediate_reference_points if return_dict else outputs[3]
        )

        # class logits + predicted points
        outputs_classes = []
        outputs_coords = []

        # References in Deformable DETR are 2D points corresponding to object
        # centers. This naturally leads to keypoints anyway.
        for level in range(hidden_states.shape[1]):
            if level == 0:
                reference = init_reference
            else:
                reference = inter_references[:, level - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[level](hidden_states[:, level])
            delta_point = self.point_embed[level](hidden_states[:, level])

            # For keypoints: reference points are (x, y) with shape [..., 2]
            if reference.shape[-1] != 2:
                raise ValueError(
                    f"Keypoint detection requires 2D reference points (x, y), "
                    f"but got shape [..., {reference.shape[-1]}]"
                )

            # Add predicted delta to reference in logit space
            outputs_coord_logits = delta_point + reference
            # Convert back to [0, 1] normalized coordinates
            outputs_coord = outputs_coord_logits.sigmoid()

            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        logits = outputs_class[-1]
        pred_points = outputs_coord[-1]

        loss, loss_dict, auxiliary_outputs = None, None, None
        if labels is not None:
            loss, loss_dict, auxiliary_outputs = self.loss_function(
                logits,
                labels,
                self.device,
                pred_points,
                self.config,
                outputs_class,
                outputs_coord,
            )
        if not return_dict:
            if auxiliary_outputs is not None:
                output = (logits, pred_points) + auxiliary_outputs + outputs
            else:
                output = (logits, pred_points) + outputs
            tuple_outputs = ((loss, loss_dict) + output) if loss is not None else output

            return tuple_outputs

        dict_outputs = DeformableDetrKeypointDetectionOutput(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            pred_points=pred_points,
            auxiliary_outputs=auxiliary_outputs,
            last_hidden_state=outputs.last_hidden_state,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            intermediate_hidden_states=outputs.intermediate_hidden_states,
            intermediate_reference_points=outputs.intermediate_reference_points,
            init_reference_points=outputs.init_reference_points,
            enc_outputs_class=outputs.enc_outputs_class,
            enc_outputs_coord_logits=outputs.enc_outputs_coord_logits,
        )

        return dict_outputs


def keypoint_to_coco(targets):
    if not isinstance(targets, list):
        targets = [targets]

    coco_targets = []
    for target in targets:
        annotations_for_target = []
        for i, (label, point) in enumerate(
            zip(target["labels"], target["points"], strict=False)
        ):
            if isinstance(point, torch.Tensor):
                point = point.tolist()
            if isinstance(label, torch.Tensor):
                label = label.item()

            annotations_for_target.append(
                {
                    "id": i,
                    "image_id": i,
                    "category_id": label,
                    "keypoints": point,  # [x, y]
                }
            )

        coco_target = {"image_id": 0, "annotations": annotations_for_target}

        # Preserve orig_size if available for coordinate scaling during preprocessing
        if "orig_size" in target:
            coco_target["orig_size"] = target["orig_size"]

        coco_targets.append(coco_target)

    return coco_targets


class KeypointDetrWrapper(nn.Module):
    """Wrapper for DeformableDetrForKeypointDetection that handles
    preprocessing and postprocessing transparently.

    This class translates between DeepForest's KeypointDataset format
    and the transformers keypoint model format.
    """

    def __init__(self, config, name, revision, **hf_args):
        """Initialize a DeformableDetrForKeypointDetection model.

        Args:
            config: DeepForest config object
            name: HuggingFace model name or path
            revision: Model revision/branch
            **hf_args: Additional arguments for from_pretrained
        """
        super().__init__()
        self.config = config

        # Import here to avoid circular imports
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)

            # Create keypoint model config with keypoint-specific parameters
            from transformers import DeformableDetrConfig

            # Load base config from pretrained model
            model_config = DeformableDetrConfig.from_pretrained(
                name, revision=revision, **hf_args
            )

            # Add keypoint-specific loss parameters if they exist in config
            if hasattr(self.config, "point_cost"):
                model_config.point_cost = self.config.point_cost
            if hasattr(self.config, "point_loss_coefficient"):
                model_config.point_loss_coefficient = self.config.point_loss_coefficient
            if hasattr(self.config, "point_loss_type"):
                model_config.point_loss_type = self.config.point_loss_type

            # Create model from config
            self.net = DeformableDetrForKeypointDetection(model_config)

            # Load pretrained weights if available (backbone only, not prediction heads)
            if name is not None:
                from transformers import DeformableDetrModel

                pretrained_base = DeformableDetrModel.from_pretrained(
                    name, revision=revision, **hf_args
                )
                # Copy only the base model weights (encoder/decoder, not classification/bbox heads)
                self.net.model.load_state_dict(pretrained_base.state_dict(), strict=False)

            # Create processor
            self.processor = DeformableDetrKeypointImageProcessor()

            # Update label mappings
            if hasattr(self.config, "label_dict"):
                self.label_dict = self.config.label_dict
            else:
                self.label_dict = {"Tree": 0}
            self.num_classes = model_config.num_labels

    def _prepare_targets(self, targets):
        """Translate KeypointDataset targets to COCO keypoint format.

        Args:
            targets: List of dicts with "points" (N, 2) and "labels" (N,)

        Returns:
            List of dicts in COCO annotation format for keypoints
        """
        return keypoint_to_coco(targets)

    def forward(self, images, targets=None, prepare_targets=True):
        """Forward pass for keypoint detection.

        Args:
            images: Input images (list of tensors or batch tensor)
            targets: Optional targets for training
            prepare_targets: Whether to convert targets to COCO format

        Returns:
            If training: loss dictionary
            If inference: list of dicts with "keypoints", "scores", "labels"
        """
        if targets and prepare_targets:
            targets = self._prepare_targets(targets)

        encoded_inputs = self.processor.preprocess(
            images=images,
            annotations=targets,
            return_tensors="pt",
            do_rescale=False,  # Dataset already normalized [0,255]→[0,1]
            # Processor still does: resize, normalize (ImageNet), pad
        )

        # Move tensors to model device
        for k, v in encoded_inputs.items():
            if isinstance(v, torch.Tensor):
                encoded_inputs[k] = v.to(self.net.device)

        if isinstance(encoded_inputs.get("labels"), list):
            encoded_inputs["labels"] = [
                {
                    key: val.to(self.net.device) if isinstance(val, torch.Tensor) else val
                    for key, val in target.items()
                }
                for target in encoded_inputs["labels"]
            ]

        preds = self.net(**encoded_inputs)

        if targets is None or not self.training:
            # Inference mode: post-process and return predictions
            # Use original image sizes from targets to scale predictions back to original coordinate space
            target_sizes = [t["orig_size"].cpu().tolist() for t in targets]

            results = self.processor.post_process_keypoint_detection(
                preds,
                threshold=self.config.score_thresh,
                target_sizes=target_sizes,
            )
            return results
        else:
            # Training mode: return loss dict and predictions for logging
            return {
                "loss_dict": preds.loss_dict,
                "pred_points": preds.pred_points,
                "targets": encoded_inputs.get("labels"),
            }


class Model:
    """Model factory for keypoint detection following DeepForest interface.

    This class provides a simple interface to create keypoint detection
    models compatible with the DeepForest training pipeline.
    """

    def __init__(self, config, **kwargs):
        """Initialize model factory.

        Args:
            config: DeepForest configuration object
        """
        self.config = config

    def create_model(
        self,
        pretrained: str | None = "SenseTime/deformable-detr",
        *,
        revision: str | None = "main",
        map_location: str | torch.device | None = None,
        **hf_args,
    ) -> KeypointDetrWrapper:
        """Create a keypoint detection model from pretrained weights.

        The model starts from a pretrained Deformable DETR backbone (encoder/decoder)
        but initializes new prediction heads for the configured number of classes.

        Args:
            pretrained: HuggingFace model name or path. If None, random initialization.
            revision: Model revision/branch
            map_location: Device to load model onto
            **hf_args: Additional arguments for from_pretrained

        Returns:
            KeypointDetrWrapper model ready for training or inference
        """
        # Set label mapping if provided
        if pretrained is None:
            hf_args.setdefault("id2label", self.config.numeric_to_label_dict)

        model = KeypointDetrWrapper(
            self.config,
            name=pretrained,
            revision=revision,
            num_labels=self.config.num_classes,
            **hf_args,
        )

        if map_location is not None:
            model = model.to(map_location)

        return model


__all__ = [
    "DeformableDetrKeypointConfig",
    "DeformableDetrKeypointDetectionOutput",
    "DeformableDetrForKeypointDetection",
    "DeformableDetrKeypointMatcher",
    "DeformableDetrKeypointLoss",
    "DeformableDetrKeypointImageProcessor",
    "KeypointDetrWrapper",
    "Model",
]
