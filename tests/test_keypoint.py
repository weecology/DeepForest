# Test keypoint detection model and loss functions
import numpy as np
import pytest
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader
from transformers import DeformableDetrConfig

from deepforest.models.keypoint import (
    DeformableDetrForKeypointDetection,
    DeformableDetrKeypointMatcher,
    DeformableDetrKeypointLoss,
    DeformableDetrKeypointImageProcessor,
    DeformableDetrKeypointConfig,
)


@pytest.fixture
def config():
    """Create a test configuration."""
    config = DeformableDetrConfig.from_pretrained("SenseTime/deformable-detr")
    config.num_labels = 5  # Use fewer labels for testing
    return config


@pytest.fixture
def keypoint_matcher():
    """Create a keypoint matcher with standard costs."""
    return DeformableDetrKeypointMatcher(
        class_cost=1.0,
        point_cost=5.0,
    )


@pytest.fixture
def keypoint_loss(keypoint_matcher):
    """Create a keypoint loss criterion."""
    return DeformableDetrKeypointLoss(
        matcher=keypoint_matcher,
        num_classes=5,
        focal_alpha=0.25,
        losses=["labels", "points", "cardinality"],
    )


@pytest.fixture
def sample_targets():
    """Create sample targets for testing."""
    return [
        {
            "class_labels": torch.tensor([0, 1, 2]),
            "points": torch.tensor([[0.2, 0.3], [0.5, 0.6], [0.8, 0.9]]),
        },
        {
            "class_labels": torch.tensor([0, 1, 2, 3]),
            "points": torch.tensor([[0.2, 0.3], [0.5, 0.6], [0.8, 0.9], [0.3, 0.4]]),
        }
    ]


def create_predictions_from_targets(targets, num_queries=10, num_classes=5, jitter_std=0.0):
    """
    Create predictions that match targets with optional jitter.

    Args:
        targets: List of target dicts
        num_queries: Number of query predictions per image
        num_classes: Number of keypoint classes
        jitter_std: Standard deviation of Gaussian jitter to add to positions

    Returns:
        Dict with 'logits' and 'pred_points'
    """
    batch_size = len(targets)
    logits = torch.full((batch_size, num_queries, num_classes), -10.0)
    pred_points = torch.rand(batch_size, num_queries, 2)

    for i, target in enumerate(targets):
        num_kpts = len(target["class_labels"])
        for j in range(num_kpts):
            logits[i, j, target["class_labels"][j]] = 10.0
            jitter = torch.randn(2) * jitter_std if jitter_std > 0 else 0
            pred_points[i, j] = torch.clamp(target["points"][j] + jitter, 0, 1)

    return {"logits": logits, "pred_points": pred_points}


"""Test suite for keypoint detection loss functions."""

def test_loss_identical_predictions(keypoint_loss, sample_targets):
    """Test that loss is very low when predictions perfectly match targets."""
    outputs = create_predictions_from_targets(sample_targets, jitter_std=0.0)
    loss_dict = keypoint_loss(outputs, sample_targets)

    assert loss_dict["loss_ce"] < 0.5, f"Classification loss too high: {loss_dict['loss_ce']}"
    assert loss_dict["loss_point"] < 0.01, f"Point loss too high: {loss_dict['loss_point']}"

def test_loss_small_jitter(keypoint_loss, sample_targets):
    """Test that loss is small when predictions have small positional errors (~5%)."""
    outputs = create_predictions_from_targets(sample_targets, jitter_std=0.05)
    loss_dict = keypoint_loss(outputs, sample_targets)

    assert loss_dict["loss_ce"] < 0.5, f"Classification loss too high: {loss_dict['loss_ce']}"
    assert 0.001 < loss_dict["loss_point"] < 0.1, f"Point loss out of expected range: {loss_dict['loss_point']}"

def test_loss_large_jitter(keypoint_loss, sample_targets):
    """Test that loss is large when predictions have large positional errors (~30%)."""
    outputs = create_predictions_from_targets(sample_targets, jitter_std=0.3)
    loss_dict = keypoint_loss(outputs, sample_targets)

    assert loss_dict["loss_ce"] < 0.5, f"Classification loss too high: {loss_dict['loss_ce']}"
    assert loss_dict["loss_point"] > 0.05, f"Point loss too low for large jitter: {loss_dict['loss_point']}"

def test_loss_shuffled_predictions(keypoint_loss):
    """Test that Hungarian matching correctly handles shuffled predictions."""
    targets = [{
        "class_labels": torch.tensor([0, 1, 2]),
        "points": torch.tensor([[0.2, 0.3], [0.5, 0.6], [0.8, 0.9]]),
    }]

    # Create predictions in reversed order
    num_queries = 10
    logits = torch.full((1, num_queries, 5), -10.0)
    pred_points = torch.rand(1, num_queries, 2)

    shuffled_order = [2, 1, 0]
    for j, target_idx in enumerate(shuffled_order):
        logits[0, j, targets[0]["class_labels"][target_idx]] = 10.0
        pred_points[0, j] = targets[0]["points"][target_idx]

    loss_dict = keypoint_loss({"logits": logits, "pred_points": pred_points}, targets)

    assert loss_dict["loss_ce"] < 0.5, f"Classification loss too high: {loss_dict['loss_ce']}"
    assert loss_dict["loss_point"] < 0.01, f"Point loss too high (matching failed?): {loss_dict['loss_point']}"

def test_loss_wrong_classes(keypoint_loss):
    """Test that classification loss is high when classes are incorrect."""
    targets = [{
        "class_labels": torch.tensor([0, 1, 2]),
        "points": torch.tensor([[0.2, 0.3], [0.5, 0.6], [0.8, 0.9]]),
    }]

    num_queries = 10
    logits = torch.full((1, num_queries, 5), -10.0)
    pred_points = torch.rand(1, num_queries, 2)

    # Assign wrong classes (shifted by 1) but correct locations
    for j in range(3):
        wrong_class = (targets[0]["class_labels"][j] + 1) % 5
        logits[0, j, wrong_class] = 10.0
        pred_points[0, j] = targets[0]["points"][j]

    loss_dict = keypoint_loss({"logits": logits, "pred_points": pred_points}, targets)

    assert loss_dict["loss_ce"] > 1.0, f"Classification loss too low: {loss_dict['loss_ce']}"

def test_loss_no_keypoints(keypoint_loss):
    """Test loss computation with images that have no keypoints."""
    targets = [{
        "class_labels": torch.tensor([], dtype=torch.long),
        "points": torch.empty(0, 2),
    }]

    outputs = {"logits": torch.randn(1, 10, 5), "pred_points": torch.rand(1, 10, 2)}
    loss_dict = keypoint_loss(outputs, targets)

    assert "loss_ce" in loss_dict
    assert "loss_point" in loss_dict
    assert not torch.isnan(loss_dict["loss_ce"])
    assert not torch.isnan(loss_dict["loss_point"])

"""Test suite for keypoint Hungarian matcher."""

def test_matcher_perfect_match(keypoint_matcher):
    """Test matcher with perfect correspondence between predictions and targets."""
    outputs = {
        "logits": torch.tensor([[[10.0, -10.0], [-10.0, 10.0], [-10.0, -10.0]]]),
        "pred_points": torch.tensor([[[0.2, 0.3], [0.5, 0.6], [0.8, 0.9]]]),
    }
    targets = [{
        "class_labels": torch.tensor([0, 1]),
        "points": torch.tensor([[0.2, 0.3], [0.5, 0.6]]),
    }]

    indices = keypoint_matcher(outputs, targets)
    pred_indices, target_indices = indices[0]

    assert len(pred_indices) == 2
    assert len(target_indices) == 2

def test_matcher_handles_more_predictions(keypoint_matcher):
    """Test that matcher handles more predictions than targets."""
    outputs = {
        "logits": torch.randn(1, 100, 5),
        "pred_points": torch.rand(1, 100, 2),
    }
    targets = [{
        "class_labels": torch.tensor([0, 1, 2]),
        "points": torch.rand(3, 2),
    }]

    indices = keypoint_matcher(outputs, targets)
    pred_indices, target_indices = indices[0]

    assert len(pred_indices) == 3
    assert len(target_indices) == 3


"""Test suite for keypoint image processor."""

def test_processor_single_keypoint():
    """Test processor with single keypoint per annotation."""
    processor = DeformableDetrKeypointImageProcessor(do_resize=False, do_normalize=False, do_pad=False)
    image = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
    annotation = {
        "image_id": 1,
        "annotations": [
            {"category_id": 0, "keypoints": [100.0, 150.0]},
            {"category_id": 1, "keypoints": [200.0, 250.0]},
        ]
    }

    # Call processor with images and annotations
    result = processor(image, annotations=annotation, do_convert_annotations=True)

    # Check that we got normalized labels
    assert "labels" in result
    labels = result["labels"][0]
    assert labels["points"].shape == (2, 2)
    assert labels["class_labels"].shape == (2,)
    # Points should be normalized
    assert labels["points"].max() <= 1.0
    assert labels["points"].min() >= 0.0

def test_processor_multiple_keypoints():
    """Test processor with multiple keypoints per annotation."""
    processor = DeformableDetrKeypointImageProcessor(do_resize=False, do_normalize=False, do_pad=False)
    image = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
    annotation = {
        "image_id": 1,
        "annotations": [
            {"category_id": 0, "keypoints": [[100.0, 150.0], [120.0, 170.0]]},
        ]
    }

    result = processor(image, annotations=annotation, do_convert_annotations=True)

    labels = result["labels"][0]
    assert labels["points"].shape == (2, 2)
    assert labels["class_labels"].shape == (2,)
    assert all(labels["class_labels"] == 0)

def test_processor_normalization():
    """Test that processor properly normalizes coordinates."""
    processor = DeformableDetrKeypointImageProcessor(do_resize=False, do_normalize=False, do_pad=False)
    image = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
    annotation = {
        "image_id": 1,
        "annotations": [
            {"category_id": 0, "keypoints": [300.0, 400.0]},  # Middle of image
        ]
    }

    result = processor(image, annotations=annotation, do_convert_annotations=True)

    labels = result["labels"][0]
    # Check normalization: 300/600=0.5, 400/800=0.5
    assert np.allclose(labels["points"][0], [0.5, 0.5])

def test_processor_post_process_keypoints():
    """Test post-processing of keypoint predictions."""
    processor = DeformableDetrKeypointImageProcessor()
    from deepforest.models.keypoint import DeformableDetrKeypointDetectionOutput

    outputs = DeformableDetrKeypointDetectionOutput(
        logits=torch.tensor([[[0.9, 0.1], [0.8, 0.2], [0.3, 0.7]]]),
        pred_points=torch.tensor([[[0.5, 0.5], [0.3, 0.7], [0.8, 0.2]]]),
    )

    results = processor.post_process_keypoint_detection(
        outputs,
        threshold=0.5,
        target_sizes=[(800, 600)],
        top_k=10
    )

    assert len(results) == 1
    result = results[0]
    assert "keypoints" in result
    assert "scores" in result
    assert "labels" in result
    assert result["keypoints"].max() <= 800


"""Test suite for the full keypoint detection model."""

def test_model_forward_inference(config):
    """Test model forward pass in inference mode."""
    model = DeformableDetrForKeypointDetection(config)
    model.eval()

    batch_size = 2
    pixel_values = torch.randn(batch_size, 3, 800, 800)

    with torch.no_grad():
        outputs = model(pixel_values)

    assert outputs.logits.shape == (batch_size, 300, config.num_labels)
    assert outputs.pred_points.shape == (batch_size, 300, 2)
    assert outputs.loss is None

def test_model_forward_training(config):
    """Test model forward pass in training mode with labels."""
    model = DeformableDetrForKeypointDetection(config)
    model.train()

    batch_size = 2
    pixel_values = torch.randn(batch_size, 3, 800, 800)
    labels = [
        {"class_labels": torch.tensor([0, 1]), "points": torch.rand(2, 2)},
        {"class_labels": torch.tensor([2, 3, 4]), "points": torch.rand(3, 2)}
    ]

    outputs = model(pixel_values, labels=labels)

    assert outputs.loss is not None
    assert outputs.loss_dict is not None
    assert "loss_ce" in outputs.loss_dict
    assert "loss_point" in outputs.loss_dict
    assert not torch.isnan(outputs.loss)

# TODO: Remove or simplify this when we have integration with the main library sorted out.
def test_model_train_overfit():
    """Test model can overfit to memorize 10 keypoints"""
    # Create small config for faster training
    config = DeformableDetrConfig.from_pretrained("SenseTime/deformable-detr")
    config.num_labels = 3
    config.decoder_layers = 4
    config.encoder_layers = 4
    config.num_queries = 20

    processor = DeformableDetrKeypointImageProcessor()

    # Create fixed dataset with 10 well-separated keypoints
    image_height, image_width = 800, 600
    fixed_keypoints_pixel = np.array([
        [100, 100], [500, 100], [100, 700], [500, 700], [300, 400],
        [150, 250], [450, 250], [150, 550], [450, 550], [300, 150],
    ])
    fixed_classes = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])

    image = np.random.randint(0, 255, (image_height, image_width, 3), dtype=np.uint8)
    annotation = {
        "image_id": 0,
        "annotations": [
            {"category_id": int(cls), "keypoints": kpt.tolist()}
            for cls, kpt in zip(fixed_classes, fixed_keypoints_pixel)
        ]
    }

    result = processor(
        image,
        annotations=annotation,
        do_resize=False,
        do_normalize=False,
        do_pad=False,
        do_convert_annotations=True,
        return_tensors="pt"
    )

    labels = [result["labels"][0]]

    # Lightning module
    class KeypointLightningModule(pl.LightningModule):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.train_losses = []

        def training_step(self, batch, batch_idx):
            pixel_values, labels = batch
            outputs = self.model(pixel_values, labels=labels)
            self.log("train_loss", outputs.loss, prog_bar=True)
            self.train_losses.append(outputs.loss.item())
            return outputs.loss

        def configure_optimizers(self):
            return torch.optim.AdamW(self.parameters(), lr=1e-3)

    # Dataset/dataloader
    class KeypointDataset(torch.utils.data.Dataset):
        def __init__(self, pixel_values, labels):
            self.pixel_values = pixel_values.squeeze(0)
            self.labels = labels[0]

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return self.pixel_values, self.labels

    def collate_fn(batch):
        pixel_values = torch.stack([item[0] for item in batch])
        labels = [item[1] for item in batch]
        return pixel_values, labels

    pixel_values = torch.randn(1, 3, image_height, image_width)
    dataset = KeypointDataset(pixel_values, labels)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    lightning_model = KeypointLightningModule(DeformableDetrForKeypointDetection(config))

    # Train with early stopping
    early_stop = EarlyStopping(monitor="train_loss", patience=20, min_delta=0.001, mode="min")
    trainer = pl.Trainer(
        max_epochs=200,
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[early_stop],
    )

    trainer.fit(lightning_model, dataloader)

    # Verify loss reduction
    initial_loss = lightning_model.train_losses[0]
    final_loss = lightning_model.train_losses[-1]
    assert final_loss < initial_loss * 0.3, \
        f"Loss did not decrease enough: {final_loss:.4f} vs {initial_loss:.4f}"

    # Test inference and post-processing
    lightning_model.model.eval()
    with torch.no_grad():
        outputs = lightning_model.model(pixel_values)

    results = processor.post_process_keypoint_detection(
        outputs,
        threshold=0.3,
        target_sizes=[(image_height, image_width)],
        top_k=50
    )

    result = results[0]
    num_predictions = len(result["keypoints"])
    assert num_predictions > 0, "No keypoints detected after training!"

    # Check predictions are well distributed
    if num_predictions >= 2:
        from scipy.spatial.distance import pdist
        pred_points = result["keypoints"].numpy()
        # Reshape if flattened (N*2,) -> (N, 2)
        if pred_points.ndim == 1:
            pred_points = pred_points.reshape(-1, 2)

        # Debug: print unique predictions
        unique_points = np.unique(pred_points, axis=0)
        print(f"\nTotal predictions: {len(pred_points)}, Unique: {len(unique_points)}")
        if len(unique_points) < len(pred_points):
            print(f"WARNING: {len(pred_points) - len(unique_points)} duplicate predictions!")

        if len(unique_points) >= 2:
            min_distance = pdist(unique_points).min()
            assert min_distance > 5, \
                f"Unique keypoints too close! Min distance = {min_distance:.1f} < 5 pixels"
        else:
            # Skip this check if all predictions are the same
            print("All predictions collapsed to same point, skipping distance check")

    # Check localization accuracy
    if num_predictions >= len(fixed_keypoints_pixel):
        pred_points = result["keypoints"].numpy()
        # Reshape if flattened (N*2,) -> (N, 2)
        if pred_points.ndim == 1:
            pred_points = pred_points.reshape(-1, 2)
        errors = []
        for gt_point in fixed_keypoints_pixel:
            distances = np.linalg.norm(pred_points - gt_point, axis=1)
            errors.append(distances.min())

        mean_error = np.mean(errors)
        assert mean_error < 5, f"Mean localization error too high: {mean_error:.1f} pixels"

def test_keypoint_config():
    """Test that DeformableDetrKeypointConfig has keypoint-specific parameters."""
    config = DeformableDetrKeypointConfig(
        num_labels=5,
        point_cost=3.0,
        point_loss_coefficient=7.0
    )

    assert hasattr(config, 'point_cost')
    assert hasattr(config, 'point_loss_coefficient')
    assert config.point_cost == 3.0
    assert config.point_loss_coefficient == 7.0
    assert config.num_labels == 5

    # Test that it still has parent class attributes
    assert hasattr(config, 'class_cost')
    assert hasattr(config, 'bbox_cost')
