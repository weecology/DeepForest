"""Unit tests for main.py keypoint detection integration."""
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image
import numpy as np

from deepforest import main


def test_main_load_keypoint_config():
    """Test that main.py can load keypoint configuration."""
    model = main.deepforest(config="keypoint")

    assert model.config.task == "keypoint"
    assert model.config.architecture == "DeformableDetr"
    assert hasattr(model, 'model')


def test_main_create_keypoint_model():
    """Test that main.py creates keypoint model correctly."""
    model = main.deepforest(config="keypoint")

    assert model.model is not None
    assert hasattr(model.model, 'net')
    assert hasattr(model.model, 'processor')


def test_main_load_keypoint_dataset():
    """Test that main.py can load keypoint dataset."""
    model = main.deepforest(config="keypoint")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create dummy image
        img = Image.fromarray(np.random.randint(0, 255, (400, 300, 3), dtype=np.uint8))
        img_path = tmpdir / "test.jpg"
        img.save(img_path)

        # Create keypoint CSV
        csv_path = tmpdir / "keypoints.csv"
        df = pd.DataFrame({
            "image_path": ["test.jpg", "test.jpg", "test.jpg"],
            "x": [100, 200, 150],
            "y": [150, 250, 200],
            "label": ["Tree", "Tree", "Tree"]
        })
        df.to_csv(csv_path, index=False)

        # Load dataset
        dataloader = model.load_dataset(
            csv_file=str(csv_path),
            root_dir=str(tmpdir),
            batch_size=1,
            shuffle=False
        )

        assert len(dataloader.dataset) == 1

        # Get one sample
        images, targets, paths = dataloader.dataset[0]
        assert images.shape[0] == 3  # channels
        assert "points" in targets
        assert "labels" in targets
        assert targets["points"].shape == (3, 2)  # 3 keypoints, (x,y)
        assert targets["labels"].shape == (3,)


def test_main_invalid_task_raises():
    """Test that invalid task type raises ValueError."""
    from deepforest.conf.schema import Config

    config = Config()
    config.task = "invalid_task"

    with pytest.raises(ValueError, match="Invalid task type"):
        model = main.deepforest()
        model.config = config
        model.create_model()


def test_main_box_task_still_works():
    """Test that box detection still works after keypoint changes."""
    model = main.deepforest()  # Default is box task

    assert model.config.task == "box"
    assert hasattr(model, 'model')


def test_main_keypoint_with_box_csv():
    """Test that KeypointDataset auto-converts box CSV to keypoints."""
    model = main.deepforest(config="keypoint")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create dummy image
        img = Image.fromarray(np.random.randint(0, 255, (400, 300, 3), dtype=np.uint8))
        img_path = tmpdir / "test.jpg"
        img.save(img_path)

        # Create box CSV (should auto-convert to keypoints)
        csv_path = tmpdir / "boxes.csv"
        df = pd.DataFrame({
            "image_path": ["test.jpg"],
            "xmin": [50],
            "ymin": [100],
            "xmax": [150],
            "ymax": [200],
            "label": ["Tree"]
        })
        df.to_csv(csv_path, index=False)

        # Load dataset - should convert boxes to keypoints (center)
        dataloader = model.load_dataset(
            csv_file=str(csv_path),
            root_dir=str(tmpdir),
            batch_size=1,
            shuffle=False
        )

        images, targets, paths = dataloader.dataset[0]

        # Check that keypoint is at center of box
        assert targets["points"].shape == (1, 2)
        expected_x = (50 + 150) / 2
        expected_y = (100 + 200) / 2
        assert targets["points"][0, 0].item() == pytest.approx(expected_x)
        assert targets["points"][0, 1].item() == pytest.approx(expected_y)
