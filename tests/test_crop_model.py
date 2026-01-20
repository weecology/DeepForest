import os

import pandas as pd
import pytest
import numpy as np
import torch
from torchvision import transforms
import numpy as np

from deepforest import get_data
from deepforest import model
from deepforest.model import CropModel

# The model object is architecture agnostic container.
def test_model_no_args(config):
    with pytest.raises(ValueError):
        model.BaseModel.create_model(config)


@pytest.fixture()
def crop_model():
    crop_model = model.CropModel()
    crop_model.create_model(num_classes=2)

    return crop_model


@pytest.fixture()
def crop_model_data(crop_model, tmp_path):
    df = pd.read_csv(get_data("testfile_multi.csv"))
    boxes = df[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
    root_dir = os.path.dirname(get_data("SOAP_061.png"))
    images = df.image_path.values

    crop_model.write_crops(boxes=boxes,
                           labels=df.label.values,
                           root_dir=root_dir,
                           images=images,
                           savedir=tmp_path)

    return None

def test_crop_model(crop_model):
    # Test forward pass
    x = torch.rand(4, 3, 224, 224)
    output = crop_model.forward(x)
    assert output.shape == (4, 2)

    # Test training step
    batch = (x, torch.tensor([0, 1, 0, 1]))
    loss = crop_model.training_step(batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor)

    # Test validation step
    val_batch = (x, torch.tensor([0, 1, 0, 1]))
    val_loss = crop_model.validation_step(val_batch, batch_idx=0)
    assert isinstance(val_loss, torch.Tensor)

def test_crop_model_train(crop_model, tmp_path, crop_model_data):
    # Create a trainer
    crop_model.create_trainer(fast_dev_run=True, default_root_dir=tmp_path)
    crop_model.load_from_disk(train_dir=tmp_path, val_dir=tmp_path)

    # Test training dataloader
    train_loader = crop_model.train_dataloader()
    assert isinstance(train_loader, torch.utils.data.DataLoader)

    # Test validation dataloader
    val_loader = crop_model.val_dataloader()
    assert isinstance(val_loader, torch.utils.data.DataLoader)

    crop_model.trainer.fit(crop_model)
    crop_model.trainer.validate(crop_model)


def test_crop_model_custom_transform(crop_model):
    # Create a dummy instance of CropModel

    def custom_transform(self, augment):
        data_transforms = []
        data_transforms.append(transforms.ToTensor())
        data_transforms.append(self.normalize)
        # Add transforms here
        data_transforms.append(transforms.Resize([300, 300]))
        if augment:
            data_transforms.append(transforms.RandomHorizontalFlip(0.5))
        return transforms.Compose(data_transforms)

    # Test custom transform
    x = torch.rand(4, 3, 300, 300)
    crop_model.get_transform = custom_transform
    output = crop_model.forward(x)
    assert output.shape == (4, 2)

def test_crop_model_configurable_resize():
    """Test that CropModel resize dimensions can be configured"""
    # Test with default resize dimensions
    default_model = model.CropModel()
    default_model.create_model(num_classes=2)
    transform = default_model.get_transform(augmentations=None)

    # Check that default resize is [224, 224]
    resize_transform = [t for t in transform.transforms if isinstance(t, transforms.Resize)]
    assert len(resize_transform) == 1
    assert resize_transform[0].size == [224, 224]

    # Test with custom resize dimensions
    custom_config = {"resize": [300, 300]}
    custom_model = model.CropModel(config_args=custom_config)
    custom_model.create_model(num_classes=2)
    custom_transform = custom_model.get_transform(augmentations=None)

    # Check that custom resize is applied
    custom_resize_transform = [t for t in custom_transform.transforms if isinstance(t, transforms.Resize)]
    assert len(custom_resize_transform) == 1
    assert custom_resize_transform[0].size == [300, 300]

    # Test forward pass with custom resize
    x = torch.rand(4, 3, 300, 300)
    output = custom_model.forward(x)
    assert output.shape == (4, 2)


def test_crop_model_load_checkpoint(tmp_path_factory, crop_model):
    """Test loading crop model from checkpoint with different numbers of classes"""
    for num_classes in [2, 5]:
        # Create data for example
        df = pd.read_csv(get_data("testfile_multi.csv"))
        boxes = df[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
        root_dir = os.path.dirname(get_data("SOAP_061.png"))
        images = df.image_path.values
        labels = np.random.randint(0, num_classes, size=len(df)).astype(str)
        out_tmp = tmp_path_factory.mktemp(f"num_classes_{num_classes}")
        crop_data_dir = str(out_tmp)
        crop_model.write_crops(boxes=boxes,
                            labels=labels,
                            root_dir=root_dir,
                            images=images,
                            savedir=crop_data_dir)

        # Create initial model and save checkpoint
        crop_model = model.CropModel()
        crop_model.create_trainer(fast_dev_run=False, limit_train_batches=1, limit_val_batches=1, max_epochs=1, default_root_dir=tmp_path_factory.mktemp("logs"))
        crop_model.load_from_disk(train_dir=crop_data_dir, val_dir=crop_data_dir)
        # Initialize classification head based on discovered labels
        crop_model.create_model(num_classes=len(crop_model.label_dict))
        crop_model.trainer.fit(crop_model)
        checkpoint_path = out_tmp.joinpath("epoch=0-step=0.ckpt")
        crop_model.trainer.save_checkpoint(checkpoint_path)

        # Load from checkpoint
        loaded_model = model.CropModel.load_from_checkpoint(checkpoint_path)

        # Test forward pass
        x = torch.rand(4, 3, 224, 224)
        output = loaded_model(x)

        # Check output shape matches number of classes
        assert output.shape == (4, num_classes)

        # Check label dictionary was loaded
        assert loaded_model.label_dict == crop_model.label_dict

        # Check model parameters were loaded
        for p1, p2 in zip(crop_model.parameters(), loaded_model.parameters()):
            assert torch.equal(p1, p2)

def test_expand_bbox_to_square_edge_cases(crop_model):
    """Test cases for the expand_bbox_to_square function."""
    # Test Case 1: Bounding box at the image edge (0,0)
    bbox = [0, 0, 20, 30]
    image_width, image_height = 100, 100
    # Expected to expand vertically to make a square while remaining within image bounds
    expected = [0.0, 0.0, 30.0, 30.0]
    result = crop_model.expand_bbox_to_square(bbox, image_width, image_height)
    assert result == expected

    # Test Case 2: Side length exceeds both image dimensions
    bbox = [10, 10, 180, 180]
    image_width, image_height = 100, 100
    # Expected to be clamped to the size of the image: full image bounding box
    expected = [0.0, 0.0, 100.0, 100.0]
    result = crop_model.expand_bbox_to_square(bbox, image_width, image_height)
    assert result == expected

    # Test Case 3: Basic case - bounding box well within image boundaries
    bbox = [40, 30, 60, 70]
    image_width, image_height = 100, 100
    # Expected to expand width to match height while maintaining the center point (50,50)
    expected = [30.0, 30.0, 70.0, 70.0]
    result = crop_model.expand_bbox_to_square(bbox, image_width, image_height)
    assert result == expected

def test_crop_model_val_dataset_confusion(tmp_path, crop_model, crop_model_data):
    crop_model.create_trainer(fast_dev_run=True, default_root_dir=tmp_path)
    crop_model.load_from_disk(train_dir=tmp_path, val_dir=tmp_path)
    crop_model.trainer.fit(crop_model)

    crop_model.create_trainer(fast_dev_run=False, default_root_dir=tmp_path)
    images, labels, predictions = crop_model.val_dataset_confusion(return_images=True)

    # There are 37 images in the testfile_multi.csv
    assert len(images) == 37

    # There was just one batch in the fast_dev_run
    assert len(labels) ==37
    assert len(predictions) == 37
