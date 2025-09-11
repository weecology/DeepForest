import os

import pandas as pd
import pytest
import torch
from torchvision import transforms

from deepforest import get_data
from deepforest import model


# The model object is architecture agnostic container.
def test_model_no_args(config):
    with pytest.raises(ValueError):
        model.BaseModel.create_model(config)


@pytest.fixture()
def crop_model():
    crop_model = model.CropModel(num_classes=2)

    return crop_model


@pytest.fixture()
def crop_model_data(crop_model, tmpdir):
    df = pd.read_csv(get_data("testfile_multi.csv"))
    boxes = df[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
    root_dir = os.path.dirname(get_data("SOAP_061.png"))
    images = df.image_path.values

    crop_model.write_crops(boxes=boxes,
                           labels=df.label.values,
                           root_dir=root_dir,
                           images=images,
                           savedir=tmpdir)

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


def test_crop_model_train(crop_model, tmpdir, crop_model_data):
    # Create a trainer
    crop_model.create_trainer(fast_dev_run=True)
    crop_model.load_from_disk(train_dir=tmpdir, val_dir=tmpdir)

    # Test training dataloader
    train_loader = crop_model.train_dataloader()
    assert isinstance(train_loader, torch.utils.data.DataLoader)

    # Test validation dataloader
    val_loader = crop_model.val_dataloader()
    assert isinstance(val_loader, torch.utils.data.DataLoader)

    crop_model.trainer.fit(crop_model)
    crop_model.trainer.validate(crop_model)


def test_crop_model_recreate_model(tmpdir, crop_model_data):
    crop_model = model.CropModel()
    crop_model.load_from_disk(train_dir=tmpdir, val_dir=tmpdir, recreate_model=True)
    assert crop_model.model is not None
    assert crop_model.model.fc.out_features == 2


def test_crop_model_custom_transform():
    # Create a dummy instance of CropModel
    crop_model = model.CropModel(num_classes=2)

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


def test_crop_model_load_checkpoint(tmpdir, crop_model_data):
    """Test loading crop model from checkpoint with different numbers of classes"""
    for num_classes in [2, 5]:
        # Create initial model and save checkpoint
        crop_model = model.CropModel(num_classes=num_classes)
        crop_model.create_trainer(fast_dev_run=True)
        crop_model.load_from_disk(train_dir=tmpdir, val_dir=tmpdir)

        crop_model.trainer.fit(crop_model)
        checkpoint_path = os.path.join(tmpdir, "epoch=0-step=0.ckpt")

        crop_model.trainer.save_checkpoint(checkpoint_path)

        # Load from checkpoint
        loaded_model = model.CropModel.load_from_checkpoint(checkpoint_path)

        # Test forward pass
        x = torch.rand(4, 3, 224, 224)
        output = loaded_model(x)

        # Check output shape matches number of classes
        assert output.shape == (4, num_classes)

        # Check model parameters were loaded
        for p1, p2 in zip(crop_model.parameters(), loaded_model.parameters()):
            assert torch.equal(p1, p2)


def test_crop_model_load_checkpoint_from_disk(tmpdir, crop_model_data):
    """Test loading crop model from checkpoint with different numbers of classes"""
    # Create initial model and save checkpoint
    crop_model = model.CropModel()
    crop_model.create_trainer(fast_dev_run=True)
    crop_model.load_from_disk(train_dir=tmpdir, val_dir=tmpdir, recreate_model=True)

    crop_model.trainer.fit(crop_model)
    checkpoint_path = os.path.join(tmpdir, "epoch=0-step=0.ckpt")

    crop_model.trainer.save_checkpoint(checkpoint_path)

    # Load from checkpoint
    loaded_model = model.CropModel.load_from_checkpoint(checkpoint_path)
    assert loaded_model.label_dict == crop_model.label_dict


def test_crop_model_maintain_label_dict(tmpdir, crop_model_data):
    """
    Test that the label dictionary is maintained when loading a checkpoint.
    """
    crop_model = model.CropModel(num_classes=2)
    crop_model.create_trainer(fast_dev_run=True)
    crop_model.load_from_disk(train_dir=tmpdir, val_dir=tmpdir)

    crop_model.trainer.fit(crop_model)
    checkpoint_path = os.path.join(tmpdir, "epoch=0-step=0.ckpt")
    crop_model.trainer.save_checkpoint(checkpoint_path)

    # Load from checkpoint
    loaded_model = model.CropModel.load_from_checkpoint(checkpoint_path)

    # Check that the label dictionary is maintained
    assert crop_model.label_dict == loaded_model.label_dict


def test_crop_model_init_no_num_classes():
    """
    Test that initializing CropModel() without num_classes
    (and not loading from checkpoint) will fail on forward pass.
    This confirms that the user either needs to provide num_classes
    or load from a checkpoint that contains it.
    """
    crop_model = model.CropModel()  # num_classes=None
    x = torch.rand(4, 3, 224, 224)

    # Expect the forward pass to fail because the model is uninitialized.
    with pytest.raises(AttributeError):
        _ = crop_model.forward(x)


def test_crop_model_load_checkpoint_with_explicit_num_classes(tmpdir, crop_model_data):
    """
    Test loading a checkpoint while explicitly providing the same num_classes.
    This verifies that passing num_classes during load_from_checkpoint
    remains valid (backward compatible).
    """
    num_classes = 3
    # Create initial model and save checkpoint
    crop_model = model.CropModel(num_classes=num_classes)
    crop_model.create_trainer(fast_dev_run=True)
    crop_model.load_from_disk(train_dir=tmpdir, val_dir=tmpdir)
    crop_model.label_dict = {0: "label1", 1: "label2", 2: "label3"}
    crop_model.trainer.fit(crop_model)
    checkpoint_path = os.path.join(tmpdir, "epoch=0-step=0.ckpt")
    crop_model.trainer.save_checkpoint(checkpoint_path)

    # Load from checkpoint, explicitly specifying num_classes
    loaded_model = model.CropModel.load_from_checkpoint(checkpoint_path, num_classes=num_classes)

    # Test forward pass
    x = torch.rand(4, 3, 224, 224)
    output = loaded_model(x)

    # Check output shape
    assert output.shape == (4, num_classes)

    # Check model parameters match
    for p1, p2 in zip(crop_model.parameters(), loaded_model.parameters()):
        assert torch.equal(p1, p2)


def test_expand_bbox_to_square_edge_cases():
    """Test cases for the expand_bbox_to_square function."""
    crop_model = model.CropModel(num_classes=2)

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


def test_crop_model_val_dataset_confusion(tmpdir, crop_model_data):
    crop_model = model.CropModel()
    crop_model.create_trainer(fast_dev_run=True)
    crop_model.load_from_disk(train_dir=tmpdir, val_dir=tmpdir, recreate_model=True)
    crop_model.trainer.fit(crop_model)
    images, labels, predictions = crop_model.val_dataset_confusion(return_images=True)

    # There are 37 images in the testfile_multi.csv
    assert len(images) == 37

    # There was just one batch in the fast_dev_run
    assert len(labels) ==37
    assert len(predictions) == 4
