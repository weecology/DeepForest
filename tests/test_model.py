import pytest
import torch
from deepforest import model
from deepforest import get_data
import pandas as pd
import os
from torchvision import transforms
import pytorch_lightning as pl
import numpy as np
from deepforest.predict import _predict_crop_model_

# The model object is achitecture agnostic container.
def test_model_no_args(config):
    with pytest.raises(ValueError):
        model.Model(config)


# The model object is achitecture agnostic container.
def test_model_no_args(config):
    with pytest.raises(ValueError):
        model.Model(config)


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

def test_crop_model(
        crop_model):  # Use pytest tempdir fixture to create a temporary directory
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

def test_predict_crop_model_labels(crop_model):
    # Set up the label mappings manually.
    crop_model.label_dict = {"Bird": 0, "Mammal": 1}
    crop_model.numeric_to_label_dict = {0: "Bird", 1: "Mammal"}

    # Create a dummy results DataFrame with the required columns.
    results = pd.DataFrame({
        "xmin": [0, 0],
        "xmax": [1, 1],
        "ymin": [0, 0],
        "ymax": [1, 1],
        "image_path": ["dummy1.png", "dummy2.png"]
    })

    # Define a dummy raster path.
    raster_path = "dummy/path/dummy.tif"

    # Patch predict_dataloader to bypass actual dataset creation.
    crop_model.predict_dataloader = lambda ds: None

    # Define a dummy trainer that returns fixed prediction outputs.
    class DummyTrainer:
        def predict(self, model, dataloader):
            # Simulate predictions for 2 samples with 2 classes:
            # Sample 1: [0.2, 0.8] -> argmax gives index 1 ("Mammal")
            # Sample 2: [0.7, 0.3] -> argmax gives index 0 ("Bird")
            return [np.array([[0.2, 0.8], [0.7, 0.3]])]

    dummy_trainer = DummyTrainer()

    # Call the prediction function.
    updated_results = _predict_crop_model_(crop_model,
                                            dummy_trainer,
                                            results,
                                            raster_path,
                                            transform=None,
                                            augment=False)

    # Expected labels based on dummy predictions.
    expected_labels = ["Mammal", "Bird"]
    expected_scores = np.array([0.8, 0.7])

    # Check that the labels and scores in the results are as expected.
    assert updated_results["cropmodel_label"].tolist() == expected_labels
    np.testing.assert_allclose(updated_results["cropmodel_score"].values, expected_scores)
