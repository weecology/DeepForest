import pytest
import torch
from deepforest import model
from deepforest import get_data
import pandas as pd
import rasterio
import cv2
import os
import numpy as np

# The model object is achitecture agnostic container.
def test_model_no_args(config):
    with pytest.raises(ValueError):
        model.Model(config)

# The model object is achitecture agnostic container.
def test_model_no_args(config):
    with pytest.raises(ValueError):
        model.Model(config)

def test_crop_model(tmpdir):  # Use pytest tempdir fixture to create a temporary directory
    df = pd.read_csv(get_data("testfile_multi.csv"))
    
    # Create a dummy instance of CropModel
    crop_model = model.CropModel(num_classes=2)
    boxes = df[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
    image_path = os.path.join(os.path.dirname(get_data("SOAP_061.png")),df["image_path"].iloc[0])
    crop_model.write_crops(boxes=boxes,labels=df.label.values,image_path=image_path, savedir=tmpdir)

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