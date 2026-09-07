import os

import numpy as np
from PIL import Image

from deepforest import get_data, main
from deepforest.datasets import prediction


def test_default_batch_sizes():
    """Verify default train and predict batch sizes in config schema."""
    m = main.deepforest()
    assert m.config.train_batch_size == 2
    assert m.config.predict_batch_size == 8
    assert m.config.batch_size == 1


def test_train_dataloader_uses_train_batch_size():
    """Verify train_dataloader uses train_batch_size."""
    csv_file = get_data("example.csv")
    root_dir = os.path.dirname(csv_file)
    m = main.deepforest(config_args={"train_batch_size": 4})
    m.config.train.csv_file = csv_file
    m.config.train.root_dir = root_dir
    loader = m.train_dataloader()
    assert loader.batch_size == 4


def test_train_dataloader_default_batch_size():
    """Verify train_dataloader defaults to batch size 2."""
    csv_file = get_data("example.csv")
    root_dir = os.path.dirname(csv_file)
    m = main.deepforest()
    m.config.train.csv_file = csv_file
    m.config.train.root_dir = root_dir
    loader = m.train_dataloader()
    assert loader.batch_size == 2


def test_val_dataloader_uses_predict_batch_size():
    """Verify val_dataloader uses predict_batch_size."""
    csv_file = get_data("example.csv")
    root_dir = os.path.dirname(csv_file)
    m = main.deepforest(config_args={"predict_batch_size": 6})
    m.config.validation.csv_file = csv_file
    m.config.validation.root_dir = root_dir
    loader = m.val_dataloader()
    assert loader.batch_size == 6


def test_val_dataloader_default_batch_size():
    """Verify val_dataloader defaults to batch size 8."""
    csv_file = get_data("example.csv")
    root_dir = os.path.dirname(csv_file)
    m = main.deepforest()
    m.config.validation.csv_file = csv_file
    m.config.validation.root_dir = root_dir
    loader = m.val_dataloader()
    assert loader.batch_size == 8


def test_predict_dataloader_uses_predict_batch_size():
    """Verify predict_dataloader uses predict_batch_size."""
    path = get_data("OSBS_029.png")
    tile = np.array(Image.open(path))
    ds = prediction.SingleImage(image=tile, path=path, patch_overlap=0.1, patch_size=100)

    m = main.deepforest(config_args={"predict_batch_size": 16})
    loader = m.predict_dataloader(ds)
    assert loader.batch_size == 16


def test_predict_dataloader_default_batch_size():
    """Verify predict_dataloader defaults to batch size 8."""
    path = get_data("OSBS_029.png")
    tile = np.array(Image.open(path))
    ds = prediction.SingleImage(image=tile, path=path, patch_overlap=0.1, patch_size=100)

    m = main.deepforest()
    loader = m.predict_dataloader(ds)
    assert loader.batch_size == 8


def test_predict_dataloader_explicit_arg_override():
    """Verify explicit batch_size argument to predict_dataloader takes precedence."""
    path = get_data("OSBS_029.png")
    tile = np.array(Image.open(path))
    ds = prediction.SingleImage(image=tile, path=path, patch_overlap=0.1, patch_size=100)

    m = main.deepforest(config_args={"predict_batch_size": 16})
    loader = m.predict_dataloader(ds, batch_size=4)
    assert loader.batch_size == 4


def test_independent_train_and_predict_batch_sizes():
    """Verify train_batch_size and predict_batch_size can be set independently."""
    csv_file = get_data("example.csv")
    root_dir = os.path.dirname(csv_file)
    path = get_data("OSBS_029.png")
    tile = np.array(Image.open(path))
    ds = prediction.SingleImage(image=tile, path=path, patch_overlap=0.1, patch_size=100)

    m = main.deepforest(config_args={"train_batch_size": 3, "predict_batch_size": 7})
    m.config.train.csv_file = csv_file
    m.config.train.root_dir = root_dir
    m.config.validation.csv_file = csv_file
    m.config.validation.root_dir = root_dir

    train_loader = m.train_dataloader()
    val_loader = m.val_dataloader()
    predict_loader = m.predict_dataloader(ds)

    assert train_loader.batch_size == 3
    assert val_loader.batch_size == 7
    assert predict_loader.batch_size == 7


def test_batch_size_one_edge_case():
    """Verify batch_size=1 works for both training and prediction."""
    csv_file = get_data("example.csv")
    root_dir = os.path.dirname(csv_file)
    path = get_data("OSBS_029.png")
    tile = np.array(Image.open(path))
    ds = prediction.SingleImage(image=tile, path=path, patch_overlap=0.1, patch_size=100)

    m = main.deepforest(config_args={"train_batch_size": 1, "predict_batch_size": 1})
    m.config.train.csv_file = csv_file
    m.config.train.root_dir = root_dir

    train_loader = m.train_dataloader()
    predict_loader = m.predict_dataloader(ds)

    assert train_loader.batch_size == 1
    assert predict_loader.batch_size == 1


def test_legacy_batch_size_config_args():
    """Verify passing legacy batch_size in config_args sets both train and predict batch sizes."""
    csv_file = get_data("example.csv")
    root_dir = os.path.dirname(csv_file)
    path = get_data("OSBS_029.png")
    tile = np.array(Image.open(path))
    ds = prediction.SingleImage(image=tile, path=path, patch_overlap=0.1, patch_size=100)

    m = main.deepforest(config_args={"batch_size": 5})
    m.config.train.csv_file = csv_file
    m.config.train.root_dir = root_dir
    m.config.validation.csv_file = csv_file
    m.config.validation.root_dir = root_dir

    assert m.config.train_batch_size == 5
    assert m.config.predict_batch_size == 5
    assert m.train_dataloader().batch_size == 5
    assert m.val_dataloader().batch_size == 5
    assert m.predict_dataloader(ds).batch_size == 5


def test_legacy_batch_size_direct_assignment():
    """Verify directly mutating m.config.batch_size falls back in dataloaders."""
    csv_file = get_data("example.csv")
    root_dir = os.path.dirname(csv_file)
    path = get_data("OSBS_029.png")
    tile = np.array(Image.open(path))
    ds = prediction.SingleImage(image=tile, path=path, patch_overlap=0.1, patch_size=100)

    m = main.deepforest()
    m.config.train.csv_file = csv_file
    m.config.train.root_dir = root_dir
    m.config.validation.csv_file = csv_file
    m.config.validation.root_dir = root_dir

    m.config.batch_size = 6
    assert m.train_dataloader().batch_size == 6
    assert m.val_dataloader().batch_size == 6
    assert m.predict_dataloader(ds).batch_size == 6
