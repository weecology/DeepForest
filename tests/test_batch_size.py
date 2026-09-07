import os

import numpy as np
from PIL import Image

from deepforest import get_data, main, utilities
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


def test_legacy_batch_size_train_override():
    """Verify legacy batch_size + explicit train_batch_size uses train override."""
    csv_file = get_data("example.csv")
    root_dir = os.path.dirname(csv_file)
    path = get_data("OSBS_029.png")
    tile = np.array(Image.open(path))
    ds = prediction.SingleImage(image=tile, path=path, patch_overlap=0.1, patch_size=100)

    m = main.deepforest(config_args={"batch_size": 4, "train_batch_size": 6})
    m.config.train.csv_file = csv_file
    m.config.train.root_dir = root_dir
    m.config.validation.csv_file = csv_file
    m.config.validation.root_dir = root_dir

    assert m.config.train_batch_size == 6
    assert m.config.predict_batch_size == 4
    assert m.train_dataloader().batch_size == 6
    assert m.val_dataloader().batch_size == 4
    assert m.predict_dataloader(ds).batch_size == 4


def test_legacy_batch_size_predict_override():
    """Verify legacy batch_size + explicit predict_batch_size uses predict override."""
    csv_file = get_data("example.csv")
    root_dir = os.path.dirname(csv_file)
    path = get_data("OSBS_029.png")
    tile = np.array(Image.open(path))
    ds = prediction.SingleImage(image=tile, path=path, patch_overlap=0.1, patch_size=100)

    m = main.deepforest(config_args={"batch_size": 4, "predict_batch_size": 16})
    m.config.train.csv_file = csv_file
    m.config.train.root_dir = root_dir
    m.config.validation.csv_file = csv_file
    m.config.validation.root_dir = root_dir

    assert m.config.train_batch_size == 4
    assert m.config.predict_batch_size == 16
    assert m.train_dataloader().batch_size == 4
    assert m.val_dataloader().batch_size == 16
    assert m.predict_dataloader(ds).batch_size == 16


def test_legacy_batch_size_all_three_supplied():
    """Verify when all three are supplied, explicit settings take precedence."""
    csv_file = get_data("example.csv")
    root_dir = os.path.dirname(csv_file)
    path = get_data("OSBS_029.png")
    tile = np.array(Image.open(path))
    ds = prediction.SingleImage(image=tile, path=path, patch_overlap=0.1, patch_size=100)

    m = main.deepforest(
        config_args={"batch_size": 4, "train_batch_size": 6, "predict_batch_size": 16}
    )
    m.config.train.csv_file = csv_file
    m.config.train.root_dir = root_dir
    m.config.validation.csv_file = csv_file
    m.config.validation.root_dir = root_dir

    assert m.config.train_batch_size == 6
    assert m.config.predict_batch_size == 16
    assert m.train_dataloader().batch_size == 6
    assert m.val_dataloader().batch_size == 16
    assert m.predict_dataloader(ds).batch_size == 16


def test_legacy_batch_size_one_compatibility():
    """Verify passing legacy batch_size=1 in config_args works deterministically."""
    csv_file = get_data("example.csv")
    root_dir = os.path.dirname(csv_file)
    path = get_data("OSBS_029.png")
    tile = np.array(Image.open(path))
    ds = prediction.SingleImage(image=tile, path=path, patch_overlap=0.1, patch_size=100)

    m = main.deepforest(config_args={"batch_size": 1})
    m.config.train.csv_file = csv_file
    m.config.train.root_dir = root_dir
    m.config.validation.csv_file = csv_file
    m.config.validation.root_dir = root_dir

    assert m.config.train_batch_size == 1
    assert m.config.predict_batch_size == 1
    assert m.train_dataloader().batch_size == 1
    assert m.val_dataloader().batch_size == 1
    assert m.predict_dataloader(ds).batch_size == 1


def test_legacy_batch_size_in_config_dict():
    """Verify passing legacy batch_size in a custom config dictionary translates properly."""
    csv_file = get_data("example.csv")
    root_dir = os.path.dirname(csv_file)
    path = get_data("OSBS_029.png")
    tile = np.array(Image.open(path))
    ds = prediction.SingleImage(image=tile, path=path, patch_overlap=0.1, patch_size=100)

    m = main.deepforest(config={"batch_size": 5})
    m.config.train.csv_file = csv_file
    m.config.train.root_dir = root_dir
    m.config.validation.csv_file = csv_file
    m.config.validation.root_dir = root_dir

    assert m.config.train_batch_size == 5
    assert m.config.predict_batch_size == 5
    assert m.train_dataloader().batch_size == 5
    assert m.val_dataloader().batch_size == 5
    assert m.predict_dataloader(ds).batch_size == 5


def test_legacy_yaml_by_path(tmp_path):
    """Verify loading a legacy YAML config with batch_size propagates to both train and predict."""
    config_path = tmp_path / "legacy_config.yaml"
    config_path.write_text("batch_size: 5\n")

    csv_file = get_data("example.csv")
    root_dir = os.path.dirname(csv_file)
    path = get_data("OSBS_029.png")
    tile = np.array(Image.open(path))
    ds = prediction.SingleImage(image=tile, path=path, patch_overlap=0.1, patch_size=100)

    m = main.deepforest(config=str(config_path))
    m.config.train.csv_file = csv_file
    m.config.train.root_dir = root_dir
    m.config.validation.csv_file = csv_file
    m.config.validation.root_dir = root_dir

    assert m.config.train_batch_size == 5
    assert m.config.predict_batch_size == 5
    assert m.config.batch_size == 5
    assert m.train_dataloader().batch_size == 5
    assert m.val_dataloader().batch_size == 5
    assert m.predict_dataloader(ds).batch_size == 5


def test_legacy_yaml_with_explicit_train_override(tmp_path):
    """Verify legacy YAML batch_size + explicit train_batch_size override."""
    config_path = tmp_path / "legacy_config.yaml"
    config_path.write_text("batch_size: 5\n")

    csv_file = get_data("example.csv")
    root_dir = os.path.dirname(csv_file)
    path = get_data("OSBS_029.png")
    tile = np.array(Image.open(path))
    ds = prediction.SingleImage(image=tile, path=path, patch_overlap=0.1, patch_size=100)

    m = main.deepforest(config=str(config_path), config_args={"train_batch_size": 10})
    m.config.train.csv_file = csv_file
    m.config.train.root_dir = root_dir
    m.config.validation.csv_file = csv_file
    m.config.validation.root_dir = root_dir

    assert m.config.train_batch_size == 10
    assert m.config.predict_batch_size == 5
    assert m.train_dataloader().batch_size == 10
    assert m.val_dataloader().batch_size == 5
    assert m.predict_dataloader(ds).batch_size == 5


def test_legacy_yaml_with_explicit_predict_override(tmp_path):
    """Verify legacy YAML batch_size + explicit predict_batch_size override."""
    config_path = tmp_path / "legacy_config.yaml"
    config_path.write_text("batch_size: 5\n")

    csv_file = get_data("example.csv")
    root_dir = os.path.dirname(csv_file)
    path = get_data("OSBS_029.png")
    tile = np.array(Image.open(path))
    ds = prediction.SingleImage(image=tile, path=path, patch_overlap=0.1, patch_size=100)

    m = main.deepforest(config=str(config_path), config_args={"predict_batch_size": 12})
    m.config.train.csv_file = csv_file
    m.config.train.root_dir = root_dir
    m.config.validation.csv_file = csv_file
    m.config.validation.root_dir = root_dir

    assert m.config.train_batch_size == 5
    assert m.config.predict_batch_size == 12
    assert m.train_dataloader().batch_size == 5
    assert m.val_dataloader().batch_size == 12
    assert m.predict_dataloader(ds).batch_size == 12


def test_legacy_yaml_with_both_explicit_overrides(tmp_path):
    """Verify legacy YAML batch_size + explicit train and predict overrides."""
    config_path = tmp_path / "legacy_config.yaml"
    config_path.write_text("batch_size: 5\n")

    csv_file = get_data("example.csv")
    root_dir = os.path.dirname(csv_file)
    path = get_data("OSBS_029.png")
    tile = np.array(Image.open(path))
    ds = prediction.SingleImage(image=tile, path=path, patch_overlap=0.1, patch_size=100)

    m = main.deepforest(
        config=str(config_path),
        config_args={"train_batch_size": 4, "predict_batch_size": 16},
    )
    m.config.train.csv_file = csv_file
    m.config.train.root_dir = root_dir
    m.config.validation.csv_file = csv_file
    m.config.validation.root_dir = root_dir

    assert m.config.train_batch_size == 4
    assert m.config.predict_batch_size == 16
    assert m.train_dataloader().batch_size == 4
    assert m.val_dataloader().batch_size == 16
    assert m.predict_dataloader(ds).batch_size == 16


def test_legacy_yaml_batch_size_one(tmp_path):
    """Verify legacy YAML batch_size=1 translates to both settings."""
    config_path = tmp_path / "legacy_config.yaml"
    config_path.write_text("batch_size: 1\n")

    csv_file = get_data("example.csv")
    root_dir = os.path.dirname(csv_file)
    path = get_data("OSBS_029.png")
    tile = np.array(Image.open(path))
    ds = prediction.SingleImage(image=tile, path=path, patch_overlap=0.1, patch_size=100)

    m = main.deepforest(config=str(config_path))
    m.config.train.csv_file = csv_file
    m.config.train.root_dir = root_dir
    m.config.validation.csv_file = csv_file
    m.config.validation.root_dir = root_dir

    assert m.config.train_batch_size == 1
    assert m.config.predict_batch_size == 1
    assert m.train_dataloader().batch_size == 1
    assert m.val_dataloader().batch_size == 1
    assert m.predict_dataloader(ds).batch_size == 1


def test_load_config_direct_legacy_yaml(tmp_path):
    """Verify utilities.load_config directly translates legacy batch_size in YAML."""
    config_path = tmp_path / "legacy_config.yaml"
    config_path.write_text("batch_size: 7\n")

    cfg = utilities.load_config(str(config_path))
    assert cfg.train_batch_size == 7
    assert cfg.predict_batch_size == 7
    assert cfg.batch_size == 7
