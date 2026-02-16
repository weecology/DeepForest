import os

import math
import pandas as pd
import pytest
import shutil
import torch

from deepforest import get_data, main, utilities


@pytest.fixture
def config(architecture, tmp_path_factory):
    # Basic from-scratch config for fast tests
    config = {
        "architecture": architecture,
        "model": {"name": None},
        "num_classes": 1,
        "label_dict": {"Tree": 0},
        "train": {
            "csv_file": get_data("OSBS_029.csv"),
            "root_dir": os.path.dirname(get_data("OSBS_029.csv")),
            "epochs": 1,
            "fast_dev_run": False,  # Required for multi-epoch testing
        },
        "validation": {
            "csv_file": get_data("OSBS_029.csv"),
            "root_dir": os.path.dirname(get_data("OSBS_029.csv")),
        },
        "log_root": str(tmp_path_factory.mktemp("logs"))
    }

    return config


def state_dicts_equal(model_a, model_b):
    """Helper function to compare model state dicts"""
    state_dict_a = model_a.state_dict()
    state_dict_b = model_b.state_dict()

    assert state_dict_a.keys() == state_dict_b.keys(), "State dict keys do not match"

    for key in state_dict_a:
        tensor_a = state_dict_a[key]
        tensor_b = state_dict_b[key]

        assert torch.equal(tensor_a, tensor_b), f"Mismatch found in key: {key}"

    return True


@pytest.mark.parametrize("architecture", ["retinanet", "DeformableDetr"])
def test_train_reload_checkpoint(config):
    """Test train and reload checkpoint"""

    m = main.deepforest(config=config)

    # Train and save model
    m.trainer.fit(m)
    checkpoint_path = f"{m.config.log_root}/pretrain.ckpt"
    m.save_model(checkpoint_path)

    # Load checkpoint
    loaded = main.deepforest.load_from_checkpoint(checkpoint_path)

    assert loaded.config.architecture == config["architecture"]
    os.remove(checkpoint_path)


@pytest.mark.parametrize("architecture", ["retinanet", "DeformableDetr"])
def test_train_and_resume(config):
    """Test resume training from checkpoint with modified config (typically for continuing training)"""

    m = main.deepforest(config=config)

    # Train and save model
    m.trainer.fit(m)
    assert m.trainer.current_epoch == 1
    checkpoint_path = f"{m.config.log_root}/pretrain.ckpt"
    m.save_model(checkpoint_path)

    # Load checkpoint
    loaded = main.deepforest.load_from_checkpoint(
        checkpoint_path, config_args={"train": {"epochs": 3}}
    )
    assert loaded.config.train.epochs == 3
    assert loaded.trainer.max_epochs == 3
    loaded.trainer.fit(loaded)
    assert loaded.trainer.current_epoch == 3
    os.remove(checkpoint_path)

@pytest.mark.parametrize("architecture", ["retinanet", "DeformableDetr"])
def test_train_fit_ckpt(config):
    """Test resume directly via fit (typically for interrupted training)"""

    m = main.deepforest(config=config)

    # Train and save model
    m.trainer.fit(m)
    assert m.trainer.current_epoch == 1
    checkpoint_path = f"{m.config.log_root}/pretrain.ckpt"
    m.save_model(checkpoint_path)

    # Load checkpoint
    m_resume = main.deepforest(config=config, config_args={"train": {"epochs": 3}})

    m_resume.trainer.fit(m_resume, ckpt_path=checkpoint_path)
    assert m_resume.trainer.current_epoch == 3
    os.remove(checkpoint_path)


@pytest.mark.parametrize("architecture", ["retinanet", "DeformableDetr"])
def test_pretrain_finetune(config):
    # Pretrain model
    pretrain = main.deepforest(config=config)
    pretrain.trainer.fit(pretrain)

    # Save checkpoint
    pretrain_checkpoint = f"{pretrain.config.log_root}/pretrain.ckpt"
    pretrain.save_model(pretrain_checkpoint)

    # Fine-tune on "new" data (same data for test)
    log_dir = str(config["log_root"]) + "_finetune"
    finetune = main.deepforest.load_from_checkpoint(
        pretrain_checkpoint,
        config_args={
            "train": {
                "epochs": 1,
            },
            "log_root": log_dir,
        },
    )
    finetune.trainer.fit(finetune)
    finetune.save_model(f"{log_dir}/finetune.ckpt")
    assert finetune.config.log_root == log_dir

    os.remove(pretrain_checkpoint)
    os.remove(f"{log_dir}/finetune.ckpt")


@pytest.mark.parametrize("architecture", ["retinanet", "DeformableDetr"])
def test_local_hf_checkpoint(config):
    """Test loading model.name from local checkpoint path"""

    # Pretrain model
    pretrain = main.deepforest(config=config)
    pretrain.trainer.fit(pretrain)

    # Save HF compatible checkpoints
    pretrain_checkpoint = f"{pretrain.config.log_root}/pretrain_ckpt"
    pretrain.model.save_pretrained(pretrain_checkpoint)

    # Load via model.name
    config_load = pretrain.config.copy()
    config_load.model.name = pretrain_checkpoint

    loaded = main.deepforest(config=config_load)

    assert loaded.config.architecture == config["architecture"]
    assert loaded.config.model.name == pretrain_checkpoint

    shutil.rmtree(pretrain_checkpoint)

@pytest.mark.parametrize("architecture", ["retinanet", "DeformableDetr"])
def test_checkpoint_label_dict(config, tmp_path, architecture):
    """Test that the label dict is saved and loaded correctly from a checkpoint"""
    # Modify config to use a custom label
    csv_file = get_data("example.csv")
    df = pd.read_csv(csv_file)
    df["label"] = "Object"
    df.to_csv(os.path.join(tmp_path, "example.csv"), index=False)

    config["train"]["csv_file"] = os.path.join(tmp_path, "example.csv")
    config["train"]["root_dir"] = os.path.dirname(csv_file)
    config["train"]["fast_dev_run"] = True
    config["validation"]["csv_file"] = os.path.join(tmp_path, "example.csv")
    config["validation"]["root_dir"] = os.path.dirname(csv_file)
    config["label_dict"] = {"Object": 0}

    m = main.deepforest(config=config)
    m.trainer.fit(m)

    checkpoint_path = f"{tmp_path}/checkpoint.ckpt"
    m.trainer.save_checkpoint(checkpoint_path)
    m.model.save_pretrained(os.path.join(tmp_path, "hf_weights"))

    # Load and verify label dicts are preserved
    loaded = main.deepforest.load_from_checkpoint(checkpoint_path)
    assert loaded.label_dict == {"Object": 0}
    del loaded

    loaded_hf = main.deepforest(
        config={"model": {"name": os.path.join(tmp_path, "hf_weights")},
                "architecture": architecture}
    )
    assert loaded_hf.label_dict == {"Object": 0}

    os.remove(checkpoint_path)
    shutil.rmtree(os.path.join(tmp_path, "hf_weights"))

# This test requires "good' weights for tree detection
@pytest.mark.parametrize("architecture", ["retinanet"])
def test_save_and_reload_checkpoint(config):
    """Test that predictions are identical after saving and reloading checkpoint"""
    img_path = get_data(path="2019_YELL_2_528000_4978000_image_crop2.png")

    # Train model
    config["model"]["name"] = "weecology/deepforest-tree"
    m = main.deepforest(config=config)
    m.create_trainer()
    m.trainer.fit(m)
    pred_after_train = m.predict_image(path=img_path)

    # Save checkpoint
    checkpoint_path = f"{m.config.log_root}/checkpoint.ckpt"
    m.save_model(checkpoint_path)

    # Reload and predict
    loaded = main.deepforest.load_from_checkpoint(
        checkpoint_path,
        config_args={
            "train": {
                "csv_file": config["train"]["csv_file"],
                "root_dir": config["train"]["root_dir"],
            },
            "validation": {
                "csv_file": config["validation"]["csv_file"],
                "root_dir": config["validation"]["root_dir"],
            },
        },
    )
    pred_after_reload = loaded.predict_image(path=img_path)

    # Verify predictions match
    assert not pred_after_train.empty
    assert not pred_after_reload.empty
    assert m.config == loaded.config
    assert state_dicts_equal(m.model, loaded.model)
    pd.testing.assert_frame_equal(pred_after_train, pred_after_reload)

    os.remove(checkpoint_path)

@pytest.mark.parametrize("architecture", ["retinanet", "DeformableDetr"])
def test_load_from_checkpoint_with_overrides(config):
    """Test that config_args can override saved config when loading from checkpoint"""
    # Train and save model
    m = main.deepforest(config=config)
    m.trainer.fit(m)

    checkpoint_path = f"{m.config.log_root}/pretrain.ckpt"
    m.save_model(checkpoint_path)

    # Load with config_args overrides
    new_batch_size = 4
    new_epochs = 15
    loaded = main.deepforest.load_from_checkpoint(
        checkpoint_path,
        weights_only=True,
        config_args={"batch_size": new_batch_size,
                     "score_thresh": 0.9,
                     "train": {"epochs": new_epochs}},
    )

    # Verify the overrides were applied
    assert loaded.config.batch_size == new_batch_size
    assert loaded.config.train.epochs == new_epochs
    assert loaded.config.score_thresh != m.config.score_thresh
    assert loaded.config.score_thresh == 0.9
    # Verify other config values from checkpoint are preserved
    assert loaded.config.architecture == config["architecture"]

    os.remove(checkpoint_path)

@pytest.mark.parametrize("architecture", ["retinanet", "DeformableDetr"])
def test_load_from_checkpoint_with_config_dict(config):
    """Test that a full config dict can be passed directly when loading from checkpoint"""
    # Train and save model
    m = main.deepforest(config=config)
    m.trainer.fit(m)

    checkpoint_path = f"{m.config.log_root}/pretrain.ckpt"
    m.save_model(checkpoint_path)

    # Create a modified config with different training parameters
    modified_config = utilities.load_config(
        overrides={
            "architecture": config["architecture"],
            "batch_size": 8,
            "train": {"epochs": 20},
            "model": {"name": None},
            "label_dict": config["label_dict"],
            "num_classes": config["num_classes"],
        }
    )

    # Load with full config dict (as done in CLI)
    loaded = main.deepforest.load_from_checkpoint(
        checkpoint_path, weights_only=True, config=modified_config
    )

    # Verify the config overrides were applied
    assert loaded.config.batch_size == 8
    assert loaded.config.train.epochs == 20
    # Verify architecture is preserved
    assert loaded.config.architecture == config["architecture"]
    # Verify label_dict is preserved
    assert loaded.label_dict == m.label_dict
    assert loaded.numeric_to_label_dict == m.numeric_to_label_dict

    os.remove(checkpoint_path)

# This test requires "good' weights for tree detection
@pytest.mark.parametrize("architecture", ["retinanet"])
def test_save_and_reload_weights(config, tmp_path):
    """Test saving and loading raw PyTorch state dict"""
    img_path = get_data(path="2019_YELL_2_528000_4978000_image_crop2.png")

    # Train model
    config["train"]["fast_dev_run"] = True
    config["model"]["name"] = "weecology/deepforest-tree"
    m = main.deepforest(config=config)
    m.create_trainer()
    m.trainer.fit(m)
    pred_after_train = m.predict_image(path=img_path)

    # Save raw state dict
    weights_path = f"{tmp_path}/weights.pt"
    torch.save(m.model.state_dict(), weights_path)

    # Create new model and load state dict
    loaded = main.deepforest(config=config)
    loaded.model.load_state_dict(torch.load(weights_path, weights_only=True))
    pred_after_reload = loaded.predict_image(path=img_path)

    # Verify predictions match
    assert not pred_after_train.empty
    assert not pred_after_reload.empty
    pd.testing.assert_frame_equal(pred_after_train, pred_after_reload)

    os.remove(weights_path)


@pytest.mark.parametrize("architecture", ["retinanet"])
def test_reload_multi_class(config, tmp_path):
    """Test reloading a multi-class model checkpoint"""
    # Configure for 2 classes with matching data
    csv_file = get_data("testfile_multi.csv")
    config["num_classes"] = 2
    config["label_dict"] = {"Alive": 0, "Dead": 1}
    config["batch_size"] = 2
    config["train"]["csv_file"] = csv_file
    config["train"]["root_dir"] = os.path.dirname(csv_file)
    config["train"]["fast_dev_run"] = True
    config["validation"]["csv_file"] = csv_file
    config["validation"]["root_dir"] = os.path.dirname(csv_file)

    # Train and save
    m = main.deepforest(config=config)
    m.trainer.fit(m)
    checkpoint_path = f"{tmp_path}/checkpoint.ckpt"
    m.save_model(checkpoint_path)
    before = m.trainer.validate(m)

    # Reload and validate
    loaded = main.deepforest.load_from_checkpoint(
        checkpoint_path,
        weights_only=True,
    )
    assert loaded.config.num_classes == 2
    loaded.create_trainer()
    after = loaded.trainer.validate(loaded)

    assert math.isclose(after[0]["val_loss"], before[0]["val_loss"], rel_tol=5e-2, abs_tol=5e-2)

    os.remove(checkpoint_path)
