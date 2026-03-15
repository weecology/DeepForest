# test callbacks
import glob
import os
import types
from unittest.mock import MagicMock, Mock

import pandas as pd
import pytest
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.logger import DummyLogger

from deepforest import get_data
from deepforest import callbacks, main
from deepforest.datasets.training import BoxDataset, KeypointDataset

class MockCometLogger(DummyLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup()

    def setup(self):
        probe = Mock(spec_set=("log_image",))
        probe.log_image = MagicMock(name="log_image")
        self._experiment = probe

    @property
    def experiment(self):
        return self._experiment


class MockTBLogger(MockCometLogger):
    def setup(self):
        probe = Mock(spec_set=("add_image",))
        probe.add_image = MagicMock(name="add_image")
        self._experiment = probe

@pytest.fixture(scope="module")
def m(download_release, tmp_path_factory):
    m = main.deepforest()
    m.config.train.csv_file = get_data("example.csv")
    m.config.train.root_dir = os.path.dirname(get_data("example.csv"))
    m.config.train.fast_dev_run = True
    m.config.batch_size = 2
    m.config.validation.csv_file = get_data("example.csv")
    m.config.validation.root_dir = os.path.dirname(get_data("example.csv"))
    m.config.workers = 0
    m.config.validation.val_accuracy_interval = 1
    m.config.train.epochs = 2
    m.config.log_root = str(tmp_path_factory.mktemp("logs"))

    m.create_trainer()
    m.load_model("weecology/deepforest-tree")

    return m


def test_log_images_dummy_comet(m, tmp_path):
    """Test for Comet-style loggers with log_image method"""
    logger = MockCometLogger()
    im_callback = callbacks.ImagesCallback(save_dir=tmp_path,
                                           every_n_epochs=1,
                                           prediction_samples=1,
                                           dataset_samples=1)

    m.create_trainer(callbacks=[im_callback], logger=logger, fast_dev_run=False)
    m.trainer.fit(m)

    # Expect 1 from each dataset, 1 from prediction
    assert logger.experiment.log_image.call_count >= 3

def test_log_images_dummy_tb(m, tmp_path):
    """Test for Tensorboard-style loggers with add_image method"""
    logger = MockTBLogger()
    im_callback = callbacks.ImagesCallback(save_dir=tmp_path,
                                           every_n_epochs=1,
                                           prediction_samples=1,
                                           dataset_samples=1)

    m.create_trainer(callbacks=[im_callback], logger=logger, fast_dev_run = False)
    m.trainer.fit(m)

    # Expect 1 from each dataset, 1 from prediction
    assert logger.experiment.add_image.call_count >= 3


def test_log_images_dummy_both(m, tmp_path):
    """Test for the correct logger precedence is respected (TB > Comet)."""
    comet = MockCometLogger()
    tensorboard = MockTBLogger()
    loggers = [comet, tensorboard]
    im_callback = callbacks.ImagesCallback(save_dir=tmp_path,
                                           every_n_epochs=1,
                                           prediction_samples=1,
                                           dataset_samples=1)

    m.create_trainer(callbacks=[im_callback], logger=loggers, fast_dev_run = False)
    m.trainer.fit(m)

    # Expect 1 from each dataset, 1 from prediction
    assert tensorboard.experiment.add_image.call_count >= 3
    assert comet.experiment.log_image.call_count == 0

def test_log_images_file(m, tmp_path):
    im_callback = callbacks.ImagesCallback(save_dir=tmp_path, every_n_epochs=1, prediction_samples=5)

    m.create_trainer(callbacks=[im_callback], fast_dev_run = False)
    m.trainer.fit(m)

    assert (tmp_path / "predictions").exists()
    saved_images = list((tmp_path / "predictions").glob("*.png"))
    assert len(saved_images) == m.current_epoch*min(len(m.val_dataloader().dataset), im_callback.prediction_samples)
    saved_meta = list((tmp_path / "predictions").glob("*.json"))
    assert len(saved_meta) == m.current_epoch*min(len(m.val_dataloader().dataset), im_callback.prediction_samples)

    assert (tmp_path / "train_sample").exists()
    train_images = list((tmp_path / "train_sample").glob("*"))
    assert len(train_images) == min(len(m.train_dataloader().dataset), im_callback.dataset_samples)

    assert (tmp_path / "validation_sample").exists()
    val_images = list((tmp_path / "validation_sample").glob("*"))
    assert len(val_images) == min(len(m.val_dataloader().dataset), im_callback.dataset_samples)

def test_log_images_fast(m, tmp_path):
    """Test that no images are logged if fast_dev_run is active"""
    im_callback = callbacks.ImagesCallback(save_dir=tmp_path, every_n_epochs=1)

    m.create_trainer(callbacks=[im_callback], fast_dev_run=True)
    m.trainer.fit(m)

    assert not (tmp_path / "predictions").exists()
    assert not (tmp_path / "train_sample").exists()
    assert not (tmp_path / "validation_sample").exists()

def test_log_images_no_pred(m, tmp_path):
    """Test disabling prediction logging"""
    im_callback = callbacks.ImagesCallback(save_dir=tmp_path, prediction_samples=0, every_n_epochs=1)

    m.create_trainer(callbacks=[im_callback], fast_dev_run=False)
    m.trainer.fit(m)

    assert not (tmp_path / "predictions").exists()
    assert (tmp_path / "train_sample").exists()
    assert (tmp_path / "validation_sample").exists()

def test_log_images_no_dataset(m, tmp_path):
    """Test disabling dataset sample logging"""
    im_callback = callbacks.ImagesCallback(save_dir=tmp_path, dataset_samples=0, every_n_epochs=1)

    m.create_trainer(callbacks=[im_callback], fast_dev_run=False)
    m.trainer.fit(m)

    assert (tmp_path / "predictions").exists()
    assert not (tmp_path / "train_sample").exists()
    assert not (tmp_path / "validation_sample").exists()

def test_log_image_empty_annotations(m, tmp_path):
    """Test that images with no annotations are logged without error"""
    im_callback = callbacks.ImagesCallback(save_dir=tmp_path, every_n_epochs=1, prediction_samples=1)
    im_callback.trainer = m.trainer

    # Drop targets from all samples
    class EmptyDataset(BoxDataset):
        def __getitem__(self, idx):
            image, targets, img_name = super().__getitem__(idx)
            targets = {"boxes": [], "labels": []}
            return image, targets, img_name

    empty_dataset = EmptyDataset(
        csv_file=m.config.validation.csv_file,
        root_dir=m.config.validation.root_dir
    )

    im_callback._log_dataset_sample(empty_dataset, split="validation")
    assert os.path.exists(os.path.join(tmp_path, "validation_sample"))
    val_images = list((tmp_path / "validation_sample").glob("*"))
    assert len(val_images) == 1

def test_create_checkpoint(m, tmp_path):
    """Test checkpoint creation"""
    checkpoint_callback = ModelCheckpoint(
        filename='model',
        dirpath=tmp_path,
        save_top_k=1,
        monitor="val_classification",
        mode="max",
        every_n_epochs=1,
    )
    m.load_model("weecology/deepforest-tree")
    m.create_trainer(callbacks=[checkpoint_callback], fast_dev_run=False)
    m.trainer.fit(m)

    assert (tmp_path / 'model.ckpt').exists()


# --- Keypoint / point-data callback tests ---


@pytest.fixture(scope="module")
def keypoint_ds():
    csv_file = get_data("2019_BLAN_3_751000_4330000_image_crop_keypoints.csv")
    root_dir = os.path.dirname(csv_file)
    return KeypointDataset(
        csv_file=csv_file,
        root_dir=root_dir,
        label_dict={"Tree": 0},
    )


@pytest.fixture(scope="module")
def keypoint_preds(keypoint_ds):
    """Load keypoint predictions directly from the repo CSV file.

    Returns a list of per-image DataFrames matching the format that
    pl_module.predictions contains after validation_step.
    """
    from deepforest import utilities

    csv_file = get_data("2019_BLAN_3_751000_4330000_image_crop_keypoints.csv")
    df = utilities.read_file(csv_file)
    df["root_dir"] = keypoint_ds.root_dir
    return [group for _, group in df.groupby("image_path")]


def test_log_dataset_sample_keypoints(keypoint_ds, tmp_path):
    """_log_dataset_sample should save images from a KeypointDataset without error."""
    im_callback = callbacks.ImagesCallback(
        save_dir=tmp_path, every_n_epochs=1, dataset_samples=1
    )
    # trainer attribute is only needed for _log_to_all; wire up a minimal stand-in
    im_callback.trainer = types.SimpleNamespace(
        global_step=0,
        loggers=[],
    )

    im_callback._log_dataset_sample(keypoint_ds, split="validation")

    out_dir = tmp_path / "validation_sample"
    assert out_dir.exists()
    assert len(list(out_dir.glob("*.png"))) >= 1


def test_log_last_predictions_keypoints(keypoint_ds, keypoint_preds, tmp_path):
    """_log_last_predictions should save PNG + JSON for point predictions."""
    im_callback = callbacks.ImagesCallback(
        save_dir=tmp_path, every_n_epochs=1, prediction_samples=1
    )

    trainer = types.SimpleNamespace(
        global_step=0,
        loggers=[],
        val_dataloaders=types.SimpleNamespace(dataset=keypoint_ds),
    )
    pl_module = types.SimpleNamespace(
        predictions=keypoint_preds,
        print=lambda *a, **kw: None,
    )

    im_callback._log_last_predictions(trainer, pl_module)

    out_dir = tmp_path / "predictions"
    assert out_dir.exists()
    assert len(list(out_dir.glob("*.png"))) >= 1
    assert len(list(out_dir.glob("*.json"))) >= 1


def test_log_last_predictions_keypoints_select_random(keypoint_ds, keypoint_preds, tmp_path):
    """select_random=True should also succeed for point predictions."""
    im_callback = callbacks.ImagesCallback(
        save_dir=tmp_path, every_n_epochs=1, prediction_samples=1, select_random=True
    )

    trainer = types.SimpleNamespace(
        global_step=0,
        loggers=[],
        val_dataloaders=types.SimpleNamespace(dataset=keypoint_ds),
    )
    pl_module = types.SimpleNamespace(
        predictions=keypoint_preds,
        print=lambda *a, **kw: None,
    )

    im_callback._log_last_predictions(trainer, pl_module)

    out_dir = tmp_path / "predictions"
    assert out_dir.exists()
    assert len(list(out_dir.glob("*.png"))) >= 1

def test_log_density_plots(tmp_path):
    """_log_density_plots should save side-by-side PNG files for density samples."""
    import torch

    im_callback = callbacks.ImagesCallback(
        save_dir=tmp_path, every_n_epochs=1, prediction_samples=1
    )

    density_samples = [
        {
            "image": torch.rand(3, 128, 128),
            "pred_density": torch.rand(32, 32),
            "gt_density": torch.rand(32, 32),
            "image_name": "test_image.tif",
        }
    ]

    trainer = types.SimpleNamespace(
        global_step=10,
        current_epoch=1,
        loggers=[],
    )
    pl_module = types.SimpleNamespace(
        density_samples=density_samples,
        print=lambda *a, **kw: None,
    )

    im_callback._log_density_plots(trainer, pl_module)

    out_dir = tmp_path / "density_plots"
    assert out_dir.exists()
    assert len(list(out_dir.glob("*.png"))) == 1
