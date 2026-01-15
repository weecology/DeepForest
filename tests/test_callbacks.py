# test callbacks
import glob
import os
from unittest.mock import MagicMock, Mock

import pytest
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.logger import DummyLogger

from deepforest import get_data
from deepforest import callbacks, main
from deepforest.datasets.training import BoxDataset

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
