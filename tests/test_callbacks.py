# test callbacks
import glob
import os
from unittest.mock import MagicMock, Mock

import pytest
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.logger import DummyLogger

from deepforest import get_data
from deepforest import callbacks, main

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
def m(download_release):
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

    m.create_trainer()
    m.load_model("weecology/deepforest-tree")

    return m


def test_log_images_dummy_comet(m, tmpdir):
    """Test for Comet-style loggers with log_image method"""
    logger = MockCometLogger()
    im_callback = callbacks.ImagesCallback(save_dir=tmpdir,
                                           every_n_epochs=1,
                                           sample_batches=1,
                                           dataset_samples=1)

    m.create_trainer(callbacks=[im_callback], logger=logger, fast_dev_run=False)
    m.trainer.fit(m)

    # Expect 1 from each dataset, 1 from prediction
    assert logger.experiment.log_image.call_count >= 3

def test_log_images_dummy_tb(m, tmpdir):
    """Test for Tensorboard-style loggers with add_image method"""
    logger = MockTBLogger()
    im_callback = callbacks.ImagesCallback(save_dir=tmpdir,
                                           every_n_epochs=1,
                                           sample_batches=1,
                                           dataset_samples=1)

    m.create_trainer(callbacks=[im_callback], logger=logger, fast_dev_run = False)
    m.trainer.fit(m)

    # Expect 1 from each dataset, 1 from prediction
    assert logger.experiment.add_image.call_count >= 3


def test_log_images_dummy_both(m, tmpdir):
    """Test for the correct logger precedence is respected (TB > Comet)."""
    comet = MockCometLogger()
    tensorboard = MockTBLogger()
    loggers = [comet, tensorboard]
    im_callback = callbacks.ImagesCallback(save_dir=tmpdir,
                                           every_n_epochs=1,
                                           sample_batches=1,
                                           dataset_samples=1)

    m.create_trainer(callbacks=[im_callback], logger=loggers, fast_dev_run = False)
    m.trainer.fit(m)

    # Expect 1 from each dataset, 1 from prediction
    assert tensorboard.experiment.add_image.call_count >= 3
    assert comet.experiment.log_image.call_count == 0

def test_log_images_file(m, tmpdir):
    im_callback = callbacks.ImagesCallback(save_dir=tmpdir, every_n_epochs=1, sample_batches=5)

    m.create_trainer(callbacks=[im_callback], fast_dev_run = False)
    m.trainer.fit(m)

    assert os.path.exists(os.path.join(tmpdir, "predictions"))
    saved_images = glob.glob("{}/predictions/*.png".format(tmpdir))
    assert len(saved_images) == m.current_epoch*min(len(m.val_dataloader().dataset), im_callback.sample_batches)
    saved_meta = glob.glob("{}/predictions/*.json".format(tmpdir))
    assert len(saved_meta) == m.current_epoch*min(len(m.val_dataloader().dataset), im_callback.sample_batches)

    assert os.path.exists(os.path.join(tmpdir, "train_sample"))
    train_images = glob.glob("{}/train_sample/*".format(tmpdir))
    assert len(train_images) == min(len(m.train_dataloader().dataset), im_callback.dataset_samples)

    assert os.path.exists(os.path.join(tmpdir, "validation_sample"))
    val_images = glob.glob("{}/validation_sample/*".format(tmpdir))
    assert len(val_images) == min(len(m.val_dataloader().dataset), im_callback.dataset_samples)

def test_log_images_fast(m, tmpdir):
    """Test that no images are logged if fast_dev_run is active"""
    im_callback = callbacks.ImagesCallback(save_dir=tmpdir, every_n_epochs=1)

    m.create_trainer(callbacks=[im_callback], fast_dev_run=True)
    m.trainer.fit(m)

    assert not os.path.exists(os.path.join(tmpdir, "predictions"))
    assert not os.path.exists(os.path.join(tmpdir, "train_sample"))
    assert not os.path.exists(os.path.join(tmpdir, "validation_sample"))

def test_log_images_no_pred(m, tmpdir):
    """Test disabling prediction logging"""
    im_callback = callbacks.ImagesCallback(save_dir=tmpdir, sample_batches=0, every_n_epochs=1)

    m.create_trainer(callbacks=[im_callback], fast_dev_run=False)
    m.trainer.fit(m)

    assert not os.path.exists(os.path.join(tmpdir, "predictions"))
    assert os.path.exists(os.path.join(tmpdir, "train_sample"))
    assert os.path.exists(os.path.join(tmpdir, "validation_sample"))

def test_log_images_no_dataset(m, tmpdir):
    """Test disabling dataset sample logging"""
    im_callback = callbacks.ImagesCallback(save_dir=tmpdir, dataset_samples=0, every_n_epochs=1)

    m.create_trainer(callbacks=[im_callback], fast_dev_run=False)
    m.trainer.fit(m)

    assert os.path.exists(os.path.join(tmpdir, "predictions"))
    assert not os.path.exists(os.path.join(tmpdir, "train_sample"))
    assert not os.path.exists(os.path.join(tmpdir, "validation_sample"))

def test_create_checkpoint(m, tmpdir):
    """Test checkpoint creation"""
    checkpoint_callback = ModelCheckpoint(
        filename='model',
        dirpath=tmpdir,
        save_top_k=1,
        monitor="val_classification",
        mode="max",
        every_n_epochs=1,
    )
    m.load_model("weecology/deepforest-tree")
    m.create_trainer(callbacks=[checkpoint_callback], fast_dev_run=False)
    m.trainer.fit(m)

    assert os.path.exists(os.path.join(tmpdir, 'model.ckpt'))
