#test callbacks
from deepforest import main
from deepforest import callbacks
import glob
import os
import pytest
from pytorch_lightning.callbacks import ModelCheckpoint
from deepforest import get_data

def test_log_images(m, tmpdir):
    im_callback = callbacks.images_callback(csv_file=m.config["validation"]["csv_file"], root_dir=m.config["validation"]["root_dir"], savedir=tmpdir)
    m.create_trainer(callbacks=[im_callback])
    m.trainer.fit(m)
    saved_images = glob.glob("{}/*.png".format(tmpdir))
    assert len(saved_images) == 1
    
def test_log_images_multiclass(two_class_m, tmpdir):
    im_callback = callbacks.images_callback(
        csv_file=two_class_m.config["validation"]["csv_file"],
        root_dir=two_class_m.config["validation"]["root_dir"],
        savedir=tmpdir)
    two_class_m.create_trainer(callbacks=[im_callback])
    two_class_m.trainer.fit(two_class_m)
    saved_images = glob.glob("{}/*.png".format(tmpdir))
    assert len(saved_images) == 1

def test_create_checkpoint(m, tmpdir):    
    checkpoint_callback = ModelCheckpoint(
            dirpath=tmpdir,
            save_top_k=1,
            monitor="val_classification",
            mode="max",
            every_n_epochs=1,
        )
    m.use_release()
    m.create_trainer(callbacks = [checkpoint_callback])
    m.trainer.fit(m)