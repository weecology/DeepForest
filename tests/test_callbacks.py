#test callbacks
from deepforest import main
from deepforest import callbacks
import glob
import os
import pytest
from pytorch_lightning.callbacks import ModelCheckpoint
from deepforest import get_data

@pytest.mark.parametrize("every_n_epochs", [1, 2, 3])
def test_log_images(m, every_n_epochs, tmpdir):
    im_callback = callbacks.images_callback(savedir=tmpdir, every_n_epochs=every_n_epochs)
    m.create_trainer(callbacks=[im_callback])
    m.trainer.fit(m)
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