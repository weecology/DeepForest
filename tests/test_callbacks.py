# test callbacks
import glob

from pytorch_lightning.callbacks import ModelCheckpoint

from deepforest import callbacks


def test_log_images(m, tmpdir):
    im_callback = callbacks.images_callback(savedir=tmpdir, every_n_epochs=2)
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
    m.load_model("weecology/deepforest-tree")
    m.create_trainer(callbacks=[checkpoint_callback])
    m.trainer.fit(m)
