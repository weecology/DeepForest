#test callbacks
from deepforest import main
from deepforest import callbacks
import glob
import os
from deepforest import get_data

def test_log_images(tmpdir, config):
    m = main.deepforest()
    m.config = config
    m.use_release(check_release=False)    
    im_callback = callbacks.images_callback(csv_file=m.config["validation"]["csv_file"], root_dir=m.config["validation"]["root_dir"], savedir=tmpdir)
    m.create_trainer(callbacks=[im_callback])
    m.max_steps = 2
    m.trainer.fit(m)
    saved_images = glob.glob("{}/*.png".format(tmpdir))
    assert len(saved_images) == 1
    

def test_log_images_multiclass(tmpdir):
    m = main.deepforest(num_classes=2, label_dict={"Alive":0,"Dead":1})
    m.config["train"]["csv_file"] = get_data("testfile_multi.csv") 
    m.config["train"]["root_dir"] = os.path.dirname(get_data("testfile_multi.csv"))
    m.config["train"]["fast_dev_run"] = False
    m.config["batch_size"] = 2
       
    m.config["validation"]["csv_file"] = get_data("testfile_multi.csv") 
    m.config["validation"]["root_dir"] = os.path.dirname(get_data("testfile_multi.csv"))

    im_callback = callbacks.images_callback(csv_file=m.config["validation"]["csv_file"], root_dir=m.config["validation"]["root_dir"], savedir=tmpdir)
    m.create_trainer(callbacks=[im_callback])
    m.max_steps = 2
    m.trainer.fit(m)
    saved_images = glob.glob("{}/*.png".format(tmpdir))
    assert len(saved_images) == 1
    
    
    