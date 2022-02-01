#test main
import os
import glob
import pytest
import pandas as pd
import numpy as np
import cv2
import copy
import shutil
import torch
import tempfile

import albumentations as A
from albumentations.pytorch import ToTensorV2

from deepforest import main
from deepforest import get_data
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from PIL import Image

@pytest.fixture()
def big_file():
    tmpdir = tempfile.gettempdir()
    csv_file = get_data("OSBS_029.csv")
    df = pd.read_csv(csv_file)    
    
    big_frame = []
    for x in range(2):
        img = Image.open("{}/{}".format(os.path.dirname(csv_file), df.image_path.unique()[0]))
        cv2.imwrite("{}/{}.png".format(tmpdir, x), np.array(img))
        new_df = df.copy()
        new_df.image_path = "{}.png".format(x)
        big_frame.append(new_df)
    
    big_frame = pd.concat(big_frame)
    big_frame.to_csv("{}/annotations.csv".format(tmpdir))    
    
    return "{}/annotations.csv".format(tmpdir)

def test_use_bird_release():
    m = main.deepforest()
    m.use_bird_release(check_release=False)
    imgpath = get_data("AWPE Pigeon Lake 2020 DJI_0005.JPG")        
    boxes = m.predict_image(path=imgpath)
    assert not boxes.empty
    
def test_train_empty(config):
    m = main.deepforest()
    m.config = config
    m.use_release(check_release=False)
    tmpdir = tempfile.gettempdir()
    empty_csv = pd.DataFrame({"image_path":["OSBS_029.png","OSBS_029.tif"],"xmin":[0,10],"xmax":[0,20],"ymin":[0,20],"ymax":[0,30],"label":["Tree","Tree"]})
    empty_csv.to_csv("{}/empty.csv".format(tmpdir))
    m.config["train"]["csv_file"] = "{}/empty.csv".format(tmpdir)
    m.config["batch_size"] = 2
    m.create_trainer()
    m.trainer.fit(m)

def test_validation_step(config):
    m = main.deepforest()
    m.config = config    
    m.use_release(check_release=False)
    before = copy.deepcopy(m)
    m.create_trainer()
    m.trainer.validate(m)
    #assert no weights have changed
    for p1, p2 in zip(before.named_parameters(), m.named_parameters()):     
        assert p1[1].ne(p2[1]).sum() == 0

def test_train_preload_images(config):
    m = main.deepforest()
    m.config = config
    m.use_release(check_release=False)
    m.config["train"]["preload_images"] = True
    m.create_trainer()
    m.trainer.fit(m)
    
def test_train_no_validation(config):
    m = main.deepforest()
    m.config = config
    m.use_release(check_release=False)
    m.config["validation"]["csv_file"] = None
    m.config["validation"]["root_dir"] = None  
    m.create_trainer()
    m.trainer.fit(m)
    
def test_predict_image_empty(config):
    m = main.deepforest()
    m.config = config
    m.use_release(check_release=False)    
    image = np.random.random((400,400,3)).astype("float32")
    prediction = m.predict_image(image = image)
    
    assert prediction is None
    
def test_predict_image_fromfile(config):
    m = main.deepforest()
    m.config = config
    m.use_release(check_release=False)      
    path = get_data(path="2019_YELL_2_528000_4978000_image_crop2.png")
    prediction = m.predict_image(path = path)
    
    assert isinstance(prediction, pd.DataFrame)
    assert set(prediction.columns) == {"xmin","ymin","xmax","ymax","label","score"}

def test_predict_image_fromarray(config):
    m = main.deepforest()
    m.config = config
    m.use_release(check_release=False)       
    image_path = get_data(path="2019_YELL_2_528000_4978000_image_crop2.png")
    
    #assert error of dtype
    with pytest.raises(TypeError):
        image = Image.open(image_path)
        prediction = m.predict_image(image = image)
            
    image = np.array(Image.open(image_path).convert("RGB"))
    prediction = m.predict_image(image = image)    
    assert isinstance(prediction, pd.DataFrame)
    assert set(prediction.columns) == {"xmin","ymin","xmax","ymax","label","score"}

def test_predict_return_plot(config):
    m = main.deepforest()
    m.config = config
    m.use_release(check_release=False)       
    image = get_data(path="2019_YELL_2_528000_4978000_image_crop2.png")
    image = np.array(Image.open(image))
    image = image.astype('float32')
    plot = m.predict_image(image = image, return_plot=True)
    assert isinstance(plot, np.ndarray)

def test_predict_big_file(big_file, tmpdir, config):
    m = main.deepforest()
    m.config = config
    m.use_release(check_release=False)       
    original_file = pd.read_csv(big_file)
    df = m.predict_file(csv_file=big_file, root_dir = os.path.dirname(big_file), savedir=tmpdir)
    assert set(df.columns) == {"xmin","ymin","xmax","ymax","label","score","image_path"}
    
    printed_plots = glob.glob("{}/*.png".format(tmpdir))
    assert len(printed_plots) == len(original_file.image_path.unique())
    
def test_predict_small_file(config):
    m = main.deepforest()
    m.config = config
    m.use_release(check_release=False)      
    tmpdir = tempfile.mkdtemp()
    csv_file = get_data("OSBS_029.csv")
    original_file = pd.read_csv(csv_file)
    df = m.predict_file(csv_file, root_dir = os.path.dirname(csv_file), savedir=tmpdir)
    assert set(df.columns) == {"xmin","ymin","xmax","ymax","label","score","image_path"}
    
    printed_plots = glob.glob("{}/*.png".format(tmpdir))
    assert len(printed_plots) == len(original_file.image_path.unique())
    shutil.rmtree(tmpdir)
    
def test_predict_tile(config):
    m = main.deepforest()
    m.config = config
    m.use_release(check_release=False)   
    #test raster prediction, intentionally large windows
    raster_path = get_data(path= 'OSBS_029.tif')
    prediction = m.predict_tile(raster_path = raster_path,
                                            patch_size = 800,
                                            patch_overlap = 0.1,
                                            return_plot = False)
    assert isinstance(prediction, pd.DataFrame)
    assert set(prediction.columns) == {"xmin","ymin","xmax","ymax","label","score"}
    assert not prediction.empty

    #test soft-nms method
    soft_nms_pred = m.predict_tile(raster_path = raster_path,
                                            patch_size = 300,
                                            patch_overlap = 0.1,
                                            return_plot = False,
                                            use_soft_nms =True)
    assert isinstance(soft_nms_pred, pd.DataFrame)
    assert set(soft_nms_pred.columns) == {"xmin","ymin","xmax","ymax","label","score"}
    assert not soft_nms_pred.empty

    #test predict numpy image
    image = np.array(Image.open(raster_path))
    prediction = m.predict_tile(image = image,
                                patch_size = 300,
                                patch_overlap = 0.1,
                                return_plot = False)
    assert not prediction.empty

    # Test no non-max suppression
    prediction = m.predict_tile(raster_path = raster_path,
                                       patch_size=300,
                                       patch_overlap=0,
                                       return_plot=False)
    assert not prediction.empty
    
    #test no mosaic, return a tuple of crop and prediction
    prediction = m.predict_tile(raster_path = raster_path,
                                       patch_size=300,
                                       patch_overlap=0,
                                       return_plot=False,
                                       mosaic=False) 
    assert len(prediction) == 4
    assert len(prediction[0]) == 2
    assert prediction[0][1].shape == (300,300, 3)
    
def test_evaluate(config):
    m = main.deepforest()
    m.config = config
    m.use_release(check_release=False)    
    csv_file = get_data("OSBS_029.csv")
    root_dir = os.path.dirname(csv_file)
    tmpdir = tempfile.gettempdir()
    results = m.evaluate(csv_file, root_dir, iou_threshold = 0.4, savedir=tmpdir)
    
    #Does this make reasonable predictions, we know the model works.
    assert np.round(results["box_precision"],2) > 0.5
    assert np.round(results["box_recall"],2) > 0.5
    assert len(results["results"].predicted_label.dropna().unique()) == 1
    assert results["results"].predicted_label.dropna().unique()[0] == "Tree"
    assert results["predictions"].shape[0] > 0
    
    df = pd.read_csv(csv_file)
    assert results["results"].shape[0] == df.shape[0]

def test_evaluate_multiple_images(config):
    m = main.deepforest()
    m.config = config
    m.use_release(check_release=False)     
    tmpdir = tempfile.gettempdir()
    orignal_csv_file = get_data("OSBS_029.csv")
    original_root_dir = os.path.dirname(orignal_csv_file)
    
    df = pd.read_csv(orignal_csv_file)
    
    df2 = df.copy()
    df2["image_path"] = "OSBS_029_1.tif"
    df3 = df.copy()
    df3["image_path"] = "OSBS_029_2.tif"
    multiple_images = multiple_images = pd.concat([df, df2, df3])
    multiple_images = multiple_images.reset_index(drop=True)
    csv_file = "{}/example.csv".format(tmpdir)
    root_dir = os.path.dirname(csv_file)
    multiple_images.to_csv(csv_file)
    
    #Create multiple files
    shutil.copyfile("{}/OSBS_029.tif".format(original_root_dir), "{}/OSBS_029.tif".format(root_dir))    
    shutil.copyfile("{}/OSBS_029.tif".format(original_root_dir), "{}/OSBS_029_1.tif".format(root_dir))
    shutil.copyfile("{}/OSBS_029.tif".format(original_root_dir), "{}/OSBS_029_2.tif".format(root_dir))
    
    root_dir = os.path.dirname(csv_file)
    
    results = m.evaluate(csv_file, root_dir, iou_threshold = 0.4, savedir=tmpdir)
  
    assert results["results"].shape[0] == multiple_images.shape[0]

    assert all([x in results["results"] for x in ["xmin","xmax","ymin","ymax"]])
    
def test_train_callbacks(config):
    m = main.deepforest()
    m.config = config
    m.use_release(check_release=False)      
    csv_file = get_data("example.csv") 
    root_dir = os.path.dirname(csv_file)
    train_ds = m.load_dataset(csv_file, root_dir=root_dir)
    
    class MyPrintingCallback(Callback):
    
        def on_init_start(self, trainer):
            print('Starting to init trainer!')
    
        def on_init_end(self, trainer):
            print('trainer is init now')
    
        def on_train_end(self, trainer, pl_module):
            print('do something when training ends')
    
    trainer = Trainer(callbacks=[MyPrintingCallback()])
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(m, train_ds)

def test_custom_config_file_path():
    print(os.getcwd())
    m = main.deepforest(config_file='tests/deepforest_config_test.yml')
    assert m.config["batch_size"] == 9999
    assert m.config["nms_thresh"] == 0.9
    assert m.config["score_thresh"] == 0.9

def test_save_and_reload_checkpoint(config):
    m = main.deepforest()
    m.config = config
    #No real learning to throw off the model predictions
    m.config["train"]["fast_dev_run"] = False
    m.config["train"]["lr"] = 0.000000000001
    m.use_release(check_release=False)   
    m.create_trainer() 
    m.trainer.fit(m)
    #save the prediction dataframe after training and compare with prediction after reload checkpoint     
    tmpdir = tempfile.gettempdir()
    img_path = get_data(path="2019_YELL_2_528000_4978000_image_crop2.png")    
    pred_after_train = m.predict_image(path = img_path)
    
    m.save_model("{}/checkpoint.pl".format(tmpdir))
    
    #reload the checkpoint to model object
    after = main.deepforest.load_from_checkpoint("{}/checkpoint.pl".format(tmpdir))
    pred_after_reload = after.predict_image(path = img_path)

    assert not pred_after_train.empty
    assert not pred_after_reload.empty
    pd.testing.assert_frame_equal(pred_after_train,pred_after_reload)
    
def test_reload_multi_class():
    m = main.deepforest(num_classes=2,label_dict={"Alive":0,"Dead":1})
    m.config["train"]["csv_file"] = get_data("testfile_multi.csv") 
    m.config["train"]["root_dir"] = os.path.dirname(get_data("testfile_multi.csv"))
    m.config["train"]["fast_dev_run"] = True
    m.config["batch_size"] = 2
    m.config["workers"] = 0
        
    m.config["validation"]["csv_file"] = get_data("testfile_multi.csv") 
    m.config["validation"]["root_dir"] = os.path.dirname(get_data("testfile_multi.csv"))
    m.config["validation"]["val_accuracy_interval"] = 1

    m.create_trainer()
    m.trainer.fit(m)
    
    tmpdir = tempfile.gettempdir()
    m.save_model("{}/checkpoint.pl".format(tmpdir))
    before = m.trainer.validate(m)
    
    #reload
    old_model = main.deepforest.load_from_checkpoint("{}/checkpoint.pl".format(tmpdir))
    old_model.config = m.config
    assert old_model.num_classes == 2
    old_model.create_trainer()    
    after = old_model.trainer.validate(old_model)
    
    assert after[0]["val_classification"] == before[0]["val_classification"]
    
def test_override_transforms():
    def get_transform(augment):
        """This is the new transform"""
        if augment:
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc',label_fields=["category_ids"]))
            
        else:
            transform = ToTensorV2()
            
        return transform
    
    m = main.deepforest(transforms=get_transform)
    
    csv_file = get_data("example.csv") 
    root_dir = os.path.dirname(csv_file)
    train_ds = m.load_dataset(csv_file, root_dir=root_dir)
    
    path, image, target = next(iter(train_ds))
    assert m.transforms.__doc__ == "This is the new transform"

def test_over_score_thresh(config):
    """A user might want to change the config after model training and update the score thresh"""
    img = get_data("OSBS_029.png")
    m = main.deepforest()
    m.config = config
    m.use_release(check_release=False)       
    original_score_thresh = m.model.score_thresh
    m.config["score_thresh"] = 0.8
    
    #trigger update
    boxes = m.predict_image(path = img)
    assert m.model.score_thresh == 0.8
    assert not m.model.score_thresh == original_score_thresh
    
    