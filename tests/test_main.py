#test main
import os
import glob
import pytest
import pandas as pd
import numpy as np
import cv2
import shutil
import torch
import tempfile
import copy

import albumentations as A
from albumentations.pytorch import ToTensorV2

from deepforest import main
from deepforest import get_data
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from PIL import Image

#Import release model from global script to avoid thrasing github during testing. Just download once.
from .conftest import download_release

@pytest.fixture()
def two_class_m():
    m = main.deepforest(num_classes=2,label_dict={"Alive":0,"Dead":1})
    m.config["train"]["csv_file"] = get_data("testfile_multi.csv") 
    m.config["train"]["root_dir"] = os.path.dirname(get_data("testfile_multi.csv"))
    m.config["train"]["fast_dev_run"] = True
    m.config["batch_size"] = 2
        
    m.config["validation"]["csv_file"] = get_data("testfile_multi.csv") 
    m.config["validation"]["root_dir"] = os.path.dirname(get_data("testfile_multi.csv"))
    m.config["validation"]["val_accuracy_interval"] = 1

    m.create_trainer()
    
    return m

@pytest.fixture()
def m(download_release):
    m = main.deepforest()
    m.config["train"]["csv_file"] = get_data("example.csv") 
    m.config["train"]["root_dir"] = os.path.dirname(get_data("example.csv"))
    m.config["train"]["fast_dev_run"] = True
    m.config["batch_size"] = 2
        
    m.config["validation"]["csv_file"] = get_data("example.csv") 
    m.config["validation"]["root_dir"] = os.path.dirname(get_data("example.csv"))
    m.config["workers"] = 0 
    m.config["validation"]["val_accuracy_interval"] = 1
    m.config["train"]["epochs"] = 2
    
    m.create_trainer()
    m.use_release(check_release=False)
    
    return m

def big_file():
    tmpdir = tempfile.gettempdir()
    csv_file = get_data("OSBS_029.csv")
    image_path = get_data("OSBS_029.png")
    df = pd.read_csv(csv_file)    
    
    big_frame = []
    for x in range(3):
        img = Image.open("{}/{}".format(os.path.dirname(csv_file), df.image_path.unique()[0]))
        cv2.imwrite("{}/{}.png".format(tmpdir, x), np.array(img))
        new_df = df.copy()
        new_df.image_path = "{}.png".format(x)
        big_frame.append(new_df)
    
    big_frame = pd.concat(big_frame)
    big_frame.to_csv("{}/annotations.csv".format(tmpdir))    
    
    return "{}/annotations.csv".format(tmpdir)
    
def test_main():
    from deepforest import main

def test_use_bird_release(m):
    imgpath = get_data("AWPE Pigeon Lake 2020 DJI_0005.JPG")    
    m.use_bird_release()
    boxes = m.predict_image(path=imgpath)
    assert not boxes.empty
    
def test_train_empty(m, tmpdir):
    empty_csv = pd.DataFrame({"image_path":["OSBS_029.png","OSBS_029.tif"],"xmin":[0,10],"xmax":[0,20],"ymin":[0,20],"ymax":[0,30],"label":["Tree","Tree"]})
    empty_csv.to_csv("{}/empty.csv".format(tmpdir))
    m.config["train"]["csv_file"] = "{}/empty.csv".format(tmpdir)
    m.config["batch_size"] = 2
    m.create_trainer()    
    m.trainer.fit(m)

def test_validation_step(m):
    m.trainer = None
    #Turn off trainer to test copying on some linux devices.
    before = copy.deepcopy(m)
    m.create_trainer()
    m.trainer.validate(m)
    #assert no weights have changed
    for p1, p2 in zip(before.named_parameters(), m.named_parameters()):     
        assert p1[1].ne(p2[1]).sum() == 0

def test_train_single(m):
    m.create_trainer()
    m.trainer.fit(m)

def test_train_preload_images(m):
    m.config["train"]["preload_images"] = True
    m.trainer.fit(m)
    
def test_train_multi(two_class_m):
    two_class_m.trainer.fit(two_class_m)
    
def test_train_no_validation(m):
    m.config["validation"]["csv_file"] = None
    m.config["validation"]["root_dir"] = None  
    m.create_trainer()
    m.trainer.fit(m)
    
def test_predict_image_empty(m):
    image = np.random.random((400,400,3)).astype("float32")
    prediction = m.predict_image(image = image)
    
    assert prediction is None
    
def test_predict_image_fromfile(m):
    path = get_data(path="2019_YELL_2_528000_4978000_image_crop2.png")
    prediction = m.predict_image(path = path)
    
    assert isinstance(prediction, pd.DataFrame)
    assert set(prediction.columns) == {"xmin","ymin","xmax","ymax","label","score","image_path"}

def test_predict_image_fromarray(m):
    image_path = get_data(path="2019_YELL_2_528000_4978000_image_crop2.png")
    
    #assert error of dtype
    with pytest.raises(TypeError):
        image = Image.open(image_path)
        prediction = m.predict_image(image = image)
            
    image = np.array(Image.open(image_path).convert("RGB"))
    prediction = m.predict_image(image = image)    
    assert isinstance(prediction, pd.DataFrame)
    assert set(prediction.columns) == {"xmin","ymin","xmax","ymax","label","score"}

def test_predict_return_plot(m):
    image = get_data(path="2019_YELL_2_528000_4978000_image_crop2.png")
    image = np.array(Image.open(image))
    image = image.astype('float32')
    plot = m.predict_image(image = image, return_plot=True)
    assert isinstance(plot, np.ndarray)

def test_predict_big_file(m, tmpdir):
    csv_file = big_file()
    original_file = pd.read_csv(csv_file)
    df = m.predict_file(csv_file=csv_file, root_dir = os.path.dirname(csv_file), savedir=tmpdir)
    assert set(df.columns) == {"xmin","ymin","xmax","ymax","label","score","image_path"}
    
    printed_plots = glob.glob("{}/*.png".format(tmpdir))
    assert len(printed_plots) == len(original_file.image_path.unique())
    
def test_predict_small_file(m, tmpdir):
    csv_file = get_data("OSBS_029.csv")
    original_file = pd.read_csv(csv_file)
    df = m.predict_file(csv_file, root_dir = os.path.dirname(csv_file), savedir=tmpdir)
    assert set(df.columns) == {"xmin","ymin","xmax","ymax","label","score","image_path"}
    
    printed_plots = glob.glob("{}/*.png".format(tmpdir))
    assert len(printed_plots) == len(original_file.image_path.unique())
    
def test_predict_tile(m):
    #test raster prediction 
    raster_path = get_data(path= 'OSBS_029.tif')
    prediction = m.predict_tile(raster_path = raster_path,
                                            patch_size = 300,
                                            patch_overlap = 0.5,
                                            return_plot = False)
    assert isinstance(prediction, pd.DataFrame)
    assert set(prediction.columns) == {"xmin","ymin","xmax","ymax","label","score","image_path"}
    assert not prediction.empty

    #test soft-nms method
    soft_nms_pred = m.predict_tile(raster_path = raster_path,
                                            patch_size = 300,
                                            patch_overlap = 0.5,
                                            return_plot = False,
                                            use_soft_nms =True)
    assert isinstance(soft_nms_pred, pd.DataFrame)
    assert set(soft_nms_pred.columns) == {"xmin","ymin","xmax","ymax","label","score","image_path"}
    assert not soft_nms_pred.empty

    #test predict numpy image
    image = np.array(Image.open(raster_path))
    prediction = m.predict_tile(image = image,
                                patch_size = 300,
                                patch_overlap = 0.5,
                                return_plot = False)
    assert not prediction.empty

    # Test no non-max suppression
    prediction = m.predict_tile(raster_path = raster_path,
                                       patch_size=300,
                                       patch_overlap=0,
                                       return_plot=False)
    assert not prediction.empty
    

def test_predict_tile_no_mosaic(m):
    #test no mosaic, return a tuple of crop and prediction
    raster_path = get_data(path='OSBS_029.tif')    
    prediction = m.predict_tile(raster_path = raster_path,
                                       patch_size=300,
                                       patch_overlap=0,
                                       return_plot=False,
                                       mosaic=False) 
    assert len(prediction) == 4
    assert len(prediction[0]) == 2
    assert prediction[0][1].shape == (300,300, 3)    
    
def test_evaluate(m, tmpdir):
    csv_file = get_data("OSBS_029.csv")
    root_dir = os.path.dirname(csv_file)
    
    results = m.evaluate(csv_file, root_dir, iou_threshold = 0.4, savedir=tmpdir)
    
    #Does this make reasonable predictions, we know the model works.
    assert np.round(results["box_precision"],2) > 0.5
    assert np.round(results["box_recall"],2) > 0.5
    assert len(results["results"].predicted_label.dropna().unique()) == 1
    assert results["results"].predicted_label.dropna().unique()[0] == "Tree"
    assert results["predictions"].shape[0] > 0
    assert results["predictions"].label.dropna().unique()[0] == "Tree"
    
    df = pd.read_csv(csv_file)
    assert results["results"].shape[0] == df.shape[0]

def test_evaluate_multiple_images(m, tmpdir):
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
    
def test_train_callbacks(m):
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

def test_custom_config_file_path(tmpdir):
    print(os.getcwd())
    m = main.deepforest(config_file='tests/deepforest_config_test.yml')
    assert m.config["batch_size"] == 9999
    assert m.config["nms_thresh"] == 0.9
    assert m.config["score_thresh"] == 0.9

def test_save_and_reload_checkpoint(m, tmpdir):
    img_path = get_data(path="2019_YELL_2_528000_4978000_image_crop2.png")    
    m.config["train"]["fast_dev_run"] = True
    m.create_trainer()
    #save the prediction dataframe after training and compare with prediction after reload checkpoint 
    m.trainer.fit(m)    
    pred_after_train = m.predict_image(path = img_path)
    m.save_model("{}/checkpoint.pl".format(tmpdir))
    
    #reload the checkpoint to model object
    after = main.deepforest.load_from_checkpoint("{}/checkpoint.pl".format(tmpdir))
    pred_after_reload = after.predict_image(path = img_path)

    assert not pred_after_train.empty
    assert not pred_after_reload.empty
    pd.testing.assert_frame_equal(pred_after_train,pred_after_reload)

def test_save_and_reload_weights(m, tmpdir):
    img_path = get_data(path="2019_YELL_2_528000_4978000_image_crop2.png")    
    m.config["train"]["fast_dev_run"] = True
    m.create_trainer()
    #save the prediction dataframe after training and compare with prediction after reload checkpoint 
    m.trainer.fit(m)    
    pred_after_train = m.predict_image(path = img_path)
    torch.save(m.model.state_dict(),"{}/checkpoint.pt".format(tmpdir))
    
    #reload the checkpoint to model object
    after = main.deepforest()
    after.model.load_state_dict(torch.load("{}/checkpoint.pt".format(tmpdir)))
    pred_after_reload = after.predict_image(path = img_path)

    assert not pred_after_train.empty
    assert not pred_after_reload.empty
    pd.testing.assert_frame_equal(pred_after_train,pred_after_reload)
    
def test_reload_multi_class(two_class_m, tmpdir):
    two_class_m.config["train"]["fast_dev_run"] = True
    two_class_m.create_trainer()
    two_class_m.trainer.fit(two_class_m)
    two_class_m.save_model("{}/checkpoint.pl".format(tmpdir))
    before = two_class_m.trainer.validate(two_class_m)
    
    #reload
    old_model = main.deepforest.load_from_checkpoint("{}/checkpoint.pl".format(tmpdir))
    old_model.config = two_class_m.config
    assert old_model.num_classes == 2
    old_model.create_trainer()    
    after = old_model.trainer.validate(old_model)
    
    assert after[0]["val_classification"] == before[0]["val_classification"]
    
def test_override_transforms():
    def get_transform(augment):
        """This is the new transform"""
        if augment:
            print("I'm a new augmentation!")
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
    train_ds = m.load_dataset(csv_file, root_dir=root_dir, augment=True)
    
    path, image, target = next(iter(train_ds))
    assert m.transforms.__doc__ == "This is the new transform"

def test_over_score_thresh(m):
    """A user might want to change the config after model training and update the score thresh"""
    img = get_data("OSBS_029.png")
    original_score_thresh = m.model.score_thresh
    m.config["score_thresh"] = 0.8
    
    #trigger update
    boxes = m.predict_image(path = img)
    assert m.model.score_thresh == 0.8
    assert not m.model.score_thresh == original_score_thresh
    
    