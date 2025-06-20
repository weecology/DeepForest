# test main
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
import importlib.util

import albumentations as A
from albumentations.pytorch import ToTensorV2

from deepforest import main, get_data, model
from deepforest.utilities import read_file, format_geometry
from deepforest.datasets import prediction
from deepforest.visualize import plot_results 

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

import geopandas as gpd
from PIL import Image
import shapely

# Import release model from global script to avoid thrasing github during testing.
# Just download once.
from .conftest import download_release


@pytest.fixture()
def two_class_m():
    m = main.deepforest(config_args={"num_classes": 2},
                        label_dict={
                            "Alive": 0,
                            "Dead": 1
                        })
    m.config.train.csv_file = get_data("testfile_multi.csv")
    m.config.train.root_dir = os.path.dirname(get_data("testfile_multi.csv"))
    m.config.train.fast_dev_run = True
    m.config.batch_size = 2

    m.config.validation["csv_file"] = get_data("testfile_multi.csv")
    m.config.validation["root_dir"] = os.path.dirname(get_data("testfile_multi.csv"))
    m.config.validation["val_accuracy_interval"] = 1

    m.create_trainer()

    return m


@pytest.fixture()
def m(download_release):
    m = main.deepforest()
    m.config.train.csv_file = get_data("example.csv")
    m.config.train.root_dir = os.path.dirname(get_data("example.csv"))
    m.config.train.fast_dev_run = True
    m.config.batch_size = 2

    m.config.validation.csv_file = get_data("example.csv")
    m.config.validation.root_dir = os.path.dirname(get_data("example.csv"))
    m.config.workers= 0
    m.config.validation.val_accuracy_interval = 1
    m.config.train.epochs = 2

    m.create_trainer()
    m.load_model("weecology/deepforest-tree")

    return m


@pytest.fixture()
def m_without_release():
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
    return m


@pytest.fixture()
def path():
    return get_data(path='OSBS_029.tif')


@pytest.fixture()
def big_file():
    tmpdir = tempfile.gettempdir()
    csv_file = get_data("OSBS_029.csv")
    df = pd.read_csv(csv_file)

    big_frame = []
    for x in range(3):
        img = Image.open("{}/{}".format(os.path.dirname(csv_file),
                                        df.image_path.unique()[0]))
        cv2.imwrite("{}/{}.png".format(tmpdir, x), np.array(img))
        new_df = df.copy()
        new_df.image_path = "{}.png".format(x)
        big_frame.append(new_df)

    big_frame = pd.concat(big_frame)
    big_frame.to_csv("{}/annotations.csv".format(tmpdir))

    return "{}/annotations.csv".format(tmpdir)

def test_m_has_tree_model_loaded(m):
    boxes = m.predict_image(path=get_data("OSBS_029.tif"))
    assert not boxes.empty

def test_tensorboard_logger(m, tmpdir):
    # Check if TensorBoard is installed
    if importlib.util.find_spec("tensorboard"):
        # Create model trainer and fit model
        annotations_file = get_data("testfile_deepforest.csv")
        logger = TensorBoardLogger(save_dir=tmpdir)
        m.config.train.csv_file = annotations_file
        m.config.train.root_dir = os.path.dirname(annotations_file)
        m.config.train.fast_dev_run = False
        m.config.validation.csv_file = annotations_file
        m.config.validation.root_dir = os.path.dirname(annotations_file)
        m.config.validation.val_accuracy_interval = 1
        m.config.train.epochs = 2

        m.create_trainer(logger=logger, limit_train_batches=1, limit_val_batches=1)
        m.trainer.fit(m)

        assert m.trainer.logged_metrics["box_precision"]
        assert m.trainer.logged_metrics["box_recall"]
    else:
        print("TensorBoard is not installed. Skipping test_tensorboard_logger.")


def test_use_bird_release(m):
    imgpath = get_data("AWPE Pigeon Lake 2020 DJI_0005.JPG")
    m.load_model("Weecology/deepforest-bird")
    boxes = m.predict_image(path=imgpath)
    assert not boxes.empty

def test_load_model(m):
    imgpath = get_data("OSBS_029.png")
    m.load_model('ethanwhite/df-test')
    boxes = m.predict_image(path=imgpath)
    assert not boxes.empty


def test_train_empty(m, tmpdir):
    empty_csv = pd.DataFrame({
        "image_path": ["OSBS_029.png", "OSBS_029.tif"],
        "xmin": [0, 10],
        "xmax": [0, 20],
        "ymin": [0, 20],
        "ymax": [0, 30],
        "label": ["Tree", "Tree"]
    })
    empty_csv.to_csv("{}/empty.csv".format(tmpdir))
    m.config.train.csv_file = "{}/empty.csv".format(tmpdir)
    m.config.batch_size = 2
    m.create_trainer(fast_dev_run=True)
    m.trainer.fit(m)

def test_train_with_empty_validation(m, tmpdir):
    empty_csv = pd.DataFrame({
        "image_path": ["OSBS_029.png", "OSBS_029.tif"],
        "xmin": [0, 10],
        "xmax": [0, 20],
        "ymin": [0, 20],
        "ymax": [0, 30],
        "label": ["Tree", "Tree"]
    })
    empty_csv.to_csv("{}/empty.csv".format(tmpdir))
    m.config.train.csv_file = "{}/empty.csv".format(tmpdir)
    m.config.validation.csv_file = "{}/empty.csv".format(tmpdir)
    m.config.batch_size = 2
    m.create_trainer(fast_dev_run=True)
    m.trainer.fit(m)
    m.trainer.validate(m)


def test_validation_step(m):
    val_dataloader = m.val_dataloader()
    batch = next(iter(val_dataloader))
    m.predictions = []
    val_loss = m.validation_step(batch, 0)
    assert val_loss != 0

def test_validation_step_empty():
    """If the model returns an empty prediction, the metrics should not fail"""
    m = main.deepforest()
    m.config.validation["csv_file"] = get_data("example.csv")
    m.config.validation["root_dir"] = os.path.dirname(get_data("example.csv"))
    m.create_trainer()

    val_dataloader = m.val_dataloader()
    batch = next(iter(val_dataloader))
    m.predictions = []
    val_predictions = m.validation_step(batch, 0)
    assert m.iou_metric.compute()["iou"] == 0

def test_validate(m):
    m.trainer = None
    # Turn off trainer to test copying on some linux devices.
    before = copy.deepcopy(m)
    m.create_trainer()
    m.trainer.validate(m)
    # assert no weights have changed
    for p1, p2 in zip(before.named_parameters(), m.named_parameters()):
        assert p1[1].ne(p2[1]).sum() == 0


# Test train with each architecture
@pytest.mark.parametrize("architecture", ["retinanet", "FasterRCNN"])
def test_train_single(m_without_release, architecture):
    m_without_release.config.architecture = architecture
    m_without_release.create_model()
    m_without_release.config.train.fast_dev_run = False
    m_without_release.create_trainer(limit_train_batches=1)
    m_without_release.trainer.fit(m_without_release)


def test_train_preload_images(m):
    m.create_trainer(fast_dev_run=True)
    m.config.train.preload_images = True
    m.trainer.fit(m)


def test_train_geometry_column(m, tmpdir):
    """Test that training works with a geometry column from a shapefile"""

    # Get the source data
    df = read_file(get_data("OSBS_029.csv"))
    df["label"] = "Tree"

    # Create geodataframe with box geometries
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[
            shapely.geometry.box(xmin, ymin, xmax, ymax)
            for xmin, ymin, xmax, ymax in zip(df.xmin, df.ymin, df.xmax, df.ymax)
        ])

    # Add image path
    gdf["image_path"] = "OSBS_029.tif"

    gdf = gdf[["label", "geometry", "image_path"]]

    # Save to temporary shapefile with only a geometry and label column
    temp_shp = os.path.join(tmpdir, "temp_trees.shp")
    gdf.to_file(temp_shp)

    # Read shapefile using utilities
    df = read_file(temp_shp, root_dir=os.path.dirname(get_data("OSBS_029.tif")))
    df.to_csv(os.path.join(tmpdir, "OSBS_029.csv"), index=False)

    # Train model
    m.config.train.csv_file = os.path.join(tmpdir, "OSBS_029.csv")
    m.config.train.root_dir = os.path.dirname(get_data("OSBS_029.tif"))
    m.create_trainer(fast_dev_run=True)
    m.trainer.fit(m)

def test_train_multi(two_class_m):
    two_class_m.create_trainer(fast_dev_run=True)
    two_class_m.trainer.fit(two_class_m)

def test_train_no_validation(m):
    m.config.train.fast_dev_run = False
    m.config.validation["csv_file"] = None
    m.config.validation["root_dir"] = None
    m.create_trainer(limit_train_batches=1)
    m.trainer.fit(m)


def test_predict_image_empty(m):
    image = np.random.random((400, 400, 3)).astype("float32")
    prediction = m.predict_image(image=image)

    assert prediction is None


def test_predict_image_fromfile(m):
    path = get_data(path="2019_YELL_2_528000_4978000_image_crop2.png")
    prediction = m.predict_image(path=path)

    assert isinstance(prediction, pd.DataFrame)
    assert set(prediction.columns) == {
        "xmin", "ymin", "xmax", "ymax", "label", "score", "image_path", "geometry"
    }
    assert not prediction.empty


def test_predict_image_fromarray(m):
    image_path = get_data(path="2019_YELL_2_528000_4978000_image_crop2.png")

    # assert error of dtype
    with pytest.raises(TypeError):
        image = Image.open(image_path)
        prediction = m.predict_image(image=image)

    image = np.array(Image.open(image_path).convert("RGB"))
    with pytest.warns(UserWarning, match="Image type is uint8, transforming to float32"):
        prediction = m.predict_image(image=image)

    assert isinstance(prediction, pd.DataFrame)
    assert set(prediction.columns) == {"xmin", "ymin", "xmax", "ymax", "label", "score", "geometry"}
    assert not hasattr(prediction, 'root_dir')

def test_predict_big_file(m, big_file):
    m.config.train.fast_dev_run = False
    m.create_trainer()
    df = m.predict_file(csv_file=big_file,
                        root_dir=os.path.dirname(big_file))
    assert set(df.columns) == {
        'label', 'score', 'image_path', 'geometry', "xmin", "ymin", "xmax", "ymax"
    }

def test_predict_small_file(m):
    csv_file = get_data("OSBS_029.csv")
    df = m.predict_file(csv_file, root_dir=os.path.dirname(csv_file))
    assert set(df.columns) == {
        'label', 'score', 'image_path', 'geometry', "xmin", "ymin", "xmax", "ymax"
    }

@pytest.mark.parametrize("batch_size", [1, 2])
def test_predict_dataloader(m, batch_size, path):
    m.config.batch_size = batch_size
    tile = np.array(Image.open(path))
    ds = prediction.SingleImage(image=tile, path=path, patch_overlap=0.1, patch_size=100)
    dl = m.predict_dataloader(ds)
    batch = next(iter(dl))
    assert batch.shape[0] == batch_size

def test_predict_tile_empty(path):
    # Random weights
    m = main.deepforest()
    predictions = m.predict_tile(path=path, patch_size=300, patch_overlap=0)
    assert predictions is None

@pytest.mark.parametrize("dataloader_strategy", ["single", "window", "batch"])
def test_predict_tile(m, path, dataloader_strategy):
    m.create_model()
    m.config.train.fast_dev_run = False
    m.create_trainer()
    m.load_model("weecology/deepforest-tree")

    if dataloader_strategy == "single":
        image_path = path
    elif dataloader_strategy == "window":
        image_path = get_data("test_tiled.tif")
    else:
        image_path = [path]

    prediction = m.predict_tile(path=image_path,
                                patch_size=300,
                                dataloader_strategy=dataloader_strategy,
                                patch_overlap=0.1)

    assert isinstance(prediction, pd.DataFrame)
    assert set(prediction.columns) == {
        "xmin", "ymin", "xmax", "ymax", "label", "score", "image_path", "geometry"
    }
    assert not prediction.empty

    # Assert there are predictions in each corner of the image
    assert prediction.xmin.min() < 50
    assert prediction.xmin.max() > 350
    assert prediction.ymin.min() < 50
    assert prediction.ymin.max() > 350

    plot_results(prediction)


# Add predict_tile for serial single dataloader strategy
def test_predict_tile_serial_single(m):
    path1 = get_data("OSBS_029.png")
    path2 = get_data("SOAP_031.png")
    m.create_model()
    m.config.train.fast_dev_run = False
    m.create_trainer()
    m.load_model("weecology/deepforest-tree")
    prediction = m.predict_tile(path=[path1, path2], patch_size=300, patch_overlap=0, dataloader_strategy="batch")
    assert prediction.image_path.unique().tolist() == [os.path.basename(path1), os.path.basename(path2)]

    # view the predictions of each image
    prediction_1 = prediction[prediction.image_path == os.path.basename(path1)]
    prediction_1.root_dir = os.path.dirname(path1)
    prediction_2 = prediction[prediction.image_path == os.path.basename(path2)]
    prediction_2.root_dir = os.path.dirname(path2)

    plot_results(prediction_1)
    plot_results(prediction_2)

# test equivalence for within and out of memory dataset strategies
def test_predict_tile_equivalence(m):
    path = get_data("test_tiled.tif")
    in_memory_prediction = m.predict_tile(path=path, patch_size=300, patch_overlap=0, dataloader_strategy="single")
    not_in_memory_prediction = m.predict_tile(path=path, patch_size=300, patch_overlap=0, dataloader_strategy="window")

    # Assert same number of predictions
    assert len(in_memory_prediction) == len(not_in_memory_prediction)

def test_predict_tile_from_array(m, path):
    image = np.array(Image.open(path))
    m.config.train.fast_dev_run = False
    m.create_trainer()
    prediction = m.predict_tile(image=image, patch_size=300)

    assert not prediction.empty    

def test_evaluate(m, tmpdir):
    csv_file = get_data("OSBS_029.csv")
    results = m.evaluate(csv_file, iou_threshold=0.4)

    assert np.round(results["box_precision"], 2) > 0.5
    assert np.round(results["box_recall"], 2) > 0.5
    assert len(results["results"].predicted_label.dropna().unique()) == 1
    assert results["results"].predicted_label.dropna().unique()[0] == "Tree"
    assert results["predictions"].shape[0] > 0
    assert results["predictions"].label.dropna().unique()[0] == "Tree"

    df = pd.read_csv(csv_file)
    assert results["results"].shape[0] == df.shape[0]


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


def test_checkpoint_label_dict(m, tmpdir):
    """Test that the label dict is saved and loaded correctly from a checkpoint."""
    csv_file = get_data("example.csv")
    df = pd.read_csv(csv_file)
    df["label"] = "Object"

    #write to tmpdir
    df.to_csv(os.path.join(tmpdir, "example.csv"), index=False)

    m.config["train"]["csv_file"] = os.path.join(tmpdir, "example.csv")
    m.config["train"]["root_dir"] = os.path.dirname(csv_file)
    m.config["validation"]["csv_file"] = os.path.join(tmpdir, "example.csv")
    m.config["validation"]["root_dir"] = os.path.dirname(csv_file)
    
    m.config.train.fast_dev_run = True
    m.create_trainer()
    m.label_dict = {"Object": 0}
    m.numeric_to_label_dict = {0: "Object"}
    m.trainer.fit(m)
    m.trainer.save_checkpoint("{}/checkpoint.pl".format(tmpdir))
    after = main.deepforest.load_from_checkpoint("{}/checkpoint.pl".format(tmpdir))
    assert after.label_dict == {"Object": 0}
    assert after.numeric_to_label_dict == {0: "Object"}

def test_save_and_reload_checkpoint(m, tmpdir):
    img_path = get_data(path="2019_YELL_2_528000_4978000_image_crop2.png")
    m.config.train.fast_dev_run = True
    m.create_trainer()
    # save the prediction dataframe after training and
    # compare with prediction after reload checkpoint
    m.trainer.fit(m)
    pred_after_train = m.predict_image(path=img_path)
    m.save_model("{}/checkpoint.pl".format(tmpdir))

    # reload the checkpoint to model object
    after = main.deepforest.load_from_checkpoint("{}/checkpoint.pl".format(tmpdir))
    pred_after_reload = after.predict_image(path=img_path)

    assert not pred_after_train.empty
    assert not pred_after_reload.empty
    pd.testing.assert_frame_equal(pred_after_train, pred_after_reload)


def test_save_and_reload_weights(m, tmpdir):
    img_path = get_data(path="2019_YELL_2_528000_4978000_image_crop2.png")
    m.config.train.fast_dev_run = True
    m.create_trainer()
    # save the prediction dataframe after training and
    # compare with prediction after reload checkpoint
    m.trainer.fit(m)
    pred_after_train = m.predict_image(path=img_path)
    torch.save(m.model.state_dict(), "{}/checkpoint.pt".format(tmpdir))

    # reload the checkpoint to model object
    after = main.deepforest()
    after.model.load_state_dict(
        torch.load("{}/checkpoint.pt".format(tmpdir), weights_only=True))
    pred_after_reload = after.predict_image(path=img_path)

    assert not pred_after_train.empty
    assert not pred_after_reload.empty
    pd.testing.assert_frame_equal(pred_after_train, pred_after_reload)


def test_reload_multi_class(two_class_m, tmpdir):
    two_class_m.config.train.fast_dev_run = True
    two_class_m.create_trainer()
    two_class_m.trainer.fit(two_class_m)
    two_class_m.save_model("{}/checkpoint.pl".format(tmpdir))
    before = two_class_m.trainer.validate(two_class_m)

    # reload
    old_model = main.deepforest.load_from_checkpoint("{}/checkpoint.pl".format(tmpdir),
                                                     weights_only=True)
    old_model.config = two_class_m.config
    assert old_model.config.num_classes == 2
    old_model.create_trainer()
    after = old_model.trainer.validate(old_model)

    assert after[0]["val_classification"] == before[0]["val_classification"]


def test_override_transforms():

    def get_transform(augment):
        """This is the new transform"""
        if augment:
            print("I'm a new augmentation!")
            transform = A.Compose(
                [A.HorizontalFlip(p=0.5), ToTensorV2()],
                bbox_params=A.BboxParams(format='pascal_voc',
                                         label_fields=["category_ids"]))

        else:
            transform = ToTensorV2()
        return transform

    m = main.deepforest(transforms=get_transform)

    csv_file = get_data("example.csv")
    root_dir = os.path.dirname(csv_file)
    train_ds = m.load_dataset(csv_file, root_dir=root_dir, augment=True)

    image, target, path = next(iter(train_ds))
    assert m.transforms.__doc__ == "This is the new transform"

#TODO: Fix this test to check that predictions change as checking
# if the threshold is changed in the config is probably not what
# we actually want to test.
@pytest.mark.xfail
def test_over_score_thresh(m):
    """A user might want to change the config after model training and update the score thresh"""
    img = get_data("OSBS_029.png")
    original_score_thresh = m.model.score_thresh
    m.model.score_thresh = 0.8

    # trigger update
    boxes = m.predict_image(path=img)

    assert all(boxes.score > 0.8)
    assert m.model.score_thresh == 0.8
    assert not m.model.score_thresh == original_score_thresh


def test_iou_metric(m):
    results = m.trainer.validate(m)
    keys = ['val_classification', 'val_bbox_regression', 'iou', 'iou/cl_0']
    for x in keys:
        assert x in list(results[0].keys())


def test_config_args(m):
    assert not m.config.num_classes == 2

    m = main.deepforest(config_args={"num_classes": 2},
                        label_dict={
                            "Alive": 0,
                            "Dead": 1
                        })
    assert m.config.num_classes == 2

    # These call also be nested for train and val arguments
    assert not m.config.train.epochs == 7

    m2 = main.deepforest(config_args={"train": {"epochs": 7}})
    assert m2.config.train.epochs == 7


@pytest.fixture()
def existing_loader(m, tmpdir):
    # Create dummy loader with a different batch size to assert, we'll need a few more images to assess
    train = pd.read_csv(m.config.train.csv_file)
    train2 = train.copy(deep=True)
    train3 = train.copy(deep=True)
    train2.image_path = train2.image_path + "2"
    train3.image_path = train3.image_path + "3"
    pd.concat([train, train2, train3]).to_csv("{}/train.csv".format(tmpdir.strpath))

    # Copy the new images to the tmpdir
    train.image_path.unique()
    image_path = train.image_path.unique()[0]
    shutil.copyfile("{}/{}".format(m.config.train.root_dir, image_path),
                    tmpdir.strpath + "/{}".format(image_path))
    shutil.copyfile("{}/{}".format(m.config.train.root_dir, image_path),
                    tmpdir.strpath + "/{}".format(image_path + "2"))
    shutil.copyfile("{}/{}".format(m.config.train.root_dir, image_path),
                    tmpdir.strpath + "/{}".format(image_path + "3"))
    existing_loader = m.load_dataset(csv_file="{}/train.csv".format(tmpdir.strpath),
                                     root_dir=tmpdir.strpath,
                                     batch_size=m.config.batch_size + 1)
    return existing_loader


def test_load_existing_train_dataloader(m, tmpdir, existing_loader):
    """Allow the user to optionally create a dataloader outside
    of the DeepForest class, ensure this works for train/val/predict
    """
    # Inspect original for comparison of batch size
    m.config.train.csv_file = "{}/train.csv".format(tmpdir.strpath)
    m.config.train.root_dir = tmpdir.strpath
    batch = next(iter(m.train_dataloader()))
    assert len(batch[0]) == m.config.batch_size

    # Existing train dataloader
    m.config.train.csv_file = "{}/train.csv".format(tmpdir.strpath)
    m.config.train.root_dir = tmpdir.strpath
    m.existing_train_dataloader = existing_loader
    m.create_trainer(fast_dev_run=True)
    m.trainer.fit(m)
    batch = next(iter(m.train_dataloader()))
    assert len(batch[0]) == m.config.batch_size + 1


def test_existing_val_dataloader(m, tmpdir, existing_loader):
    m.config.validation["csv_file"] = "{}/train.csv".format(tmpdir.strpath)
    m.config.validation["root_dir"] = tmpdir.strpath
    m.existing_val_dataloader = existing_loader
    m.create_trainer()
    m.trainer.validate(m)
    batch = next(iter(m.val_dataloader()))
    assert len(batch[0]) == m.config.batch_size + 1


def test_existing_predict_dataloader(m, tmpdir):
    # Predict datasets yield only images
    ds = prediction.TiledRaster(path=get_data("test_tiled.tif"),
                             patch_overlap=0.1,
                             patch_size=100)
    existing_loader = m.predict_dataloader(ds)
    batches = m.trainer.predict(m, existing_loader)
    len(batches[0]) == m.config.batch_size + 1


# Test train with each scheduler
@pytest.mark.parametrize("scheduler,expected",
                         [("cosine", "CosineAnnealingLR"), ("lambdaLR", "LambdaLR"),
                            ("stepLR", "StepLR"),
                          ("multistepLR", "MultiStepLR"),
                          ("reduceLROnPlateau", "ReduceLROnPlateau")])
def test_configure_optimizers(scheduler, expected):
    scheduler_config = {
        "type": scheduler,
        "params": {
            "T_max": 10,
            "eta_min": 0.00001,
            "lr_lambda": "0.95**epoch",  # For lambdaLR and multiplicativeLR
            "step_size": 30,  # For stepLR
            "gamma": 0.1,  # For stepLR, multistepLR, and exponentialLR
            "milestones": [50, 100],  # For multistepLR

            # ReduceLROnPlateau parameters (used if type is not explicitly mentioned)
            "mode": "min",
            "factor": 0.1,
            "patience": 10,
            "threshold": 0.0001,
            "threshold_mode": "rel",
            "cooldown": 0,
            "min_lr": 0,
            "eps": 1e-08
        }
    }

    annotations_file = get_data("testfile_deepforest.csv")
    root_dir = os.path.dirname(annotations_file)

    config_args = {
        "train": {
            "lr": 0.01,
            "scheduler": scheduler_config,
            "csv_file": annotations_file,
            "root_dir": root_dir,
            "fast_dev_run": False,
        },
        "validation": {
            "csv_file": annotations_file,
            "root_dir": root_dir
        }
    }

    # Initialize the model with the config arguments
    m = main.deepforest(config_args=config_args)

    # Create and run the trainer
    m.create_trainer(limit_train_batches=1.0)
    m.trainer.fit(m)

    # Assert the scheduler type
    assert type(m.trainer.lr_scheduler_configs[0].scheduler).__name__ == expected, f"Scheduler type mismatch for {scheduler_config['type']}"


@pytest.fixture()
def crop_model():
    return model.CropModel(num_classes=2)


def test_predict_tile_with_crop_model(m, config):
    path = get_data("SOAP_061.png")
    patch_size = 400
    patch_overlap = 0.05
    iou_threshold = 0.15
    # Set up the crop model
    crop_model = model.CropModel(num_classes=2, label_dict = {"Dead":0, "Alive":1})

    # Call the predict_tile method with the crop_model
    m.config.train.fast_dev_run = False
    m.create_trainer()
    result = m.predict_tile(path=path,
                            patch_size=patch_size,
                            patch_overlap=patch_overlap,
                            iou_threshold=iou_threshold,
                            crop_model=crop_model)

    # Assert the result
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {
        "xmin", "ymin", "xmax", "ymax", "label", "score", "cropmodel_label", "geometry",
        "cropmodel_score", "image_path"
    }

    # Assert cropmodel_label is in the label_dict
    labels = [x for x in crop_model.label_dict]
    assert result.cropmodel_label.isin(labels).all()


def test_predict_tile_with_crop_model_empty():
    """If the model return is empty, the crop model should return an empty dataframe"""
    path = get_data("SOAP_061.png")
    m = main.deepforest()
    
    # Set up the crop model
    crop_model = model.CropModel(num_classes=2, label_dict = {"Dead": 0, "Alive": 1})
    
    # Call the predict_tile method with the crop_model
    m.config.train.fast_dev_run = False
    m.create_trainer()
    result = m.predict_tile(path=path,
                            patch_size=400,
                            patch_overlap=0.05,
                            iou_threshold=0.15,
                            crop_model=crop_model)
    

    # Assert the result
    assert result is None or result.empty

def test_predict_tile_with_multiple_crop_models(m, config):
    path = get_data("SOAP_061.png")
    patch_size = 400
    patch_overlap = 0.05
    iou_threshold = 0.15

    # Create multiple crop models
    crop_model = [model.CropModel(num_classes=2, label_dict={"Dead":0, "Alive":1}), model.CropModel(num_classes=3, label_dict={"Dead":0, "Alive":1, "Sapling":2})]

    # Call predict_tile with multiple crop models
    m.config.train.fast_dev_run = False
    m.create_trainer()
    result = m.predict_tile(path=path,
                            patch_size=patch_size,
                            patch_overlap=patch_overlap,
                            iou_threshold=iou_threshold,
                            crop_model=crop_model)

    # Assert result type
    assert isinstance(result, pd.DataFrame)

    # Check column names dynamically for multiple crop models
    expected_cols = {"xmin", "ymin", "xmax", "ymax", "label", "score", "geometry", "image_path"}
    for i in range(len(crop_model)):
        expected_cols.add(f"cropmodel_label_{i}")
        expected_cols.add(f"cropmodel_score_{i}")

    assert set(result.columns) == expected_cols
    assert not result.empty


def test_predict_tile_with_multiple_crop_models_empty():
    """If no predictions are made, result should be empty"""
    path = get_data("SOAP_061.png")
    m = main.deepforest()

    # Create multiple crop models
    crop_model_1 = model.CropModel(num_classes=2, label_dict={"Dead":0, "Alive":1})
    crop_model_2 = model.CropModel(num_classes=3, label_dict={"Dead":0, "Alive":1, "Sapling":2})

    m.config.train.fast_dev_run = False
    m.create_trainer()
    result = m.predict_tile(path=path,
                            patch_size=400,
                            patch_overlap=0.05,
                            iou_threshold=0.15,
                            crop_model=[crop_model_1, crop_model_2])

    assert result is None or result.empty  # Ensure empty result is handled properly

def test_batch_prediction(m, path):
    # Prepare input data
    ds = prediction.SingleImage(path=path, patch_overlap=0.1, patch_size=300)
    dl = DataLoader(ds, batch_size=3)

    # Perform prediction
    predictions = []
    for batch in dl:
        batch_predictions = m.predict_batch(batch)
        predictions.extend(batch_predictions)

    # Check results  
    assert len(predictions) == len(ds)
    for image_pred in predictions:
        assert isinstance(image_pred, pd.DataFrame)
        assert "label" in image_pred.columns
        assert "score" in image_pred.columns
        assert "geometry" in image_pred.columns

def test_batch_inference_consistency(m, path):
    ds = prediction.SingleImage(path=path, patch_overlap=0.1, patch_size=300)
    dl = DataLoader(ds, batch_size=4)

    batch_predictions = []
    for batch in dl:
        batch_prediction = m.predict_batch(batch)
        batch_predictions.extend(batch_prediction)

    single_predictions = []
    for image in ds:
        image = np.rollaxis(image, 0, 3) * 255
        single_prediction = m.predict_image(image=image)
        single_predictions.append(single_prediction)

    batch_df = pd.concat(batch_predictions, ignore_index=True)
    single_df = pd.concat(single_predictions, ignore_index=True)

    # Make all xmin, ymin, xmax, ymax integers
    for col in ["xmin", "ymin", "xmax", "ymax"]:
        batch_df[col] = batch_df[col].astype(int)
        single_df[col] = single_df[col].astype(int)
    pd.testing.assert_frame_equal(batch_df[["xmin", "ymin", "xmax", "ymax"]], single_df[["xmin", "ymin", "xmax", "ymax"]], check_dtype=False)


def test_epoch_evaluation_end(m, tmpdir):
    """Test the epoch evaluation end method by """
    preds = [{
        'boxes': torch.tensor([
            [690.3572, 902.9113, 781.1031, 996.5151],
            [998.1990, 655.7919, 172.4619, 321.8518]
        ]),
        'scores': torch.tensor([
            1.0, 1.0
        ]),
        'labels': torch.tensor([
            0, 0
        ])
    }]
    targets = preds

    m.iou_metric.update(preds, targets)
    m.mAP_metric.update(preds, targets)

    boxes = format_geometry(preds[0])
    boxes["image_path"] = "test"

    predictions = boxes.copy()
    assert m.iou_metric.compute()["iou"] == 1.0

    # write a csv file to the tmpdir
    boxes["label"] = "Tree"
    m.predictions = [predictions]
    boxes.to_csv(tmpdir.strpath + "/predictions.csv", index=False)
    m.config.validation.csv_file = tmpdir.strpath + "/predictions.csv"
    m.config.validation.root_dir = tmpdir.strpath

    results = m.on_validation_epoch_end()

    assert results["box_precision"] == 1.0
    assert results["box_recall"] == 1.0

def test_epoch_evaluation_end_empty(m):
    """If the model returns an empty prediction, the metrics should not fail"""
    preds = [{
        'boxes': torch.zeros((1, 4)),
        'scores': torch.zeros(1),
        'labels': torch.zeros(1, dtype=torch.int64)
    }]
    targets = preds

    m.iou_metric.update(preds, targets)
    m.mAP_metric.update(preds, targets)

    boxes = format_geometry(preds[0])
    boxes["image_path"] = "test"
    m.predictions = [boxes]
    m.on_validation_epoch_end()

def test_empty_frame_accuracy_all_empty_with_predictions(m, tmpdir):
    """Test empty frame accuracy when all frames are empty but model predicts objects.
    The accuracy should be 0 since model incorrectly predicts objects in empty frames."""
    # Create ground truth with all empty frames
    ground_df = pd.read_csv(get_data("testfile_deepforest.csv"))
    # Set all xmin, ymin, xmax, ymax to 0 to mark as empty
    ground_df.loc[:, ["xmin", "ymin", "xmax", "ymax"]] = 0
    ground_df.drop_duplicates(subset=["image_path"], keep="first", inplace=True)

    # Save the ground truth to a temporary file
    ground_df.to_csv(tmpdir.strpath + "/ground_truth.csv", index=False)
    m.config.validation["csv_file"] = tmpdir.strpath + "/ground_truth.csv"
    m.config.validation["root_dir"] = os.path.dirname(get_data("testfile_deepforest.csv"))

    m.create_trainer()
    results = m.trainer.validate(m)
    
    # This is bit of a preference, if there are no predictions, the empty frame accuracy should be 0, precision is 0, and accuracy is None.
    assert results[0]["empty_frame_accuracy"] == 0.0
    assert results[0]["box_precision"] == 0.0

def test_empty_frame_accuracy_mixed_frames_with_predictions(m, tmpdir):
    """Test empty frame accuracy with a mix of empty and non-empty frames.
    Model predicts objects in all frames, so accuracy for empty frames should be 0."""
    # Create ground truth with mix of empty and non-empty frames
    tree_ground_df = pd.read_csv(get_data("testfile_deepforest.csv"))
    empty_ground_df = pd.DataFrame({
        "image_path": ["AWPE Pigeon Lake 2020 DJI_0005.JPG"],
        "xmin": [0],
        "ymin": [0],
        "xmax": [0],
        "ymax": [0],
        "label": ["Tree"]
    })

    ground_df = pd.concat([tree_ground_df, empty_ground_df])

    # Save the ground truth to a temporary file
    ground_df.to_csv(tmpdir.strpath + "/ground_truth.csv", index=False)
    m.config.validation.csv_file = tmpdir.strpath + "/ground_truth.csv"
    m.config.validation.root_dir = os.path.dirname(get_data("testfile_deepforest.csv"))
    m.config.validation.size = 400

    m.create_trainer()
    results = m.trainer.validate(m)
    assert results[0]["empty_frame_accuracy"] == 0

def test_empty_frame_accuracy_without_predictions(tmpdir):
    """Create a ground truth with empty frames, the accuracy should be 1 with a random model"""
    m = main.deepforest()
    # Create ground truth with empty frames
    ground_df = pd.read_csv(get_data("testfile_deepforest.csv"))
    # Set all xmin, ymin, xmax, ymax to 0
    ground_df.loc[:, ["xmin", "ymin", "xmax", "ymax"]] = 0
    ground_df.drop_duplicates(subset=["image_path"], keep="first", inplace=True)

    # Save the ground truth to a temporary file
    ground_df.to_csv(tmpdir.strpath + "/ground_truth.csv", index=False)
    m.config.validation["csv_file"] = tmpdir.strpath + "/ground_truth.csv"
    m.config.validation["root_dir"] = os.path.dirname(get_data("testfile_deepforest.csv"))

    m.create_trainer()
    results = m.trainer.validate(m)
    assert results[0]["empty_frame_accuracy"] == 1

def test_multi_class_with_empty_frame_accuracy_without_predictions(two_class_m, tmpdir):
    """Create a ground truth with empty frames, the accuracy should be 1 with a random model"""
    # Create ground truth with empty frames
    ground_df = pd.read_csv(get_data("testfile_deepforest.csv"))
    # Set all xmin, ymin, xmax, ymax to 0
    ground_df.loc[:, ["xmin", "ymin", "xmax", "ymax"]] = 0
    ground_df.drop_duplicates(subset=["image_path"], keep="first", inplace=True)
    ground_df.loc[:, "label"] = "Alive"

    # Merge with a multi class ground truth
    multi_class_df = pd.read_csv(get_data("testfile_multi.csv"))
    ground_df = pd.concat([ground_df, multi_class_df])

    # Save the ground truth to a temporary file
    ground_df.to_csv(tmpdir.strpath + "/ground_truth.csv", index=False)
    two_class_m.config.validation["csv_file"] = tmpdir.strpath + "/ground_truth.csv"
    two_class_m.config.validation["root_dir"] = os.path.dirname(get_data("testfile_deepforest.csv"))

    two_class_m.create_trainer()
    results = two_class_m.trainer.validate(two_class_m)
    assert results[0]["empty_frame_accuracy"] == 1

def test_evaluate_on_epoch_interval(m):
    m.config.validation.val_accuracy_interval = 1
    m.config.train.epochs = 1
    m.create_trainer()
    m.trainer.fit(m)
    assert m.trainer.logged_metrics["box_precision"]
    assert m.trainer.logged_metrics["box_recall"]

def test_set_labels_updates_mapping(m):
    new_mapping = {"Object": 0}
    m.set_labels(new_mapping)

    # Verify that the label dictionary has been updated.
    assert m.label_dict == new_mapping

    # Verify that the inverse mapping is correctly computed.
    expected_inverse = {0: "Object"}
    assert m.numeric_to_label_dict == expected_inverse

def test_set_labels_invalid_length(m): # Expect a ValueError when setting an invalid label mapping.
    # This mapping has two entries, which should be invalid since m.config.num_classes is 1.
    invalid_mapping = {"Object": 0, "Extra": 1}
    with pytest.raises(ValueError):
        m.set_labels(invalid_mapping)
